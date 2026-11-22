"""Synthetic GIS checks for area, NoData, projected slope and grid alignment."""

from contextlib import ExitStack
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pyproj import CRS
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from prepare_rs_static import qa_figure, quality_flags
from rs_gis import (extract_cell, grid_cell, horn_slope, landcover_statistics,
                    native_projection, projected_pixel_weights, regular_pixel_weights,
                    warp_mosaic, weighted_statistics)
from temporal_folds import project_path


class StaticNumericalTests(unittest.TestCase):
    def test_area_weighted_landcover_preserves_unknown_and_missing(self):
        values = np.array([[10., 80.], [0., 255.]])
        valid = np.array([[True, True], [False, True]])
        weights = np.array([[1., 3.], [2., 4.]])
        fractions, area, unknown, missing = landcover_statistics(values, valid, weights)
        self.assertEqual(fractions[0], .25)
        self.assertEqual(fractions[7], .75)
        self.assertEqual(fractions.sum(), 1.)
        self.assertEqual((area, unknown, missing), (4., 4., 2.))

    def test_height_zero_is_valid_and_no_missing_imputation(self):
        mean, sd, area = weighted_statistics(np.array([0., 10., -9999.]),
                                              np.array([True, True, False]),
                                              np.array([1., 3., 1000.]))
        self.assertEqual(area, 4.)
        self.assertEqual(mean, 7.5)
        self.assertAlmostEqual(sd, np.sqrt(18.75))
        self.assertTrue(np.isnan(weighted_statistics(np.array([0.]), np.array([False]), np.array([1.]))[0]))

    def test_horn_plane_and_holes(self):
        x, y = np.meshgrid(np.arange(7)*30., np.arange(7)*30.)
        slope = horn_slope(.1*x+.2*y+100, 30)
        np.testing.assert_allclose(slope[1:-1, 1:-1], np.rad2deg(np.arctan(np.hypot(.1, .2))))
        z = np.zeros((7, 7));z[3, 3] = np.nan
        result = horn_slope(z, 30)
        self.assertEqual(np.isfinite(result).sum(), 16)
        self.assertTrue(np.isnan(result[2:5, 2:5]).all())

    def test_fractional_pixel_intersection(self):
        affine = from_origin(0, 2, 1, 1)
        bounds = (.25, .5, 1.75, 1.75)
        weights = projected_pixel_weights(affine, (2, 2), 'EPSG:3857', 'EPSG:3857', bounds)
        expected = np.array([[.5625, .5625], [.375, .375]])
        np.testing.assert_allclose(weights, expected)
        np.testing.assert_allclose(regular_pixel_weights(affine, (2, 2), bounds), expected)

    def test_native_grid_uses_first_point_and_scan_direction(self):
        header = dict(gridType='lambert', scanningMode=64, shapeOfTheEarth=6,
                      Latin1InDegrees=50., Latin2InDegrees=50., LaDInDegrees=50.,
                      LoVInDegrees=8., radius=6371229,
                      longitudeOfFirstGridPointInDegrees=342.514057,
                      latitudeOfFirstGridPointInDegrees=20.292281,
                      DxInMetres=5500., DyInMetres=5500.)
        crs, projection, origin = native_projection(header)
        from types import SimpleNamespace
        lon, lat = projection(origin[0]+542*5500, origin[1]+539*5500, inverse=True)
        cell = grid_cell(SimpleNamespace(center_x=542, center_y=539, grid_lon=lon, grid_lat=lat),
                         header, projection, origin)
        self.assertLess(cell['center_error_m'], 1e-6)
        self.assertAlmostEqual((cell['bounds'][2]-cell['bounds'][0])*(cell['bounds'][3]-cell['bounds'][1]), 5500**2)
        with self.assertRaises(ValueError):
            native_projection(dict(header, scanningMode=0))

    def test_complete_memory_rasters_no_zero_as_nodata(self):
        transform = from_origin(-.01, .01, .002, .002)
        land = np.full((10, 10), 10, dtype=np.uint8);land[:, 5:] = 80
        dsm = np.zeros((10, 10), dtype=np.float32)
        with ExitStack() as stack:
            datasets = []
            for values, nodata in [(land, 0), (dsm, -9999.)]:
                memory = stack.enter_context(MemoryFile())
                with memory.open(driver='GTiff', width=10, height=10, count=1,
                                 dtype=values.dtype, crs='EPSG:4326', transform=transform, nodata=nodata) as writer:
                    writer.write(values, 1)
                dataset = stack.enter_context(memory.open())
                datasets.append(dataset)
            cell = dict(bounds=(-1000., -1000., 1000., 1000.), geo_bounds=(-.012, -.012, .012, .012))
            features, quality, _ = extract_cell(cell, CRS.from_epsg(3857), [datasets[0]], [datasets[1]], 30)
            self.assertAlmostEqual(features['lc_10'], .5, places=7)
            self.assertAlmostEqual(features['lc_80'], .5, places=7)
            self.assertEqual(features['dsm_mean_m'], 0)
            self.assertEqual(features['dsm_sd_m'], 0)
            self.assertEqual(features['dsm_slope_mean_deg'], 0)
            self.assertAlmostEqual(quality['dsm_valid_fraction'], 1, places=7)
            self.assertAlmostEqual(quality['slope_valid_fraction'], 1, places=7)
            config = json.loads(project_path('configs/rs_hpc_v3_p2_static.json').read_text())
            self.assertEqual(quality_flags(features, quality, 0, config), [])
            quality['lc_valid_fraction'] = .8
            self.assertIn('lc_coverage', quality_flags(features, quality, 0, config))

    def test_adjacent_tiles_without_declared_nodata_do_not_fill_with_zero(self):
        with ExitStack() as stack:
            sources = []
            for left, height in [(0, 0.), (2, 100.)]:
                memory = stack.enter_context(MemoryFile())
                with memory.open(driver='GTiff', width=2, height=2, count=1, dtype='float32',
                                 crs='EPSG:3857', transform=from_origin(left, 2, 1, 1)) as writer:
                    writer.write(np.full((2, 2), height, dtype=np.float32), 1)
                sources.append(stack.enter_context(memory.open()))
            combined, _ = warp_mosaic(sources, CRS.from_epsg(3857), from_origin(0, 2, 1, 1), (2, 4))
            np.testing.assert_allclose(combined, [[0, 0, 100, 100], [0, 0, 100, 100]])

    def test_spherical_cerra_warp_of_geographic_raster(self):
        header = dict(gridType='lambert', scanningMode=64, shapeOfTheEarth=6,
                      Latin1InDegrees=50., Latin2InDegrees=50., LaDInDegrees=50.,
                      LoVInDegrees=8., radius=6371229,
                      longitudeOfFirstGridPointInDegrees=342.514057,
                      latitudeOfFirstGridPointInDegrees=20.292281,
                      DxInMetres=5500., DyInMetres=5500.)
        crs, projection, _ = native_projection(header)
        x, y = projection(10, 50)
        cell = dict(x=x, y=y, bounds=(x-1000, y-1000, x+1000, y+1000),
                    geo_bounds=(9.98, 49.98, 10.02, 50.02))
        with ExitStack() as stack:
            sources = []
            for dtype, value in [('uint8', 10), ('float32', 100.)]:
                memory = stack.enter_context(MemoryFile())
                with memory.open(driver='GTiff', width=20, height=20, count=1, dtype=dtype,
                                 crs='EPSG:4326', transform=from_origin(9.98, 50.02, .002, .002)) as writer:
                    writer.write(np.full((20, 20), value, dtype=dtype), 1)
                sources.append(stack.enter_context(memory.open()))
            features, quality, _ = extract_cell(cell, crs, [sources[0]], [sources[1]], 30)
            self.assertAlmostEqual(features['lc_10'], 1)
            self.assertAlmostEqual(features['dsm_mean_m'], 100, places=6)
            self.assertAlmostEqual(features['dsm_slope_mean_deg'], 0, places=6)
            self.assertAlmostEqual(quality['slope_valid_fraction'], 1, places=6)
            config = json.loads(project_path('configs/rs_hpc_v3_p2_static.json').read_text())
            with tempfile.TemporaryDirectory(dir=project_path('.')) as folder:
                destination = Path(folder)/'qa_example'
                qa_figure(cell, crs, [sources[0]], [sources[1]], config, destination, 'Synthetic cell')
                self.assertGreater(destination.with_suffix('.png').stat().st_size, 0)
                self.assertGreater(destination.with_suffix('.svg').stat().st_size, 0)


if __name__ == '__main__':
    unittest.main()
