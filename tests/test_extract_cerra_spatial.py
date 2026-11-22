"""Small synthetic checks; never opens the real CERRA GRIB."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

from extract_cerra_spatial import build_mapping, extract, parse_args, prepare_time_axis


class SpatialExtractionTests(unittest.TestCase):
    def setUp(self):
        self.args = parse_args(['--num_points', '2', '--far_offset', '3'])
        y, x = np.mgrid[:21, :21]
        self.lat = 48 + y * 0.25
        self.lon = 7 + x * 0.3
        self.dates = pd.date_range('2024-06-01', periods=4, freq='3h')
        self.values = np.arange(4 * 21 * 21, dtype=float).reshape(4, 21, 21)
        self.ds = xr.Dataset(
            {'si10': (('time', 'y', 'x'), self.values, {'units': 'm s**-1'})},
            coords={'time': self.dates, 'valid_time': ('time', self.dates),
                    'latitude': (('y', 'x'), self.lat),
                    'longitude': (('y', 'x'), self.lon)})

    def test_original_seed_and_sampling(self):
        centers, mapping, local, far = build_mapping(self.lat, self.lon, self.args)
        rng = np.random.RandomState(42)
        np.testing.assert_array_equal(centers.target_lat, rng.uniform(49.5, 51.5, 2))
        np.testing.assert_array_equal(centers.target_lon, rng.uniform(8.5, 11.5, 2))
        self.assertEqual(local[12], (0, 0))
        self.assertEqual(len(far), 8)
        da, _, _, selectors = prepare_time_axis(self.ds, self.args)
        actual = np.stack([da.isel(**selector).isel(
            y=xr.DataArray(mapping.grid_y, dims='cell'),
            x=xr.DataArray(mapping.grid_x, dims='cell')).values for selector in selectors])
        np.testing.assert_array_equal(actual, self.values[:, mapping.grid_y, mapping.grid_x])

    def test_valid_time_and_duplicates(self):
        shifted = self.ds.assign_coords(valid_time=('time', self.dates + pd.Timedelta(hours=3)))
        self.assertEqual(prepare_time_axis(shifted, self.args)[1][0], self.dates[1])
        duplicate = self.ds.assign_coords(valid_time=('time', [self.dates[0]] * 4))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            prepare_time_axis(duplicate, self.args)

    def test_output_tables_tensor_and_reference_guard(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            self.args.grib = root / 'placeholder.grib'
            self.args.grib.touch()
            self.args.reference_csv = root / 'reference.csv'
            self.args.output_dir = root / 'output'
            centers, _, _, _ = build_mapping(self.lat, self.lon, self.args)
            reference = pd.concat([
                pd.DataFrame({'datetime': self.dates, 'point_id': row.point_id,
                              'ws_10': self.values[:, row.center_y, row.center_x]})
                for row in centers.itertuples()], ignore_index=True)
            reference.to_csv(self.args.reference_csv, index=False)
            with patch.dict('os.environ', {'SLURM_JOB_ID': 'synthetic'}), \
                    patch('extract_cerra_spatial.xr.open_dataset', return_value=self.ds):
                extract(self.args)
            for name, count in [('S01', 9), ('S02', 25), ('A01', 9)]:
                frame = pd.read_csv(self.args.output_dir / f'{name}.csv')
                self.assertEqual(frame.shape, (8, count + 2))
                np.testing.assert_allclose(frame.ws_10, reference.ws_10)
            with np.load(self.args.output_dir / 'neighborhoods.npz') as tensor:
                self.assertEqual(tensor['ws_10'].shape, (2, 4, 5, 5))
                np.testing.assert_allclose(tensor['ws_10'][:, :, 2, 2].ravel(), reference.ws_10)
            from extract_cerra_spatial import check_reference
            changed = reference.copy()
            changed.loc[0, 'ws_10'] += 10
            with self.assertRaisesRegex(ValueError, 'differs'):
                check_reference(changed, self.args.reference_csv)


if __name__ == '__main__':
    unittest.main()
