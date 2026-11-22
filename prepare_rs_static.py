"""Slurm pipeline for native CERRA grid footprints, raster QA and 14 static features."""

import argparse
from contextlib import ExitStack
import importlib.metadata
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from direction_pipeline import code_hashes, output_transaction, sha256
from rs_gis import (LAND_CODES, extract_cell, grid_cell, native_projection,
                    warp_mosaic)
from rs_static_contract import STATIC_FEATURES
from temporal_folds import project_path

CONFIG = 'configs/rs_hpc_v3_p2_static.json'
LABELS = ['Trees', 'Shrubland', 'Grassland', 'Cropland', 'Built-up', 'Bare/sparse',
          'Snow/ice', 'Water', 'Wetland', 'Mangroves', 'Moss/lichen']
COLORS = ['#006400', '#ffbb22', '#ffff4c', '#f096ff', '#fa0000', '#b4b4b4',
          '#f0f0f0', '#0064c8', '#0096a0', '#00cf75', '#fae6a0']


def raster_description(dataset, product):
    if dataset.crs is None or dataset.crs.to_epsg() != 4326 or dataset.count != 1:
        raise ValueError(f'Expected single-band EPSG:4326 source: {dataset.name}')
    if (dataset.transform.b != 0 or dataset.transform.d != 0
            or dataset.transform.a <= 0 or dataset.transform.e >= 0):
        raise ValueError(f'Expected north-up source: {dataset.name}')
    if product.startswith('WorldCover') and not np.issubdtype(np.dtype(dataset.dtypes[0]), np.integer):
        raise ValueError('WorldCover classes must be integer native values.')
    if product.startswith('Copernicus') and dataset.units[0] not in [None, '', 'm', 'metre', 'meter']:
        raise ValueError('Unexpected DSM height unit.')
    nodata = dataset.nodata
    if nodata is not None and not np.isfinite(nodata):
        nodata = str(nodata)
    return dict(path=dataset.name, product=product, crs=dataset.crs.to_wkt(),
                width=dataset.width, height=dataset.height, bounds=list(dataset.bounds),
                resolution=list(dataset.res), transform=list(dataset.transform)[:6],
                dtype=dataset.dtypes[0], nodata=nodata, units=list(dataset.units),
                mask_flags=[[item.name for item in flags] for flags in dataset.mask_flag_enums],
                tags=dataset.tags(), band_tags=dataset.tags(1))


def quality_flags(features, quality, center_error, config):
    flags = []
    if center_error > config['center_tolerance_m']:
        flags.append('center_projection')
    for product in ['lc', 'dsm', 'slope']:
        fraction = quality[f'{product}_valid_fraction']
        if fraction < config['minimum_valid_fraction'] or fraction > 1+config['geometric_fraction_tolerance']:
            flags.append(f'{product}_coverage')
    for product in ['lc', 'dsm']:
        if abs(quality[f'{product}_geometric_fraction']-1) > config['geometric_fraction_tolerance']:
            flags.append(f'{product}_geometric_coverage')
    if quality['lc_unknown_fraction'] > config['maximum_unknown_fraction']:
        flags.append('unknown_landcover')
    if not np.isfinite(list(features.values())).all():
        flags.append('nonfinite_features')
    fractions = [features[f'lc_{code}'] for code in LAND_CODES]
    if not np.isclose(sum(fractions), 1, atol=1e-8, rtol=0):
        flags.append('landcover_sum')
    if np.isfinite(features['dsm_slope_mean_deg']) and not 0 <= features['dsm_slope_mean_deg'] <= 90:
        flags.append('slope_range')
    return flags


def qa_figure(cell, crs, land_sources, dsm_sources, config, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch, Rectangle

    _, _, display = extract_cell(cell, crs, land_sources, dsm_sources, config['slope_resolution_m'])
    dsm, slope, transform = display['dsm'], display['slope'], display['transform']
    land, _ = warp_mosaic(display['land_sources'], crs, transform, dsm.shape, categorical=True)
    indices = np.full(land.shape, np.nan)
    for i, code in enumerate(LAND_CODES):
        indices[land == code] = i
    height, width = dsm.shape
    extent = (transform.c-cell['x'], transform.c+width*transform.a-cell['x'],
              transform.f+height*transform.e-cell['y'], transform.f-cell['y'])
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), layout='constrained')
    cmap = ListedColormap(COLORS)
    axes[0].imshow(indices, extent=extent, origin='upper', interpolation='nearest',
                   cmap=cmap, norm=BoundaryNorm(np.arange(12)-.5, 11))
    for axis, image, label in [(axes[1], dsm, 'DSM height (m)'), (axes[2], slope, 'DSM slope (degrees)')]:
        plotted = axis.imshow(image, extent=extent, origin='upper', cmap='terrain' if axis is axes[1] else 'viridis')
        fig.colorbar(plotted, ax=axis, label=label)
    left, bottom, right, top = cell['bounds']
    for axis, name in zip(axes, ['Land cover (nearest only for display)', '30m DSM analysis grid', 'Horn 3x3 slope']):
        axis.add_patch(Rectangle((left-cell['x'], bottom-cell['y']), right-left, top-bottom,
                                 fill=False, edgecolor='black', linewidth=1.5))
        axis.set(title=name, xlabel='Native x relative to center (m)', ylabel='Native y relative to center (m)')
    fig.legend(handles=[Patch(facecolor=color, label=label) for color, label in zip(COLORS, LABELS)],
               loc='outside lower center', ncol=6, fontsize=8)
    fig.suptitle(title)
    fig.savefig(path.with_suffix('.png'), dpi=170)
    fig.savefig(path.with_suffix('.svg'))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=project_path, default=project_path(CONFIG))
    parser.add_argument('--output_dir', type=project_path)
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if tuple(config['feature_order']) != STATIC_FEATURES:
        raise ValueError('Static feature order differs from the P3 contract.')
    centers = pd.read_csv(project_path(config['centers']))
    if sha256(project_path(config['centers'])) != config['expected_centers_sha256']:
        raise ValueError('Target mapping differs from the versioned experiment plan.')
    centers['grid_id'] = centers.center_y.astype(str)+':'+centers.center_x.astype(str)
    unique = centers.drop_duplicates('grid_id').sort_values(['center_y', 'center_x']).reset_index(drop=True)
    if centers.point_id.duplicated().any() or len(centers) != 100 or len(unique) != 98:
        raise ValueError('Expected the original 100-point / 98-grid mapping.')
    if args.list:
        print('98 unique target cells; 14 static features; 0 model fits.')
        print('Minimum valid fraction:', config['minimum_valid_fraction'])
        return
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Submit slurm/run_prepare_rs_static.sh.')
    import rasterio
    import pyproj
    import shapely

    started = perf_counter()
    raw = project_path(config['raw_dir'])
    receipt = json.loads((raw/'download_complete.json').read_text())
    download_manifest = project_path(config['download_manifest'])
    if (receipt['status'] != 'downloaded_waiting_geospatial_QA' or receipt['assets'] != 16
            or sha256(download_manifest) != receipt['manifest_sha256']
            or sha256(raw/'source_manifest.json') != receipt['manifest_sha256']):
        raise ValueError('Download manifest or status differs.')
    for asset in receipt['files']:
        path = raw/asset['product']/asset['filename']
        if path.stat().st_size != asset['bytes'] or sha256(path) != asset['sha256']:
            raise ValueError(f'Changed source tile: {path}')
    provenance = json.loads(project_path(config['grib_audit']).read_text())
    if provenance['grid_hash'] != config['expected_grid_hash']:
        raise ValueError('CERRA grid differs from the audited version.')
    header = provenance['sampled_headers'][0]
    crs, projection, origin = native_projection(header)
    cells = {}
    for row in unique.itertuples():
        cells[row.grid_id] = grid_cell(row, header, projection, origin)
        if cells[row.grid_id]['center_error_m'] > config['center_tolerance_m']:
            raise ValueError(f'Native center projection mismatch: {row.grid_id}, {cells[row.grid_id]["center_error_m"]}m')
    versions = {name: importlib.metadata.version(name) for name in ['rasterio', 'pyproj', 'shapely', 'numpy', 'pandas']}
    fingerprint = dict(config_sha256=sha256(args.config),
                       centers_sha256=sha256(project_path(config['centers'])),
                       grib_audit_sha256=sha256(project_path(config['grib_audit'])),
                       tiles={asset['filename']: asset['sha256'] for asset in receipt['files']},
                       versions=versions,
                       code=code_hashes(['prepare_rs_static.py', 'rs_gis.py', 'rs_static_contract.py',
                                         'direction_pipeline.py']))
    final = args.output_dir or project_path(config['output_dir'])
    qa_pass = False
    with output_transaction(final, fingerprint) as work:
        if work is None:
            summary = json.loads((final/'qa_summary.json').read_text())
            if summary['status'] != 'passed':
                raise SystemExit('Existing extraction failed QA. Inspect it and use a new version for changed settings.')
            print(f'[P2 static] Verified existing dataset: {final}')
            return
        with ExitStack() as stack, rasterio.Env(GDAL_CACHEMAX=256*1024**2):
            sources = {'WorldCover_2021_v200': [], 'Copernicus_GLO30_Public': []}
            descriptions = []
            for asset in receipt['files']:
                dataset = stack.enter_context(rasterio.open(raw/asset['product']/asset['filename']))
                descriptions.append(raster_description(dataset, asset['product']))
                sources[asset['product']].append(dataset)
            land_sources, dsm_sources = sources['WorldCover_2021_v200'], sources['Copernicus_GLO30_Public']
            feature_rows, quality_rows = [], []
            for i, row in enumerate(unique.itertuples()):
                features, quality, _ = extract_cell(cells[row.grid_id], crs, land_sources, dsm_sources,
                                                     config['slope_resolution_m'])
                flags = quality_flags(features, quality, cells[row.grid_id]['center_error_m'], config)
                feature_rows.append(dict(grid_id=row.grid_id, **features))
                quality_rows.append(dict(grid_id=row.grid_id, **quality,
                                         center_error_m=cells[row.grid_id]['center_error_m'],
                                         qa_pass=not flags, flags='|'.join(flags)))
                print(f'[P2 static] {i+1}/98 {row.grid_id} flags={flags}', flush=True)
            table, quality = pd.DataFrame(feature_rows), pd.DataFrame(quality_rows)
            table.to_csv(work/'static_features.csv', index=False)
            quality.to_csv(work/'quality.csv', index=False)
            centers.to_csv(work/'point_grid_mapping.csv', index=False)
            geofeatures = [dict(type='Feature', properties=dict(grid_id=row.grid_id, grid_lat=row.grid_lat, grid_lon=row.grid_lon),
                                geometry=dict(type='Polygon', coordinates=[cells[row.grid_id]['coordinates']]))
                           for row in unique.itertuples()]
            (work/'grid_footprints.geojson').write_text(json.dumps(dict(type='FeatureCollection', features=geofeatures)))
            (work/'raster_metadata.json').write_text(json.dumps(descriptions, indent=2, allow_nan=False))
            selected = []
            # Deterministic surface-diversity QA choices use no wind labels.
            for column, ascending in [('dsm_sd_m', True), ('dsm_sd_m', False), ('lc_50', False)]:
                candidates = table.sort_values([column, 'grid_id'], ascending=[ascending, True])
                grid_id = next(grid for grid in candidates.grid_id if grid not in selected)
                selected.append(grid_id)
                qa_figure(cells[grid_id], crs, land_sources, dsm_sources, config,
                          work/f'qa_{grid_id.replace(":", "_")}', f'Native target cell {grid_id}')
        # Join by keys, explicitly verifying duplicate point mappings.
        connected = centers[['point_id', 'grid_id']].merge(table, on='grid_id', validate='many_to_one')
        if len(connected) != 100:
            raise ValueError('Static-to-point mapping changed the original point count.')
        qa_pass = bool(quality.qa_pass.all() and np.isfinite(table[list(STATIC_FEATURES)]).all().all())
        summary = dict(status='passed' if qa_pass else 'failed', unique_grids=len(table),
                       point_ids=len(connected), features=14,
                       failed_grids=quality.loc[~quality.qa_pass, 'grid_id'].tolist(),
                       minimum_valid_fraction=config['minimum_valid_fraction'], qa_example_grids=selected,
                       minimum_lc_coverage=float(quality.lc_valid_fraction.min()),
                       minimum_dsm_coverage=float(quality.dsm_valid_fraction.min()),
                       minimum_slope_coverage=float(quality.slope_valid_fraction.min()),
                       maximum_center_error_m=float(quality.center_error_m.max()),
                       elapsed_seconds=perf_counter()-started)
        (work/'qa_summary.json').write_text(json.dumps(summary, indent=2))
        metadata = dict(config=config, feature_order=list(STATIC_FEATURES), versions=versions,
                        gdal_version=rasterio.__gdal_version__, proj_version=pyproj.proj_version_str,
                        native_crs_wkt=crs.to_wkt(), native_first_grid_xy=list(origin),
                        area_units='m2 in native projected plane; pixel boundaries approximated by transformed quadrilaterals',
                        dsm_vertical_reference='EGM2008 orthometric metres per product specification; source TIFF tags recorded',
                        vertical_transformation='none',
                        slope_source='DSM surface including vegetation and buildings',
                        landcover_time_assumption='2021 static proxy for the 2024 forecasting experiments',
                        preprocess='Extraction is deterministic; imputation/scaling/constant-column decisions fit training grids later',
                        source_manifest=json.loads(download_manifest.read_text()),
                        qa_examples=dict(rule='Lowest/highest DSM SD and highest built-up fraction; deterministic tie by grid_id', grids=selected))
        (work/'feature_metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False))
        (work/'provenance.json').write_text(json.dumps(fingerprint, indent=2))
        report = ['# P2 spatial QA', '', f'Overall QA: {summary["status"]}.',
                  f'{len(table)} unique cells, 100 point mappings, 14 features.',
                  f'Minimum valid area requirement: {config["minimum_valid_fraction"]:.2%}.',
                  'The raw 14-column table retains all target cells and uses no weather labels.',
                  'Land-cover classes are accumulated on native raster pixels by projected intersection area.',
                  'Coverage, NoData, unknown-class and uncovered-area fractions remain in quality.csv.',
                  'DSM mean/SD use native weighted pixels. Slope uses bilinear DSM on a 30m metric grid and a complete Horn 3x3 stencil.',
                  'Three QA examples contain the actual projected cell outline. Class resampling is nearest for display only.',
                  'No spatial imputation, feature scaling or constant-column filtering occurs here.',
                  f'Failed grids: {summary["failed_grids"]}.', '']
        (work/'qa_report.md').write_text('\n'.join(report))
        if qa_pass:
            ready = dict(status='passed', unique_grids=98, features=list(STATIC_FEATURES),
                         config_sha256=fingerprint['config_sha256'],
                         static_features_sha256=sha256(work/'static_features.csv'),
                         mapping_sha256=sha256(work/'point_grid_mapping.csv'),
                         quality_sha256=sha256(work/'quality.csv'))
            (work/'READY.json').write_text(json.dumps(ready, indent=2))
    if not qa_pass:
        raise SystemExit(f'P2 static QA failed; inspect published quality report at {final}.')
    print(f'[P2 static] READY: {final}')


if __name__ == '__main__':
    main()
