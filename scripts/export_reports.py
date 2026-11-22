"""Copy small, complete experiment summaries into the GitHub-facing reports folder."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SELECTION = {
    'docs/experiments/cerra_download_month_status_2026-11-21.csv': 'downloads/month_status.csv',
    'docs/experiments/CERRA_DOWNLOAD_IMPLEMENTATION_STATUS.json': 'downloads/status.json',
    'docs/experiments/CONFIRMATION_V1_IMPLEMENTATION_STATUS.json': 'confirmation/implementation_status.json',
    'result/rs_hpc_v3/temporal_controls/DATE_REVIEW.md': 'temporal/DATE_REVIEW.md',
    'result/rs_hpc_v3/temporal_controls/date_concentration_913084.csv': 'temporal/date_concentration.csv',
    'result/rs_hpc_v3/temporal_controls/analysis_913084/date_intervals.csv': 'temporal/date_intervals.csv',
    'result/rs_hpc_v3/temporal_controls/analysis_913084/RS_gain_interactions.csv': 'temporal/RS_gain_interactions.csv',
    'result/rs_hpc_v3/temporal_controls/analysis_913084/method.json': 'temporal/method.json',
    'result/rs_hpc_v3/P3c_horizon/v1/DEVELOPMENT_REVIEW.md': 'horizon/DEVELOPMENT_REVIEW.md',
    'runs/rs_hpc_v3/P3c_horizon/v1/combined_results.csv': 'horizon/combined_results.csv',
    'runs/rs_hpc_v3/P3c_horizon/v1/paired_results.csv': 'horizon/paired_results.csv',
    'runs/rs_hpc_v3/P3c_horizon/v1/paired_seed_summary.csv': 'horizon/paired_seed_summary.csv',
    'result/rs_hpc_v3/P1b_history/v1/DEVELOPMENT_REVIEW.md': 'history/DEVELOPMENT_REVIEW.md',
    'runs/rs_hpc_v3/P1b_history/v1/combined_results.csv': 'history/combined_results.csv',
    'runs/rs_hpc_v3/P1b_history/v1/paired_results.csv': 'history/paired_results.csv',
    'runs/rs_hpc_v3/P1b_history/v1/paired_seed_summary.csv': 'history/paired_seed_summary.csv',
    'result/rs_hpc_v3/P3b/v1/DATE_REVIEW.md': 'p3b/DATE_REVIEW.md',
    'result/rs_hpc_v3/P3b/v1/date_concentration_912973.csv': 'p3b/date_concentration.csv',
    'result/rs_hpc_v3/P3b/v1/analysis_912973/date_intervals.csv': 'p3b/dates/date_intervals.csv',
    'result/rs_hpc_v3/P3b/v1/analysis_912973/method.json': 'p3b/dates/method.json',
    'result/rs_hpc_v3/P3b/v1/DEVELOPMENT_REVIEW.md': 'p3b/DEVELOPMENT_REVIEW.md',
    'runs/rs_hpc_v3/P3b/v1/combined_results.csv': 'p3b/combined_results.csv',
    'runs/rs_hpc_v3/P3b/v1/paired_results.csv': 'p3b/paired_results.csv',
    'runs/rs_hpc_v3/P3b/v1/paired_seed_summary.csv': 'p3b/paired_seed_summary.csv',
    'result/rs_hpc_v3/P3/DATE_REVIEW.md': 'p3/DATE_REVIEW.md',
    'result/rs_hpc_v3/P3/date_concentration_912912.csv': 'p3/date_concentration.csv',
    'result/rs_hpc_v3/P3/analysis_912912/date_intervals.csv': 'p3/dates/date_intervals.csv',
    'result/rs_hpc_v3/P3/analysis_912912/method.json': 'p3/dates/method.json',
    'result/rs_hpc_v3/P3/DEVELOPMENT_REVIEW.md': 'p3/DEVELOPMENT_REVIEW.md',
    'result/rs_hpc_v3/P3/development_absolute.csv': 'p3/development_absolute.csv',
    'result/rs_hpc_v3/P3/development_gains.csv': 'p3/development_gains.csv',
    'result/rs_hpc_v3/P3/PILOT_REVIEW.md': 'p3/PILOT_REVIEW.md',
    'runs/rs_hpc_v3/P3/summary.csv': 'p3/summary.csv',
    'runs/rs_hpc_v3/P3/paired_results.csv': 'p3/paired_results.csv',
    'result/rs_hpc_v3/P2/STATIC_REVIEW.md': 'p2/STATIC_REVIEW.md',
    'data/rs_static/features_v1/qa_summary.json': 'p2/qa_summary.json',
    'data/rs_static/features_v1/feature_metadata.json': 'p2/feature_metadata.json',
    'data/rs_static/features_v1/static_features.csv': 'p2/static_features.csv',
    'result/paper_review/FORMAL_RESULTS.csv': 'seasonal/formal_results.csv',
    'result/spatial_analysis/analysis_912234/FINAL_SEASONAL_REVIEW.md': 'seasonal/REVIEW.md',
    'result/spatial_analysis/date_blocks_912276/DIAGNOSTICS_REVIEW.md': 'seasonal/DATE_REVIEW.md',
    'result/spatial_analysis/date_blocks_912276/date_block_intervals.csv': 'seasonal/date_intervals.csv',
    'result/rs_hpc_v3/P1/baseline_results.csv': 'p1/baseline_results.csv',
    'result/rs_hpc_v3/P1/P1_REVIEW.md': 'p1/REVIEW.md',
    'result/rs_hpc_v3/P1/analysis_912751/P1_DATE_REVIEW.md': 'p1/DATE_REVIEW.md',
    'result/rs_hpc_v3/P1/analysis_912751/unified_results.csv': 'p1/unified_results.csv',
    'result/rs_hpc_v3/P1c/DEVELOPMENT_REVIEW.md': 'p1c/DEVELOPMENT_REVIEW.md',
    'result/rs_hpc_v3/P1c/development_results.csv': 'p1c/development_results.csv',
    'result/rs_hpc_v3/P1c/DATE_REVIEW.md': 'p1c/DATE_REVIEW.md',
    'result/rs_hpc_v3/P2/DATA_AVAILABILITY.md': 'p2/DATA_AVAILABILITY.md',
    'result/rs_hpc_v3/P0b/VERIFICATION.md': 'roi/VERIFICATION.md',
    'result/rs_hpc_v3/P0b/v1/roi_dimensions.json': 'roi/roi_dimensions.json',
    'result/rs_hpc_v3/P0b/v1/figure_manifest.json': 'roi/figure_manifest.json',
    'result/rs_hpc_v3/P0b/v1/architectures.json': 'roi/architectures.json',
    'result/rs_hpc_v3/P0b/v1/lstm_architecture.svg': 'roi/lstm_architecture.svg',
    'result/rs_hpc_v3/P0b/v1/F01_maps.png': 'roi/F01_maps.png',
    'result/rs_hpc_v3/P0b/v1/F01_series.png': 'roi/F01_series.png',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = ROOT / 'reports'
    output.mkdir(exist_ok=True)
    selected = dict(SELECTION)
    candidates = sorted((ROOT / 'result/rs_hpc_v3/P1c').glob('analysis_*/completion.json'),
                        key=lambda path: int(path.parent.name.split('_')[-1]))
    if candidates:
        latest = candidates[-1].parent
        marker = json.loads((latest / 'completion.json').read_text())
        if marker['status'] != 'completed' or any(
            digest(latest / name) != expected for name, expected in marker['artifacts'].items()
        ):
            raise ValueError(f'Incomplete or changed analysis: {latest}')
        for name in ['REVIEW.md', 'date_intervals.csv', 'method.json', 'validation_selection.json']:
            selected[str((latest / name).relative_to(ROOT))] = f'p1c/dates/{name}'
    exported, missing = [], []
    for source_name, destination_name in selected.items():
        source = ROOT / source_name
        destination = output / destination_name
        if not source.is_file():
            missing.append(source_name)
            continue
        if source.stat().st_size > 2 * 1024**2:
            raise ValueError(f'Summary exceeds 2 MiB: {source}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        exported.append(dict(source=source_name, destination=str(destination.relative_to(ROOT)),
                             sha256=digest(source), bytes=source.stat().st_size))
    (output / 'manifest.json').write_text(json.dumps(dict(
        exported_at_utc=datetime.now(timezone.utc).isoformat(),
        files=exported, unavailable_sources=missing,
        scope='Small archived tables, reports and selected ROI figures. No model fitting or metrics recomputation.'
    ), indent=2)+'\n')
    print(f'Exported {len(exported)} small artifacts; {len(missing)} unavailable sources.')


if __name__ == '__main__':
    main()
