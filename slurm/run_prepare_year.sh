#!/bin/bash
#SBATCH --job-name=cerra_year_data
#SBATCH --partition=dp-cn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --chdir=/p/project1/sdlrs/tian1/wind_prediction
#SBATCH --output=/p/project1/sdlrs/tian1/wind_prediction/logs/year_data_%j.out
#SBATCH --error=/p/project1/sdlrs/tian1/wind_prediction/logs/year_data_%j.err

set -eo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo 'Submit with sbatch run_prepare_year.sh.' >&2
    exit 1
fi
cd /p/project1/sdlrs/tian1/wind_prediction
module --force purge
module load Stages/2025 GCCcore/.13.3.0 Python/3.12.3
source /p/project1/sdlrs/tian1/wind_prediction/wind-env/bin/activate
set -u
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ANNUAL_DIR="${ANNUAL_DIR:-./data/CERRA/spatial_2024_full}"
FOLD_MANIFEST="${FOLD_MANIFEST:-./configs/seasonal_folds_2024.json}"
case "${VALIDATE_ONLY:-0}" in
    0)
        srun wind-env/bin/python -u extract_cerra_spatial.py \
            --year 2024 --months 1 2 3 4 5 6 7 8 9 10 11 12 \
            --require_complete_months --reference_mode reference_subset \
            --reference_centers ./data/CERRA/spatial_911958/centers.csv \
            --output_dir "$ANNUAL_DIR"
        ;;
    1) echo '[Prepare] Reusing existing extraction; validating only.' ;;
    *) echo 'VALIDATE_ONLY must be 0 or 1.' >&2; exit 1 ;;
esac
srun wind-env/bin/python -u validate_annual_data.py \
    --spatial_dir "$ANNUAL_DIR" --fold_manifest "$FOLD_MANIFEST" \
    --report_dir "./result/data_validation/annual_${SLURM_JOB_ID}"
