#!/bin/bash
#SBATCH --job-name=cerra_date_blocks
#SBATCH --partition=dp-cn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --chdir=/p/project1/sdlrs/tian1/wind_prediction
#SBATCH --output=/p/project1/sdlrs/tian1/wind_prediction/logs/date_blocks_%j.out
#SBATCH --error=/p/project1/sdlrs/tian1/wind_prediction/logs/date_blocks_%j.err

set -eo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo 'Submit with sbatch run_seasonal_diagnostics.sh.' >&2
    exit 1
fi
cd /p/project1/sdlrs/tian1/wind_prediction
module --force purge
module load Stages/2025 GCCcore/.13.3.0 Python/3.12.3
source /p/project1/sdlrs/tian1/wind_prediction/wind-env/bin/activate
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
srun wind-env/bin/python -u seasonal_block_diagnostics.py \
    --analysis_dir ./result/spatial_analysis/analysis_912234 \
    --output_dir "./result/spatial_analysis/date_blocks_${SLURM_JOB_ID}" \
    --block_days 3 --resamples 2000 "$@"
