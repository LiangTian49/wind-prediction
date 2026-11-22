#!/bin/bash
#SBATCH --job-name=cerra_spatial
#SBATCH --partition=dp-cn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --chdir=/p/project1/sdlrs/tian1/wind_prediction
#SBATCH --output=/p/project1/sdlrs/tian1/wind_prediction/logs/extract_%j.out
#SBATCH --error=/p/project1/sdlrs/tian1/wind_prediction/logs/extract_%j.err

set -eo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo 'Submit with sbatch run_extract_cerra.sh.' >&2
    exit 1
fi
cd /p/project1/sdlrs/tian1/wind_prediction
module --force purge
module load Stages/2025 GCCcore/.13.3.0 Python/3.12.3
source /p/project1/sdlrs/tian1/wind_prediction/wind-env/bin/activate
set -u
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
srun wind-env/bin/python -u extract_cerra_spatial.py \
    --output_dir "${OUTPUT_DIR:-./data/CERRA/spatial_${SLURM_JOB_ID}}" "$@"
