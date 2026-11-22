#!/bin/bash
#SBATCH --job-name=cerra_newyear_download
#SBATCH --partition=dp-cn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=20:00:00
#SBATCH --array=0
#SBATCH --chdir=/p/project1/sdlrs/tian1/wind_prediction
#SBATCH --output=/p/project1/sdlrs/tian1/wind_prediction/logs/cerra_download_%A_%a.out
#SBATCH --error=/p/project1/sdlrs/tian1/wind_prediction/logs/cerra_download_%A_%a.err
set -eo pipefail
: "${SLURM_ARRAY_TASK_ID:?Submit with sbatch}"
: "${CDSAPI_KEY:?Export CDSAPI_KEY in your shell before sbatch; never put it in scripts or logs}"
source /p/project1/sdlrs/tian1/wind_prediction/environment/activate_wind_env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
# One account-wide project lock also serializes independently submitted arrays.
flock /p/project1/sdlrs/tian1/wind_prediction/data/CERRA/cds_account_download.lock \
    srun wind-env/bin/python -u download_cerra_confirmation.py --task "$SLURM_ARRAY_TASK_ID"
