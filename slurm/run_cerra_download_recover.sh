#!/bin/bash
#SBATCH --job-name=cerra_remote_recover
#SBATCH --partition=dp-cn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --chdir=/p/project1/sdlrs/tian1/wind_prediction
#SBATCH --output=/p/project1/sdlrs/tian1/wind_prediction/logs/cerra_recovery_%j.out
#SBATCH --error=/p/project1/sdlrs/tian1/wind_prediction/logs/cerra_recovery_%j.err
set -eo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${CDSAPI_KEY:?Export CDSAPI_KEY before sbatch}"
source /p/project1/sdlrs/tian1/wind_prediction/environment/activate_wind_env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
flock /p/project1/sdlrs/tian1/wind_prediction/data/CERRA/cds_account_download.lock \
    srun wind-env/bin/python -u recover_cerra_downloads.py
