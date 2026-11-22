#!/bin/bash
#SBATCH --job-name=cerra_confirm_data
#SBATCH --partition=dp-cn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --chdir=/p/project1/sdlrs/tian1/wind_prediction
#SBATCH --output=/p/project1/sdlrs/tian1/wind_prediction/logs/confirmation_extract_%j.out
#SBATCH --error=/p/project1/sdlrs/tian1/wind_prediction/logs/confirmation_extract_%j.err
set -eo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
source /p/project1/sdlrs/tian1/wind_prediction/environment/activate_wind_env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
srun wind-env/bin/python -u prepare_confirmation_year.py extract "$@"
