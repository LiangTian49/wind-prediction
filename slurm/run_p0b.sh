#!/bin/bash
#SBATCH --job-name=cerra_roi_figures
#SBATCH --partition=dp-cn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --chdir=/p/project1/sdlrs/tian1/wind_prediction
#SBATCH --output=/p/project1/sdlrs/tian1/wind_prediction/logs/p0b_%j.out
#SBATCH --error=/p/project1/sdlrs/tian1/wind_prediction/logs/p0b_%j.err

set -eo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
module --force purge
module load Stages/2025 GCCcore/.13.3.0 Python/3.12.3
source /p/project1/sdlrs/tian1/wind_prediction/wind-env/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 CUDA_VISIBLE_DEVICES=-1
export MPLCONFIGDIR=/p/project1/sdlrs/tian1/wind_prediction/.mplconfig
srun wind-env/bin/python -u plot_p0b.py
