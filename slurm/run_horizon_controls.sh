#!/bin/bash
#SBATCH --job-name=cerra_horizon
#SBATCH --partition=dp-cn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-1%2
#SBATCH --chdir=/p/project1/sdlrs/tian1/wind_prediction
#SBATCH --output=/p/project1/sdlrs/tian1/wind_prediction/logs/horizon_controls_%A_%a.out
#SBATCH --error=/p/project1/sdlrs/tian1/wind_prediction/logs/horizon_controls_%A_%a.err
set -eo pipefail
: "${SLURM_ARRAY_TASK_ID:?Submit with sbatch}"
module --force purge
module load Stages/2025 GCCcore/.13.3.0 Python/3.12.3
source /p/project1/sdlrs/tian1/wind_prediction/wind-env/bin/activate
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS="$SLURM_CPUS_PER_TASK" TF_NUM_INTEROP_THREADS=1
export CUDA_VISIBLE_DEVICES=-1 PYTHONUNBUFFERED=1
srun wind-env/bin/python -u train_temporal_controls.py --stage horizon --task "$SLURM_ARRAY_TASK_ID"
