#!/bin/bash
#SBATCH --job-name=rs_hpc_p1
#SBATCH --partition=dp-cn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --array=0-47%4
#SBATCH --chdir=/p/project1/sdlrs/tian1/wind_prediction
#SBATCH --output=/p/project1/sdlrs/tian1/wind_prediction/logs/p1_%A_%a.out
#SBATCH --error=/p/project1/sdlrs/tian1/wind_prediction/logs/p1_%A_%a.err

set -eo pipefail
: "${SLURM_ARRAY_TASK_ID:?Submit via sbatch}"
module --force purge
module load Stages/2025 GCCcore/.13.3.0 Python/3.12.3
source /p/project1/sdlrs/tian1/wind_prediction/wind-env/bin/activate
set -u
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
srun wind-env/bin/python -u train_p1_baseline.py --task "$SLURM_ARRAY_TASK_ID"
