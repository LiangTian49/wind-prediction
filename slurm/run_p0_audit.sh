#!/bin/bash
#SBATCH --job-name=rs_hpc_p0
#SBATCH --partition=dp-cn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:45:00
#SBATCH --chdir=/p/project1/sdlrs/tian1/wind_prediction
#SBATCH --output=/p/project1/sdlrs/tian1/wind_prediction/logs/p0_%j.out
#SBATCH --error=/p/project1/sdlrs/tian1/wind_prediction/logs/p0_%j.err

set -eo pipefail
: "${SLURM_JOB_ID:?Submit via sbatch}"
module --force purge
module load Stages/2025 GCCcore/.13.3.0 Python/3.12.3
source /p/project1/sdlrs/tian1/wind_prediction/wind-env/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
srun wind-env/bin/python -u audit_p0_predictions.py
