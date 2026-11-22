#!/bin/bash
#SBATCH --job-name=rs_hpc_p1_dates
#SBATCH --partition=dp-cn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --chdir=/p/project1/sdlrs/tian1/wind_prediction
#SBATCH --output=/p/project1/sdlrs/tian1/wind_prediction/logs/p1_dates_%j.out
#SBATCH --error=/p/project1/sdlrs/tian1/wind_prediction/logs/p1_dates_%j.err

set -eo pipefail
: "${SLURM_JOB_ID:?Submit via sbatch}"
module --force purge
module load Stages/2025 GCCcore/.13.3.0 Python/3.12.3
source /p/project1/sdlrs/tian1/wind_prediction/wind-env/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
srun wind-env/bin/python -u summarize_p1.py --require_complete
srun wind-env/bin/python -u analyze_p1_dates.py
