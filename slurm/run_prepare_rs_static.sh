#!/bin/bash
#SBATCH --job-name=cerra_p2_static
#SBATCH --partition=dp-cn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --chdir=/p/project1/sdlrs/tian1/wind_prediction
#SBATCH --output=/p/project1/sdlrs/tian1/wind_prediction/logs/p2_static_%j.out
#SBATCH --error=/p/project1/sdlrs/tian1/wind_prediction/logs/p2_static_%j.err

set -eo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
module --force purge
module load Stages/2025 GCCcore/.13.3.0 Python/3.12.3
source /p/project1/sdlrs/tian1/wind_prediction/wind-env/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/p/project1/sdlrs/tian1/wind_prediction/.mplconfig
export PROJ_NETWORK=OFF
export TMPDIR=/p/project1/sdlrs/tian1/wind_prediction/.pip-tmp
export CPL_TMPDIR="$TMPDIR"
mkdir -p "$TMPDIR"
srun wind-env/bin/python -u prepare_rs_static.py "$@"
