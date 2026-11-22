#!/bin/bash
#SBATCH --job-name=cerra_paper_figures
#SBATCH --partition=dp-cn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --chdir=/p/project1/sdlrs/tian1/wind_prediction
#SBATCH --output=/p/project1/sdlrs/tian1/wind_prediction/logs/paper_figures_%j.out
#SBATCH --error=/p/project1/sdlrs/tian1/wind_prediction/logs/paper_figures_%j.err
set -eo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
module --force purge
module load Stages/2025 GCCcore/.13.3.0 Python/3.12.3
source /p/project1/sdlrs/tian1/wind_prediction/wind-env/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR=/p/project1/sdlrs/tian1/wind_prediction/logs/mpl_paper_figures_${SLURM_JOB_ID}
mkdir -p "$MPLCONFIGDIR"
srun wind-env/bin/python -u plot_paper_review.py "$@"
