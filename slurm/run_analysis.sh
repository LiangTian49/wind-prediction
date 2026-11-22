#!/bin/bash
#SBATCH --job-name=cerra_analysis
#SBATCH --partition=dp-cn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --chdir=/p/project1/sdlrs/tian1/wind_prediction
#SBATCH --output=/p/project1/sdlrs/tian1/wind_prediction/logs/analysis_%j.out
#SBATCH --error=/p/project1/sdlrs/tian1/wind_prediction/logs/analysis_%j.err

set -eo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo 'Submit with sbatch run_analysis.sh.' >&2
    exit 1
fi
cd /p/project1/sdlrs/tian1/wind_prediction
module --force purge
module load Stages/2025 GCCcore/.13.3.0 Python/3.12.3
source /p/project1/sdlrs/tian1/wind_prediction/wind-env/bin/activate
set -u
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TF_NUM_INTRAOP_THREADS="$SLURM_CPUS_PER_TASK"
export TF_NUM_INTEROP_THREADS=1
export CUDA_VISIBLE_DEVICES=-1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/p/project1/sdlrs/tian1/wind_prediction/.mplconfig
srun wind-env/bin/python -u analyze_experiments.py \
    --output_dir "${ANALYSIS_DIR:-./result/spatial_analysis/analysis_${SLURM_JOB_ID}}" "$@"
