#!/bin/bash
#SBATCH --job-name=cerra_v2
#SBATCH --partition=dp-cn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --array=0-19%4
#SBATCH --chdir=/p/project1/sdlrs/tian1/wind_prediction
#SBATCH --output=/p/project1/sdlrs/tian1/wind_prediction/logs/v2_%A_%a.out
#SBATCH --error=/p/project1/sdlrs/tian1/wind_prediction/logs/v2_%A_%a.err

# First check: sbatch --array=0 run_v2.sh (T01 seed=42, all four leads).
# If successful, submit remaining tasks: sbatch --array=1-19%4 run_v2.sh.
# Or submit all 80 fits once: sbatch run_v2.sh.
set -eo pipefail
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo 'Submit with sbatch run_v2.sh.' >&2
    exit 1
fi
cd /p/project1/sdlrs/tian1/wind_prediction
experiment_groups=(T01 S01 S02 A01)
seeds=(42 43 44 45 46)
task_id="$SLURM_ARRAY_TASK_ID"
if (( task_id < 0 || task_id > 19 )); then
    echo 'Array task index must be 0..19.' >&2
    exit 1
fi
export EXPERIMENT="${experiment_groups[task_id / 5]}"
export SEED="${seeds[task_id % 5]}"
export SPATIAL_DIR="${SPATIAL_DIR:-./data/CERRA/spatial_911958}"
export SAVE_DIR="${SAVE_DIR:-./runs/v2}"
export PROTOCOL=v2
export DATA_PATH=./data/CERRA/cerra_100_points_200km_summer_roi_2024.csv
export INPUT_LEN=56 EPOCHS=100 LEARNING_RATE=0.001
export FIRST_FOUR=0 SMOKE_TEST=0 CUDA_VISIBLE_DEVICES=-1
echo "[V2] task=$task_id experiment=$EXPERIMENT seed=$SEED"
# The module setup and wind-env activation occur in this allocated child shell.
bash run_ablation.sh
