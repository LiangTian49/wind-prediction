#!/bin/bash
#SBATCH --job-name=cerra_repeats
#SBATCH --partition=dp-cn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --array=0-15%4
#SBATCH --chdir=/p/project1/sdlrs/tian1/wind_prediction
#SBATCH --output=/p/project1/sdlrs/tian1/wind_prediction/logs/repeats_%A_%a.out
#SBATCH --error=/p/project1/sdlrs/tian1/wind_prediction/logs/repeats_%A_%a.err

# 16 tasks: 4 experiment groups x 4 additional seeds; 4 leads per task.
# Existing seed=42 experiments are reused, not retrained.
set -eo pipefail
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo 'Submit with sbatch run_repeats.sh.' >&2
    exit 1
fi
cd /p/project1/sdlrs/tian1/wind_prediction
experiment_groups=(T01 S01 S02 A01)
seeds=(43 44 45 46)
task_id="$SLURM_ARRAY_TASK_ID"
if (( task_id < 0 || task_id > 15 )); then
    echo 'Array task index must be 0..15.' >&2
    exit 1
fi
export EXPERIMENT="${experiment_groups[task_id / 4]}"
export SEED="${seeds[task_id % 4]}"
export SPATIAL_DIR="${SPATIAL_DIR:-./data/CERRA/spatial_911958}"
export SAVE_DIR="${SAVE_DIR:-./runs/ablation/repeats}"
# Freeze the pilot protocol and hyperparameters for these repetitions.
export DATA_PATH=./data/CERRA/cerra_100_points_200km_summer_roi_2024.csv
export INPUT_LEN=56 EPOCHS=100 LEARNING_RATE=0.001
export FIRST_FOUR=0 SMOKE_TEST=0
export PROTOCOL=v1
export CUDA_VISIBLE_DEVICES=-1
echo "[Repeat] task=$task_id experiment=$EXPERIMENT seed=$SEED"
# Inner SBATCH directives are comments here; use this task's allocation.
bash run_ablation.sh
