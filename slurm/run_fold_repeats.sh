#!/bin/bash
#SBATCH --job-name=cerra_fold_repeats
#SBATCH --partition=dp-cn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --array=0-47%4
#SBATCH --chdir=/p/project1/sdlrs/tian1/wind_prediction
#SBATCH --output=/p/project1/sdlrs/tian1/wind_prediction/logs/fold_repeats_%A_%a.out
#SBATCH --error=/p/project1/sdlrs/tian1/wind_prediction/logs/fold_repeats_%A_%a.err

# 3 development folds x 4 groups x 4 additional seeds x 4 leads = 192 fits.
# Existing seed=42 runs are reused. F04 is deliberately excluded.
set -eo pipefail
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo 'Submit with sbatch run_fold_repeats.sh.' >&2
    exit 1
fi
cd /p/project1/sdlrs/tian1/wind_prediction
folds=(F01 F02 F03)
groups=(T01 S01 S02 A01)
seeds=(43 44 45 46)
task_id="$SLURM_ARRAY_TASK_ID"
if (( task_id < 0 || task_id > 47 )); then
    echo 'Array task index must be 0..47.' >&2; exit 1
fi
export FOLD_ID="${folds[task_id / 16]}"
export EXPERIMENT="${groups[(task_id / 4) % 4]}"
export SEED="${seeds[task_id % 4]}"
export FOLD_MANIFEST="${FOLD_MANIFEST:-./configs/seasonal_folds_2024.json}"
export SPATIAL_DIR="${SPATIAL_DIR:-./data/CERRA/spatial_2024_full}"
export SAVE_DIR="${SAVE_DIR:-./runs/seasonal/repeats}"
export PROTOCOL=v2_fixed INPUT_LEN=56 EPOCHS=100 LEARNING_RATE=0.001
export FIRST_FOUR=0 SMOKE_TEST=0 ALLOW_CONFIRMATION=0 CUDA_VISIBLE_DEVICES=-1
echo "[Fold repeat] fold=$FOLD_ID experiment=$EXPERIMENT seed=$SEED"
bash run_ablation.sh
