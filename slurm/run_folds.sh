#!/bin/bash
#SBATCH --job-name=cerra_fold_check
#SBATCH --partition=dp-cn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --array=0-11%4
#SBATCH --chdir=/p/project1/sdlrs/tian1/wind_prediction
#SBATCH --output=/p/project1/sdlrs/tian1/wind_prediction/logs/folds_%A_%a.out
#SBATCH --error=/p/project1/sdlrs/tian1/wind_prediction/logs/folds_%A_%a.err

# First development batch only: 3 folds x 4 groups x 4 leads = 48 fits.
# Submit only after run_prepare_year.sh has passed readiness checks.
set -eo pipefail
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo 'Submit with sbatch run_folds.sh after annual preparation passes.' >&2
    exit 1
fi
cd /p/project1/sdlrs/tian1/wind_prediction
folds=(F01 F02 F03)
groups=(T01 S01 S02 A01)
task_id="$SLURM_ARRAY_TASK_ID"
if (( task_id < 0 || task_id > 11 )); then
    echo 'Array index must be 0..11.' >&2; exit 1
fi
export FOLD_ID="${folds[task_id / 4]}"
export EXPERIMENT="${groups[task_id % 4]}"
export FOLD_MANIFEST="${FOLD_MANIFEST:-./configs/seasonal_folds_2024.json}"
export SPATIAL_DIR="${SPATIAL_DIR:-./data/CERRA/spatial_2024_full}"
export SAVE_DIR="${SAVE_DIR:-./runs/seasonal/check_seed42}"
export PROTOCOL=v2_fixed SEED=42 INPUT_LEN=56 EPOCHS=100 LEARNING_RATE=0.001
export FIRST_FOUR=0 SMOKE_TEST=0 ALLOW_CONFIRMATION=0 CUDA_VISIBLE_DEVICES=-1
echo "[Fold check] fold=$FOLD_ID experiment=$EXPERIMENT seed=$SEED"
bash run_ablation.sh
