#!/bin/bash
#SBATCH --job-name=cerra_winter_confirm
#SBATCH --partition=dp-cn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --array=0-19%4
#SBATCH --chdir=/p/project1/sdlrs/tian1/wind_prediction
#SBATCH --output=/p/project1/sdlrs/tian1/wind_prediction/logs/confirmation_%A_%a.out
#SBATCH --error=/p/project1/sdlrs/tian1/wind_prediction/logs/confirmation_%A_%a.err

# F04 only: four unchanged groups x five seeds x four leads = 80 fits.
# Submit after development review and creation of confirmation_freeze_2024.json.
set -eo pipefail
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo 'Submit with sbatch run_confirmation.sh.' >&2
    exit 1
fi
cd /p/project1/sdlrs/tian1/wind_prediction
module --force purge
module load Stages/2025 GCCcore/.13.3.0 Python/3.12.3
source /p/project1/sdlrs/tian1/wind_prediction/wind-env/bin/activate
set -u
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=-1 PYTHONUNBUFFERED=1
CONFIRMATION_FREEZE="${CONFIRMATION_FREEZE:-./configs/confirmation_freeze_2024.json}"
wind-env/bin/python - "$CONFIRMATION_FREEZE" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
from temporal_folds import load_manifest, project_path, verify_readiness

freeze = json.loads(project_path(sys.argv[1]).read_text())
manifest, folds, digest = load_manifest(freeze['fold_manifest'])
if freeze.get('status') != 'frozen_for_confirmation' or freeze['fold_manifest_sha256'] != digest:
    raise SystemExit('Confirmation freeze does not match the current fold manifest.')
if folds['F04'].role != 'locked_confirmation':
    raise SystemExit('F04 must remain a reserved confirmation fold.')
if (freeze['groups'] != manifest['groups'] or freeze['seeds'] != manifest['seeds']
        or freeze['forecast_steps'] != manifest['forecast_steps']
        or freeze['hyperparameters'] != manifest['hyperparameters']):
    raise SystemExit('Confirmation settings differ from the frozen development design.')
for name, expected in {**freeze['code_sha256'], **freeze['development_report_sha256']}.items():
    if hashlib.sha256(project_path(name).read_bytes()).hexdigest() != expected:
        raise SystemExit(f'Frozen source/report changed: {name}')
verify_readiness(freeze['spatial_dir'], freeze['fold_manifest'])
print('[Confirmation] Frozen method and completed development report verified.')
PY

groups=(T01 S01 S02 A01)
seeds=(42 43 44 45 46)
task_id="$SLURM_ARRAY_TASK_ID"
if (( task_id < 0 || task_id > 19 )); then
    echo 'Confirmation task index must be 0..19.' >&2; exit 1
fi
export FOLD_ID=F04 EXPERIMENT="${groups[task_id / 5]}" SEED="${seeds[task_id % 5]}"
export FOLD_MANIFEST=./configs/seasonal_folds_2024.json
export SPATIAL_DIR=./data/CERRA/spatial_2024_full
export SAVE_DIR="${SAVE_DIR:-./runs/seasonal/confirmation}"
export PROTOCOL=v2_fixed INPUT_LEN=56 EPOCHS=100 LEARNING_RATE=0.001
export FIRST_FOUR=0 SMOKE_TEST=0 ALLOW_CONFIRMATION=1
echo "[Confirmation] fold=$FOLD_ID experiment=$EXPERIMENT seed=$SEED"
bash run_ablation.sh
