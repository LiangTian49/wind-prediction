#!/bin/bash
#SBATCH --job-name=cerra_ablation
#SBATCH --partition=dp-dam
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --chdir=/p/project1/sdlrs/tian1/wind_prediction
#SBATCH --output=/p/project1/sdlrs/tian1/wind_prediction/logs/ablation_%j.out
#SBATCH --error=/p/project1/sdlrs/tian1/wind_prediction/logs/ablation_%j.err

# Manual submission: sbatch run_ablation.sh
# One-epoch check: sbatch --export=ALL,SMOKE_TEST=1 run_ablation.sh
# First four configurations: sbatch --export=ALL,FIRST_FOUR=1 run_ablation.sh
# T01: sbatch --job-name=cerra_T01 --export=ALL,EXPERIMENT=T01 run_ablation.sh
# Spatial: sbatch --export=ALL,EXPERIMENT=S01,SPATIAL_DIR=./data/CERRA/spatial_JOBID run_ablation.sh
# CPU-only allocation: sbatch --partition=dp-cn --gres=none run_ablation.sh
# If required by your allocation: sbatch --account=YOUR_ACCOUNT run_ablation.sh
# All experiments run sequentially within this job's four-hour limit.
set -eo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo 'Submit with sbatch run_ablation.sh; do not execute on the login node.' >&2
    exit 1
fi

cd /p/project1/sdlrs/tian1/wind_prediction
# wind-env was created with Python/3.12.3 from Stages/2025.
# Stages/2026 + GCCcore/.14.3.0 does not supply its libpython3.12.so.1.0.
module --force purge
module load Stages/2025 GCCcore/.13.3.0 Python/3.12.3
source /p/project1/sdlrs/tian1/wind_prediction/wind-env/bin/activate
set -u

PYTHON=/p/project1/sdlrs/tian1/wind_prediction/wind-env/bin/python
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK}"
export TF_NUM_INTEROP_THREADS=1
export TF_FORCE_GPU_ALLOW_GROWTH=true

DATA_PATH="${DATA_PATH:-./data/CERRA/cerra_100_points_200km_summer_roi_2024.csv}"
PROTOCOL="${PROTOCOL:-v1}"
case "$PROTOCOL" in
    v1) SAVE_DIR="${SAVE_DIR:-./runs/ablation}" ;;
    v2) SAVE_DIR="${SAVE_DIR:-./runs/v2}" ;;
    v2_fixed)
        SAVE_DIR="${SAVE_DIR:-./runs/seasonal}"
        SPATIAL_DIR="${SPATIAL_DIR:-./data/CERRA/spatial_2024_full}"
        DATA_PATH="${SPATIAL_DIR}/center.csv"
        : "${FOLD_ID:?Set FOLD_ID for fixed-date training.}"
        if [[ ! "$FOLD_ID" =~ ^[A-Za-z0-9_-]+$ ]]; then
            echo 'Invalid FOLD_ID.' >&2; exit 1
        fi
        SAVE_DIR="${SAVE_DIR}/${FOLD_ID}"
        ;;
    *) echo 'PROTOCOL must be v1, v2 or v2_fixed.' >&2; exit 1 ;;
esac
fold_args=()
if [[ "$PROTOCOL" == v2_fixed ]]; then
    fold_args=(--fold_manifest "${FOLD_MANIFEST:-./configs/seasonal_folds_2024.json}" --fold_id "$FOLD_ID")
    case "${ALLOW_CONFIRMATION:-0}" in
        0) ;;
        1) fold_args+=(--allow_confirmation) ;;
        *) echo 'ALLOW_CONFIRMATION must be 0 or 1.' >&2; exit 1 ;;
    esac
fi
INPUT_LEN="${INPUT_LEN:-56}"
EPOCHS="${EPOCHS:-100}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
SEED="${SEED:-42}"
JOB_TAG="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    JOB_TAG="${JOB_TAG}_${SLURM_ARRAY_TASK_ID}"
fi
SMOKE_TEST="${SMOKE_TEST:-0}"
FIRST_FOUR="${FIRST_FOUR:-0}"
EXPERIMENT="${EXPERIMENT:-all}"
case "$EXPERIMENT" in
    all|T01|S01|S02|A01) ;;
    *) echo 'EXPERIMENT must be all, T01, S01, S02 or A01.' >&2; exit 1 ;;
esac
if [[ "$EXPERIMENT" != all && ( "$FIRST_FOUR" != 0 || "$SMOKE_TEST" != 0 ) ]]; then
    echo 'A named EXPERIMENT cannot be combined with FIRST_FOUR or SMOKE_TEST.' >&2
    exit 1
fi
model_types=(lstm bilstm)
feature_sets=(wind wind_time)
pred_lens=(1 2 4 8)
if [[ "$FIRST_FOUR" == 1 ]]; then
    model_types=(lstm)
    feature_sets=(wind)
    SAVE_DIR="${SAVE_DIR}/first_four_${JOB_TAG}"
    echo '[First four] LSTM, ws_10 only, forecast leads 1/2/4/8; includes Persistence evaluation.'
elif [[ "$FIRST_FOUR" != 0 ]]; then
    echo 'FIRST_FOUR must be 0 or 1.' >&2
    exit 1
fi
if [[ "$FIRST_FOUR" == 1 && "$SMOKE_TEST" == 1 ]]; then
    echo 'Choose FIRST_FOUR=1 or SMOKE_TEST=1, not both.' >&2
    exit 1
fi
if [[ "$SMOKE_TEST" == 1 ]]; then
    model_types=(lstm)
    feature_sets=(wind_time)
    pred_lens=(1)
    EPOCHS=1
    SAVE_DIR="${SAVE_DIR}/smoke_${JOB_TAG}"
    echo '[Smoke test] One LSTM configuration, one epoch; uses the full input CSV.'
elif [[ "$SMOKE_TEST" != 0 ]]; then
    echo 'SMOKE_TEST must be 0 or 1.' >&2
    exit 1
fi
if [[ "$EXPERIMENT" == T01 ]]; then
    model_types=(lstm)
    feature_sets=(wind_time)
    SAVE_DIR="${SAVE_DIR}/T01_${JOB_TAG}"
    echo '[T01] LSTM, ws_10 + hour_sin + hour_cos, leads 1/2/4/8; includes B01 Persistence.'
fi
if [[ "$EXPERIMENT" == S01 || "$EXPERIMENT" == S02 || "$EXPERIMENT" == A01 ]]; then
    : "${SPATIAL_DIR:?Set SPATIAL_DIR to the completed spatial extraction directory.}"
    DATA_PATH="${SPATIAL_DIR}/${EXPERIMENT}.csv"
    # Read only extraction metadata. Engineered hour columns are added by train.py.
    FEATURES_TEXT=$("$PYTHON" - "$SPATIAL_DIR/metadata.json" "$EXPERIMENT" <<'PY'
import json
import sys
from pathlib import Path

metadata = json.loads(Path(sys.argv[1]).read_text())
if metadata.get('status') != 'complete':
    raise SystemExit('Spatial extraction has not completed successfully.')
if metadata.get('reference_check', {}).get('matched_rows', 0) <= 0:
    raise SystemExit('Spatial centers have not been verified against the original CSV.')
features = metadata['feature_sets'][sys.argv[2]]
if not features or len(set(features)) != len(features) or 'ws_10' not in features:
    raise SystemExit('Invalid feature list in spatial metadata.')
print('\n'.join(features))
PY
)
    mapfile -t spatial_features <<< "$FEATURES_TEXT"
    model_types=(lstm)
    feature_sets=(spatial)
    SAVE_DIR="${SAVE_DIR}/${EXPERIMENT}_${JOB_TAG}"
    echo "[$EXPERIMENT] Input=$DATA_PATH; features=${spatial_features[*]}; leads=1/2/4/8"
fi

if [[ ! -f "$DATA_PATH" ]]; then
    echo "Missing CERRA CSV: $DATA_PATH" >&2
    exit 1
fi
if [[ "$PROTOCOL" == v2_fixed ]]; then
    "$PYTHON" - "$SPATIAL_DIR" "${FOLD_MANIFEST:-./configs/seasonal_folds_2024.json}" <<'PY'
import sys
from temporal_folds import verify_readiness

report = verify_readiness(sys.argv[1], sys.argv[2])
print('[Readiness] Verified annual files and fixed manifest:', report['report_dir'])
PY
fi

# Diagnose the runtime inside the allocation, before beginning experiments.
# Device detection runs inside Slurm, just like the training steps.
# Capture only the device choice; diagnostics go to the job's stderr log.
srun --ntasks=1 "$PYTHON" -m py_compile train.py
srun --ntasks=1 "$PYTHON" -m pip check
DEVICE_MODE=$(srun --ntasks=1 "$PYTHON" - <<'PY'
import sys
import tensorflow as tf

print('Python:', sys.executable, file=sys.stderr)
print('TensorFlow:', tf.__version__, file=sys.stderr)
try:
    gpus = tf.config.list_physical_devices('GPU')
    print('Visible GPUs:', gpus, file=sys.stderr)
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        gpus = tf.config.list_logical_devices('GPU')
except (RuntimeError, ValueError, tf.errors.OpError) as error:
    print(f'GPU initialization failed: {error}', file=sys.stderr)
    gpus = []
print('GPU' if gpus else 'CPU')
PY
)
if [[ "$DEVICE_MODE" == CPU ]]; then
    # Apply the decision to every subsequent Python/srun process.
    export CUDA_VISIBLE_DEVICES=-1
    echo '[Device] GPU unavailable; continuing all experiments on CPU.'
elif [[ "$DEVICE_MODE" == GPU ]]; then
    echo '[Device] GPU initialized successfully; using TensorFlow GPU execution.'
else
    echo "Unexpected device detection result: $DEVICE_MODE" >&2
    exit 1
fi

for model_type in "${model_types[@]}"; do
    for feature_set in "${feature_sets[@]}"; do
        if [[ "$feature_set" == wind ]]; then
            features=(ws_10)
        elif [[ "$feature_set" == spatial ]]; then
            features=("${spatial_features[@]}")
        else
            features=(ws_10 hour_sin hour_cos)
        fi
        experiment_label="$EXPERIMENT"
        if [[ "$EXPERIMENT" == all ]]; then
            experiment_label=custom
            if [[ "$model_type" == lstm && "$feature_set" == wind ]]; then
                experiment_label=T00
            elif [[ "$model_type" == lstm && "$feature_set" == wind_time ]]; then
                experiment_label=T01
            fi
        fi
        for pred_len in "${pred_lens[@]}"; do
            echo "[$(date --iso-8601=seconds)] model=$model_type features=$feature_set input=$INPUT_LEN lead=$pred_len seed=$SEED"
            srun --ntasks=1 "$PYTHON" -u train.py \
                --model_type "$model_type" \
                --input_len "$INPUT_LEN" \
                --pred_len "$pred_len" \
                --features "${features[@]}" \
                --lr "$LEARNING_RATE" \
                --save_dir "$SAVE_DIR" \
                --data_path "$DATA_PATH" \
                --epochs "$EPOCHS" \
                --seed "$SEED" \
                --protocol "$PROTOCOL" \
                "${fold_args[@]}" \
                --experiment "$experiment_label"
        done
    done
done

echo "Completed ablations. Results: $SAVE_DIR/results.csv"
