#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/saturn-jax-numba}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/saturn-jax-matplotlib}"

PYTHON="${PYTHON:-python}"
VALIDATION_PROFILE="${VALIDATION_PROFILE:-short}"
case "$VALIDATION_PROFILE" in
  short) DEFAULT_PRETRAIN_EPOCHS=2; DEFAULT_METRIC_EPOCHS=2 ;;
  full) DEFAULT_PRETRAIN_EPOCHS=20; DEFAULT_METRIC_EPOCHS=30 ;;
  *) echo "VALIDATION_PROFILE must be short or full" >&2; exit 2 ;;
esac
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-$DEFAULT_PRETRAIN_EPOCHS}"
METRIC_EPOCHS="${METRIC_EPOCHS:-$DEFAULT_METRIC_EPOCHS}"
WORK_DIR="${WORK_DIR:-out/human_monkey_mouse_jax}"
PRETRAIN_MODEL_PATH="${PRETRAIN_MODEL_PATH:-$WORK_DIR/shared/pretrain_orbax}"
METRIC_MODEL_PATH="${METRIC_MODEL_PATH:-$WORK_DIR/shared/metric_orbax}"
PRETRAIN_NATIVE="$PRETRAIN_MODEL_PATH"
METRIC_NATIVE="$METRIC_MODEL_PATH"
if [[ "$PRETRAIN_NATIVE" == *.pt ]]; then PRETRAIN_NATIVE="${PRETRAIN_NATIVE%.pt}_orbax"; fi
if [[ "$METRIC_NATIVE" == *.pt ]]; then METRIC_NATIVE="${METRIC_NATIVE%.pt}_orbax"; fi
PRETRAIN_NATIVE="${ORBAX_CHECKPOINT_DIR:-$PRETRAIN_NATIVE}"
OPTIONS=()
if [[ -n "${MIXED_PRECISION:-}" ]]; then OPTIONS+=(--mixed-precision "$MIXED_PRECISION"); fi
if [[ -n "${PRETRAIN_MIXED_PRECISION:-}" ]]; then OPTIONS+=(--pretrain-mixed-precision "$PRETRAIN_MIXED_PRECISION"); fi
if [[ "${PRETRAIN_DISTRIBUTED:-0}" == "1" ]]; then OPTIONS+=(--pretrain-distributed); fi
if [[ "${METRIC_DISTRIBUTED:-0}" == "1" ]]; then OPTIONS+=(--metric-distributed); fi
if [[ -n "${EXPECTED_LOCAL_DEVICE_COUNT:-}" ]]; then
  OPTIONS+=(--expected-local-device-count "$EXPECTED_LOCAL_DEVICE_COUNT")
fi
if [[ "${RESUME:-0}" == "1" ]]; then
  OPTIONS+=(--resume "$PRETRAIN_NATIVE")
  if [[ -d "$METRIC_NATIVE/checkpoints" ]]; then OPTIONS+=(--metric-resume "$METRIC_NATIVE"); fi
fi
if [[ "${KEEP_PYTORCH_COMPAT_CHECKPOINT:-0}" == "1" ]]; then
  PRETRAIN_COMPAT="$PRETRAIN_MODEL_PATH"
  METRIC_COMPAT="$METRIC_MODEL_PATH"
  if [[ "$PRETRAIN_COMPAT" != *.pt ]]; then PRETRAIN_COMPAT="$PRETRAIN_COMPAT.pt"; fi
  if [[ "$METRIC_COMPAT" != *.pt ]]; then METRIC_COMPAT="$METRIC_COMPAT.pt"; fi
  OPTIONS+=(--pytorch-compat-pretrain-checkpoint "$PRETRAIN_COMPAT")
  if [[ "$METRIC_EPOCHS" != "0" ]]; then OPTIONS+=(--pytorch-compat-metric-checkpoint "$METRIC_COMPAT"); fi
fi

"$PYTHON" scripts/train_saturn_jax.py \
  --in_data "${IN_DATA:-data/human_monkey_mouse.csv}" \
  --work_dir "$WORK_DIR" --device "${DEVICE:-cuda}" --device_num "${DEVICE_NUM:-0}" \
  --seed "${SEED:-0}" --ref_label_col cellType \
  --hv_genes "${HV_GENES:-2000}" --num_macrogenes "${NUM_MACROGENES:-200}" \
  --model_dim "${MODEL_DIM:-256}" --hidden_dim "${HIDDEN_DIM:-256}" \
  --pretrain_epochs "$PRETRAIN_EPOCHS" --epochs "$METRIC_EPOCHS" \
  --pretrain_batch_size "${PRETRAIN_BATCH_SIZE:-512}" --batch_size "${METRIC_BATCH_SIZE:-512}" \
  --pretrain_lr "${PRETRAIN_LR:-0.0005}" --metric_lr "${METRIC_LR:-0.001}" \
  --pretrain "${PRETRAIN:-true}" --pretrain_model_path "$PRETRAIN_NATIVE" \
  --metric_model_path "$METRIC_NATIVE" --polling_freq "${POLLING_FREQ:-5}" \
  --embedding_model ESM1b --centroid_score_func "${CENTROID_SCORE_FUNC:-default}" \
  --centroids_init_path "${CENTROIDS_INIT_PATH:-$WORK_DIR/shared/centroids.npz}" \
  --pe_sim_penalty "${PE_SIM_PENALTY:-0.2}" --l1_penalty "${L1_PENALTY:-0.0}" "${OPTIONS[@]}"
