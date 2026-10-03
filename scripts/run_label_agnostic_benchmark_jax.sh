#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/saturn-jax-numba}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/saturn-jax-matplotlib}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-1}"
export PYTHON="${PYTHON:-python}"
OUT_DIR="$(realpath -m "${OUT_DIR:-out/human_monkey_mouse_jax_benchmark}")"
SHARED_DIR="$OUT_DIR/shared"
BASELINE_DIR="$OUT_DIR/baseline"
export VALIDATION_PROFILE="${VALIDATION_PROFILE:-short}"
case "$VALIDATION_PROFILE" in
  short) DEFAULT_EPOCHS=2 ;;
  full) DEFAULT_EPOCHS=30 ;;
  *) echo "VALIDATION_PROFILE must be short or full" >&2; exit 2 ;;
esac
EPOCHS="${EPOCHS:-$DEFAULT_EPOCHS}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-512}"
PRETRAIN_MODEL="$SHARED_DIR/pretrain_orbax"
METRIC_MODEL="$BASELINE_DIR/metric_orbax"

WORK_DIR="$BASELINE_DIR" PRETRAIN_MODEL_PATH="$PRETRAIN_MODEL" METRIC_MODEL_PATH="$METRIC_MODEL" \
  CENTROIDS_INIT_PATH="$SHARED_DIR/centroids_seed${SEED}.npz" \
  METRIC_EPOCHS="$EPOCHS" METRIC_BATCH_SIZE="$BATCH_SIZE" SEED="$SEED" DEVICE="$DEVICE" \
  bash scripts/human_monkey_mouse_jax.sh

"$PYTHON" scripts/prepare_label_agnostic_artifacts.py \
  --pretrain-adata "$BASELINE_DIR/saturn_results/adata_pretrain.h5ad" \
  --artifact "$SHARED_DIR/label_free_artifact.npz" --triplets "$SHARED_DIR/evaluation_triplets.npz" \
  --metadata "$SHARED_DIR/artifact_metadata.json" --seed "$SEED" --batch-size "$BATCH_SIZE"

PRETRAIN_CHECKPOINT="$("$PYTHON" -c 'import sys; from jax_saturn.distributed.checkpoint import latest_checkpoint; print(latest_checkpoint(sys.argv[1]))' "$PRETRAIN_MODEL")"
for OBJECTIVE in infonce mmd ot; do
  OPTIONS=()
  if [[ -n "${MIXED_PRECISION:-}" ]]; then OPTIONS+=(--mixed-precision "$MIXED_PRECISION"); fi
  if [[ "${LABEL_DISTRIBUTED:-0}" == "1" ]]; then OPTIONS+=(--distributed); fi
  if [[ -n "${EXPECTED_LOCAL_DEVICE_COUNT:-}" ]]; then
    OPTIONS+=(--expected-local-device-count "$EXPECTED_LOCAL_DEVICE_COUNT")
  fi
  if [[ "${RESUME:-0}" == "1" && -d "$OUT_DIR/$OBJECTIVE/checkpoints" ]]; then
    LAST_CHECKPOINT="$("$PYTHON" -c 'import sys; from jax_saturn.distributed.checkpoint import latest_checkpoint; print(latest_checkpoint(sys.argv[1]))' "$OUT_DIR/$OBJECTIVE")"
    OPTIONS+=(--resume "$LAST_CHECKPOINT")
  fi
  "$PYTHON" scripts/train_label_agnostic_jax.py \
    --objective "$OBJECTIVE" --artifact "$SHARED_DIR/label_free_artifact.npz" \
    --pretrain-checkpoint "$PRETRAIN_CHECKPOINT" --output-dir "$OUT_DIR/$OBJECTIVE" \
    --device "$DEVICE" --device-num "${DEVICE_NUM:-0}" --seed "$SEED" --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" --hidden-dim "${HIDDEN_DIM:-256}" --model-dim "${MODEL_DIM:-256}" "${OPTIONS[@]}"
done

"$PYTHON" scripts/evaluate_label_agnostic_benchmark.py --root "$OUT_DIR" \
  --truth-adata "$BASELINE_DIR/saturn_results/adata_pretrain.h5ad" \
  --triplets "$SHARED_DIR/evaluation_triplets.npz" --seed "$SEED"
