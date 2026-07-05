#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-1}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/macrogenes-numba-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/macrogenes-matplotlib}"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"

OUT_DIR="$(realpath -m "${OUT_DIR:-out/label_agnostic_benchmark_clean}")"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
DEVICE_NUM="${DEVICE_NUM:-0}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-512}"
FORCE="${FORCE:-0}"

SHARED_DIR="$OUT_DIR/shared"
BASELINE_DIR="$OUT_DIR/baseline"
PRETRAIN_MODEL="$SHARED_DIR/pretrain_model.pt"
BASELINE_MODEL="$BASELINE_DIR/final_model.pt"
PRETRAIN_ADATA="$BASELINE_DIR/saturn_results/adata_pretrain.h5ad"
BASELINE_ADATA="$BASELINE_DIR/saturn_results/final_adata.h5ad"
LABEL_FREE_ARTIFACT="$SHARED_DIR/label_free_artifact.npz"
EVAL_TRIPLETS="$SHARED_DIR/evaluation_triplets.npz"

mkdir -p "$SHARED_DIR" "$BASELINE_DIR"

if [[ "$FORCE" == "1" || ! -f "$BASELINE_ADATA" || ! -f "$PRETRAIN_MODEL" ]]; then
  PRETRAIN=true
  if [[ -f "$PRETRAIN_MODEL" ]]; then
    PRETRAIN=false
  fi
  SEED="$SEED" \
  DEVICE="$DEVICE" \
  DEVICE_NUM="$DEVICE_NUM" \
  WORK_DIR="$BASELINE_DIR/" \
  CENTROIDS_INIT_PATH="$SHARED_DIR/centroids_seed${SEED}.pkl" \
  PRETRAIN_MODEL_PATH="$PRETRAIN_MODEL" \
  METRIC_MODEL_PATH="$BASELINE_MODEL" \
  PRETRAIN="$PRETRAIN" \
  METRIC_EPOCHS="$EPOCHS" \
  METRIC_BATCH_SIZE="$BATCH_SIZE" \
  POLLING_FREQ=5 \
    bash scripts/run_saturn_tiny_benchmark.sh
fi

if [[ "$FORCE" == "1" || ! -f "$LABEL_FREE_ARTIFACT" || ! -f "$EVAL_TRIPLETS" ]]; then
  python scripts/prepare_label_agnostic_artifacts.py \
    --pretrain-adata "$PRETRAIN_ADATA" \
    --artifact "$LABEL_FREE_ARTIFACT" \
    --triplets "$EVAL_TRIPLETS" \
    --metadata "$SHARED_DIR/artifact_metadata.json" \
    --seed "$SEED" \
    --batch-size "$BATCH_SIZE"
fi

for OBJECTIVE in infonce mmd ot; do
  TRIAL_DIR="$OUT_DIR/$OBJECTIVE"
  NEEDS_RUN=0
  if [[ \
    "$FORCE" == "1" \
    || ! -f "$TRIAL_DIR/final_embeddings.npz" \
    || ! -f "$TRIAL_DIR/run_summary.json" \
  ]]; then
    NEEDS_RUN=1
  elif ! grep -q '"trainer_version": 4' "$TRIAL_DIR/run_summary.json"; then
    NEEDS_RUN=1
  fi
  if [[ "$NEEDS_RUN" == "1" ]]; then
    python scripts/train_label_agnostic.py \
      --objective "$OBJECTIVE" \
      --artifact "$LABEL_FREE_ARTIFACT" \
      --pretrain-checkpoint "$PRETRAIN_MODEL" \
      --output-dir "$TRIAL_DIR" \
      --device "$DEVICE" \
      --device-num "$DEVICE_NUM" \
      --seed "$SEED" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE"
  fi
done

python scripts/evaluate_label_agnostic_benchmark.py \
  --root "$OUT_DIR" \
  --truth-adata "$PRETRAIN_ADATA" \
  --triplets "$EVAL_TRIPLETS" \
  --seed "$SEED"

LABEL_AGNOSTIC_OUT="$OUT_DIR" \
  jupyter nbconvert \
    --to notebook \
    --execute notebooks/label_agnostic_benchmark.ipynb \
    --output-dir "$OUT_DIR" \
    --output "label_agnostic_benchmark.executed.ipynb" \
    --ExecutePreprocessor.timeout=-1

echo "Benchmark complete: $OUT_DIR"
echo "Comparison table: $OUT_DIR/comparison.csv"
echo "Acceptance result: $OUT_DIR/acceptance.json"
