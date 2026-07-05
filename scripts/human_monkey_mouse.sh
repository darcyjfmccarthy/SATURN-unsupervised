#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
DEVICE_NUM="${DEVICE_NUM:-0}"

IN_DATA="${IN_DATA:-data/human_monkey_mouse.csv}"
WORK_DIR="${WORK_DIR:-out/human_monkey_mouse_benchmark/}"
CENTROIDS_INIT_PATH="${CENTROIDS_INIT_PATH:-out/human_monkey_mouse_benchmark_centroids_seed${SEED}.pkl}"

HV_GENES="${HV_GENES:-2000}"
NUM_MACROGENES="${NUM_MACROGENES:-200}"
MODEL_DIM="${MODEL_DIM:-256}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"

PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-20}"
METRIC_EPOCHS="${METRIC_EPOCHS:-20}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-512}"
METRIC_BATCH_SIZE="${METRIC_BATCH_SIZE:-512}"

PRETRAIN_LR="${PRETRAIN_LR:-0.0005}"
METRIC_LR="${METRIC_LR:-0.001}"
PE_SIM_PENALTY="${PE_SIM_PENALTY:-0.2}"
L1_PENALTY="${L1_PENALTY:-0.0}"
CENTROID_SCORE_FUNC="${CENTROID_SCORE_FUNC:-default}"
PRETRAIN="${PRETRAIN:-true}"
PRETRAIN_MODEL_PATH="${PRETRAIN_MODEL_PATH:-out/human_monkey_mouse_benchmark_pretrain_seed${SEED}.pt}"
METRIC_MODEL_PATH="${METRIC_MODEL_PATH:-out/human_monkey_mouse_benchmark_metric_seed${SEED}.pt}"
POLLING_FREQ="${POLLING_FREQ:-5}"

mkdir -p \
  "$WORK_DIR" \
  "$(dirname "$CENTROIDS_INIT_PATH")" \
  "$(dirname "$PRETRAIN_MODEL_PATH")" \
  "$(dirname "$METRIC_MODEL_PATH")"

python train-saturn.py \
  --in_data "$IN_DATA" \
  --work_dir "$WORK_DIR" \
  --device "$DEVICE" \
  --device_num "$DEVICE_NUM" \
  --seed "$SEED" \
  --ref_label_col cellType \
  --hv_genes "$HV_GENES" \
  --num_macrogenes "$NUM_MACROGENES" \
  --model_dim "$MODEL_DIM" \
  --hidden_dim "$HIDDEN_DIM" \
  --pretrain_epochs "$PRETRAIN_EPOCHS" \
  --epochs "$METRIC_EPOCHS" \
  --pretrain_batch_size "$PRETRAIN_BATCH_SIZE" \
  --batch_size "$METRIC_BATCH_SIZE" \
  --pretrain_lr "$PRETRAIN_LR" \
  --metric_lr "$METRIC_LR" \
  --pretrain "$PRETRAIN" \
  --pretrain_model_path "$PRETRAIN_MODEL_PATH" \
  --metric_model_path "$METRIC_MODEL_PATH" \
  --polling_freq "$POLLING_FREQ" \
  --embedding_model ESM1b \
  --centroid_score_func "$CENTROID_SCORE_FUNC" \
  --centroids_init_path "$CENTROIDS_INIT_PATH" \
  --pe_sim_penalty "$PE_SIM_PENALTY" \
  --l1_penalty "$L1_PENALTY"
