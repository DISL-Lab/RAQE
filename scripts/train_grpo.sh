#!/usr/bin/env bash
set -euo pipefail

# Camera-ready Stage 2: cached RSDG offline GRPO.
# BASE_MODEL_DIR contains five teacher pseudo_train files and pseudo_ndcg_approx caches.
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
cd "$ROOT_DIR"

BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-artifacts/train/Meta-Llama-3.1-70B-Instruct}"
ADAPTER_DIR="${ADAPTER_DIR:-outputs/sft_llama31_8b}"
QIDS_PATH="${QIDS_PATH:-artifacts/sft/pr_final}"
OUTPUT_NAME="${OUTPUT_NAME:-raqe_llama31_8b_sparse}"
RETRIEVAL_TYPE="${RETRIEVAL_TYPE:-sparse}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"

accelerate launch --num_processes "$NUM_PROCESSES" grpo/step1/test_grpo_offline_real.py \
  --grpo --seed "${SEED:-42}" --shuffle-seed "${SEED:-42}" \
  --dataset nfcorpus,fiqa,fever,hotpotqa,msmarco --qids_path "$QIDS_PATH" \
  --base-model "$BASE_MODEL" --base-model-dir "$BASE_MODEL_DIR" --adapter-dir "$ADAPTER_DIR" \
  --retrieval_type "$RETRIEVAL_TYPE" --rollouts 8 --epochs 2 --lr 5e-6 \
  --batch-size "${BATCH_SIZE:-4}" --grad-accum "${GRAD_ACCUM:-4}" --clip-eps 0.2 --save-name "$OUTPUT_NAME" "$@"
