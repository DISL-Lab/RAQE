#!/usr/bin/env bash
set -euo pipefail

# Camera-ready Stage 1: LoRA supervised distillation (5 epochs).
#
# The offline artifact supplies the curated qid lists and the teacher
# pseudo-passage rollouts; see docs/DATA.md.
#
# With the defaults below (6,453 curated queries, per-device batch 1,
# gradient accumulation 16, 2 processes) each rank sees ceil(6453/2)=3227 examples,
# i.e. ceil(3227/16)=202 updates per epoch and 5 x 202 = 1,010 in total, matching
# the Stage-1 step count reported in Appendix A.2.
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
cd "$ROOT_DIR"

BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
PSEUDO_ROOT="${PSEUDO_ROOT:-artifacts/train/Meta-Llama-3.1-70B-Instruct}"
FILTERED_DIR="${FILTERED_DIR:-artifacts/sft/pr_final}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/sft_llama31_8b}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
EPOCHS="${EPOCHS:-5}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
LORA_R="${LORA_R:-32}"
SEED="${SEED:-42}"

accelerate launch --num_processes "$NUM_PROCESSES" --mixed_precision bf16 sft/train_raqe.py \
  --model "$BASE_MODEL" --pseudo-root "$PSEUDO_ROOT" --filtered-dir "$FILTERED_DIR" \
  --output-dir "$OUTPUT_DIR" --epochs "$EPOCHS" --learning-rate "$LEARNING_RATE" \
  --per-device-batch-size "$PER_DEVICE_BATCH_SIZE" --gradient-accumulation-steps "$GRAD_ACCUM" \
  --lora-r "$LORA_R" --seed "$SEED" "$@"
