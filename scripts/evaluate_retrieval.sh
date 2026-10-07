#!/usr/bin/env bash
set -euo pipefail

# Generate one RAQE expansion per test query, then evaluate sparse BM25 NDCG@10.
#
# Generation is resumable at dataset granularity: a dataset whose ours.json
# already exists is skipped unless GENERATE_ARGS="--overwrite" is passed.
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
cd "$ROOT_DIR"

BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
MODEL_DIR="${MODEL_DIR:?Set MODEL_DIR to a trained adapter directory}"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_DIR")}"
DATASETS="${DATASETS:-nfcorpus,fiqa,fever,hotpotqa,msmarco,arguana,nq,scidocs}"
RESULTS_DIR="${RESULTS_DIR:-evaluation/results/${MODEL_NAME}}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
BATCH_SIZE="${BATCH_SIZE:-8}"

python evaluation/generate_raqe.py \
  --model-dir "$MODEL_DIR" --base-model "$BASE_MODEL" --datasets "$DATASETS" \
  --output-dir "$RESULTS_DIR" --max-new-tokens "$MAX_NEW_TOKENS" \
  --batch-size "$BATCH_SIZE" ${GENERATE_ARGS:-}

python evaluation/evaluate_raqe.py --results-dir "$RESULTS_DIR" --datasets "$DATASETS"
