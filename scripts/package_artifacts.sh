#!/usr/bin/env bash
set -euo pipefail

# Package the offline training artifact for release.
#
# Collects the curated qid lists, the teacher rollouts and the sparse RSDG reward
# caches from a local working tree into a single tarball plus a SHA256 manifest,
# in the exact layout `scripts/check_artifacts.py` and the training scripts
# expect. See configs/artifacts.md.
#
#   SRC_QIDS=/path/to/pr_final \
#   SRC_TRAIN=/path/to/Meta-Llama-3.1-70B-Instruct \
#   bash scripts/package_artifacts.sh
#
# Add INCLUDE_DENSE=1 to also ship the dense reward caches (Appendix D.1).
#
# By default the original query strings are stripped from the packaged files:
# MS MARCO, NFCorpus and FiQA grant no right to redistribute them, and the
# training scripts read the text from the local datasets/IR/ instead. See
# docs/LICENSES.md. Set KEEP_QUERY_TEXT=1 to leave them in.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
cd "$ROOT_DIR"

SRC_QIDS="${SRC_QIDS:?Set SRC_QIDS to the directory holding per_qid_ndcg_*_filtered_qids_pr.json}"
SRC_TRAIN="${SRC_TRAIN:?Set SRC_TRAIN to the teacher output root (contains <ds>/ and pseudo_ndcg_approx/)}"
TEACHER_NAME="${TEACHER_NAME:-Meta-Llama-3.1-70B-Instruct}"
OUT_DIR="${OUT_DIR:-dist}"
STAGE_DIR="${OUT_DIR}/artifacts"
DATASETS="${DATASETS:-nfcorpus fiqa fever hotpotqa msmarco}"
INCLUDE_DENSE="${INCLUDE_DENSE:-0}"
KEEP_QUERY_TEXT="${KEEP_QUERY_TEXT:-0}"

strip_query_text() {
  # Drop the `query` and `expanded_query` fields from a JSONL file in place.
  local path="$1"
  [[ "$KEEP_QUERY_TEXT" == "1" ]] && return 0
  python3 - "$path" <<'PY'
import json, sys
path = sys.argv[1]
out = []
with open(path, encoding="utf-8") as fh:
    for line in fh:
        if not line.strip():
            continue
        row = json.loads(line)
        row.pop("query", None)
        row.pop("expanded_query", None)
        out.append(json.dumps(row, ensure_ascii=False))
with open(path, "w", encoding="utf-8") as fh:
    fh.write("\n".join(out) + "\n")
PY
}

rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR/sft/pr_final" "$STAGE_DIR/train/${TEACHER_NAME}/pseudo_ndcg_approx"

for ds in $DATASETS; do
  cp "${SRC_QIDS}/per_qid_ndcg_${ds}_filtered_qids_pr.json" "$STAGE_DIR/sft/pr_final/"

  mkdir -p "$STAGE_DIR/train/${TEACHER_NAME}/${ds}"
  cp "${SRC_TRAIN}/${ds}/${ds}_pseudo_train.jsonl" "$STAGE_DIR/train/${TEACHER_NAME}/${ds}/"
  strip_query_text "$STAGE_DIR/train/${TEACHER_NAME}/${ds}/${ds}_pseudo_train.jsonl"

  cp "${SRC_TRAIN}/pseudo_ndcg_approx/${ds}_pseudo_ndcg_approx_sparse.jsonl" \
     "$STAGE_DIR/train/${TEACHER_NAME}/pseudo_ndcg_approx/"
  strip_query_text "$STAGE_DIR/train/${TEACHER_NAME}/pseudo_ndcg_approx/${ds}_pseudo_ndcg_approx_sparse.jsonl"

  if [[ "$INCLUDE_DENSE" == "1" ]]; then
    cp "${SRC_TRAIN}/pseudo_ndcg_approx/${ds}_pseudo_ndcg_approx_dense.jsonl" \
       "$STAGE_DIR/train/${TEACHER_NAME}/pseudo_ndcg_approx/"
    strip_query_text "$STAGE_DIR/train/${TEACHER_NAME}/pseudo_ndcg_approx/${ds}_pseudo_ndcg_approx_dense.jsonl"
  fi
done

cp NOTICE "$STAGE_DIR/NOTICE"

( cd "$STAGE_DIR" && find . -type f -print0 | sort -z | xargs -0 sha256sum ) > "${OUT_DIR}/SHA256SUMS"
tar -czf "${OUT_DIR}/raqe-artifacts.tar.gz" -C "$OUT_DIR" artifacts

echo
echo "Wrote ${OUT_DIR}/raqe-artifacts.tar.gz ($(du -h "${OUT_DIR}/raqe-artifacts.tar.gz" | cut -f1))"
echo "Wrote ${OUT_DIR}/SHA256SUMS"
echo
if [[ "$KEEP_QUERY_TEXT" == "1" ]]; then
  echo "WARNING: query strings were kept in the artifact; see docs/LICENSES.md before publishing."
else
  echo "Query strings stripped; consumers recover them from their own datasets/IR/."
fi
echo
echo "Consumers unpack it at the repository root:"
echo "  tar -xzf raqe-artifacts.tar.gz && python scripts/check_artifacts.py"
