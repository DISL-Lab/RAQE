#!/usr/bin/env python3
"""Sparse BM25 NDCG@10 evaluation for RAQE expansions.

Rebuilds the sparse expanded query (original query repeated five times followed
by the pseudo-passage, Appendix A.2), retrieves with BM25 over the prebuilt BEIR
`.flat` indexes, and scores NDCG@10 against `qrels/test.tsv`.
"""
import argparse
import json
import sys
from pathlib import Path

# Allow `python evaluation/evaluate_raqe.py` from the repository root.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import raqe_jvm  # noqa: F401  (must precede the pyserini import)

import numpy as np
from pyserini.search.lucene import LuceneSearcher

NDCG_CUTOFF = 10


def load_qrels(path: Path):
    qrels: dict[str, dict[str, int]] = {}
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        if i == 0 and "query-id" in line.lower():
            continue
        qid, docid, score = line.split("\t")[:3]
        qrels.setdefault(str(qid), {})[str(docid)] = int(score)
    return qrels


def ndcg(qrels, rankings, k=NDCG_CUTOFF):
    """NDCG@k averaged over the judged queries that were actually retrieved for."""
    discount = lambda xs: sum(g / np.log2(i + 2) for i, g in enumerate(xs))
    values = []
    for qid, relevant in qrels.items():
        if qid not in rankings:
            continue
        gains = [relevant.get(docid, 0) for docid in rankings[qid][:k]]
        ideal = sorted(relevant.values(), reverse=True)[:k]
        values.append(discount(gains) / discount(ideal) if discount(ideal) else 0.0)
    return float(np.mean(values)) if values else 0.0


def index_key(dataset: str) -> str:
    return "msmarco-v1-passage" if dataset == "msmarco" else f"beir-v1.0.0-{dataset}.flat"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="evaluation/results/raqe")
    parser.add_argument("--dataset-root", default="datasets/IR")
    parser.add_argument("--datasets", default="nfcorpus,fiqa,fever,hotpotqa,msmarco,arguana,nq,scidocs")
    parser.add_argument("--retrieval-depth", type=int, default=NDCG_CUTOFF,
                        help="Documents retrieved per query; NDCG is always reported at 10.")
    parser.add_argument("--query-repeat", type=int, default=5,
                        help="Times the original query is repeated in the sparse expansion.")
    args = parser.parse_args()

    depth = max(args.retrieval_depth, NDCG_CUTOFF)
    summary = {}
    for dataset in [x for x in args.datasets.split(",") if x]:
        expansions_path = Path(args.results_dir) / dataset / "ours.json"
        if not expansions_path.exists():
            print(f"Skipping {dataset}: {expansions_path} not found; run generate_raqe.py first.")
            continue
        expansions = json.loads(expansions_path.read_text(encoding="utf-8"))
        searcher = LuceneSearcher.from_prebuilt_index(index_key(dataset))
        rankings = {}
        for row in expansions:
            expanded = " ".join([row["query"]] * args.query_repeat + [row["pseudo"]])
            rankings[str(row["qid"])] = [hit.docid for hit in searcher.search(expanded, k=depth)]
        qrels = load_qrels(Path(args.dataset_root) / dataset / dataset / "qrels" / "test.tsv")
        summary[dataset] = {"ndcg@10": ndcg(qrels, rankings), "queries": len(rankings)}
        print(f"{dataset}: NDCG@10 = {summary[dataset]['ndcg@10']:.4f} "
              f"over {summary[dataset]['queries']} queries")

    if summary:
        macro = float(np.mean([v["ndcg@10"] for v in summary.values()]))
        summary["macro_avg_ndcg@10"] = macro
        print(f"macro average NDCG@10 over {len(summary) - 1} datasets = {macro:.4f}")
        out = Path(args.results_dir) / "summary_sparse.json"
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
