#!/usr/bin/env python3
"""Reranker-Shaped Discounted Gain (RSDG) over cached teacher rollouts.

Implements Eq. (1)-(2) of the paper for the sparse (BM25) setting:

    s(q, d)      = max(CE(q, d), 0)                                    (Eq. 1)
    RSDG@k(q')   = sum_i s(q, d'_i) / log2(i+1)                        (Eq. 2)
                   / sum_j s(q, d'_(j)) / log2(j+1)

where ``i`` indexes the retriever ranking induced by the expanded query
``q' = concat(q, e)`` and ``j`` indexes the same documents sorted by reranker
score (the surrogate for IDCG). Cross-encoder scores are always conditioned on
the *original* query q, never on q'.

For every ``(dataset, qid)`` this script scores the original query once
(``base_ndcg``) and each of the g cached teacher rollouts (``approx_ndcg``), and
writes one JSONL record per rollout. The resulting files are exactly the reward
cache consumed by ``grpo/step1/test_grpo_offline_real.py``.

Input :  <pseudo-root>/<dataset>/<dataset>_pseudo_train.jsonl
Output:  <out-dir>/<dataset>_pseudo_ndcg_approx_sparse.jsonl

Example
-------
    python rsdg/compute_rsdg.py \
        --datasets nfcorpus,fiqa,fever,hotpotqa,msmarco \
        --pseudo-root artifacts/train/Meta-Llama-3.1-70B-Instruct \
        --out-dir artifacts/train/Meta-Llama-3.1-70B-Instruct/pseudo_ndcg_approx \
        --topk 30
"""

import argparse
import json
import math
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import raqe_jvm  # noqa: E402,F401  (must precede the pyserini import)

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

from pyserini.search.lucene import LuceneSearcher  # noqa: E402

_SEARCHER_CACHE: Dict[str, LuceneSearcher] = {}

DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"


# ---------------------------------------------------------------------------
# Query expansion and retrieval
# ---------------------------------------------------------------------------

def query2doc_expand(query: str, passage: str, repeat: int = 5) -> str:
    """Sparse expansion used everywhere in RAQE: the query repeated `repeat`
    times followed by the pseudo-passage (Appendix A.2, "Inference")."""
    parts = [p for p in [query] * int(repeat) + [passage] if p]
    return " ".join(parts)


def index_key_for_dataset(dataset: str) -> str:
    if dataset == "msmarco":
        return "msmarco-v1-passage"
    return f"beir-v1.0.0-{dataset}.flat"


def get_searcher(dataset: str) -> LuceneSearcher:
    key = index_key_for_dataset(dataset)
    searcher = _SEARCHER_CACHE.get(key)
    if searcher is None:
        print(f"[rsdg] loading prebuilt Lucene index '{key}'")
        searcher = LuceneSearcher.from_prebuilt_index(key)
        _SEARCHER_CACHE[key] = searcher
    return searcher


def retrieve_topk_batch(dataset: str, queries: List[str], keys: List[str],
                        topk: int, threads: int = 8) -> Dict[str, List[Dict[str, str]]]:
    """BM25 top-k for a batch of queries. Returns key -> [{doc_id, text}, ...]."""
    assert len(queries) == len(keys)
    searcher = get_searcher(dataset)
    hits_dict = searcher.batch_search([str(q) for q in queries], [str(k) for k in keys],
                                      k=topk, threads=threads)
    results: Dict[str, List[Dict[str, str]]] = {}
    for key, hits in hits_dict.items():
        docs = []
        for hit in hits:
            doc = searcher.doc(hit.docid)
            if doc is None:
                continue
            raw = doc.raw()
            try:
                obj = json.loads(raw)
                text = obj.get("text") or obj.get("contents") or raw
            except Exception:
                text = raw
            docs.append({"doc_id": str(hit.docid), "text": str(text)})
        results[str(key)] = docs
    return results


# ---------------------------------------------------------------------------
# Cross-encoder reranker
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """HuggingFace sequence-classification cross-encoder (default BGE-reranker-v2-m3)."""

    def __init__(self, model_name: str = DEFAULT_RERANKER, device: str | None = None,
                 dtype: str = "fp32", max_length: int = 512):
        self.model_name = model_name
        self.max_length = int(max_length)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        if self.device.type == "cuda":
            if dtype == "fp16":
                model.half()
            elif dtype == "bf16":
                model.to(torch.bfloat16)
        model.to(self.device)
        model.eval()
        self.model = model

    @torch.no_grad()
    def score(self, queries: List[str], docs: List[str], batch_size: int = 256) -> List[float]:
        assert len(queries) == len(docs)
        scores: List[float] = []
        for i in range(0, len(queries), batch_size):
            enc = self.tokenizer(queries[i:i + batch_size], docs[i:i + batch_size],
                                 padding=True, truncation=True, max_length=self.max_length,
                                 return_tensors="pt")
            enc = {k: v.to(self.device, non_blocking=True) for k, v in enc.items()}
            out = self.model(**enc)
            logits = getattr(out, "logits", out[0]).squeeze(-1)
            scores.extend(logits.float().detach().cpu().tolist())
        return scores


# ---------------------------------------------------------------------------
# RSDG (Eq. 1-2)
# ---------------------------------------------------------------------------

def dcg_from_scores(scores: List[float]) -> float:
    """Discounted gain with negative reranker scores clamped to zero (Eq. 1)."""
    return sum(max(float(s), 0.0) / math.log2(i + 2) for i, s in enumerate(scores))


def rsdg_from_scores(scores: List[float]) -> Tuple[float, float, float]:
    """Return (RSDG@k, DCG, IDCG-surrogate) for one ranking (Eq. 2)."""
    if not scores:
        return 0.0, 0.0, 0.0
    dcg = dcg_from_scores(scores)
    idcg = dcg_from_scores(sorted(scores, reverse=True))
    if idcg <= 0:
        return 0.0, dcg, idcg
    return dcg / idcg, dcg, idcg


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def load_dataset_queries(dataset_root: Path, dataset: str) -> Dict[str, str]:
    """qid -> query text from the locally downloaded BEIR files.

    The packaged artifact strips the original query strings (docs/LICENSES.md),
    so the text is recovered from datasets/IR/ when a rollout does not carry it.
    """
    path = dataset_root / dataset / dataset / "queries.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; it is needed because the artifact does not carry "
            "query text. See docs/DATA.md."
        )
    queries: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        queries[str(row.get("_id", row.get("id")))] = row.get("text", "")
    return queries


def group_rollouts_by_qid(path: Path) -> "OrderedDict[str, List[Dict[str, Any]]]":
    grouped: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        grouped.setdefault(str(row["qid"]), []).append(row)
    return grouped


def process_dataset(dataset: str, pseudo_root: Path, out_dir: Path, reranker: CrossEncoderReranker,
                    topk: int, rerank_batch_size: int, repeat: int, max_queries: int | None,
                    save_raw_scores: bool, dataset_root: Path) -> None:
    in_path = pseudo_root / dataset / f"{dataset}_pseudo_train.jsonl"
    if not in_path.exists():
        print(f"[rsdg] skip {dataset}: missing {in_path}")
        return

    grouped = group_rollouts_by_qid(in_path)
    qids = list(grouped.keys())
    if max_queries:
        qids = qids[:max_queries]
    print(f"[rsdg] {dataset}: {len(qids)} queries, {sum(len(grouped[q]) for q in qids)} rollouts")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}_pseudo_ndcg_approx_sparse.jsonl"

    with out_path.open("w", encoding="utf-8") as wf:
        queries: Dict[str, str] | None = None
        if not grouped[qids[0]][0].get("query"):
            queries = load_dataset_queries(dataset_root, dataset)

        for qid in tqdm(qids, desc=f"{dataset}"):
            rows = grouped[qid]
            original_query = rows[0].get("query") or (queries or {}).get(qid, "")
            if not original_query:
                continue

            base_key = f"{qid}|base"
            batch_queries = [original_query]
            batch_keys = [base_key]
            meta: Dict[str, Dict[str, Any]] = {}
            for row in rows:
                pseudo = row.get("pseudo") or ""
                if not pseudo:
                    continue
                rollout = int(row.get("rollout", 0))
                key = f"{qid}|rollout={rollout}"
                expanded = query2doc_expand(original_query, pseudo, repeat=repeat)
                batch_queries.append(expanded)
                batch_keys.append(key)
                meta[key] = {"rollout": rollout, "expanded_query": expanded}
            if len(batch_queries) <= 1:
                continue

            docs_by_key = retrieve_topk_batch(dataset, batch_queries, batch_keys, topk=topk)

            base_docs = docs_by_key.get(base_key, [])
            if base_docs:
                base_scores = reranker.score([original_query] * len(base_docs),
                                             [d["text"] for d in base_docs],
                                             batch_size=rerank_batch_size)
                base_ndcg, base_dcg, base_idcg = rsdg_from_scores(base_scores)
            else:
                base_scores, base_ndcg, base_dcg, base_idcg = [], 0.0, 0.0, 0.0

            for key, info in meta.items():
                docs = docs_by_key.get(key, [])
                if docs:
                    # Reranker scores are conditioned on the ORIGINAL query q, not on q'.
                    scores = reranker.score([original_query] * len(docs),
                                            [d["text"] for d in docs],
                                            batch_size=rerank_batch_size)
                    ndcg, dcg, idcg = rsdg_from_scores(scores)
                else:
                    scores, ndcg, dcg, idcg = [], 0.0, 0.0, 0.0

                record: Dict[str, Any] = {
                    "dataset": dataset,
                    "task": "pseudo",
                    "qid": qid,
                    "rollout": info["rollout"],
                    "query": original_query,
                    "expanded_query": info["expanded_query"],
                    "importance_empty": False,
                    "topk": topk,
                    "approx_ndcg": ndcg,
                    "approx_dcg": dcg,
                    "approx_idcg": idcg,
                    "base_ndcg": base_ndcg,
                    "base_dcg": base_dcg,
                    "base_idcg": base_idcg,
                    "doc_ids": [d["doc_id"] for d in docs],
                    "retrieval_type_used": "sparse",
                }
                if save_raw_scores:
                    record["reranker_scores"] = [float(s) for s in scores]
                    record["base_reranker_scores"] = [float(s) for s in base_scores]
                    record["reranker_model"] = reranker.model_name
                wf.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[rsdg] wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default="nfcorpus,fiqa,fever,hotpotqa,msmarco")
    parser.add_argument("--pseudo-root", type=Path,
                        default=Path("artifacts/train/Meta-Llama-3.1-70B-Instruct"))
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Default: <pseudo-root>/pseudo_ndcg_approx")
    parser.add_argument("--reranker", default=DEFAULT_RERANKER)
    parser.add_argument("--topk", type=int, default=30, help="Reward depth k (paper default: 30)")
    parser.add_argument("--repeat", type=int, default=5, help="Query repetitions in the sparse expansion")
    parser.add_argument("--rerank-batch-size", type=int, default=256)
    parser.add_argument("--reranker-dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/IR"),
                        help="Used to recover query text when the artifact omits it.")
    parser.add_argument("--max-queries", type=int, default=None, help="Debug only: cap queries per dataset")
    parser.add_argument("--no-raw-scores", action="store_true",
                        help="Omit per-document reranker scores from the cache (smaller files)")
    args = parser.parse_args()

    out_dir = args.out_dir or (args.pseudo_root / "pseudo_ndcg_approx")
    reranker = CrossEncoderReranker(args.reranker, dtype=args.reranker_dtype)

    for dataset in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        process_dataset(dataset, args.pseudo_root, out_dir, reranker, topk=args.topk,
                        rerank_batch_size=args.rerank_batch_size, repeat=args.repeat,
                        max_queries=args.max_queries, save_raw_scores=not args.no_raw_scores,
                        dataset_root=args.dataset_root)


if __name__ == "__main__":
    main()
