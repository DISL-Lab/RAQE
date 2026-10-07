#!/usr/bin/env python3
"""Label-free data curation (Eq. 3): keep queries whose teacher expansion helps.

For every training query q with teacher expansion e* (rollout 0) and
q' = concat(q, e*), the reranker-shaped gain is

    delta(q', q) = RSDG@k(q', D_q') - RSDG@k(q, D_q)                   (Eq. 3)

and only queries with ``delta > margin`` are retained for Stage-1 distillation
and Stage-2 GRPO. Both terms come straight out of the RSDG cache written by
``rsdg/compute_rsdg.py``, so this step needs no retrieval, no reranker and no
human relevance labels.

Outputs one JSON list of qids per dataset, named the way Stage 1 and Stage 2
expect: ``per_qid_ndcg_<dataset>_filtered_qids_pr.json``.

    python rsdg/curate_queries.py \
        --ndcg-root artifacts/train/Meta-Llama-3.1-70B-Instruct/pseudo_ndcg_approx \
        --out-dir artifacts/sft/pr_final

Note on the released artifact
-----------------------------
The qid lists shipped with the paper artifact were produced by an earlier
variant of this step that thresholded labelled NDCG@10 on the BEIR *train*
qrels (``pr > original + 0.05``) instead of the label-free gain above. Pass
``--reproduce-release`` together with ``--labelled-ndcg-dir`` to reproduce those
exact lists; see docs/DATA.md.
"""

import argparse
import json
from collections import OrderedDict
from pathlib import Path


def load_rsdg(path: Path, rollout: int):
    """Return qid -> (base_rsdg, rollout_rsdg) for the requested rollout index."""
    values: "OrderedDict[str, tuple[float, float]]" = OrderedDict()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row.get("rollout", 0)) != rollout:
            continue
        values[str(row["qid"])] = (float(row.get("base_ndcg", 0.0)),
                                   float(row.get("approx_ndcg", 0.0)))
    return values


def curate_from_rsdg(ndcg_root: Path, dataset: str, rollout: int, margin: float,
                     retrieval_type: str):
    path = ndcg_root / f"{dataset}_pseudo_ndcg_approx_{retrieval_type}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing RSDG cache {path}; run rsdg/compute_rsdg.py first.")
    values = load_rsdg(path, rollout)
    kept = [qid for qid, (base, expanded) in values.items() if (expanded - base) > margin]
    return kept, len(values)


def curate_from_labelled_ndcg(labelled_dir: Path, dataset: str, margin: float):
    """Reproduce the released qid lists from labelled per-query NDCG@10 files.

    Each ``per_qid_ndcg_<dataset>*.json`` is a list of records with the NDCG@10 of
    the original query (``original``) and of the pseudo-passage expansion (``pr``),
    both computed against the BEIR train qrels.
    """
    kept, total = [], 0
    for path in sorted(labelled_dir.glob(f"per_qid_ndcg_{dataset}_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            continue
        for entry in data:
            pr, original, qid = entry.get("pr"), entry.get("original"), entry.get("qid")
            if pr is None or original is None or qid is None:
                continue
            total += 1
            if float(pr) > float(original) + margin:
                kept.append(str(qid))
    if total == 0:
        raise FileNotFoundError(f"No per_qid_ndcg_{dataset}_*.json records under {labelled_dir}")
    return kept, total


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default="nfcorpus,fiqa,fever,hotpotqa,msmarco")
    parser.add_argument("--ndcg-root", type=Path,
                        default=Path("artifacts/train/Meta-Llama-3.1-70B-Instruct/pseudo_ndcg_approx"))
    parser.add_argument("--retrieval-type", default="sparse", choices=["sparse", "dense"])
    parser.add_argument("--rollout", type=int, default=0,
                        help="Rollout index treated as the teacher expansion e* (default: 0)")
    parser.add_argument("--margin", type=float, default=None,
                        help="Keep queries with gain > margin. Default: 0.0 for the label-free "
                             "criterion, 0.05 for --reproduce-release.")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/sft/pr_final"))
    parser.add_argument("--reproduce-release", action="store_true",
                        help="Use labelled train-qrels NDCG@10 instead of RSDG (see module docstring).")
    parser.add_argument("--labelled-ndcg-dir", type=Path, default=None,
                        help="Directory with per_qid_ndcg_<dataset>_*.json files.")
    args = parser.parse_args()

    margin = args.margin if args.margin is not None else (0.05 if args.reproduce_release else 0.0)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    total_kept = 0
    for dataset in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        if args.reproduce_release:
            if args.labelled_ndcg_dir is None:
                parser.error("--reproduce-release requires --labelled-ndcg-dir")
            kept, total = curate_from_labelled_ndcg(args.labelled_ndcg_dir, dataset, margin)
        else:
            kept, total = curate_from_rsdg(args.ndcg_root, dataset, args.rollout, margin,
                                           args.retrieval_type)
        out_path = args.out_dir / f"per_qid_ndcg_{dataset}_filtered_qids_pr.json"
        out_path.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
        total_kept += len(kept)
        print(f"{dataset}: kept {len(kept)}/{total} queries -> {out_path}")
    print(f"Total retained training queries: {total_kept}")


if __name__ == "__main__":
    main()
