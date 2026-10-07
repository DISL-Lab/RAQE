#!/usr/bin/env python3
"""Verify that datasets/ and artifacts/ are laid out the way training expects.

Run this before the first training job; it is cheap (no model, no retrieval) and
catches the mistakes that otherwise surface as an empty training set an hour in.

    python scripts/check_artifacts.py
    python scripts/check_artifacts.py --skip-datasets     # artifacts only
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

TRAIN_DATASETS = ["nfcorpus", "fiqa", "fever", "hotpotqa", "msmarco"]
EVAL_DATASETS = TRAIN_DATASETS + ["arguana", "nq", "scidocs"]

# Curated query counts of the released artifact (Appendix B).
EXPECTED_QIDS = {"nfcorpus": 753, "fiqa": 837, "fever": 2103, "hotpotqa": 1882, "msmarco": 878}
# Queries surviving --zero_ndcg_threshold 0.7 in the sparse setting; these are the
# counts Stage 2 prints, and they determine the 1,730 reported update steps.
EXPECTED_AFTER_ZERO_FILTER = {"nfcorpus": 263, "fiqa": 313, "fever": 158,
                              "hotpotqa": 1852, "msmarco": 876}
ROLLOUTS = 8
ZERO_THRESHOLD = 0.7


class Report:
    def __init__(self):
        self.problems = []

    def ok(self, msg):
        print(f"  ok    {msg}")

    def warn(self, msg):
        print(f"  warn  {msg}")

    def fail(self, msg):
        print(f"  FAIL  {msg}")
        self.problems.append(msg)


def check_datasets(root: Path, report: Report):
    print(f"\ndatasets ({root})")
    for dataset in EVAL_DATASETS:
        base = root / dataset / dataset
        missing = [name for name in ("queries.jsonl", "corpus.jsonl", "qrels/test.tsv")
                   if not (base / name).exists()]
        if missing:
            report.fail(f"{dataset}: missing {', '.join(missing)} under {base}")
            continue
        qids = set()
        with (base / "qrels" / "test.tsv").open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i == 0 and line.lower().startswith("query-id"):
                    continue
                if line.strip():
                    qids.add(line.split("\t")[0].strip())
        report.ok(f"{dataset}: {len(qids)} test queries with judgements")


def check_qid_lists(filtered_dir: Path, report: Report):
    print(f"\ncurated qid lists ({filtered_dir})")
    total = 0
    for dataset in TRAIN_DATASETS:
        path = filtered_dir / f"per_qid_ndcg_{dataset}_filtered_qids_pr.json"
        if not path.exists():
            report.fail(f"{dataset}: missing {path.name}")
            continue
        qids = json.loads(path.read_text(encoding="utf-8"))
        count = len(qids)
        total += count
        expected = EXPECTED_QIDS[dataset]
        if count == expected:
            report.ok(f"{dataset}: {count} curated queries")
        else:
            report.warn(f"{dataset}: {count} curated queries (release artifact has {expected})")
    print(f"  total: {total} (release artifact: {sum(EXPECTED_QIDS.values())})")


def check_rollouts_and_rewards(train_root: Path, filtered_dir: Path, report: Report):
    print(f"\nteacher rollouts and RSDG cache ({train_root})")
    for dataset in TRAIN_DATASETS:
        pseudo_path = train_root / dataset / f"{dataset}_pseudo_train.jsonl"
        ndcg_path = train_root / "pseudo_ndcg_approx" / f"{dataset}_pseudo_ndcg_approx_sparse.jsonl"
        if not pseudo_path.exists():
            report.fail(f"{dataset}: missing {pseudo_path}")
            continue
        if not ndcg_path.exists():
            report.fail(f"{dataset}: missing {ndcg_path}")
            continue

        pseudo_keys = set()
        empty = 0
        with pseudo_path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                pseudo_keys.add((str(row["qid"]), int(row.get("rollout", 0))))
                if not str(row.get("pseudo", "")).strip():
                    empty += 1

        scores = defaultdict(list)
        joined = 0
        topks = set()
        with ndcg_path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (str(row["qid"]), int(row.get("rollout", 0)))
                if key not in pseudo_keys:
                    continue
                joined += 1
                topks.add(int(row.get("topk", -1)))
                scores[key[0]].append(float(row.get("approx_ndcg", 0.0)))

        kept = sum(1 for vals in scores.values()
                   if sum(1 for v in vals if v == 0.0) / len(vals) < ZERO_THRESHOLD)
        expected_kept = EXPECTED_AFTER_ZERO_FILTER[dataset]
        detail = (f"{dataset}: {len(pseudo_keys) // ROLLOUTS} queries x {ROLLOUTS} rollouts, "
                  f"{joined} reward rows joined, {len(scores)} queries with rewards, "
                  f"{kept} survive the zero-RSDG filter")
        if kept == expected_kept:
            report.ok(detail)
        else:
            report.warn(detail + f" (release artifact: {expected_kept})")
        if empty:
            report.warn(f"{dataset}: {empty} rollouts have an empty pseudo-passage")
        if topks and topks != {30}:
            report.warn(f"{dataset}: reward depth k={sorted(topks)} (paper default: 30)")

        qid_path = filtered_dir / f"per_qid_ndcg_{dataset}_filtered_qids_pr.json"
        if qid_path.exists():
            curated = {str(q) for q in json.loads(qid_path.read_text(encoding="utf-8"))}
            orphans = curated - {q for q, _ in pseudo_keys}
            if orphans:
                report.warn(f"{dataset}: {len(orphans)} curated qids have no teacher rollouts")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/IR"))
    parser.add_argument("--train-root", type=Path,
                        default=Path("artifacts/train/Meta-Llama-3.1-70B-Instruct"))
    parser.add_argument("--filtered-dir", type=Path, default=Path("artifacts/sft/pr_final"))
    parser.add_argument("--skip-datasets", action="store_true")
    args = parser.parse_args()

    report = Report()
    if not args.skip_datasets:
        check_datasets(args.dataset_root, report)
    check_qid_lists(args.filtered_dir, report)
    check_rollouts_and_rewards(args.train_root, args.filtered_dir, report)

    print()
    if report.problems:
        print(f"{len(report.problems)} blocking problem(s); see docs/DATA.md.")
        sys.exit(1)
    print("All required inputs are present. Warnings above only mean your artifact "
          "differs from the released one.")


if __name__ == "__main__":
    main()
