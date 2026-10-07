# Data preparation

RAQE needs three kinds of data:

| What | Where it goes | Who produces it |
|---|---|---|
| BEIR / MS MARCO corpora, queries and qrels | `datasets/IR/` | `get_IR_dataset.py` (§1) |
| Prebuilt BM25 Lucene indexes | `~/.cache/pyserini/indexes/` | `evaluation/download_lucene_indexes.py` (§2) |
| Teacher rollouts, RSDG reward caches, curated qid lists | `artifacts/` | released artifact, or §4–§5 |

None of it is stored in Git. `datasets/`, `indexes/` and `artifacts/` are in
`.gitignore`. Before redistributing anything derived from the corpora, read
[LICENSES.md](LICENSES.md) — several of the BEIR datasets grant no
redistribution right.

---

## 1. BEIR datasets

The paper uses eight datasets. Five have official training splits and are used
for both training and evaluation; three are held out for the out-of-distribution
evaluation.

| Dataset | Role | Corpus docs | Test queries | On-disk |
|---|---|---:|---:|---:|
| NFCorpus | train + test | 3,633 | 323 | 11 MB |
| FiQA | train + test | 57,638 | 648 | 49 MB |
| FEVER | train + test | 5,416,568 | 6,666 | 3.2 GB |
| HotpotQA | train + test | 5,233,329 | 7,405 | 2.1 GB |
| MS MARCO | train + test | 8,841,823 | 43 | 3.4 GB |
| ArguAna | test only (OOD) | 8,674 | 1,406 | 16 MB |
| NQ | test only (OOD) | 2,681,468 | 3,452 | 1.5 GB |
| SciDocs | test only (OOD) | 25,657 | 1,000 | 254 MB |

Total ≈ **10.5 GB**. "Test queries" is the number of distinct query ids in
`qrels/test.tsv`; `queries.jsonl` is much larger because it also holds the
train/dev queries (509,962 rows for MS MARCO), and every RAQE script filters
`queries.jsonl` through `qrels/test.tsv` before generating or retrieving.

### Download

```bash
for ds in nfcorpus fiqa fever hotpotqa msmarco arguana nq scidocs; do
  python get_IR_dataset.py -dataset "$ds"
done
# or, equivalently:
python get_IR_dataset.py -dataset all
```

The script fetches the official BEIR archives from
`https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/<name>.zip`,
unzips them, and writes `test_subsample_processed.json` (the test queries that
carry a relevance judgement). Expect the FEVER/HotpotQA/MS MARCO/NQ downloads to
take a while; re-running is safe because an existing `queries.jsonl` is not
re-downloaded.

Resulting layout (note the doubled directory name — every script expects it):

```
datasets/IR/<dataset>/<dataset>/
├── corpus.jsonl
├── queries.jsonl
├── test_subsample_processed.json
└── qrels/
    ├── train.tsv      # present for the five training datasets
    ├── dev.tsv
    └── test.tsv       # used for all reported NDCG@10 numbers
```

`download_beir_parallel_resume.py` is an optional multi-threaded downloader with
resume support for the large archives; `get_IR_dataset.py` is the reference path.

### Licensing

BEIR datasets keep the licences of their original sources and are **not**
redistributed here. See the [BEIR repository](https://github.com/beir-cellar/beir)
for the per-dataset licence table before using them beyond research.

---

## 2. BM25 indexes

Retrieval uses Pyserini's prebuilt `beir-v1.0.0-<dataset>.flat` Lucene indexes
(`msmarco-v1-passage` for MS MARCO) — plain BM25, **not** the learned-sparse
`.splade-pp-ed` variants.

```bash
python evaluation/download_lucene_indexes.py
# or a subset:
python evaluation/download_lucene_indexes.py --datasets nfcorpus fiqa
```

Indexes land in `~/.cache/pyserini/indexes/` and total roughly 30 GB for the
eight datasets. This step needs a working JDK 21 — see
[ENVIRONMENT.md](ENVIRONMENT.md).

---

## 3. Training artifacts (`artifacts/`)

Stage 1 and Stage 2 read three precomputed inputs. Together they are ≈71 MB for
the sparse setting, which is what lets GRPO run offline: no 70B teacher, no
retrieval and no reranker calls during training.

```
artifacts/
├── sft/pr_final/
│   └── per_qid_ndcg_<dataset>_filtered_qids_pr.json     # curated training qids
└── train/Meta-Llama-3.1-70B-Instruct/
    ├── <dataset>/<dataset>_pseudo_train.jsonl           # g=8 teacher rollouts per query
    └── pseudo_ndcg_approx/
        └── <dataset>_pseudo_ndcg_approx_sparse.jsonl    # RSDG reward cache (k=30)
```

### Curated qid lists

| Dataset | Retained queries |
|---|---:|
| NFCorpus | 753 |
| FiQA | 837 |
| FEVER | 2,103 |
| HotpotQA | 1,882 |
| MS MARCO | 878 |
| **Total** | **6,453** |

These 6,453 queries are the Stage-1 SFT set (Appendix B) and the pool Stage 2
draws from.

> **How these particular lists were built.** Eq. (3) selects queries by the
> label-free gain `Δ(q′, q) = RSDG@k(q′) − RSDG@k(q) > 0`, and that is what
> `rsdg/curate_queries.py` does. The lists in the released artifact predate that
> script: they were produced by thresholding *labelled* NDCG@10 on the BEIR
> **train** qrels (`pr > original + 0.05`). The two criteria are not
> interchangeable — of the 4,519 released queries whose teacher rollout is
> covered by the sparse RSDG cache, only 2,635 (58 %) also satisfy Δ > 0, and the
> agreement is far lower on NFCorpus (15 %) and FiQA (22 %) than on HotpotQA and
> MS MARCO (80 % each). Rebuilding the lists the paper's way therefore means
> redoing §4–§5 from the 3,000-query random pool, not re-filtering these 6,453.

### Teacher rollouts — `<dataset>_pseudo_train.jsonl`

One JSON object per (query, rollout); `g = 8` rollouts per query, generated by
`Llama-3.1-70B-Instruct` with the Appendix G.1 prompt, sampled at
`temperature=0.7` / `top_p=0.9` with `max_new_tokens=128`.

```json
{"dataset": "nfcorpus", "qid": "PLAIN-3", "rollout": 0,
 "pseudo": "Researchers have discovered that ..."}
```

`rollout: 0` is the SFT target e*; all eight rollouts form the GRPO rollout group.

The published artifact carries no `query` field: MS MARCO, NFCorpus and FiQA
grant no right to redistribute their query strings, so `scripts/package_artifacts.sh`
strips them and the training scripts read the text from your own
`datasets/IR/<ds>/<ds>/queries.jsonl` instead (`--dataset-root`). See
[LICENSES.md](LICENSES.md).

### RSDG cache — `<dataset>_pseudo_ndcg_approx_sparse.jsonl`

One record per (query, rollout) holding the reward terms of Eq. (2):

```json
{"qid": "PLAIN-3", "rollout": 0, "topk": 30,
 "approx_ndcg": 1.0,          // RSDG@30 of the expanded query q'
 "base_ndcg": 1.0,            // RSDG@30 of the original query q
 "doc_ids": ["MED-2421", ...],
 "reranker_scores": [...], "base_reranker_scores": [...],
 "retrieval_type_used": "sparse"}
```

Document *text* is never stored, only ids and scores; the `expanded_query` field
is stripped from the published artifact for the same reason as `query` above.

The GRPO reward is `r_i = approx_ndcg - base_ndcg` (Eq. 3), so both terms live in
the same record.

Alternative reranker caches for the Sec. 4.2 robustness study live in
sub-directories named after the model (`pseudo_ndcg_approx/BAAI__bge-reranker-v2-m3/`,
`.../Alibaba-NLP__gte-reranker-modernbert-base/`, `.../mixedbread-ai__mxbai-rerank-large-v2/`)
and are selected with `--reranker`. The files directly under
`pseudo_ndcg_approx/` are the paper default (BGE-reranker-v2-m3, k=30).

### Sanity check

After placing the artifact, Stage 2 should report exactly these counts on its
first lines:

```
[offline-ndcg] loaded dataset=nfcorpus (sparse): qids=263 (filtered 753 -> 263)
[offline-ndcg] loaded dataset=fiqa     (sparse): qids=313 (filtered 837 -> 313)
[offline-ndcg] loaded dataset=fever    (sparse): qids=158 (filtered 169 -> 158)
[offline-ndcg] loaded dataset=hotpotqa (sparse): qids=1852 (filtered 1882 -> 1852)
[offline-ndcg] loaded dataset=msmarco  (sparse): qids=876 (filtered 878 -> 876)
[offline-ndcg] Total filtered qids: 3462, Total candidates: 27693, Avg rollouts/qid: 8.0
```

The second number is after `--zero_ndcg_threshold 0.7` drops queries where more
than 70 % of the rollouts score RSDG = 0 (no positive reranker evidence anywhere
in the top-k). 27,693 candidates at an effective batch of 32 give 865 optimizer
steps per epoch, i.e. the 1,730 Stage-2 updates reported in Appendix A.2.

> **FEVER caveat.** The released FEVER sparse cache covers 169 of the 2,103
> curated queries; the remaining rows were never written back after the run that
> produced the paper numbers. Training reproduces exactly as published with the
> file as shipped, and `rsdg/compute_rsdg.py` regenerates the full set if you
> want complete FEVER coverage (which changes the step count and the results).

---

## 4. Regenerating the teacher rollouts (optional)

Sec. 3.2.1 starts from 3,000 training queries sampled uniformly at random per
dataset, and generates `g = 8` pseudo-passages for each with the Appendix G.1
prompt:

```bash
accelerate launch --num_processes 2 sft/generate_train_pseudo_refs.py \
  --model_dir meta-llama/Llama-3.1-70B-Instruct \
  --datasets nfcorpus,fiqa,fever,hotpotqa,msmarco \
  --sample_queries 3000 --sample_seed 42 \
  --rollouts 8 --max_new_tokens 128 --temperature 0.7 --top_p 0.9 \
  --out_root artifacts/train/Meta-Llama-3.1-70B-Instruct
```

`--sample_queries` draws the pool from `qrels/train.tsv`; the curated qid lists
do not exist yet at this point, which is why this step samples rather than
reading `artifacts/sft/pr_final`. Budget several GPU-hours and ~150 GB of weights
for the 70B teacher; any strong instruct model can stand in.

## 5. Regenerating the RSDG cache and the curated qids

```bash
# Eq. (1)-(2): BM25 top-k for q and each q', scored by the cross-encoder w.r.t. q
python rsdg/compute_rsdg.py \
  --datasets nfcorpus,fiqa,fever,hotpotqa,msmarco \
  --pseudo-root artifacts/train/Meta-Llama-3.1-70B-Instruct \
  --topk 30 --reranker BAAI/bge-reranker-v2-m3

# Eq. (3): keep queries whose teacher expansion improves RSDG
python rsdg/curate_queries.py \
  --ndcg-root artifacts/train/Meta-Llama-3.1-70B-Instruct/pseudo_ndcg_approx \
  --out-dir artifacts/sft/pr_final
```

`compute_rsdg.py` needs the BM25 indexes from §2 and one GPU for the reranker.
Cost is dominated by `(g + 1) × k` cross-encoder forwards per query — about
1.7 M document scorings for the 6,453-query pool at `k = 30`.

The two commands above are the paper's pipeline end to end: `compute_rsdg.py`
scores the rollouts, `curate_queries.py` keeps the queries with Δ > 0 (Eq. 3).
Neither step touches a relevance label.

> The released qid lists were **not** produced this way — see the note in §3.
> `curate_queries.py --reproduce-release --labelled-ndcg-dir <dir>` rebuilds them
> verbatim from labelled per-query NDCG files if you need the exact published
> training set.
