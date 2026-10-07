# RAQE: Reranker-Aligned Query Expansion

Official implementation of the COLM 2026 paper
*RAQE: Reranker-Aligned Query Expansion via Label-Free Group-Relative Policy Optimization*.

RAQE trains a compact query-expansion policy (3B/8B) **without human relevance
labels**. A cross-encoder reranker replaces the labels: its scores over the
retriever's top-*k* documents are turned into a Reranker-Shaped Discounted Gain
(RSDG), which both curates the distillation data and defines the reward for
group-relative policy optimization. At inference the policy emits a single short
pseudo-passage in one forward pass, so there is no repeated generation and no
reranking at query time.

```
                    ┌─ Stage 1 ─────────────┐   ┌─ Stage 2 ──────────────────────┐
 teacher rollouts ─▶│ supervised distillation│──▶│ reranker-aligned offline GRPO  │──▶ LoRA policy
 (g = 8 per query)  │ (Eq. 4)               │   │ (Eq. 5, reward = ΔRSDG@30)     │
                    └───────────────────────┘   └────────────────────────────────┘
```

---

## Repository layout

| Path | Contents |
|---|---|
| `rsdg/` | RSDG computation (Eq. 1–2) and label-free query curation (Eq. 3) |
| `sft/` | teacher rollout generation, shared prompt, Stage-1 distillation |
| `grpo/step1/` | Stage-2 offline GRPO over the cached RSDG rewards |
| `evaluation/` | single-pass RAQE generation and sparse BM25 NDCG@10 evaluation |
| `scripts/` | thin launchers with the paper defaults baked in |
| `configs/` | paper hyperparameters and artifact inventory |
| `docs/` | [data preparation](docs/DATA.md), [environment setup](docs/ENVIRONMENT.md), [licensing](docs/LICENSES.md) |

`scripts/check_artifacts.py` validates a local data/artifact tree before you
spend GPU hours on it; `scripts/package_artifacts.sh` builds the release
tarball plus checksums from a working tree.

---

## Quick start

```bash
# 0. environment — see docs/ENVIRONMENT.md
conda create -n raqe python=3.11 -y && conda activate raqe
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
sudo apt install openjdk-21-jdk   # Pyserini needs a JDK 21; RAQE finds it itself
export HF_TOKEN=hf_...            # Llama checkpoints are gated

# 1. data — see docs/DATA.md
python get_IR_dataset.py -dataset all              # ~10.5 GB into datasets/IR/
python evaluation/download_lucene_indexes.py       # ~30 GB of BM25 indexes

# 2. training artifacts (teacher rollouts + RSDG cache + curated qids) into artifacts/
#    Download the release artifact, or regenerate them with rsdg/ (docs/DATA.md §4-5).
python scripts/check_artifacts.py                  # verifies layout and row counts

# 3. two-stage training
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_sft.sh
CUDA_VISIBLE_DEVICES=0,1 ADAPTER_DIR=outputs/sft_llama31_8b bash scripts/train_grpo.sh

# 4. generate one expansion per test query and score NDCG@10
CUDA_VISIBLE_DEVICES=0 \
  MODEL_DIR=grpo/step1/grpo_output/raqe_llama31_8b_sparse \
  bash scripts/evaluate_retrieval.sh
```

Every script takes its paths and model ids from environment variables documented
at the top of the file, and forwards extra flags to the underlying Python entry
point.

---

## The pipeline in detail

### Reward: RSDG (Eq. 1–2)

For an expanded query `q' = concat(q, e)`, retrieve the BM25 top-*k*
(`k = 30`), score each document with the cross-encoder **conditioned on the
original query** `q`, clamp negatives to zero, and normalise the discounted gain
by the gain of the reranker's own ideal ordering of the same documents:

```
s(q, d)    = max(CE(q, d), 0)
RSDG@k(q') = Σ_i s(q, d'_i)/log2(i+1)  /  Σ_j s(q, d'_(j))/log2(j+1)
```

Conditioning on `q` rather than `q'` is what stops the reward from rewarding
expansions that merely restate their own generated text.

```bash
python rsdg/compute_rsdg.py --topk 30 --reranker BAAI/bge-reranker-v2-m3
```

Writes `artifacts/train/<teacher>/pseudo_ndcg_approx/<dataset>_pseudo_ndcg_approx_sparse.jsonl`,
one record per (query, rollout) with `approx_ndcg` (RSDG of `q'`) and
`base_ndcg` (RSDG of `q`).

### Curation (Eq. 3)

Keep only queries whose teacher expansion actually improves the ranking,
`Δ(q', q) = RSDG@k(q') − RSDG@k(q) > 0`:

```bash
python rsdg/curate_queries.py --out-dir artifacts/sft/pr_final
```

The released artifact contains 6,453 curated queries. See the caveat in
[docs/DATA.md §5](docs/DATA.md) about how those particular lists were produced.

### Stage 1 — supervised distillation (Eq. 4)

LoRA fine-tuning on `(q, e*)` pairs, where `e*` is the first teacher rollout;
only the assistant span is supervised.

```bash
bash scripts/train_sft.sh
# 5 epochs, effective batch 32 on 2 GPUs -> 1,010 updates, ~12 min
```

### Stage 2 — reranker-aligned GRPO (Eq. 5)

Fully offline: rollouts, retrieval results and reranker scores are all cached,
so no retriever or reranker call happens during training. For each query the
eight cached candidates form one rollout group; rewards `r_i = Δ(q'_i, q)` are
standardised within the group into advantages, and the policy is updated with
the Eq. (5) clipped surrogate against a frozen copy of the Stage-1 adapter.

```bash
ADAPTER_DIR=outputs/sft_llama31_8b bash scripts/train_grpo.sh
# 2 epochs, g=8, k=30, lr 5e-6, eps 0.2 -> 1,730 updates, ~24 min
```

Stage 2 prints the surviving query counts before training; the expected numbers
are listed in [docs/DATA.md §3](docs/DATA.md) and are a quick check that the
artifact is placed correctly.

### Evaluation

```bash
MODEL_DIR=grpo/step1/grpo_output/raqe_llama31_8b_sparse bash scripts/evaluate_retrieval.sh
```

1. `evaluation/generate_raqe.py` produces one pseudo-passage of at most 128
   tokens per **test** query (the `queries.jsonl` rows whose id appears in
   `qrels/test.tsv`) and writes `evaluation/results/<model>/<dataset>/ours.json`.
   Decoding is greedy, so the output is a deterministic function of the
   checkpoint.
2. `evaluation/evaluate_raqe.py` builds the sparse expansion (original query ×5 +
   pseudo-passage), runs BM25, and reports NDCG@10 per dataset plus
   `summary_sparse.json`.

Generation is the expensive part (≈21k queries across the eight datasets);
`--overwrite` is off by default so an interrupted run resumes at dataset
granularity.

---

## Reproduction status

Verified on 2 × RTX PRO 6000 with the released artifact, the published Stage-2
checkpoint and `meta-llama/Llama-3.1-8B-Instruct`.

**End-to-end sparse retrieval (BM25, NDCG@10 ×100).** Expansions regenerated from
scratch with this repository's defaults, against the seed-42 run reported in
Table 19:

| | NFC | FiQA | FEV | HP | MSM | Arg | NQ | SciD | **Avg** |
|---|---|---|---|---|---|---|---|---|---|
| Paper (Table 19, seed 42) | 36.0 | 24.5 | 84.5 | 69.1 | 65.5 | 28.3 | 44.4 | 14.7 | **45.9** |
| This repository | 35.3 | 24.3 | 81.7 | 70.4 | 65.0 | 30.1 | 47.5 | 15.5 | **46.2** |

The macro average lands inside the seed band the paper itself reports
(45.9 / 46.0 / 46.2 across seeds 42 / 123 / 3407). Per-dataset movement of a few
points comes from decoding: the numbers in the paper were produced by a sampled
decoder, whereas this repository decodes greedily (see
`configs/paper_defaults.md`). Scoring the paper's own stored expansions with
`evaluation/evaluate_raqe.py` returns 0.3608 on NFCorpus, matching the published
36.0 exactly, which isolates the difference to generation rather than scoring.

**Pipeline checks.**

| Check | Result |
|---|---|
| Stage-2 query counts | 263 / 313 / 158 / 1,852 / 876 = **3,462** queries, 27,693 candidates → 865 steps/epoch, i.e. the 1,730 updates of Appendix A.2 |
| Stage-1 SFT | 6,453 curated examples load and train; adapter matches the released config (r=32, α=32, `q_proj`/`v_proj`) |
| Stage-2 GRPO | runs fully offline and writes a loadable adapter |
| `rsdg/compute_rsdg.py` | reproduces the shipped RSDG cache to within 1e-3 (cross-encoder numerics) |
| Stripped artifact | trains identically after recovering query text from `datasets/IR/` |

---

## Scope of this release

Included: RSDG, data curation, both training stages, single-pass inference, and
sparse BM25 evaluation — everything needed to reproduce the RAQE rows of
Tables 1 and 17.

Not included: the training-free baselines (HyDE, Query2Doc, MuGI, Word2Passage),
ExpandR, RaFe-DPO, the downstream QA/fact-verification experiments, and the
dense Contriever retrieval path. Cite and use the official implementations of
those methods when comparing. Prompt templates for every baseline are in
Appendix G of the paper.

---

## Licensing

The code in this repository is Apache-2.0 (`LICENSE`).

**Built with Llama.** The policies RAQE trains are LoRA adapters on
`meta-llama/Llama-3.1-8B-Instruct` / `Llama-3.2-3B-Instruct`, so any adapter you
release is governed by the corresponding Meta community license — its name must
begin with `Llama`, and it must ship the agreement, the acceptable-use policy and
the `NOTICE` attribution.

The BEIR and MS MARCO corpora are downloaded from their official sources and are
not redistributed here. Several of them (MS MARCO, NFCorpus, FiQA) grant no
redistribution right at all, which is why the packaged training artifact contains
no query or document text — only ids, reranker scores and Llama-generated
passages.

Full details, including what a released adapter and artifact must carry, are in
[docs/LICENSES.md](docs/LICENSES.md).

---

## Citation

```bibtex
@inproceedings{sun2026raqe,
  title     = {RAQE: Reranker-Aligned Query Expansion via Label-Free Group-Relative Policy Optimization},
  author    = {Sun, Gyeonghun and Choi, Jeonghwan and Kim, Sundong and Song, Hwanjun},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2026}
}
```
