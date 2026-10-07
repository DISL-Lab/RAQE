# Camera-ready RAQE defaults

Settings transcribed from the camera-ready paper, annotated with the value each
script in this repository actually uses. The minimal release implements the RAQE
training path and sparse BM25 evaluation; the dense Contriever result is an
optional paper experiment and is not bundled here.

## Data and models

| Component | Default |
|---|---|
| Training datasets | NFCorpus, FiQA, FEVER, HotpotQA, MS MARCO |
| Evaluation datasets | Training datasets + ArguAna, NQ, SciDocs |
| Teacher | `meta-llama/Llama-3.1-70B-Instruct`, g = 8 rollouts, sampled at T=0.7 / top-p 0.9, max 128 new tokens |
| Student backbones | `Llama-3.1-8B-Instruct` (main), `Llama-3.2-3B-Instruct`, `Qwen3-4B-Instruct` (robustness) |
| Reward reranker | `BAAI/bge-reranker-v2-m3` |
| Reward depth *k* | 30 |
| Curated training queries | 6,453 (753 / 837 / 2,103 / 1,882 / 878) |

## Stage 1 — supervised distillation (`scripts/train_sft.sh`)

| Hyperparameter | Value | Source |
|---|---|---|
| Epochs | 5 | Appendix A.2 |
| Optimizer | AdamW, weight decay 0.01 | code |
| Learning rate | 2e-5, cosine schedule, warmup ratio 0.1 | code |
| Per-device batch / grad. accumulation / processes | 1 / 16 / 2 → effective 32 | code |
| Optimizer updates | 5 × 202 = 1,010 | Appendix A.2 |
| LoRA | r = 32, α = 32, dropout 0.05, `q_proj`/`v_proj` | released adapter config |
| Precision | bf16 | Appendix A.2 |
| Max sequence length | 2048 | code |

> On the learning rate, see "Reading of Appendix A.2's optimizer sentence" below.

## Stage 2 — reranker-aligned GRPO (`scripts/train_grpo.sh`)

| Hyperparameter | Value | Source |
|---|---|---|
| Epochs | 2 | Appendix A.2 |
| Rollout group size *g* | 8 | Appendix A.2 / Appendix E.1 |
| Learning rate | 5e-6, constant | Appendix A.2 |
| Per-device batch / grad. accumulation / processes | 4 / 4 / 2 → effective 32 | code |
| Optimizer updates | 2 × 865 = 1,730 | Appendix A.2 |
| Clip epsilon ε | 0.2 | Appendix A.2 |
| Surrogate | Eq. (5), `min(ρÂ, clip(ρ,1±ε)Â)` | Appendix A.2 |
| Reward | `r_i = Δ(q′_i, q) = approx_ndcg − base_ndcg` | Eq. (3) / Appendix A.1 |
| Advantage | `Â_i = (r_i − r̄)/σ_r` within the rollout group, then clipped to ±5 | Eq. (5); the clip is code-only |
| Grad-norm clip | 2.0 | code |
| Reference policy | frozen copy of the Stage-1 adapter | code |
| Zero-RSDG query filter | drop qids with >70 % zero-RSDG rollouts (3,462 of 6,453 survive) | code |
| Precision | bf16 | Appendix A.2 |

The reward transformation and the advantage clipping are implementation details
that the paper does not spell out; Eq. (5) describes the clipped surrogate that
sits on top of them.

## Inference and evaluation (`scripts/evaluate_retrieval.sh`)

| Component | Default |
|---|---|
| Generation | one pseudo-passage per query, greedy, at most 128 new tokens |
| Sparse expansion | original query repeated 5× followed by the pseudo-passage |
| Retriever | BM25 over `beir-v1.0.0-<dataset>.flat` (`msmarco-v1-passage` for MS MARCO) |
| Metric | NDCG@10 against `qrels/test.tsv` |
| Evaluation queries | `queries.jsonl` filtered to the qids in `qrels/test.tsv` (20,943 across the eight sets) |
| Dense retrieval | Contriever, `query [SEP] pseudo` (paper-only optional experiment) |

## Hardware and runtime

Two NVIDIA RTX PRO 6000 (96 GB). Stage 1 ≈ 12 min, Stage 2 ≈ 24 min
(Appendix A.2). Generation for the eight evaluation sets (≈21k queries)
dominates end-to-end wall-clock.

## Where this code differs from the pre-release code

Four behaviours of the internal code did not match the paper. This repository
follows the paper by default and keeps the old behaviour behind a flag, so a
checkpoint trained before the fix can still be reproduced exactly.

| | Paper (default here) | Pre-release code | Flag to restore |
|---|---|---|---|
| Stage-2 reward | `r_i = Δ(q′_i, q)` (Eq. 3) | squashed to `tanh(10 · Δ)` before standardisation | `--reward-transform tanh` |
| Stage-2 surrogate | `min(ρÂ, clip(ρ)Â)` for every sample (Eq. 5) | took a *maximum* when `Â < 0`, inverting the trust region for those samples | `--surrogate released` |
| Decoding | greedy, single pass | one top-p 0.9 sample at `temperature = 1.0` | `--do-sample` |
| Expansion length | at most 128 tokens (Table 16) | additionally forced `min_new_tokens = 96` | `--min-new-tokens 96` |

The surrogate variants are identical while the likelihood ratio stays inside
`[1−ε, 1+ε]`, so they only diverge once the policy drifts from the reference.
`tanh` is monotone and the rewards are standardised afterwards either way, so it
only changes how strongly large gains are compressed relative to small ones.
On NFCorpus the decoding change is worth a few tenths of NDCG@10 (0.3548 greedy
vs 0.3573 sampled, against 0.3608 for the released expansions).

To reproduce the published checkpoint exactly, combine the restore flags:

```bash
ADAPTER_DIR=outputs/sft_llama31_8b bash scripts/train_grpo.sh \
  --reward-transform tanh --surrogate released
MODEL_DIR=... bash scripts/evaluate_retrieval.sh   # with:
GENERATE_ARGS="--do-sample --min-new-tokens 96"
```

## Data-selection details not stated in the paper

- **Zero-RSDG query filter.** Stage 2 drops queries where more than 70 % of the
  rollouts score RSDG = 0, taking 6,453 curated queries down to 3,462. Appendix B
  says the 6,453 are "shared by both Stage 1 distillation and Stage 2 GRPO", but
  Appendix A.2's 1,730 updates only follow from the 3,462. `--zero_ndcg_threshold
  1.0` disables the filter, at the cost of no longer matching the reported step
  count.
- **Advantage clipping** to ±5 after standardisation; it almost never binds for
  a group of 8.

## Reading of Appendix A.2's optimizer sentence

"Training uses the AdamW optimizer with learning rate 5×10⁻⁶ and batch size 8"
describes **Stage 2**: its per-step batch is exactly 4 × 2 GPUs = 8 and its
learning rate is exactly 5e-6. Stage 1 has its own paper-stated anchor — 5
epochs over ~1,010 updates — which fixes its effective batch at 32 and is what
`scripts/train_sft.sh` reproduces. Its learning rate (2e-5, cosine, warmup 0.1)
is not constrained by the paper and is taken from the run that produced the
released adapter; override with `LEARNING_RATE=...`.
