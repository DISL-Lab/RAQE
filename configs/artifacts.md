# External artifacts required for exact reproduction

None of these belong in Git. Publish them as a GitHub release asset or a Hugging
Face dataset repository, with checksums, and point `scripts/download_artifacts.sh`
at the resulting URL. `docs/DATA.md` describes the file formats and the expected
row counts.

## Required for the cached training path (≈71 MB, sparse)

| Path in `artifacts/` | Contents | Size |
|---|---|---:|
| `sft/pr_final/per_qid_ndcg_<ds>_filtered_qids_pr.json` | curated training qids, 6,453 total | 116 KB |
| `train/Meta-Llama-3.1-70B-Instruct/<ds>/<ds>_pseudo_train.jsonl` | 8 teacher rollouts per query | 21 MB |
| `train/Meta-Llama-3.1-70B-Instruct/pseudo_ndcg_approx/<ds>_pseudo_ndcg_approx_sparse.jsonl` | RSDG@30 reward cache | 50 MB |

`<ds> ∈ {nfcorpus, fiqa, fever, hotpotqa, msmarco}`.

Together these allow exact offline GRPO reproduction without rerunning the 70B
teacher, the BM25 retrieval, or the cross-encoder scoring. Verify a download
with:

```bash
python scripts/check_artifacts.py
```

## Optional extras

- `..._pseudo_ndcg_approx_dense.jsonl` — reward caches for the dense (Contriever)
  reward source, needed for Appendix D.1. Adds ~60 MB.
- `pseudo_ndcg_approx/<reranker>/` — caches for the Sec. 4.2 reranker-robustness
  study (`BAAI__bge-reranker-v2-m3` at k=10, `Alibaba-NLP__gte-reranker-modernbert-base`,
  `mixedbread-ai__mxbai-rerank-large-v2`), selected with `--reranker`.
- Labelled per-query NDCG files (`per_qid_ndcg_<ds>_*.json`) if you want to
  rebuild the released qid lists with `rsdg/curate_queries.py --reproduce-release`.

## Known gap in the released cache

`fever_pseudo_ndcg_approx_sparse.jsonl` covers 169 of the 2,103 curated FEVER
queries. The paper's Stage-2 run used the file in this state, which is why it
reports 1,730 update steps; regenerating the full FEVER cache with
`rsdg/compute_rsdg.py` changes both the step count and the results. Ship the file
as-is for reproduction and document the gap.

## Publish separately as model artifacts

- Final LoRA adapters for the 8B, 3B and Qwen robustness runs
  (`adapter_config.json` + `adapter_model.safetensors` + tokenizer files).
- A model card naming the base model, the data artifacts, the licence and the
  intended use.

Adapter configs should read `r=32, lora_alpha=32, lora_dropout=0.05,
target_modules=["q_proj","v_proj"]` with
`base_model_name_or_path=meta-llama/Llama-3.1-8B-Instruct` for the main run.

**Naming is not free-form.** Adapters on a Llama base are derivative works under
the Llama Community License, whose Sec. 1.b.i requires the released model name to
*begin* with `Llama` — e.g. `Llama-3.1-8B-RAQE-LoRA` and
`Llama-3.2-3B-RAQE-LoRA`, not `RAQE-Llama-3.1-8B`. Each release also needs
"Built with Llama" on its card, a copy of the agreement and use policy, the
`NOTICE` attribution string, and `license: llama3.1` / `llama3.2` in the card
front matter. The Qwen adapter is Apache-2.0 and has no such constraint. See
`docs/LICENSES.md` §2.1.

## Query text is stripped

`scripts/package_artifacts.sh` removes the `query` and `expanded_query` fields
before building the tarball, because MS MARCO, NFCorpus and FiQA grant no
redistribution right over their query strings. Consumers recover the text from
their own `datasets/IR/`, so a stripped artifact trains and evaluates
identically. `KEEP_QUERY_TEXT=1` disables the stripping — only use it if you have
established that redistributing those strings is acceptable for your release.

Publish the artifact under its own terms (for example CC BY-NC-SA 4.0), not under
the repository's Apache-2.0 licence, and include the `NOTICE` file the packaging
script copies in. See `docs/LICENSES.md` §4.

## Do not redistribute

Raw BEIR/MS MARCO datasets, Pyserini indexes, optimizer checkpoints, W&B logs
and cache directories. Provide source URLs and dataset licences instead — see
`docs/DATA.md §1` and `docs/LICENSES.md §3`.
