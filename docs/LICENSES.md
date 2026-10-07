# Licensing

This repository is **source code only** — no model weights, no dataset files.
The code is Apache-2.0 (see `LICENSE`). Everything the code *downloads* or
*produces* carries its own terms, and several of them are more restrictive than
Apache-2.0. This page records what applies to what.

Nothing here is legal advice; it is a summary of the terms as published by each
rights holder, with links so you can check them yourself.

---

## 1. Code

| Component | License |
|---|---|
| This repository | Apache-2.0 |
| [Pyserini](https://github.com/castorini/pyserini) / Anserini | Apache-2.0 |
| [BEIR](https://github.com/beir-cellar/beir) toolkit | Apache-2.0 |
| transformers, peft, accelerate, trl, datasets | Apache-2.0 |
| [pytrec_eval](https://github.com/cvangysel/pytrec_eval) | MIT |
| PyTorch, NumPy | BSD-3-Clause |
| faiss | MIT |

All permissive and mutually compatible. No third-party source is vendored into
this repository.

---

## 2. Models

| Model | Role | License |
|---|---|---|
| `meta-llama/Llama-3.1-8B-Instruct` | main student policy | **Llama 3.1 Community License** |
| `meta-llama/Llama-3.1-70B-Instruct` | teacher | **Llama 3.1 Community License** |
| `meta-llama/Llama-3.2-3B-Instruct` | 3B student (Table 17) | **Llama 3.2 Community License** |
| `Qwen/Qwen3-4B-Instruct-2507` | Qwen robustness study (Sec. F.2) | Apache-2.0 |
| `BAAI/bge-reranker-v2-m3` | RSDG reward reranker | Apache-2.0 |
| `facebook/contriever` | optional dense retriever | **CC BY-NC 4.0 — non-commercial** |

### 2.1 Releasing a RAQE adapter (Llama obligations)

A LoRA adapter trained by `scripts/train_sft.sh` / `scripts/train_grpo.sh` is a
derivative of Llama Materials. Sec. 1.b of the Llama 3.1 / 3.2 Community License
then requires all of the following when you distribute it:

1. **Name it with a `Llama` prefix.** "you shall also include 'Llama' at the
   beginning of any such AI model name" (§1.b.i). For example
   `Llama-3.1-8B-RAQE-LoRA`, not `RAQE-Llama-3.1-8B`. Sec. 5.a grants the "Llama"
   trademark *only* for this purpose.
2. **Display "Built with Llama"** prominently on the model card / repository
   page (§1.b.i(B)).
3. **Ship a copy of the agreement** with the adapter (§1.b.i(A)), plus the
   Acceptable Use Policy it incorporates (§1.b.iv).
4. **Ship a `Notice` file** containing, verbatim (§1.b.iii):
   `Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.`
   (and the Llama 3.2 wording for the 3B adapter).
5. On the Hub, set `license: llama3.1` (or `llama3.2`) and `base_model:` in the
   model-card front matter.

The repository-level `NOTICE` already carries items 2 and 4; copy it alongside
each released adapter.

Llama 3.1 and 3.2 place **no restriction on using model outputs to train other
models** — the Llama 2 clause that forbade it was removed — but the naming rule
in §1.b.i still applies to anything trained on Llama outputs. That covers the
teacher pseudo-passages in the training artifact.

### 2.2 Contriever

`facebook/contriever` is CC BY-NC 4.0. It is used only by the optional dense
retrieval experiment, which this release does not bundle. If you add it, the
resulting embeddings, indexes and numbers are non-commercial-research-only, and
that restriction does not flow from this repository's Apache-2.0 license.

---

## 3. Datasets

RAQE downloads BEIR and MS MARCO from their official sources and redistributes
none of them. Their terms still govern what you may do with derived files.

| Dataset | Terms | Redistribution |
|---|---|---|
| MS MARCO | non-commercial research only; ["made available free of charge without extending any license or other intellectual property rights"](https://microsoft.github.io/msmarco/) | **no grant** |
| NFCorpus | ["free to use for academic purposes"](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/); other uses need the NutritionFacts.org terms | **no grant** |
| FiQA-2018 | ["available only for non-commercial use"](https://sites.google.com/view/fiqa/home) | non-commercial |
| FEVER | CC BY-SA 3.0 | share-alike |
| NQ | CC BY-SA 3.0 | share-alike |
| HotpotQA | CC BY-SA 4.0 | share-alike |
| ArguAna | CC BY 4.0 | attribution |
| SciDocs | CC BY 4.0 ([`allenai/scidocs`](https://github.com/allenai/scidocs/blob/master/LICENSE); the BEIR paper's appendix says GPL-3.0 — the repository file is the better authority) | attribution |

BEIR itself is explicit that it makes no licensing claim: "we do not vouch for
their quality or fairness, or claim that you have license to use the dataset."
The `BeIR/*` mirrors on the Hub are blanket-tagged `cc-by-sa-4.0` by the
uploader; that tag is not the rights holder's and should not be relied on.

---

## 4. The training artifact

The offline artifact described in `configs/artifacts.md` is where the dataset
terms actually bite, because it is the only thing this project publishes that is
derived from the corpora.

It contains, per training query: the qid, the eight Llama-generated
pseudo-passages, the retrieved doc ids, and the reranker scores. It does **not**
contain any document text.

By default `scripts/package_artifacts.sh` also **strips the original query
strings** (`query` / `expanded_query`) from the packaged files, because MS MARCO,
NFCorpus and FiQA grant no redistribution right over them. The training and RSDG
scripts read the query text from your local `datasets/IR/` instead, so a stripped
artifact is fully functional. Pass `KEEP_QUERY_TEXT=1` only if you have
separately satisfied yourself that redistributing those strings is acceptable in
your jurisdiction and use case.

What remains in a stripped artifact is Llama output plus identifiers and
numbers. Ship it with:

- the "Built with Llama" notice and the Llama licence texts (§2.1);
- a statement that it is for non-commercial research use, since it derives from
  MS MARCO, NFCorpus and FiQA queries even though their text is not included;
- attribution for FEVER, NQ, HotpotQA, ArguAna and SciDocs.

Keep it out of the Apache-2.0 scope: license the artifact separately (for example
CC BY-NC-SA 4.0) rather than implying the repository licence covers it.

---

## 5. Citation obligations

Cite the dataset papers you evaluate on — BEIR (Thakur et al., 2021), MS MARCO
(Bajaj et al., 2016), NFCorpus (Boteva et al., 2016), FiQA (Maia et al., 2018),
FEVER (Thorne et al., 2018), HotpotQA (Yang et al., 2018), NQ (Kwiatkowski et
al., 2019), ArguAna (Wachsmuth et al., 2018), SciDocs (Cohan et al., 2020) — plus
Pyserini (Lin et al., 2021) for the BM25 implementation.
