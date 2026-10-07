#!/usr/bin/env python3
"""Generate one RAQE pseudo-passage per evaluation query.

Evaluation queries are the BEIR *test* queries, i.e. the entries of
``queries.jsonl`` whose ``_id`` appears in ``qrels/test.tsv``. ``queries.jsonl``
itself also contains the train/dev queries (509,962 rows for MS MARCO against 43
test queries), so the qrels filter is what keeps inference at the paper's scale.
"""
import argparse
import json
import sys
from pathlib import Path

# Allow `python evaluation/generate_raqe.py` from the repository root.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from sft.train.prompts import build_user_only_messages


def load_test_qids(dataset_dir: Path) -> set[str]:
    """Return the qids that have at least one judgement in qrels/test.tsv."""
    qrels_path = dataset_dir / "qrels" / "test.tsv"
    if not qrels_path.exists():
        raise FileNotFoundError(
            f"Missing {qrels_path}. Run `python get_IR_dataset.py -dataset <name>` first "
            "(see docs/DATA.md)."
        )
    qids: set[str] = set()
    for i, line in enumerate(qrels_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        if i == 0 and line.lower().startswith("query-id"):
            continue
        qids.add(line.split("\t")[0].strip())
    return qids


def load_queries(dataset_dir: Path, limit: int | None):
    """Load (qid, text) for the test queries of one BEIR dataset, in file order."""
    test_qids = load_test_qids(dataset_dir)
    queries = []
    for line in (dataset_dir / "queries.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        qid = str(row.get("_id", row.get("id")))
        if qid not in test_qids:
            continue
        queries.append((qid, row["text"]))
        if limit and len(queries) >= limit:
            break
    return queries


def render(tokenizer, query):
    messages = build_user_only_messages("pseudo", query)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n\n".join(f"<{m['role']}>\n{m['content']}" for m in messages) + "\n<assistant>\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--dataset-root", default="datasets/IR")
    parser.add_argument("--datasets", default="nfcorpus,fiqa,fever,hotpotqa,msmarco,arguana,nq,scidocs")
    parser.add_argument("--output-dir", default="evaluation/results/raqe")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    # Appendix A.2 specifies a single pseudo-passage per query of at most 128
    # tokens and says nothing further about decoding, so the defaults here are
    # deterministic: greedy, no length floor. The generator used before release
    # instead drew one top-p 0.9 sample at temperature 1.0 with a 96-token floor;
    # `--do-sample --min-new-tokens 96` restores that.
    parser.add_argument("--min-new-tokens", type=int, default=0,
                        help="Suppress EOS until this many tokens have been emitted (0 = off).")
    parser.add_argument("--do-sample", action="store_true",
                        help="Sample instead of decoding greedily.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Only used with --do-sample.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Only used with --do-sample.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (ignored without --do-sample)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate datasets whose ours.json already exists.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, min_new_tokens=args.min_new_tokens,
                      num_beams=1, num_return_sequences=1, use_cache=True)
    if args.do_sample:
        gen_kwargs.update(do_sample=True, top_p=args.top_p, temperature=args.temperature)
    else:
        gen_kwargs["do_sample"] = False

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only batched generation needs left padding, otherwise the pad run
    # sits between the prompt and the first generated token.
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, args.model_dir).eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    for dataset in [x for x in args.datasets.split(",") if x]:
        out = Path(args.output_dir) / dataset / "ours.json"
        if out.exists() and not args.overwrite:
            print(f"Skipping {dataset}: {out} already exists (use --overwrite to regenerate).")
            continue

        dataset_dir = Path(args.dataset_root) / dataset / dataset
        queries = load_queries(dataset_dir, args.max_queries)
        print(f"{dataset}: generating {len(queries)} expansions")

        records = []
        for start in range(0, len(queries), args.batch_size):
            chunk = queries[start:start + args.batch_size]
            prompts = [render(tokenizer, query) for _qid, query in chunk]
            # Tokenizer settings kept identical to the batch generator used for the
            # paper numbers (left padding, truncation at 2048, pad to a multiple of 8).
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                               max_length=2048, pad_to_multiple_of=8).to(model.device)
            with torch.inference_mode():
                output = model.generate(**inputs, **gen_kwargs,
                                        pad_token_id=tokenizer.pad_token_id,
                                        eos_token_id=tokenizer.eos_token_id)
            generated = output[:, inputs["input_ids"].shape[1]:]
            for (qid, query), tokens in zip(chunk, generated):
                pseudo = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                records.append({"qid": qid, "query": query, "pseudo": pseudo})
            if start % (args.batch_size * 25) == 0:
                print(f"  {dataset}: {len(records)}/{len(queries)}", flush=True)

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {len(records)} RAQE expansions to {out}")


if __name__ == "__main__":
    main()
