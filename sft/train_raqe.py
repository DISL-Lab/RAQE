#!/usr/bin/env python3
"""Stage-1 RAQE supervised distillation on filtered teacher pseudo-passages.

Implements Eq. (4) of the paper: minimise the negative log-likelihood of the
teacher pseudo-passage e* given the query q, over the curated training queries.
Only the assistant span is supervised; the prompt is masked with -100.
"""

import argparse
import json
import sys
from pathlib import Path

# Allow `python sft/train_raqe.py` from the repository root.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

from sft.train.prompts import build_messages_for_task

# The five datasets with official training splits used for distillation (Sec. 3.2.1).
TRAIN_DATASETS = ("nfcorpus", "fiqa", "fever", "hotpotqa", "msmarco")


class Records(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def load_filtered_qids(filtered_dir: Path) -> dict[str, set[str]]:
    """Map dataset name -> retained qids.

    Qids are only unique within a dataset (``PLAIN-3`` exists in NFCorpus, ``1``
    in several others), so the lists are kept per dataset rather than pooled.
    """
    qids: dict[str, set[str]] = {}
    for path in sorted(filtered_dir.glob("per_qid_ndcg_*_filtered_qids_pr.json")):
        dataset = path.name[len("per_qid_ndcg_"):-len("_filtered_qids_pr.json")]
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            selected = set(map(str, data.keys()))
        else:
            selected = {str(item.get("qid")) if isinstance(item, dict) else str(item) for item in data}
        qids[dataset] = selected
    if not qids:
        raise RuntimeError(
            f"No per_qid_ndcg_*_filtered_qids_pr.json files under {filtered_dir}. See docs/DATA.md."
        )
    return qids


def load_dataset_queries(dataset_root: Path, dataset: str) -> dict[str, str]:
    """qid -> query text, read from the locally downloaded BEIR files.

    The packaged artifact strips the original query strings (docs/LICENSES.md),
    so the text is recovered from datasets/IR/ when a record does not carry it.
    """
    path = dataset_root / dataset / dataset / "queries.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; it is needed because the artifact does not carry "
            "query text. See docs/DATA.md."
        )
    queries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        queries[str(row.get("_id", row.get("id")))] = row.get("text", "")
    return queries


def load_teacher_examples(pseudo_root: Path, selected_qids: dict[str, set[str]],
                          dataset_root: Path):
    examples = []
    per_dataset: dict[str, int] = {}
    for path in sorted(pseudo_root.glob("*/*_pseudo_train.jsonl")):
        dataset = path.parent.name
        keep = selected_qids.get(dataset)
        if keep is None:
            continue
        queries: dict[str, str] | None = None
        seen: set[str] = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["qid"])
            if qid not in keep or qid in seen or not row.get("pseudo"):
                continue
            # The first cached teacher rollout is the SFT target e*.
            if int(row.get("rollout", 0)) != 0:
                continue
            query = row.get("query")
            if not query:
                if queries is None:
                    queries = load_dataset_queries(dataset_root, dataset)
                query = queries.get(qid)
                if not query:
                    continue
            examples.append((query, row["pseudo"]))
            seen.add(qid)
        per_dataset[dataset] = len(seen)
    if not examples:
        raise RuntimeError("No filtered teacher pseudo-passages found. Check --pseudo-root and --filtered-dir.")
    print(f"Teacher examples per dataset: {per_dataset}")
    return examples


def render(tokenizer, messages, add_generation_prompt=False):
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    return "\n\n".join(f"<{m['role']}>\n{m['content']}" for m in messages)


def tokenize_examples(tokenizer, examples, max_length):
    records = []
    for query, pseudo in examples:
        messages = build_messages_for_task("pseudo", query, pseudo)
        prompt = render(tokenizer, messages[:-1], add_generation_prompt=True)
        full = render(tokenizer, messages)
        encoded = tokenizer(full, truncation=True, max_length=max_length)
        prompt_len = len(tokenizer(prompt, truncation=True, max_length=max_length)["input_ids"])
        labels = encoded["input_ids"].copy()
        labels[:prompt_len] = [-100] * min(prompt_len, len(labels))
        encoded["labels"] = labels
        records.append(encoded)
    return Records(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--pseudo-root", type=Path, required=True)
    parser.add_argument("--filtered-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/IR"),
                        help="Used to recover query text when the artifact omits it.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Debug only: truncate the training set to this many examples.")
    args = parser.parse_args()

    selected_qids = load_filtered_qids(args.filtered_dir)
    examples = load_teacher_examples(args.pseudo_root, selected_qids, args.dataset_root)
    if args.max_examples:
        examples = examples[: args.max_examples]
    print(f"Loaded {len(examples)} filtered RAQE SFT examples.")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.config.use_cache = False
    # Matches the released adapters: r=32, alpha=32, dropout=0.05 on q_proj/v_proj
    # (PEFT's default target modules for Llama).
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_r, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    train_dataset = tokenize_examples(tokenizer, examples, args.max_length)
    train_args = TrainingArguments(
        output_dir=str(args.output_dir), num_train_epochs=args.epochs,
        learning_rate=args.learning_rate, per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps, bf16=True, tf32=True,
        weight_decay=args.weight_decay, warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type, max_grad_norm=args.max_grad_norm,
        seed=args.seed, logging_steps=10, save_strategy="epoch", save_total_limit=1, report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=train_args, train_dataset=train_dataset,
                      data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100, padding=True))
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))


if __name__ == "__main__":
    main()
