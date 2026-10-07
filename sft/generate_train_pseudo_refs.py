import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Iterable, Tuple

import random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from accelerate import Accelerator
from contextlib import nullcontext
from glob import glob

# Project-local imports.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from sft.train.prompts import build_user_only_messages


# ===== 공통 유틸들 =====

def load_train_queries(dataset_root: str, dataset: str, max_queries: int | None = None) -> Dict[str, str]:
    """
    Train 용 query 로더.
    datasets/IR/{dataset}/{dataset}/queries.jsonl 에서 _id, text 읽어서
    {qid(str): text(str)} 딕셔너리로 반환.

    max_queries 가 주어지면 앞에서부터 그 개수만 사용.
    """
    queries_file = os.path.join(dataset_root, dataset, dataset, "queries.jsonl")
    if not os.path.exists(queries_file):
        raise FileNotFoundError(f"Train queries file not found: {queries_file}")

    queries: Dict[str, str] = {}
    with open(queries_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            qid = obj.get("_id") or obj.get("id")
            text = obj.get("text") or ""
            if qid is None or not text:
                continue
            qid_str = str(qid)
            queries[qid_str] = text  # ★ 여기에서 'ㅎ' 이었던 부분을 text로 수정
            if max_queries is not None and len(queries) >= int(max_queries):
                break

    print(f"[pseudo-gen] Loaded {len(queries)} train queries from {queries_file}")
    return queries


def sample_train_qids(dataset_root: str, dataset: str, n: int, seed: int) -> List[str]:
    """Uniformly sample `n` query ids from the official training split.

    Sec. 3.2.1: "For each dataset, we sample 3,000 queries uniformly at random to
    construct an initial pool of training data." The pool is the set of query ids
    that carry at least one judgement in qrels/train.tsv.
    """
    qrels_path = os.path.join(dataset_root, dataset, dataset, "qrels", "train.tsv")
    if not os.path.exists(qrels_path):
        raise FileNotFoundError(f"Train qrels not found: {qrels_path}")

    qids: List[str] = []
    seen = set()
    with open(qrels_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0 and line.lower().startswith("query-id"):
                continue
            if not line.strip():
                continue
            qid = line.split("\t")[0].strip()
            if qid and qid not in seen:
                seen.add(qid)
                qids.append(qid)

    rng = random.Random(seed)
    if n and n < len(qids):
        qids = rng.sample(qids, n)
    print(f"[pseudo-gen] Sampled {len(qids)} train qids for {dataset} (pool={len(seen)}, seed={seed})")
    return qids


def _is_sft_dir(path: Path) -> bool:
    """진짜 SFT/모델 디렉토리인지 여부 판단."""
    if not path.is_dir():
        return False
    candidate_files: Iterable[str] = [
        "config.json",
        "adapter_config.json",
        "pytorch_model.bin",
        "model.safetensors",
        "adapter_model.bin",
        "adapter_model.safetensors",
    ]
    return any((path / fname).exists() for fname in candidate_files)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")


def _load_llama_tokenizer(name: str):
    """Llama 계열 토크나이저 로더."""
    # 1차 시도: fast + legacy=False
    try:
        tok = AutoTokenizer.from_pretrained(
            name,
            trust_remote_code=True,
            use_fast=True,
            legacy=False,
        )
        return tok
    except TypeError:
        pass
    except Exception as e:
        print(f"[load_tok] fast+legacy=False 실패 ({name}): {e}")

    # 2차 시도: fast
    try:
        tok = AutoTokenizer.from_pretrained(
            name,
            trust_remote_code=True,
            use_fast=True,
        )
        return tok
    except Exception as e:
        print(f"[load_tok] fast tokenizer 실패 ({name}): {e}")

    # 3차 시도: slow
    try:
        tok = AutoTokenizer.from_pretrained(
            name,
            trust_remote_code=True,
            use_fast=False,
        )
        return tok
    except Exception as e:
        print(f"[load_tok] slow tokenizer 실패 ({name}): {e}")
        raise


def load_sft_model(model_dir: str, base_model: str, bf16: bool = True, use_task_tokens: bool = False):
    """evaluate_sft.py에 있던 load_sft_model 거의 그대로."""
    model_dir_path = Path(model_dir)
    tokenizer = None

    is_local_sft = _is_sft_dir(model_dir_path)

    # 1) local SFT 디렉토리에서 토크나이저 우선 로드
    if is_local_sft:
        try:
            tokenizer = _load_llama_tokenizer(model_dir)
            print(f"[pseudo-gen] Loaded tokenizer from local model_dir: {model_dir}")
        except Exception as e:
            print(f"[pseudo-gen] tokenizer load from model_dir failed: {e}; falling back to base_model")

    # 2) 아니면 base_model에서 토크나이저 로드
    if tokenizer is None:
        tokenizer = _load_llama_tokenizer(base_model)
        print(f"[pseudo-gen] Loaded tokenizer from base_model: {base_model}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_task_tokens:
        special = {"additional_special_tokens": ["<TASK_PSEUDO>", "<TASK_IMPORTANCE>"]}
        tokenizer.add_special_tokens(special)

    try:
        tokenizer.padding_side = "left"
    except Exception:
        pass

    dtype = torch.bfloat16 if bf16 and torch.cuda.is_available() else None

    has_adapter = any(
        (Path(model_dir) / fname).exists()
        for fname in ["adapter_model.safetensors", "adapter_model.bin", "adapter_config.json"]
    )

    if has_adapter and model_dir_path.exists():
        try:
            from peft import PeftModel

            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
            try:
                if base.get_input_embeddings().weight.shape[0] != len(tokenizer):
                    base.resize_token_embeddings(len(tokenizer))
            except Exception:
                pass

            if use_task_tokens:
                try:
                    base.resize_token_embeddings(len(tokenizer))
                except Exception:
                    pass

            model = PeftModel.from_pretrained(base, model_dir)
            try:
                model = model.merge_and_unload()
            except Exception:
                pass
            return model, tokenizer
        except Exception as e:
            print(f"[pseudo-gen] PEFT adapter load failed: {e}; falling back to full model load")

    # full model 로드
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir if model_dir_path.exists() else base_model,
            trust_remote_code=True,
            torch_dtype=dtype,
            ignore_mismatched_sizes=True,
        )
        if use_task_tokens and model.get_input_embeddings().weight.shape[0] != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))
        return model, tokenizer
    except Exception as e:
        print(f"[pseudo-gen] Full model load failed: {e}; trying partial state dict load")

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        if use_task_tokens:
            base.resize_token_embeddings(len(tokenizer))

        sd_path = None
        for fname in ["pytorch_model.bin", "model.safetensors", "pytorch_model.safetensors"]:
            p = Path(model_dir) / fname
            if p.exists():
                sd_path = p
                break
        if sd_path is None:
            raise RuntimeError("No loadable weights found in model_dir")
        state_dict = torch.load(sd_path, map_location="cpu")

        model_emb = base.get_input_embeddings().weight
        head = base.get_output_embeddings()
        filtered = {}
        for k, v in state_dict.items():
            if "embed_tokens.weight" in k and v.shape != model_emb.shape:
                continue
            if head is not None and ("lm_head.weight" in k) and v.shape != head.weight.shape:
                continue
            filtered[k] = v
        base.load_state_dict(filtered, strict=False)
        return base, tokenizer


def apply_chat_template(tokenizer, messages: List[Dict[str, str]]) -> str:
    """evaluate_sft.py와 동일한 chat template 적용."""
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    parts = []
    for m in messages:
        parts.append(f"<{m['role']}>\n{m['content']}\n</{m['role']}>")
    parts.append("<assistant>\n")
    return "\n\n".join(parts)


def batch_generate(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    batch_size: int = 4,
    device: torch.device | None = None,
) -> List[str]:
    """evaluate_sft.py의 batch_generate를 약간 수정해서 사용."""
    outs: List[str] = []

    if device is None:
        raise ValueError("batch_generate must be called with an explicit device (use accelerator.device)")

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating", unit="batch"):
            chunk = prompts[i : i + batch_size]
            enc = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(device)

            # do_sample 여부 결정
            if temperature is None:
                temperature = 0.0
            if top_p is None:
                top_p = 1.0

            do_sample = (float(temperature) > 0.0) or (float(top_p) < 1.0)

            gen_kwargs = dict(
                **enc,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=do_sample,
            )
            if do_sample:
                gen_kwargs.update(
                    {
                        "temperature": float(temperature),
                        "top_p": float(top_p),
                    }
                )

            gen = model.generate(**gen_kwargs)

            # prompt 길이 이후만 디코딩
            for j in range(gen.size(0)):
                input_len = enc["input_ids"][j].size(0)
                new_tokens = gen[j, input_len:]
                text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                outs.append(text.strip())

    return outs


# ===== 실제 pseudo ref 생성 로직 =====

def build_pseudo_prompts_for_qids(
    tokenizer,
    queries: Dict[str, str],
    qids: List[str],
    use_task_tokens: bool,
) -> Tuple[List[str], List[str]]:
    """주어진 qids 순서에 맞게 pseudo 생성용 프롬프트와 qid 리스트를 반환."""
    prompts: List[str] = []
    selected_qids: List[str] = []
    for qid in qids:
        if qid not in queries:
            continue
        q = queries[qid]
        msgs = build_user_only_messages("pseudo", q, use_task_tokens=use_task_tokens)
        prompt = apply_chat_template(tokenizer, msgs)
        prompts.append(prompt)
        selected_qids.append(qid)
    return selected_qids, prompts


def generate_pseudo_for_dataset(
    accelerator,
    model,
    tokenizer,
    dataset_root: str,
    dataset: str,
    qids_path: Path | None,
    rollouts: int,
    batch_size: int,
    max_new_tokens: int,
    use_task_tokens: bool,
    out_dir: Path,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: torch.device | None = None,
    max_queries: int | None = None,
    qid_list_override: List[str] | None = None,
):
    """
    한 dataset에 대해:
    - qids 파일을 읽고
    - 해당 qid의 *train* query를 로드해서
    - 각 qid마다 rollouts 개수만큼 pseudo ref를 샘플링 생성
    - JSONL로 저장
    """
    print(f"\n=== Dataset: {dataset} ===")
    if qid_list_override is not None:
        qid_list_raw = qid_list_override
    else:
        if qids_path is None or not qids_path.exists():
            raise FileNotFoundError(
                f"QID file not found: {qids_path}. Pass --sample_queries 3000 to draw the "
                "initial training pool instead (the curated lists only exist after RSDG)."
            )
        with open(qids_path, "r", encoding="utf-8") as f:
            qid_list_raw = json.load(f)
        print(f"[pseudo-gen] Loaded {len(qid_list_raw)} qids from {qids_path}")

    # qid를 문자열로 통일
    qid_list = [str(q) for q in qid_list_raw]

    # ★★ Train queries 로딩 (evaluation loader 말고 queries.jsonl 기반)
    queries_all = load_train_queries(dataset_root, dataset, max_queries=None)

    # qid_list 중 실제 존재하는 것만 사용
    existing_qids = [qid for qid in qid_list if qid in queries_all]
    missing = len(qid_list) - len(existing_qids)
    if missing > 0:
        print(f"[pseudo-gen] WARNING: {missing} qids not found in train queries; they will be skipped")

    # max_queries 제한 적용 (테스트 목적)
    if max_queries is not None and max_queries > 0:
        before = len(existing_qids)
        existing_qids = existing_qids[:max_queries]
        print(f"[pseudo-gen] Limited to first {len(existing_qids)} queries (from {before}) for testing")

    # qid 순서대로 pseudo 프롬프트 만들기
    # 1) qid → prompt 미리 만듦
    base_qids, base_prompts = build_pseudo_prompts_for_qids(
        tokenizer, queries_all, existing_qids, use_task_tokens=use_task_tokens
    )
    print(f"[pseudo-gen] Will generate pseudo refs for {len(base_qids)} queries")

    # ★ qid → prompt 매핑 만들어두면 편함
    qid_to_prompt = {qid: prompt for qid, prompt in zip(base_qids, base_prompts)}

        # 2) qid 단위로 rank에 분배 (accelerate의 split_between_processes는 context manager임)
    if accelerator.num_processes > 1:
        ctx = accelerator.split_between_processes(base_qids)
    else:
        # single-process일 때는 그냥 base_qids를 그대로 넘기는 dummy context manager
        ctx = nullcontext(base_qids)

    # is_local_main_process가 없을 수도 있어서 안전하게 처리
    is_local_main = getattr(accelerator, "is_local_main_process", accelerator.is_main_process)

    with ctx as local_qids:
        if is_local_main:
            print(
                f"[pseudo-gen] rank={accelerator.local_process_index}: "
                f"num_qids={len(local_qids)} (world_size={accelerator.num_processes})"
            )

        # 3) 각 rank는 자기 qid 들에 대해서만 rollout 프롬프트 생성
        local_prompts: List[str] = []
        local_meta: List[Tuple[str, int, str]] = []

        for qid in local_qids:
            prompt = qid_to_prompt[qid]
            qtext = queries_all[qid]
            for r in range(rollouts):
                local_prompts.append(prompt)
                local_meta.append((qid, r, qtext))

        print(
            f"[pseudo-gen] rank={accelerator.local_process_index}: "
            f"local_prompts={len(local_prompts)}"
        )

        # 4) 각 rank에서 local_prompts 로 generate
        gen_texts = batch_generate(
            model=model,
            tokenizer=tokenizer,
            prompts=local_prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            batch_size=batch_size,
            device=device,
        )

        assert len(gen_texts) == len(local_meta)

        # ✅ 레코드 만들기 + rank별 파일 저장
        out_dir.mkdir(parents=True, exist_ok=True)
        rank = accelerator.local_process_index
        out_path = out_dir / f"{dataset}_pseudo_train.rank{rank}.jsonl"

        with open(out_path, "w", encoding="utf-8") as wf:
            for (qid, ridx, query_text), pseudo in zip(local_meta, gen_texts):
                rec = {
                    "dataset": dataset,
                    "qid": qid,
                    "rollout": int(ridx),
                    "query": query_text,
                    "pseudo": pseudo,
                }
                wf.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if is_local_main:
            print(f"[pseudo-gen] rank={rank} wrote {len(local_meta)} records to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate train pseudo refs for multiple IR datasets")
    parser.add_argument("--model_dir", type=str, default="meta-llama/Llama-3.1-70B-Instruct",
                        help="Teacher model: a Hugging Face id, a full local model dir, or a LoRA adapter dir")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--datasets", type=str, default="nfcorpus,fiqa,fever,hotpotqa,msmarco")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="datasets/IR",
        help="Root directory containing IR datasets (expects {root}/{dataset}/{dataset}/queries.jsonl)",
    )
    parser.add_argument("--qids_root", type=str, default="artifacts/sft/pr_final",
                        help="Directory with per_qid_ndcg_{dataset}_filtered_qids_pr.json. Ignored "
                             "when --sample_queries is set, which is the from-scratch path: "
                             "curation (Eq. 3) can only run after these rollouts exist.")
    parser.add_argument("--out_root", type=str, default=None,
                        help="Where to write <dataset>/<dataset>_pseudo_train.jsonl. "
                             "Default: artifacts/train/<basename of --model_dir>")
    parser.add_argument("--sample_queries", type=int, default=None,
                        help="Sample this many training queries per dataset uniformly at random "
                             "from qrels/train.tsv instead of reading a qid list (paper: 3000).")
    parser.add_argument("--sample_seed", type=int, default=42, help="Seed for --sample_queries")
    parser.add_argument("--max_queries", type=int, default=None, help="Maximum number of queries to process per dataset (for testing)")
    parser.add_argument("--rollouts", type=int, default=8, help="Number of pseudo refs per query")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Teacher rollout length used for the released artifact")
    parser.add_argument("--use_task_tokens", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="If set, overwrite existing pseudo ref files")
    args = parser.parse_args()

    accelerator = Accelerator()
    set_seed(args.seed + accelerator.process_index)

    # 모델 / 토크나이저 로드
    model, tokenizer = load_sft_model(
        model_dir=args.model_dir,
        base_model=args.base_model,
        bf16=True,
        use_task_tokens=args.use_task_tokens,
    )

    # ✅ 모델을 accelerator에 올리기
    model = accelerator.prepare(model)
    device = accelerator.device
    if accelerator.is_main_process:
        print(f"[pseudo-gen] Using device: {device}, world_size={accelerator.num_processes}")

    # Output layout must match what Stage 1 and Stage 2 read:
    #   <out_root>/<dataset>/<dataset>_pseudo_train.jsonl
    out_root = Path(args.out_root) if args.out_root else (
        Path("artifacts") / "train" / Path(args.model_dir.rstrip("/")).name
    )
    if accelerator.is_main_process:
        out_root.mkdir(parents=True, exist_ok=True)
        print(f"[pseudo-gen] Output root: {out_root}")
    accelerator.wait_for_everyone()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    for dset in datasets:
        qids_path = Path(args.qids_root) / f"per_qid_ndcg_{dset}_filtered_qids_pr.json"
        if not qids_path.exists():
            legacy = Path(args.qids_root) / f"per_qid_ndcg_{dset}_filtered_qids.json"
            if legacy.exists():
                qids_path = legacy
        out_dir = out_root / dset

        # 이미 완성된 merged 파일이 있고 overwrite 안할거면 스킵
        merged_file = out_dir / f"{dset}_pseudo_train.jsonl"
        if merged_file.exists() and not args.overwrite:
            if accelerator.is_main_process:
                print(f"[pseudo-gen] {merged_file} already exists, skip (use --overwrite to regenerate)")
            # 모든 rank에서 동일하게 continue 되도록 그냥 continue만 쓰면 됨
            continue

        qid_list_override = None
        if args.sample_queries:
            qid_list_override = sample_train_qids(
                args.dataset_root, dset, args.sample_queries, args.sample_seed
            )

        generate_pseudo_for_dataset(
            qid_list_override=qid_list_override,
            model=model,
            tokenizer=tokenizer,
            dataset=dset,
            qids_path=qids_path,
            rollouts=args.rollouts,
            batch_size=args.batch_size,
            dataset_root=args.dataset_root,
            max_new_tokens=args.max_new_tokens,
            use_task_tokens=args.use_task_tokens,
            out_dir=out_dir,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
            max_queries=args.max_queries,
            accelerator=accelerator,
        )

        # ====== 여기서부터 merge 단계 ======
        accelerator.wait_for_everyone()  # 모든 rank가 partial 파일 다 쓴 후

        if accelerator.is_main_process:
            pattern = str(out_dir / f"{dset}_pseudo_train.rank*.jsonl")
            part_files = sorted(glob(pattern))

            print(f"[pseudo-gen] Merging {len(part_files)} rank files into {merged_file}")

            with open(merged_file, "w", encoding="utf-8") as wf:
                for pf in part_files:
                    with open(pf, "r", encoding="utf-8") as rf:
                        for line in rf:
                            wf.write(line)
                    os.remove(pf)  # partial 파일 삭제

            print(f"[pseudo-gen] Done merge → {merged_file} (and removed rank*.jsonl)")
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
