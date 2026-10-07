import os
import json
import sys
from pathlib import Path
import torch
import argparse
import math
import random
import numpy as np

# Ensure repository-local imports resolve when this file is launched directly.
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from pathlib import Path as _Path
import raqe_jvm  # noqa: F401  (must precede the pyserini import)
try:
    from pyserini.search.lucene import LuceneSearcher
    _PYSERINI_IMPORT_ERROR = None
except Exception as _e:
    LuceneSearcher = None
    _PYSERINI_IMPORT_ERROR = _e
import pytrec_eval
# 파일 상단 import 근처에 추가
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
from accelerate import Accelerator

# ==== Offline RL Dataset ====

from sft.train.prompts import build_user_only_messages

class OfflineRLDataset(Dataset):
    """
    각 샘플 = (input_ids, attention_mask, labels, advantage, 메타)
    - input_ids: prompt + pseudo
    - labels: prompt 부분은 -100, pseudo 부분은 token id
    - advantage: REINFORCE용 스칼라 (reward - baseline)
    """
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def offline_rl_collate_fn(batch, pad_token_id: int):
    """
    batch: list of dicts with keys:
      - input_ids (list[int])
      - labels (list[int])
      - advantage (float)
    """
    max_len = max(len(ex["input_ids"]) for ex in batch)

    input_ids = []
    attention_mask = []
    labels = []
    advantages = []

    for ex in batch:
        ids = ex["input_ids"]
        labs = ex["labels"]
        pad_len = max_len - len(ids)

        input_ids.append(ids + [pad_token_id] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)
        labels.append(labs + [-100] * pad_len)
        advantages.append(ex["advantage"])

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    advantages = torch.tensor(advantages, dtype=torch.float32)

    # Preserve metadata for debug / NDGC printing: qid, dataset, approx_ndcg, base_ndcg
    qids = []
    datasets = []
    approx_ndcgs = []
    base_ndcgs = []
    for ex in batch:
        qids.append(str(ex.get("qid")))
        datasets.append(str(ex.get("dataset")))
        try:
            approx_ndcgs.append(float(ex.get("approx_ndcg", 0.0)))
        except Exception:
            approx_ndcgs.append(0.0)
        try:
            base_ndcgs.append(float(ex.get("base_ndcg", 0.0)))
        except Exception:
            base_ndcgs.append(0.0)

    approx_ndcgs = torch.tensor(approx_ndcgs, dtype=torch.float32)
    base_ndcgs = torch.tensor(base_ndcgs, dtype=torch.float32)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "advantages": advantages,
        "qids": qids,
        "datasets": datasets,
        "approx_ndcg": approx_ndcgs,
        "base_ndcg": base_ndcgs,
    }

def build_offline_rl_samples(
    qids,                    # [(qid, ds), ...]
    queries_by_dataset,      # {ds: {qid: query_text}}
    offline_ndcg_table,      # load_offline_ndcg_table 결과
    tokenizer,
    dataset_weights,         # {ds: weight} (이미 코드에 있음)
    delta_scale: float = 5.0,
    reward_clip: float = 2.0,
    adv_clip: float = 5.0,
    max_cands_per_qid: int | None = None,
    cand_select_mode: str = "rollout",
    use_task_tokens: bool = False,
    reward_transform: str = "identity",
):
    """
    각 (dataset, qid)에 대해:
      - offline_ndcg_table[ds][qid]["cands"] 안의 pseudo / approx_ndcg 사용
      - reward_raw = (approx_ndcg - base_ndcg) * delta_scale * dataset_weight
      - 같은 qid 그룹 내에서 reward를 평균 0으로 center -> advantage
      - prompt + pseudo를 토크나이즈해서 input_ids / labels 생성
    
    Note: 데이터는 이미 생성 시 max_tokens=128로 제한되었고,
          prompt도 짧아서 전체 길이가 512를 초과하지 않음.
    """
    samples = []
    _sample_append_count = 0

    # 1) (ds, qid)별로 후보 모으기
    grouped = {}  # (ds, qid) -> list[dict]
    for (qid, ds) in qids:
        ds = str(ds)
        qid = str(qid)
        if ds not in offline_ndcg_table:
            continue
        if qid not in offline_ndcg_table[ds]:
            continue
        qinfo = offline_ndcg_table[ds][qid]
        base_ndcg = float(qinfo.get("base_ndcg", 0.0))
        cands = qinfo.get("cands", [])
        if not cands:
            continue

        if max_cands_per_qid is not None and len(cands) > max_cands_per_qid:
            # rollout 수 축소 실험(8->4/2)에서는 후보 선택 정책을 명시적으로 고정한다.
            if cand_select_mode == "approx":
                cands = sorted(cands, key=lambda x: x.get("approx_ndcg", 0.0), reverse=True)[:max_cands_per_qid]
            else:
                cands = sorted(cands, key=lambda x: int(x.get("rollout", 0)))[:max_cands_per_qid]

        # query 텍스트
        qtext = queries_by_dataset.get(ds, {}).get(qid)
        if not qtext:
            continue

        grouped.setdefault((ds, qid), {
            "base_ndcg": base_ndcg,
            "cands": [],
            "query": qtext,
        })
        grouped[(ds, qid)]["cands"].extend(cands)

    safe_log(f"[offline-rl] grouped qids: {len(grouped)} (from qids={len(qids)})")

    # Diagnostic: count total candidate entries and how many have non-empty pseudo texts.
    total_cands = sum(len(v.get("cands", [])) for v in grouped.values())
    non_empty_pseudos = 0
    sample_nonempty_examples = []
    for (ds, qid), v in grouped.items():
        for c in v.get("cands", []):
            p = str(c.get("pseudo", "")).strip()
            if p:
                non_empty_pseudos += 1
                if len(sample_nonempty_examples) < 5:
                    sample_nonempty_examples.append((ds, qid, p[:80]))

    if total_cands > 0 and non_empty_pseudos == 0:
        # No usable pseudo texts found in the offline table — common causes:
        #  - pseudo_refs files were generated but contained empty outputs
        #  - merge step produced empty records (e.g., generate ran with 0 queries)
        msg_lines = [
            "[offline-rl][ERROR] Found candidate slots but ALL pseudo texts are empty.",
            f"datasets_in_group={sorted({ds for (ds, _q) in grouped.keys()})}",
            f"total_candidate_slots={total_cands}",
            "Common causes: pseudo generation produced empty strings, or merged pseudo refs are empty.",
            "Suggested actions:",
            " - Re-run pseudo generation with --max_queries>N or without --max_queries to ensure non-empty outputs.",
            " - Inspect artifacts/train/<teacher>/<dataset>/<dataset>_pseudo_train.jsonl for empty pseudo fields.",
        ]
        # Include a small example if available
        if sample_nonempty_examples:
            msg_lines.append("Example non-empty pseudos (truncated):")
            for ds, qid, p in sample_nonempty_examples:
                msg_lines.append(f"  ds={ds} qid={qid} pseudo_preview={p}")

        # Emit message and raise to make failure explicit (so caller can see reason)
        full_msg = "\n".join(msg_lines)
        raise RuntimeError(full_msg)

    # 2) 그룹별 reward_raw 계산 후, 평균 0으로 center -> advantage
    all_group_keys = list(grouped.keys())
    for (ds, qid) in all_group_keys:
        entry = grouped[(ds, qid)]
        base_ndcg = float(entry["base_ndcg"])
        qtext = entry["query"]
        cands = entry["cands"]
        if not cands:
            continue

        # dataset weight
        w_ds = dataset_weights.get(ds, 1.0)

        system_msg = "You are a helpful assistant for query expansion. Return only the requested pseudo-passage; do not include commentary."
        user_prompt = raqe_passage_prompt(qtext)
        
        # Prefix with task token if enabled
        if use_task_tokens:
            user_prompt = "<TASK_PSEUDO>" + user_prompt

        # chat template 기반 prompt string 만들기
        try:
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt},
            ]
            prompt_str = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,  # assistant 시작까지 포함
            )
        except Exception:
            prompt_str = system_msg + "\n\n" + user_prompt + "\n\nAssistant:"

        # prompt 토크나이즈 (special token은 chat template에 이미 포함되어 있을 가능성 크므로 add_special_tokens=False)
        prompt_ids = tokenizer(
            prompt_str,
            add_special_tokens=False,
            return_tensors=None,
        )["input_ids"]

        # 2-1) Reward r_i = Delta(q'_i, q) = RSDG@k(q'_i) - RSDG@k(q)  (Eq. 3).
        # The pre-release code squashed this with tanh(delta_scale * Delta) before
        # standardisation, which compresses large gains; --reward-transform tanh
        # restores it.
        reward_raw_list = []
        for cand in cands:
            approx_ndcg = float(cand.get("approx_ndcg", base_ndcg))
            delta = approx_ndcg - base_ndcg
            if reward_transform == "tanh":
                delta = math.tanh(delta_scale * delta)
            reward_raw_list.append(delta)

        # 2-2) 그룹 내 평균 0으로 center -> advantage
        # ✅ 중요: 여기서 같은 qid의 **모든 rollout(cands)** 을 한 번에 상대평가합니다
        num_rollouts = len(reward_raw_list)
        if reward_raw_list:
            mean_r = float(sum(reward_raw_list) / len(reward_raw_list))
        else:
            mean_r = 0.0

        advantages = [r - mean_r for r in reward_raw_list]
        std_r = float(np.std(advantages)) + 1e-8
        advantages = [(a / std_r) for a in advantages]
        advantages = [max(min(a, adv_clip), -adv_clip) for a in advantages]
        # 첫 3개 qid에 대해서만 상세 로그 출력
        if len(samples) < 30:  # 대략 첫 3-4개 qid
            safe_log(f"[offline-rl][ADVANTAGE] ds={ds} qid={qid}: {num_rollouts} rollouts, mean_r={mean_r:.4f}, std_r={std_r:.4f}")

        # clip은 weight 적용 후
        advantages = [max(min(a, adv_clip), -adv_clip) for a in advantages]
        # 3) 각 candidate에 대해 input_ids / labels 생성
        for cand, adv in zip(cands, advantages):
            # 음수는 0.2배만 반영
            #if adv < 0:
            #    adv = 0.2 * adv

            pseudo = str(cand.get("pseudo", "")).strip()
            if not pseudo:
                continue

            pseudo_ids = tokenizer(
                pseudo,
                add_special_tokens=False,
                return_tensors=None,
            )["input_ids"]

            # prompt + pseudo
            full_ids = prompt_ids + pseudo_ids
            # labels: prompt 부분은 -100, pseudo 부분은 token id
            labels = [-100] * len(prompt_ids) + pseudo_ids

            samples.append({
                "input_ids": full_ids,
                "labels": labels,
                "advantage": float(adv),
                "dataset": ds,
                "qid": qid,
                "base_ndcg": base_ndcg,
                # debugging용
                "approx_ndcg": float(cand.get("approx_ndcg", base_ndcg)),
            })
            _sample_append_count += 1
            # Log a tiny sample when first appended to help debug multi-process behavior
            if _sample_append_count <= 3:
                safe_log(f"[offline-rl][DEBUG] appended sample #{_sample_append_count} ds={ds} qid={qid} approx_ndcg={cand.get('approx_ndcg')}")
    # If no samples were produced despite grouped candidates / non-empty pseudos,
    # produce a detailed diagnostic message to help root-cause analysis.
    # Final debug: report how many samples were appended
    safe_log(f"[offline-rl] final_sample_count={len(samples)} grouped_qids={len(grouped)} total_candidate_slots={total_cands} non_empty_pseudos={non_empty_pseudos}")
    if not samples:
        # gather per-group diagnostics (limit to first 10 groups)
        diag_lines = []
        diag_lines.append(f"[offline-rl][DIAG] samples=0 grouped_qids={len(grouped)} total_cands={total_cands} non_empty_pseudos={non_empty_pseudos}")
        cnt = 0
        for (ds, qid), v in list(grouped.items())[:10]:
            cands = v.get("cands", [])
            non_empty = 0
            previews = []
            for c in cands:
                p = str(c.get("pseudo", "")).strip()
                if p:
                    non_empty += 1
                    if len(previews) < 3:
                        previews.append(p[:120].replace("\n", " "))
            diag_lines.append(f"  group ds={ds} qid={qid} cands={len(cands)} non_empty_pseudos={non_empty}")
            if previews:
                diag_lines.append(f"    previews: {previews}")
            # try tokenization diagnostics if possible
            try:
                qtext = v.get("query", "")
                pids = tokenizer(qtext, add_special_tokens=False, return_tensors=None)["input_ids"]
                diag_lines.append(f"    prompt_token_len={len(pids)}")
                if previews:
                    for i, pv in enumerate(previews):
                        try:
                            toks = tokenizer(pv, add_special_tokens=False, return_tensors=None)["input_ids"]
                            diag_lines.append(f"    preview[{i}] token_len={len(toks)}")
                        except Exception as e:
                            diag_lines.append(f"    preview[{i}] tokenization_failed: {e}")
            except Exception as e:
                diag_lines.append(f"    prompt_tokenization_failed: {e}")
            cnt += 1
        diag_lines.append("Suggested checks: run scripts/check_artifacts.py; verify qid keys are strings and match queries_by_dataset keys.")
        full_diag = "\n".join(diag_lines)
        raise RuntimeError(full_diag)
    # Return constructed samples
    return samples

def sequence_logprob_from_logits(logits, labels):
    # logits: [B, T, V], labels: [B, T]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    mask = (shift_labels != -100)
    safe_labels = shift_labels.clone()
    safe_labels[~mask] = 0

    log_probs = F.log_softmax(shift_logits, dim=-1)
    chosen = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

    token_sum = (chosen * mask).sum(dim=1)                 # [B]
    token_count = mask.sum(dim=1).clamp_min(1)             # [B]
    return token_sum, token_count

def run_offline_grpo(
    adapter_dir: str | None,
    base_model: str,
    qids: list,
    queries_by_dataset: dict,
    steps: int = 1000,
    lr: float = 5e-6,
    batch_size: int = 2,
    grad_accum: int = 2,
    save_name: str | None = None,
    wandb_enable: bool = False,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
    use_task_tokens: bool = False,
    bf16: bool = True,
    retrieval_type: str = "sparse",
    zero_ndcg_threshold: float = 0.5,
    reranker: str | None = None,
    clip_eps: float = 0.2,
    logrho_clip: float = 15.0,
    kl_beta: float = 0.0,
    length_norm: bool = False,
    max_cands_per_qid: int | None = None,
    cand_select_mode: str = "rollout",
    surrogate_mode: str = "paper",
    reward_transform: str = "identity",
):
    """
    완전 offline GRPO-style 학습:
      - generate() 안 함
      - 미리 계산된 pseudo + approx_ndcg 기반 advantage로 policy gradient 수행
      - REINFORCE: loss = - E[ advantage * log p(pseudo | prompt) ]
    zero_ndcg_threshold: qid별 rollout 중 ndcg=0.0인 비율이 이 값 이상이면 제외
    
    Note: 데이터는 이미 max_tokens=128, 전체 컨텍스트 ~512 이하로 생성되었음.
    """
    # ==== 1. 모델 / 토크나이저 로드 (기존 코드 재사용, 8bit OFF, bfloat16) ====

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0") or 0)

    # Make process-level CUDA device explicit to avoid accidental allocations on cuda:0.
    if torch.cuda.is_available():
        ndev = max(1, torch.cuda.device_count())
        dev = local_rank % ndev
        torch.cuda.set_device(dev)
        safe_log(
            f"[offline-rl] CUDA binding: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')}, "
            f"LOCAL_RANK={local_rank}, torch.cuda.device_count={ndev}, current_device={torch.cuda.current_device()}"
        )

    # adapter_dir is optional: when omitted, start GRPO from base model with fresh LoRA.
    adapter_dir = (adapter_dir or "").strip()
    use_adapter_init = bool(adapter_dir)
    if use_adapter_init and not os.path.isabs(adapter_dir):
        adapter_dir = os.path.abspath(adapter_dir)
    
    dtype = torch.bfloat16 if bf16 and torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=dtype,
    )

    if use_adapter_init and not os.path.exists(adapter_dir):
        raise FileNotFoundError(f"[offline-rl] SFT adapter not found at {adapter_dir}")

    if use_adapter_init:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                adapter_dir,
                trust_remote_code=True,
                local_files_only=True
            )
            safe_log(f"[offline-rl] Loaded tokenizer from adapter_dir: {adapter_dir}")
        except Exception as e:
            safe_log(f"[offline-rl] Failed to load tokenizer from adapter_dir ({e}), using BASE_MODEL")
            tokenizer = AutoTokenizer.from_pretrained(
                base_model,
                trust_remote_code=True
            )
    else:
        safe_log("[offline-rl] No adapter_dir provided. Initializing fresh LoRA from base model.")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Resize base model to match tokenizer vocab BEFORE loading PEFT adapter
    base.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    if use_task_tokens:
        if "<TASK_PSEUDO>" not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": ["<TASK_PSEUDO>"]})
            base.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    
    if use_adapter_init:
        # SFT adapter를 로드하고 그대로 fine-tune (새 LoRA 추가 안 함)
        try:
            model = PeftModel.from_pretrained(base, adapter_dir, local_files_only=True)
            safe_log(f"[offline-rl] Loaded SFT adapter from: {adapter_dir}")
            safe_log(f"[offline-rl] Will continue training the loaded adapter with RL objective")
            # ---- Reference policy (frozen) ----
            # Reference는 "학습 시작 시점의 SFT 정책"이어야 함.
            base_ref = AutoModelForCausalLM.from_pretrained(
                base_model,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
            base_ref.resize_token_embeddings(len(tokenizer), mean_resizing=False)
            if use_task_tokens and "<TASK_PSEUDO>" in tokenizer.get_vocab():
                base_ref.resize_token_embeddings(len(tokenizer), mean_resizing=False)

            ref_model = PeftModel.from_pretrained(base_ref, adapter_dir, local_files_only=True)
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad = False
        except Exception as e:
            safe_log(f"[offline-rl] Failed to load PEFT adapter: {e}")
            raise RuntimeError(f"Cannot load SFT adapter from {adapter_dir}")
    else:
        # Base model에서 시작: fresh LoRA를 붙여 policy를 만들고, reference는 base 모델로 고정.
        lora_cfg = LoraConfig(
            r=32,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, lora_cfg)
        safe_log("[offline-rl] Initialized fresh LoRA adapter from base model.")
        base_ref = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        base_ref.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        ref_model = base_ref
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
    
    # LoRA 파라미터만 trainable하게 설정
    for n, p in model.named_parameters():
        if any(k in n.lower() for k in ("lora_", "adapter")):
            p.requires_grad = True
        else:
            p.requires_grad = False

    # gradient 설정 확인 (LoRA만 학습)
    model.train()
    total_params = 0
    trainable_params = 0
    lora_params = 0
    for n, p in model.named_parameters():
        total_params += 1
        if p.requires_grad:
            trainable_params += 1
            if any(k in n.lower() for k in ("lora_", "adapter")):
                lora_params += 1
    
    safe_log(f"[offline-rl] Total params: {total_params}, Trainable: {trainable_params}, LoRA: {lora_params}")
    
    if trainable_params == 0:
        raise RuntimeError("[offline-rl] No trainable parameters found! Check adapter loading.")
    
    if lora_params == 0:
        safe_log("[offline-rl] WARNING: No LoRA parameters detected in trainable params!")

    # ==== 2. dataset_weights (inbound datasets 우선) ====
    ds_sizes: dict[str, int] = {}
    for item in qids:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            _, ds = item[0], item[1]
        elif isinstance(item, dict) and "qid" in item and "dataset" in item:
            ds = item["dataset"]
        else:
            continue
        if not ds:
            continue
        ds_sizes[ds] = ds_sizes.get(ds, 0) + 1

    inbound_datasets = {"nfcorpus", "fiqa", "fever", "hotpotqa", "msmarco"}
    dataset_weights: dict[str, float] = {ds: 1.0 if ds in inbound_datasets else 0.5 for ds in ds_sizes}

    # ==== 3. offline NDCG 테이블 로드 ====
    try:
        ds_list = sorted({ds for (_qid, ds) in qids})
    except Exception:
        ds_list = specified_datasets
    offline_ndcg_table = load_offline_ndcg_table(
        pseudo_ref_root=args.base_model_dir,
        pseudo_ndcg_root=args.base_model_dir + "/pseudo_ndcg_real" if args.use_real_ndcg else args.base_model_dir + "/pseudo_ndcg_approx",
        datasets=ds_list,
        retrieval_type=retrieval_type,
        zero_ndcg_threshold=zero_ndcg_threshold,
        reranker=reranker,
        use_real_ndcg=args.use_real_ndcg,
    )
    safe_log(f"[offline-rl] loaded offline_ndcg_table for datasets: {list(offline_ndcg_table.keys())} (retrieval_type={retrieval_type}, zero_ndcg_threshold={zero_ndcg_threshold})")
    # ✅ filtering 이후 유효 qid 기준으로 ds_sizes 재계산
    ds_sizes_filtered = {}
    for (qid, ds) in qids:
        ds = str(ds); qid = str(qid)
        if ds not in offline_ndcg_table: 
            continue
        if qid not in offline_ndcg_table[ds]:
            continue
        if qid not in queries_by_dataset.get(ds, {}):
            continue
        ds_sizes_filtered[ds] = ds_sizes_filtered.get(ds, 0) + 1

    inbound = {"nfcorpus","fiqa","fever","hotpotqa","msmarco"}
    dataset_weights = {ds: (1.0 if ds in inbound else 0.5) for ds in ds_sizes_filtered}
    #dataset_weights["fever"] = 1.3
    #dataset_weights["hotpotqa"] = 1.3
    safe_log(f"[offline-rl] ds_sizes_filtered={ds_sizes_filtered}")
    safe_log(f"[offline-rl] dataset_weights(filtered)={dataset_weights}")
    
    # 필터링된 전체 qid 수 계산
    total_filtered_qids = sum(ds_sizes_filtered.values())
    safe_log(f"[offline-rl] Total filtered qids across all datasets: {total_filtered_qids}")
    safe_log(f"[offline-rl] Candidate truncation: max_cands_per_qid={max_cands_per_qid}, cand_select_mode={cand_select_mode}")

    # ==== 4. offline 샘플 구성 ====
    samples = build_offline_rl_samples(
        qids=qids,
        queries_by_dataset=queries_by_dataset,
        offline_ndcg_table=offline_ndcg_table,
        tokenizer=tokenizer,
        dataset_weights=dataset_weights,
        delta_scale=10.0,
        reward_clip=99999.0,
        adv_clip=5.0,
        max_cands_per_qid=max_cands_per_qid,
        cand_select_mode=cand_select_mode,
        use_task_tokens=use_task_tokens,
        reward_transform=reward_transform,
    )
    if not samples:
        raise RuntimeError("No offline RL samples were built. Check pseudo_ref / ndcg files.")
    
    # 실제 샘플 수 기반으로 1 epoch steps 계산
    num_samples = len(samples)
    
    # world_size 확인 (이미 위에서 정의됨)
    effective_batch_size = batch_size * world_size * grad_accum
    steps_per_epoch = num_samples // effective_batch_size
    
    safe_log(f"[offline-rl] ===== EPOCH/STEPS CALCULATION =====")
    safe_log(f"[offline-rl] Total samples: {num_samples}")
    safe_log(f"[offline-rl] Batch size per device: {batch_size}")
    safe_log(f"[offline-rl] Num devices (world_size): {world_size}")
    safe_log(f"[offline-rl] Gradient accumulation steps: {grad_accum}")
    safe_log(f"[offline-rl] Effective batch size: {effective_batch_size}")
    safe_log(f"[offline-rl] Steps per epoch (calculated): {steps_per_epoch}")
    
    # epochs argument가 주어지면 이를 우선 사용
    if args.epochs is not None:
        original_steps = steps
        steps = args.epochs * steps_per_epoch
        safe_log(f"[offline-rl] EPOCHS MODE: {args.epochs} epochs requested")
        safe_log(f"[offline-rl] AUTO-CALCULATED: steps {original_steps} -> {steps} ({args.epochs} epochs)")
    else:
        # 기존 로직: 사용자가 기본값(1148)을 사용한 경우, 계산된 값으로 자동 조정
        original_steps = steps
        if steps == 1148:  # 기본값
            steps = steps_per_epoch
            safe_log(f"[offline-rl] AUTO-ADJUSTED: steps {original_steps} -> {steps} (1 epoch)")
        else:
            epochs = steps / steps_per_epoch if steps_per_epoch > 0 else 0
            safe_log(f"[offline-rl] Requested steps will run approximately {epochs:.2f} epochs")
    
    safe_log(f"[offline-rl] Final steps: {steps}")
    safe_log(f"[offline-rl] =================================")

    dataset = OfflineRLDataset(samples)
    collate = lambda batch: offline_rl_collate_fn(batch, pad_token_id=tokenizer.pad_token_id)

    # NOTE: keep shuffle=False so that samples built per-qid remain contiguous
    # and we can print per-qid rollout nDCG lines (one line per query/group).
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        drop_last=True,
    )

        # ==== 5. Accelerator + optimizer 설정 ====
    accelerator = Accelerator()
    
    # LoRA 파라미터만 optimizer에 포함
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and any(k in n.lower() for k in ("lora_", "adapter"))]
    if not lora_params:
        raise RuntimeError("[offline-rl] No LoRA parameters found for optimizer!")
    safe_log(f"[offline-rl] Optimizer will update {len(lora_params)} LoRA parameters")
    
    optimizer = optim.AdamW(lora_params, lr=lr)

    model, ref_model, optimizer, dataloader = accelerator.prepare(model, ref_model, optimizer, dataloader)
    lora_params = [p for p in model.parameters() if p.requires_grad]  # Re-fetch after prepare
    # ==== run_dir 생성 (online GRPO와 동일한 grpo_output 구조) ====
    run_dir = None
    if accelerator.is_main_process:
        from datetime import datetime
        out_root = _Path(ROOT_DIR) / "grpo" / "step1" / "grpo_output"
        out_root.mkdir(parents=True, exist_ok=True)

        if save_name is not None:
            # 사용자가 --save-name 을 주면 고정 이름 사용
            run_dir = out_root / save_name
        else:
            # 없으면 timestamp 기반 디렉토리 생성
            ts = datetime.now().strftime("offline_%Y%m%d_%H%M%S")
            run_dir = out_root / ts

        run_dir.mkdir(parents=True, exist_ok=True)
        safe_log(f"[offline-rl] run output directory: {run_dir}")

    # ==== wandb init (main process only) ====
    if wandb_enable and accelerator.is_main_process:
        try:
            import wandb
            # 디폴트 프로젝트 / 런 이름 설정 (online 코드와 맞춰줌)
            if wandb_project is None:
                wandb_project = "raqe-grpo"
            if wandb_run_name is None:
                wandb_run_name = "offline-grpo-step1"

            # env 도 같이 맞춰주기 (online 경로와 동일한 동작)
            os.environ.setdefault("WANDB_PROJECT", str(wandb_project))
            os.environ.setdefault("WANDB_NAME", str(wandb_run_name))

            wandb.init(project=wandb_project, name=wandb_run_name)
            safe_log(f"[offline-rl] wandb initialized (project={wandb_project}, name={wandb_run_name})")
        except Exception as e:
            safe_log(f"[offline-rl] wandb init failed: {e}")
            wandb_enable = False

    # ==== 6. training loop (offline policy gradient) ====

    global_step = 0
    model.train()

    # Create a local progress bar for the custom training loop when running on main process.
    pbar = None
    try:
        if accelerator.is_main_process:
            try:
                from tqdm.auto import tqdm
                pbar = tqdm(total=int(steps), desc="GRPO steps", unit="step")
            except Exception:
                pbar = None
    except Exception:
        pbar = None

    # Initialize dataloader iterator before loop
    safe_log("[offline-rl] Creating dataloader iterator...")
    dataloader_iter = iter(dataloader)
    safe_log("[offline-rl] Starting training loop...")

    for step in range(steps):
        if accelerator.is_main_process and step == 0:
            safe_log("[offline-rl] Starting step 0 (first step)...")
        
        for grad_step in range(grad_accum):
            if accelerator.is_main_process and step == 0 and grad_step == 0:
                safe_log("[offline-rl] Loading first batch...")
            
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)
            
            if accelerator.is_main_process and step == 0 and grad_step == 0:
                safe_log(f"[offline-rl] First batch loaded. Batch keys: {batch.keys() if isinstance(batch, dict) else 'not a dict'}")
                safe_log(f"[offline-rl] Starting forward pass...")

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            advantages = batch["advantages"]

            if accelerator.is_main_process and step == 0 and grad_step == 0:
                safe_log(f"[offline-rl] input_ids.shape={input_ids.shape}, max_len={input_ids.shape[1]}")

        
            # forward (current policy) - ONLY ONCE
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = out.logits
            logp_cur_sum, tok_count = sequence_logprob_from_logits(logits, labels)

            with torch.no_grad():
                ref_out = ref_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                logp_ref_sum, _ = sequence_logprob_from_logits(ref_out.logits, labels)

            # optional length normalization (stabilize ratio)
            if length_norm:
                logp_cur = logp_cur_sum / tok_count
                logp_ref = logp_ref_sum / tok_count
            else:
                logp_cur = logp_cur_sum
                logp_ref = logp_ref_sum

            # log-ratio and ratio
            log_rho = logp_cur - logp_ref
            log_rho = torch.clamp(log_rho, -logrho_clip, logrho_clip)
            rho = torch.exp(log_rho)

            A = advantages.to(rho.device)

            # Clipped surrogate, Eq. (5): min(rho*A, clip(rho)*A), maximized.
            #
            # The pre-release training code special-cased A < 0 with a maximum, which
            # inverts where the trust region bites for negative-advantage samples.
            # The two agree whenever rho stays inside [1-eps, 1+eps]; pass
            # --surrogate released to reproduce a checkpoint trained before the fix.
            unclipped = rho * A
            clipped = torch.clamp(rho, 1.0 - clip_eps, 1.0 + clip_eps) * A
            if surrogate_mode == "released":
                surrogate = torch.where(A >= 0, torch.minimum(unclipped, clipped), torch.maximum(unclipped, clipped))
            else:
                surrogate = torch.minimum(unclipped, clipped)

            # Optional KL penalty (대체로 logrho가 KL proxy라서 간단히 사용 가능)
            # 여기서는 reverse-KL 근사로 log_rho를 사용(정확 KL은 토큰단위 KL 필요)
            if kl_beta > 0.0:
                surrogate = surrogate - kl_beta * log_rho  # penalize drift

            loss = -surrogate.mean()
            loss = loss / grad_accum
            
            if accelerator.is_main_process and step == 0 and grad_step == 0:
                safe_log(f"[offline-rl] Loss computed: {loss.item():.4f}, starting backward pass...")
            
            accelerator.backward(loss)
            
            if accelerator.is_main_process and step == 0 and grad_step == 0:
                safe_log(f"[offline-rl] Backward pass complete (grad_step {grad_step + 1}/{grad_accum})")
        
        if accelerator.is_main_process and step == 0:
            safe_log(f"[offline-rl] All grad accumulation done. Updating optimizer...")
        
        accelerator.clip_grad_norm_(lora_params, 2.0)
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1
        
        if accelerator.is_main_process and step == 0:
            safe_log(f"[offline-rl] Step 0 complete!")
        if accelerator.is_main_process and pbar is not None:
            pbar.update(1)
        # --- checkpoint 저장: 100 step마다 (main process만) ---
        # 제거됨
        
        # --- step별 scalar 통계 계산 ---
            with torch.no_grad():
                # loss는 grad_accum 나누기 전 값으로 보고 싶으면 * grad_accum 해도 됨
                step_loss = float(loss.item())
                avg_adv = float(advantages.mean().item())
                std_adv = float(advantages.std().item()) if advantages.numel() > 1 else 0.0
                avg_logp = float(logp_cur.mean().item())      # length_norm 여부에 따라 정의된 logp_cur
                avg_logrho = float(log_rho.mean().item())
                avg_rho = float(rho.mean().item())

                # ndcg 통계 (approx vs base)
                approx_b = batch.get("approx_ndcg")
                base_b = batch.get("base_ndcg")
                if approx_b is not None and base_b is not None:
                    approx_mean = float(approx_b.mean().item())
                    base_mean = float(base_b.mean().item())
                    delta_mean = approx_mean - base_mean
                else:
                    approx_mean = base_mean = delta_mean = 0.0

            # 콘솔 로그: 매 step마다 한 줄
            safe_log(
                f"[offline-rl] step={global_step} "
                f"loss={step_loss:.4f} "
                f"adv_mean={avg_adv:.4f} adv_std={std_adv:.4f} "
                f"logp_mean={avg_logp:.4f} "
                f"ndcg_base={base_mean:.4f} ndcg_approx={approx_mean:.4f} "
                f"ndcg_delta={delta_mean:.4f}"
            )

            # wandb logging: 매 step마다 기록
            if wandb_enable:
                try:
                    import wandb
                    log_dict = {
                        "train/offline_loss": step_loss,
                        "train/adv_mean": avg_adv,
                        "train/adv_std": std_adv,
                        "train/logp_mean": avg_logp,
                        "train/ndcg_base_mean": base_mean,
                        "train/ndcg_approx_mean": approx_mean,
                        "train/ndcg_delta_mean": delta_mean,
                        "train/lr": float(optimizer.param_groups[0]["lr"]),
                        "train/step": int(global_step),
                    }
                    wandb.log(log_dict, step=int(global_step))
                except Exception as e:
                    safe_log(f"[offline-rl] wandb.log failed: {e}")

            # --- Per-qid NDGC print (batch-local grouping): 매 step마다 실행 ---
            try:
                qids_b = batch.get("qids") if isinstance(batch, dict) else None
                ds_b = batch.get("datasets") if isinstance(batch, dict) else None
                approx_b = batch.get("approx_ndcg") if isinstance(batch, dict) else None
                base_b = batch.get("base_ndcg") if isinstance(batch, dict) else None

                if qids_b and ds_b and approx_b is not None and base_b is not None:
                    try:
                        approx_list = approx_b.detach().cpu().tolist() if hasattr(approx_b, "detach") else list(approx_b)
                    except Exception:
                        approx_list = list(approx_b)
                    try:
                        base_list = base_b.detach().cpu().tolist() if hasattr(base_b, "detach") else list(base_b)
                    except Exception:
                        base_list = list(base_b)

                    # contiguous group by qid (shuffle=False 가정)
                    groups = []
                    cur_q = None
                    cur_ds = None
                    cur_base = None
                    cur_vals = []
                    for qi, dsi, av, bv in zip(qids_b, ds_b, approx_list, base_list):
                        if qi != cur_q:
                            if cur_q is not None:
                                groups.append((cur_q, cur_ds, cur_base, cur_vals))
                            cur_q = str(qi)
                            cur_ds = str(dsi)
                            cur_base = float(bv)
                            cur_vals = [float(av)]
                        else:
                            cur_vals.append(float(av))
                    if cur_q is not None:
                        groups.append((cur_q, cur_ds, cur_base, cur_vals))

                    local_rank = int(os.environ.get("LOCAL_RANK", "0") or 0)

                    for (qid_g, ds_g, base_g, vals_g) in groups:
                        try:
                            vals_str = ",".join(f"{float(v):.6f}" for v in vals_g)
                            print(
                                f"[ndcg] rank={local_rank} step={global_step} "
                                f"qid={qid_g} ds={ds_g} base={base_g:.6f} "
                                f"rollouts_in_batch={len(vals_g)} (※배치 내 일부만, advantage는 전체 기준) "
                                f"approx=[{vals_str}]",
                                flush=True,
                            )
                        except Exception:
                            continue

                    # (선택) wandb에 배치 평균 ndcg도 기록
                    if wandb_enable:
                        try:
                            import numpy as _np, wandb
                            ndcg_mean = float(_np.mean(approx_list))
                            wandb.log({"train/ndcg_mean_batch": ndcg_mean}, step=int(global_step))
                        except Exception:
                            pass

            except Exception:
                pass

        if global_step >= steps:
            break
    if accelerator.is_main_process and pbar is not None:
        pbar.close()    

    # ==== 7. 저장 ====
    if accelerator.is_main_process and run_dir is not None:
        safe_log(f"[offline-rl] Saving final adapter to {run_dir}")

        unwrapped = accelerator.unwrap_model(model)
        # SFT+GRPO 하나의 adapter로 저장 (LoRA 파라미터만 업데이트됨)
        unwrapped.save_pretrained(str(run_dir))
        tokenizer.save_pretrained(str(run_dir))
        safe_log(f"[offline-rl] Saved SFT+GRPO unified adapter")

        meta = {
            "adapter_dir": adapter_dir if use_adapter_init else None,
            "init_from_base": (not use_adapter_init),
            "base_model": base_model,
            "steps": steps,
            "lr": lr,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "mode": "sft+grpo_unified" if use_adapter_init else "base+grpo_unified",
            "description": "SFT adapter continued with RL objective - single unified adapter" if use_adapter_init else "Fresh LoRA initialized from base model and trained with RL objective",
        }
        with open(run_dir / "offline_grpo_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    accelerator.wait_for_everyone()
    safe_log("[offline-rl] Training complete.")
    safe_log(f"[offline-rl] built offline samples: {len(samples)}")
    return samples

def evaluate_retrieval_train(dataset_name: str, queries_map: dict, method_name: str = "train_eval") -> float:
    """Train qrels 기반 nDCG@10 계산 (evaluator.py/evaluate_sft.py 패턴 적용).
    
    Follows the successful pattern from evaluate_sft.py and evaluator.py:
    - Try pyserini get_qrels() first, fallback to local qrels files
    - Use prebuilt pyserini indices with proper caching
    - Query normalization (handle dict/list query formats)
    - Return 0.0 on errors instead of sys.exit for training stability
    """

    if not hasattr(evaluate_retrieval_train, "_cache"):
        evaluate_retrieval_train._cache = {"searcher": {}, "qrels": {}}

    cache = evaluate_retrieval_train._cache

    try:
        # Prefer the local implementation that explicitly uses the TRAIN qrels
        # (train.tsv) because training qid lists come from the train split.
        # Fall back to the more featureful `evaluate_retrieval` only if the
        # local path cannot produce results.

        # Local implementation (copied/adapted): load qrels and searcher once.
        base = os.path.join(os.getcwd(), "datasets", "IR")

        # Load and cache qrels for this dataset (evaluator.py pattern: pyserini first, then local)
        # Use train + dev splits only (avoid using test qrels for training/eval).
        if dataset_name not in cache["qrels"]:
            qrels = {}
            
            # Try pyserini get_qrels first (evaluator.py pattern)
            try:
                from pyserini.search import get_qrels
                if dataset_name in ["dl19", "dl20", "msmarco"]:
                    qrels_key = f"{dataset_name}-passage-train"
                else:
                    qrels_key = f"beir-v1.0.0-{dataset_name}-train"
                
                try:
                    raw_qrels = get_qrels(qrels_key)
                    qrels = {
                        str(k): {str(dk): int(dv) for dk, dv in v.items()}
                        for k, v in raw_qrels.items()
                    }
                except Exception as e:
                    qrels = {}
            except Exception:
                qrels = {}
            
            # Fallback to local qrels files if pyserini failed (evaluator.py pattern)
            if not qrels:
                qrels_base_path = _Path(base) / dataset_name / dataset_name / "qrels"
                
                # Try to load train and dev splits
                qrels_splits_to_load = ["train.tsv", "dev.tsv"]
                for split_file in qrels_splits_to_load:
                    qrels_path = qrels_base_path / split_file
                    if qrels_path.exists():
                        try:
                            with qrels_path.open("r", encoding="utf-8") as qf:
                                first = True
                                for line in qf:
                                    if first:
                                        first = False
                                        if line.lower().startswith("query-id"):
                                            continue
                                    if not line or not line.strip():
                                        continue
                                    parts = line.strip().split()
                                    if len(parts) < 2:
                                        continue
                                    qid = str(parts[0])
                                    docid = str(parts[1])
                                    try:
                                        rel = int(float(parts[-1]))
                                    except Exception:
                                        rel = 1
                                    if qid not in qrels:
                                        qrels[qid] = {}
                                    qrels[qid][docid] = int(rel)
                        except Exception as e:
                            continue
            
            if not qrels:
                print(f"[eval][ERROR] No qrels loaded for dataset '{dataset_name}'. Cannot evaluate.")
                return 0.0
            
            cache["qrels"][dataset_name] = qrels

        qrels = cache["qrels"].get(dataset_name, {})

        # --- [수정 시작] evaluate_sft.py/evaluator.py에서 수정한 Pyserini 전용 로직 적용 ---
        beir_datasets = {
            "fever": "beir-v1.0.0-fever.flat",
            "fiqa": "beir-v1.0.0-fiqa.flat",
            "hotpotqa": "beir-v1.0.0-hotpotqa.flat",
            "nfcorpus": "beir-v1.0.0-nfcorpus.flat",
            "scifact": "beir-v1.0.0-scifact.flat",
            "arguana": "beir-v1.0.0-arguana.flat",
            # add more BEIR mappings here as needed
        }

        index_key = None
        if dataset_name == "msmarco":
            index_key = "msmarco-v1-passage"
        elif dataset_name in beir_datasets:
            index_key = beir_datasets[dataset_name]

        searcher = None
        if not index_key:
            print(f"[eval][ERROR] No prebuilt index mapping for dataset '{dataset_name}'.")
            return 0.0

        # Try cached searcher first (evaluator.py pattern)
        searcher = cache["searcher"].get(index_key)
        if searcher is None:
            try:
                if LuceneSearcher is None:
                    raise RuntimeError(f"Pyserini/Lucene unavailable: {_PYSERINI_IMPORT_ERROR}")
                searcher = LuceneSearcher.from_prebuilt_index(index_key)
                cache["searcher"][index_key] = searcher
            except Exception as e:
                import traceback
                # ALWAYS print critical errors (ignore verbose flag for errors)
                try:
                    rank_info = f"[rank={os.environ.get('LOCAL_RANK', '?')}]"
                except Exception:
                    rank_info = ""
                print(f"[eval][CRITICAL]{rank_info} Failed to load index '{index_key}': {e}")
                print(traceback.format_exc())
                print(f"{rank_info} This is likely the cause of nDCG 0.0 scores. Pyserini/Java 환경을 확인하세요.")
                print(f"{rank_info} Multi-process 환경에서 Pyserini Java VM 충돌 가능성 높음!")
                return 0.0
        else:
            pass  # cached searcher found

        # Build run results using the cached searcher (evaluator.py pattern with query normalization)
        run = {}
        if searcher is not None:
            for qid, query_text in queries_map.items():
                if not query_text:
                    run[qid] = {}
                    continue
                try:
                    # Normalize query text (evaluator.py pattern: handle dict/list)
                    q = query_text
                    if isinstance(q, dict):
                        q = q.get('text') or ''
                    if isinstance(q, list):
                        q = q[0]
                    
                    hits = searcher.search(str(q), k=10)
                    run[qid] = {hit.docid: float(hit.score) for hit in hits}
                    
                except Exception as e:
                    run[qid] = {}
        # If we somehow reached here without a searcher, return 0.0 (evaluator.py pattern)
        if searcher is None:
            print(f"[eval][CRITICAL] No searcher available for dataset '{dataset_name}'. Cannot perform retrieval.")
            return 0.0

        # Filter run to qrels keys and report stats when verbose
        pre_count = len(run)
        run = {q: d for q, d in run.items() if q in qrels}
        post_count = len(run)
        
        if not run:
            # qrels에 매칭되는 qid가 하나도 없는 경우 0.0 반환
            return 0.0

        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10"})
        eval_res = evaluator.evaluate(run)
        
        # qid별 점수 반환이 아닌, 평균 nDCG 계산
        ndcg_scores = [res.get("ndcg_cut_10", 0) for res in eval_res.values()]
        import numpy as _np
        mean_ndcg = float(_np.mean(ndcg_scores))
        return mean_ndcg
    # --- [수정 끝] ---

    except Exception as e:
        return 0.0
from sft.train.prompts import build_user_only_messages

from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset
from datasets import Dataset as HFDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainerCallback
from peft import PeftModel, LoraConfig, get_peft_model
from torch.optim import AdamW
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm


# Safe logging helper that keeps tqdm progress bar at the bottom by using
# `tqdm.write` for any log lines produced while the progress bar is active.
def safe_log(*args, **kwargs):
    try:
        # Build a single string similar to print()
        msg = ' '.join(str(a) for a in args)
        tqdm.write(msg)
    except Exception:
        try:
            print(*args, **kwargs)
        except Exception:
            pass

# Adapter-only dir (your SFT output)
# Optional runtime-loaded pseudos map for offline training. Format: {qid: [pseudo1, pseudo2, ...]}
# This can be set by an external script before calling run_grpo_with_trl, e.g.:
# import test_grpo; test_grpo._LOADED_PSEUDOS = loaded_map
_LOADED_PSEUDOS = None

def raqe_passage_prompt(query: str) -> str:
    return (
        "Write a single cohesive paragraph (at least three sentences) that directly addresses the query. "
        "Include concrete details when possible, avoid meta commentary, and return only the passage text.\n"
        f"Query: {query}"
    )

def load_queries(dataset_root: str, dataset: str, max_queries: int | None = None) -> dict[str, str]:
    """Load queries for a dataset from disk.

    Expected layout: <dataset_root>/<dataset>/<dataset>/queries.jsonl
    Returns a mapping qid -> text. Honors max_queries if provided (keeps file order).
    """
    # Ensure absolute path
    if not os.path.isabs(dataset_root):
        dataset_root = os.path.abspath(dataset_root)
    queries_file = os.path.join(dataset_root, dataset, dataset, "queries.jsonl")
    print(f"[debug] load_queries: looking for {queries_file}")
    if not os.path.exists(queries_file):
        # Try one alternative layout: <dataset_root>/<dataset>/queries.jsonl
        alt = os.path.join(dataset_root, dataset, "queries.jsonl")
        print(f"[debug] load_queries: primary path not found, trying {alt}")
        if os.path.exists(alt):
            queries_file = alt
        else:
            # Nothing found
            print(f"[debug] load_queries: no queries file found for dataset '{dataset}'")
            return {}

    print(f"[debug] load_queries: reading from {queries_file}")
    out: dict[str, str] = {}
    try:
        with open(queries_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except Exception as e:
                    print(f"[debug] load_queries: malformed JSON line {i+1}: {e}")
                    continue
                qid = data.get("_id") or data.get("id")
                if qid is None:
                    print(f"[debug] load_queries: no qid in line {i+1}")
                    continue
                text = data.get("text", "")
                if not text:
                    print(f"[debug] load_queries: empty text in line {i+1}")
                    continue
                out[str(qid)] = text
                if max_queries is not None and len(out) >= int(max_queries):
                    break
    except Exception as e:
        print(f"[debug] load_queries: error reading file: {e}")
        return {}

    print(f"[debug] load_queries: loaded {len(out)} queries")
    return out
# ===== Offline approx NDCG 테이블 로더 =====

_OFFLINE_NDCG_TABLE = None  # (ds -> qid -> {"base_ndcg": float, "cands": [ {rollout, pseudo, approx_ndcg} ]})

def load_offline_ndcg_table(
    pseudo_ref_root: str = "artifacts/train/Meta-Llama-3.1-70B-Instruct",
    pseudo_ndcg_root: str = "artifacts/train/Meta-Llama-3.1-70B-Instruct/pseudo_ndcg_approx",
    datasets: list[str] | None = None,
    retrieval_type: str = "sparse",
    zero_ndcg_threshold: float = 0.7,
    reranker: str | None = None,
    use_real_ndcg: bool = False,
):
    """
    pseudo_refs_train + pseudo_ndcg (approx or real) 를 조인해서
    (dataset, qid, rollout) 별 ndcg 를 pseudo text 와 함께 메모리에 올린다.
    retrieval_type: 'sparse', 'dense', 'mixed' 중 하나 선택
    - mixed: 기존 _sparse.jsonl과 _dense.jsonl을 읽어서 dataset별로 qid 반반 섞음
    zero_ndcg_threshold: qid별 rollout 중 ndcg=0.0인 비율이 이 값 이상이면 해당 qid 제외
    reranker: reranker 모델 이름 (예: 'Alibaba-NLP__gte-reranker-modernbert-base'). 지정 시 해당 폴더 내 jsonl 사용
    use_real_ndcg: True이면 pseudo_ndcg_real 폴더의 real_ndcg 값 사용, False이면 pseudo_ndcg_approx 폴더의 approx_ndcg 값 사용
    """
    global _OFFLINE_NDCG_TABLE
    if _OFFLINE_NDCG_TABLE is not None:
        return _OFFLINE_NDCG_TABLE

    pseudo_ref_root = Path(pseudo_ref_root)
    pseudo_ndcg_root = Path(pseudo_ndcg_root)

    # cqadupstack의 경우 다른 모델 사용
    if "cqadupstack" in datasets:
        pseudo_ref_root_cqadupstack = Path("evaluation/results/train/Llama-3.1-70B-Instruct")
        ndcg_dir = "pseudo_ndcg_real" if use_real_ndcg else "pseudo_ndcg_approx"
        pseudo_ndcg_root_cqadupstack = Path(f"evaluation/results/train/Llama-3.1-70B-Instruct/{ndcg_dir}")
        if reranker:
            pseudo_ndcg_root_cqadupstack = pseudo_ndcg_root_cqadupstack / reranker

    table: dict[str, dict[str, dict]] = {}

    if datasets is None:
        # 디렉토리 안의 *_pseudo_refs_train.jsonl 기준으로 dataset 자동 추출
        for p in pseudo_ref_root.glob("*_pseudo_refs_train.jsonl"):
            name = p.name
            # 예: nfcorpus_pseudo_refs_train.jsonl
            ds = name.split("_pseudo_refs_train.jsonl")[0]
            if ds:
                table.setdefault(ds, {})
        datasets = sorted(table.keys())

    for ds in datasets:
        if ds == "cqadupstack":
            current_pseudo_ref_root = pseudo_ref_root_cqadupstack
            current_pseudo_ndcg_root = pseudo_ndcg_root_cqadupstack
        else:
            current_pseudo_ref_root = pseudo_ref_root
            current_pseudo_ndcg_root = pseudo_ndcg_root
        
        if reranker:
            current_pseudo_ndcg_root = current_pseudo_ndcg_root / reranker
        
        ref_path = current_pseudo_ref_root / f"{ds}" / f"{ds}_pseudo_train.jsonl"
        
        # Mixed 모드: sparse와 dense 파일 둘 다 읽어서 반반 섞기
        if retrieval_type == "mixed":
            ndcg_file_type = "pseudo_ndcg_real" if use_real_ndcg else "pseudo_ndcg_approx"
            ndcg_field = "real_ndcg" if use_real_ndcg else "approx_ndcg"
            # real_ndcg는 sparse/dense suffix 없음
            if use_real_ndcg:
                sparse_path = current_pseudo_ndcg_root / f"{ds}_{ndcg_file_type}.jsonl"
                dense_path = current_pseudo_ndcg_root / f"{ds}_{ndcg_file_type}.jsonl"
            else:
                sparse_path = current_pseudo_ndcg_root / f"{ds}_{ndcg_file_type}_sparse.jsonl"
                dense_path = current_pseudo_ndcg_root / f"{ds}_{ndcg_file_type}_dense.jsonl"
            
            if not ref_path.exists():
                print(f"[offline-ndcg] skip dataset={ds}: missing ref file ({ref_path})")
                continue
            if not sparse_path.exists() or not dense_path.exists():
                print(f"[offline-ndcg] skip dataset={ds} (mixed): missing sparse or dense files")
                continue
            
            # 1) pseudo text 로드
            by_qid_rollout: dict[tuple[str, int], str] = {}
            with open(ref_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    qid = str(obj.get("qid"))
                    rollout = int(obj.get("rollout", 0))
                    pseudo = str(obj.get("pseudo", "")).strip()
                    by_qid_rollout[(qid, rollout)] = pseudo
            
            # 2) sparse NDCG 로드
            sparse_data: dict[str, dict] = {}
            with open(sparse_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    qid = str(obj.get("qid"))
                    rollout = int(obj.get("rollout", 0))
                    ndcg_value = float(obj.get(ndcg_field, 0.0))
                    base_ndcg = float(obj.get("base_ndcg", 0.0))
                    
                    pseudo = by_qid_rollout.get((qid, rollout))
                    if pseudo is None:
                        continue
                    
                    entry = sparse_data.setdefault(qid, {"base_ndcg": base_ndcg, "cands": []})
                    entry["cands"].append({
                        "rollout": rollout,
                        "pseudo": pseudo,
                        "approx_ndcg": ndcg_value,  # 내부적으로는 approx_ndcg key 사용
                    })
            
            # 3) dense NDCG 로드
            dense_data: dict[str, dict] = {}
            with open(dense_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    qid = str(obj.get("qid"))
                    rollout = int(obj.get("rollout", 0))
                    ndcg_value = float(obj.get(ndcg_field, 0.0))
                    base_ndcg = float(obj.get("base_ndcg", 0.0))
                    
                    pseudo = by_qid_rollout.get((qid, rollout))
                    if pseudo is None:
                        continue
                    
                    entry = dense_data.setdefault(qid, {"base_ndcg": base_ndcg, "cands": []})
                    entry["cands"].append({
                        "rollout": rollout,
                        "pseudo": pseudo,
                        "approx_ndcg": ndcg_value,  # 내부적으로는 approx_ndcg key 사용
                    })
            
            # 4) 각 데이터에서 0.0 비율 계산 및 필터링
            def filter_by_zero_ratio(data: dict, threshold: float) -> dict:
                """
                각 qid별로 rollout 중 ndcg=0.0인 비율을 계산하고,
                threshold 이상이면 제외한 필터링된 dict 반환
                """
                filtered = {}
                for qid, entry in data.items():
                    cands = entry.get("cands", [])
                    if not cands:
                        continue
                    zero_count = sum(1 for c in cands if c.get("approx_ndcg", 0.0) == 0.0)
                    zero_ratio = zero_count / len(cands)
                    if zero_ratio < threshold:
                        filtered[qid] = entry
                return filtered
            
            sparse_filtered = filter_by_zero_ratio(sparse_data, zero_ndcg_threshold)
            dense_filtered = filter_by_zero_ratio(dense_data, zero_ndcg_threshold)
            
            # 5) mixed 로직: sparse와 dense 중 하나만 통과해도 사용
            # - 둘 다 통과: 공통 qid로 취급하고 반반 나눔
            # - 하나만 통과: 통과한 것 사용
            # - 둘 다 실패: 제외
            sparse_only_qids = set(sparse_filtered.keys()) - set(dense_filtered.keys())
            dense_only_qids = set(dense_filtered.keys()) - set(sparse_filtered.keys())
            common_qids = sorted(set(sparse_filtered.keys()) & set(dense_filtered.keys()))
            
            # 공통 qid를 반반 나눔
            mid_idx = len(common_qids) // 2
            sparse_from_common = set(common_qids[:mid_idx])
            dense_from_common = set(common_qids[mid_idx:])
            
            # 최종 sparse, dense qid 집합
            final_sparse_qids = sparse_only_qids | sparse_from_common
            final_dense_qids = dense_only_qids | dense_from_common
            
            # 6) 반반 섞어서 최종 테이블 구성
            by_qid: dict[str, dict] = {}
            for qid in final_sparse_qids:
                by_qid[qid] = sparse_filtered[qid]
            for qid in final_dense_qids:
                by_qid[qid] = dense_filtered[qid]
            
            if not by_qid:
                print(f"[offline-ndcg] skip dataset={ds} (mixed): all qids filtered out by zero_ndcg_threshold={zero_ndcg_threshold}")
                continue
            
            table[ds] = by_qid
            print(f"[offline-ndcg] loaded dataset={ds} (mixed): {len(final_sparse_qids)} sparse + {len(final_dense_qids)} dense = {len(by_qid)} total qids")
            print(f"  (filtered: sparse {len(sparse_data)} -> {len(sparse_filtered)}, dense {len(dense_data)} -> {len(dense_filtered)})")
        
        else:
            # sparse 또는 dense 단일 모드
            ndcg_file_type = "pseudo_ndcg_real" if use_real_ndcg else "pseudo_ndcg_approx"
            ndcg_field = "real_ndcg" if use_real_ndcg else "approx_ndcg"
            # real_ndcg는 sparse/dense suffix 없음
            if use_real_ndcg:
                ndcg_path = current_pseudo_ndcg_root / f"{ds}_{ndcg_file_type}.jsonl"
            else:
                ndcg_suffix = f"_{retrieval_type}" if retrieval_type in ["sparse", "dense"] else ""
                ndcg_path = current_pseudo_ndcg_root / f"{ds}_{ndcg_file_type}{ndcg_suffix}.jsonl"
            
            if not ref_path.exists() or not ndcg_path.exists():
                print(f"[offline-ndcg] skip dataset={ds}: missing files ({ref_path}, {ndcg_path})")
                continue

            # 1) (qid, rollout) -> pseudo text
            by_qid_rollout: dict[tuple[str, int], str] = {}
            with open(ref_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    qid = str(obj.get("qid"))
                    rollout = int(obj.get("rollout", 0))
                    pseudo = str(obj.get("pseudo", "")).strip()
                    by_qid_rollout[(qid, rollout)] = pseudo

            # 2) approx ndcg 파일 읽어서 조인
            by_qid: dict[str, dict] = {}
            with open(ndcg_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    qid = str(obj.get("qid"))
                    rollout = int(obj.get("rollout", 0))
                    ndcg_value = float(obj.get(ndcg_field, 0.0))
                    base_ndcg = float(obj.get("base_ndcg", 0.0))

                    pseudo = by_qid_rollout.get((qid, rollout))
                    if pseudo is None:
                        # 혹시 조인 실패하면 스킵
                        continue

                    entry = by_qid.setdefault(qid, {"base_ndcg": base_ndcg, "cands": []})
                    entry["cands"].append(
                        {
                            "rollout": rollout,
                            "pseudo": pseudo,
                            "approx_ndcg": ndcg_value,  # 내부적으로는 approx_ndcg key 사용
                        }
                    )

            # 3) 0.0 비율 필터링
            before_filter = len(by_qid)
            filtered_by_qid: dict[str, dict] = {}
            for qid, entry in by_qid.items():
                cands = entry.get("cands", [])
                if not cands:
                    continue
                zero_count = sum(1 for c in cands if c.get("approx_ndcg", 0.0) == 0.0)
                zero_ratio = zero_count / len(cands)
                if zero_ratio < zero_ndcg_threshold:
                    filtered_by_qid[qid] = entry
            
            if not filtered_by_qid:
                print(f"[offline-ndcg] skip dataset={ds} ({retrieval_type}): all qids filtered out by zero_ndcg_threshold={zero_ndcg_threshold}")
                continue

            table[ds] = filtered_by_qid
            print(f"[offline-ndcg] loaded dataset={ds} ({retrieval_type}): qids={len(filtered_by_qid)} (filtered {before_filter} -> {len(filtered_by_qid)}) from {ref_path.name}, {ndcg_path.name}")

    _OFFLINE_NDCG_TABLE = table
    
    # 전체 필터링된 qid 수와 평균 rollout 수 계산
    total_qids = sum(len(qid_dict) for qid_dict in table.values())
    total_rollouts = 0
    total_cands = 0
    for ds, qid_dict in table.items():
        for qid, entry in qid_dict.items():
            cands = entry.get("cands", [])
            total_cands += len(cands)
            total_rollouts += len(cands)
    
    avg_rollouts = total_rollouts / total_qids if total_qids > 0 else 0
    
    safe_log(f"[offline-ndcg] SUMMARY: retrieval_type={retrieval_type}, zero_threshold={zero_ndcg_threshold}")
    safe_log(f"[offline-ndcg] Total filtered qids: {total_qids}, Total candidates: {total_cands}, Avg rollouts/qid: {avg_rollouts:.1f}")
    
    return _OFFLINE_NDCG_TABLE


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single-qid pseudo-ref test or print model input (--test).")
    parser.add_argument("--test", action="store_true", help="Do not load model; print model input for the first available qid.")
    parser.add_argument("--qids_path", default="artifacts/sft/pr_final", help="Directory (or file) with per_qid_ndcg_{dataset}_filtered_qids_pr.json")
    parser.add_argument("--dataset-root", default="datasets/IR", help="Dataset root path")
    parser.add_argument("--dataset", default="nfcorpus,fever,fiqa,hotpotqa,msmarco", help="Dataset name")
    parser.add_argument("--grpo", action="store_true", help="Run GRPO training using TRL GRPOTrainer")
    parser.add_argument("--rollouts", type=int, default=8, help="Group size per prompt. If using manual grouping (group_id/slot), num_generations=num_return_sequences=1")
    parser.add_argument(
        "--rollout_subset_mode",
        type=str,
        default="rollout",
        choices=["rollout", "approx"],
        help="When --rollouts is smaller than available candidates, keep first-k by rollout index or top-k by approx_ndcg.",
    )
    # Defaults tailored for RL (GRPO): 1 epoch = 1148 steps
    # (3444 queries × 8 rollouts / (batch_size=8 × num_gpus=3 × grad_accum=1) = 27552 / 24 = 1148)
    # Use lower LR (5e-6) by default for stability; 1e-6 for extra caution.
    parser.add_argument("--grpo-steps", type=int, default=1148, help="Number of GRPO training steps (default: 1148 = 1 epoch)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs to train (alternative to --grpo-steps). If specified, --grpo-steps is ignored and calculated as epochs * steps_per_epoch")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate for GRPO training (recommend 5e-6 or 1e-6 for stability)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for TRL training")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps (effective batch multiplication)")
    parser.add_argument("--shuffle-seed", type=int, default=None, help="Optional seed to make qid shuffling reproducible when combining datasets")
    parser.add_argument("--save-name", type=str, default=None, help="Name for output dir under grpo/step1/outputs to save the trained model/tokenizer")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging (main process only). Off by default; needs a configured W&B account.")
    parser.add_argument("--wandb-project", type=str, default=None, help="WandB project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="WandB run name")
    parser.add_argument("--pseudo_print", action="store_true", help="Print all generated pseudo refs per qid/rollout for debugging diversity (includes process rank)")
    parser.add_argument("--task_tokens", action="store_true", help="Prefix prompts with <TASK_PSEUDO> special token")
    parser.add_argument("--no-bf16", action="store_true", help="Disable BF16 and use FP32 instead")
    parser.add_argument("--retrieval_type", type=str, default="sparse", choices=["sparse", "dense", "mixed"], help="Retrieval type for loading pseudo_ndcg files (default: sparse)")
    parser.add_argument("--use_real_ndcg", action="store_true", help="Use real NDCG (pseudo_ndcg_real) instead of approx NDCG (pseudo_ndcg_approx)")
    parser.add_argument("--zero_ndcg_threshold", type=float, default=0.7, help="Threshold for filtering qids based on zero NDCG ratio (default: 0.5)")
    parser.add_argument(
        "--reranker",
        type=str,
        default=None,
        choices=[
            "BAAI__bge-reranker-v2-m3",
            "Alibaba-NLP__gte-reranker-modernbert-base",
            "mixedbread-ai__mxbai-rerank-large-v2",
        ],
        help=(
            "Reranker sub-directory of the RSDG cache. Leave unset for the paper default "
            "(BAAI/bge-reranker-v2-m3 caches stored directly under pseudo_ndcg_approx/); "
            "set it to reproduce the reranker-robustness ablation in Sec. 4.2."
        ),
    )
    parser.add_argument("--config-file", type=str, default=None, help="Optional Accelerate config path; leave unset to use the local Accelerate default.")
    
    parser.add_argument("--clip-eps", type=float, default=0.2, help="GRPO/PPO clip epsilon")
    parser.add_argument(
        "--reward-transform",
        type=str,
        default="identity",
        choices=["identity", "tanh"],
        help=(
            "How the RSDG gain becomes the scalar reward before group standardisation. "
            "'identity' is Eq. (3), r_i = Delta(q'_i, q). 'tanh' reproduces the "
            "pre-release squashing, tanh(10 * Delta)."
        ),
    )
    parser.add_argument(
        "--surrogate",
        type=str,
        default="paper",
        choices=["paper", "released"],
        help=(
            "Clipped-surrogate variant. 'paper' is Eq. (5), min(rho*A, clip(rho)*A). "
            "'released' reproduces the pre-release behaviour, which took a maximum for "
            "negative-advantage samples. They differ only once a ratio leaves "
            "[1-eps, 1+eps]."
        ),
    )
    parser.add_argument("--logrho-clip", type=float, default=20.0, help="Clamp log-ratio for numeric stability")
    parser.add_argument("--kl-beta", type=float, default=0.0, help="Optional KL penalty coefficient (0 disables)")
    parser.add_argument("--length-norm", default=True, action="store_true", help="Use length-normalized logprob for ratio")
    parser.add_argument("--base-model", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Base model name for GRPO training")
    parser.add_argument("--base-model-dir", type=str, default="artifacts/train/Meta-Llama-3.1-70B-Instruct", help="Base model directory for pseudo refs and ndcg files")
    parser.add_argument("--adapter-dir", type=str, default="", help="Optional SFT adapter directory. Leave empty to start GRPO from base model with fresh LoRA.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed for Python/NumPy/PyTorch.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # If --grpo is passed, run the TRL training entrypoint directly. Do NOT first run
    # the single-qid smoke test since that will cause every spawned process to attempt
    # to load the model (and gated HF models will fail on processes without auth).
    if getattr(args, "grpo", False):
        # Load qids and queries. Support two modes:
        # - args.qids_path is a directory containing files named
        #   per_qid_ndcg_{dataset}_filtered_qids.json -> load all datasets found
        # - args.qids_path is a single file -> behave as before using args.dataset
        import glob
        import re

        qids_list = []
        queries_by_dataset = {}
        # If user passed a single per_qid file path but intended the directory,
        # treat the parent directory as the qids directory when multiple
        # `per_qid_ndcg_*_filtered_qids_pr.json` files are present there.
        try:
            if os.path.isfile(args.qids_path):
                parent = os.path.dirname(args.qids_path)
                import glob as _glob
                cand = _glob.glob(os.path.join(parent, "per_qid_ndcg_*_filtered_qids_pr.json"))
                if cand:
                    args.qids_path = parent
        except Exception:
            pass

        # Parse specified datasets
        specified_datasets = [d.strip() for d in args.dataset.split(",") if d.strip()]

        if os.path.isdir(args.qids_path):
            pattern = re.compile(r"per_qid_ndcg_(?P<dataset>.+?)_filtered_qids_pr\.json$")
            paths = sorted(glob.glob(os.path.join(args.qids_path, "per_qid_ndcg_*_filtered_qids_pr.json")))
            # Filter paths to only specified datasets
            paths = [p for p in paths if any(ds in os.path.basename(p) for ds in specified_datasets)]
            # Load qids per-dataset first, then interleave across datasets so that
            # the training sequence alternates datasets instead of processing all
            # qids from one dataset first. If a shuffle seed is provided, shuffle
            # each dataset's qid list deterministically before interleaving.
            per_dataset_qids: dict[str, list] = {}
            ds_order: list[str] = []
            
            # cqadupstack의 경우 별도 처리: qid filtering 없이 pseudo_train.jsonl에서 직접 로드
            if "cqadupstack" in specified_datasets:
                ds = "cqadupstack"
                ds_order.append(ds)
                pseudo_train_file = f"{args.base_model_dir}/cqadupstack/cqadupstack_pseudo_train.jsonl"
                print(f"[debug] Loading cqadupstack directly from {pseudo_train_file}")
                
                qid_to_query = {}
                qid_to_type = {}
                if os.path.exists(pseudo_train_file):
                    with open(pseudo_train_file, "r", encoding="utf-8") as f:
                        for line in f:
                            obj = json.loads(line)
                            qid = str(obj["qid"])
                            query = obj.get("query", "")
                            qtype = obj.get("type", "")
                            qid_to_query[qid] = query
                            qid_to_type[qid] = qtype
                    print(f"[debug] Loaded {len(qid_to_query)} queries from pseudo_train.jsonl for cqadupstack")
                    queries_by_dataset[ds] = qid_to_query
                    per_dataset_qids[ds] = list(qid_to_query.keys())
                else:
                    print(f"[debug] WARNING: pseudo_train.jsonl not found: {pseudo_train_file}")
                    queries_by_dataset[ds] = {}
                    per_dataset_qids[ds] = []
            
            for p in paths:
                m = pattern.search(os.path.basename(p))
                if not m:
                    continue
                ds = m.group("dataset")
                
                # cqadupstack은 이미 처리했으므로 스킵
                if ds == "cqadupstack":
                    continue
                
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        file_qids = json.load(f)
                except Exception:
                    continue
                # load queries for this dataset (add diagnostics to help debug empty mappings)
                try:
                    # Print diagnostic info about dataset_root and expected query file paths
                    try:
                        root = args.dataset_root
                        # Convert to absolute path if relative
                        if not os.path.isabs(root):
                            root = os.path.abspath(root)
                    except Exception:
                        root = os.path.abspath("datasets/IR")
                    qpath1 = os.path.join(root, ds, ds, "queries.jsonl")
                    qpath2 = os.path.join(root, ds, "queries.jsonl")
                    print(f"[debug] Loading queries for dataset '{ds}' from root '{root}'")
                    print(f"[debug] Trying path 1: {qpath1} (exists: {os.path.exists(qpath1)})")
                    print(f"[debug] Trying path 2: {qpath2} (exists: {os.path.exists(qpath2)})")
                    queries_by_dataset[ds] = load_queries(root, ds, max_queries=None)
                    print(f"[debug] Loaded {len(queries_by_dataset[ds])} queries for dataset '{ds}'")
                    if len(queries_by_dataset[ds]) == 0:
                        print(f"[debug] WARNING: No queries loaded for dataset '{ds}' - check file format and paths")
                except Exception as e:
                    print(f"[debug] load_queries failed for ds={ds}: {e}")
                    import traceback
                    print(f"[debug] Full traceback: {traceback.format_exc()}")
                    queries_by_dataset[ds] = {}

                # Filter qids to ones that exist in the loaded queries (preserve order)
                valid = [q for q in file_qids if q in queries_by_dataset.get(ds, {})]
                per_dataset_qids[ds] = valid
                ds_order.append(ds)

            # Optionally shuffle each dataset's qid list deterministically using the seed.
            try:
                import random
                if args.shuffle_seed is not None:
                    seed = int(args.shuffle_seed)
                    for i, ds in enumerate(ds_order):
                        r = random.Random(seed + i)
                        r.shuffle(per_dataset_qids[ds])
                    print(f"Shuffled per-dataset qid lists with seed={args.shuffle_seed}")
                # Interleave (round-robin) across datasets so order alternates datasets
                # Build an ordered round-robin list: pop from each dataset in turn
                more = True
                counts = {ds: len(lst) for ds, lst in per_dataset_qids.items()}
                while any(counts.get(ds, 0) > 0 for ds in ds_order):
                    for ds in ds_order:
                        lst = per_dataset_qids.get(ds, [])
                        if lst:
                            q = lst.pop(0)
                            qids_list.append((q, ds))
                            counts[ds] -= 1
                print(f"Interleaved qids across datasets: {ds_order} -> total qids={len(qids_list)}")
            except Exception:
                # Fallback: previous behavior (append all and shuffle combined if requested)
                for p in paths:
                    m = pattern.search(os.path.basename(p))
                    if not m:
                        continue
                    ds = m.group("dataset")
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            file_qids = json.load(f)
                    except Exception:
                        continue
                    for q in file_qids:
                        qids_list.append((q, ds))
                try:
                    if len(qids_list) > 1:
                        if args.shuffle_seed is not None:
                            random.seed(int(args.shuffle_seed))
                        random.shuffle(qids_list)
                        print(f"Shuffled combined qids list (seed={args.shuffle_seed}) — total qids={len(qids_list)}")
                except Exception:
                    pass
        else:
            # single-file (legacy) behavior: use args.dataset to load queries
            try:
                with open(args.qids_path, "r", encoding="utf-8") as f:
                    file_qids = json.load(f)
            except Exception:
                file_qids = []
            queries_map = load_queries(args.dataset_root, args.dataset, max_queries=None)
            selected = [qid for qid in file_qids if qid in queries_map]
            queries_by_dataset[args.dataset] = queries_map
            for q in selected:
                qids_list.append((q, args.dataset))

        # If we failed to collect any qids, abort early with helpful diagnostics
        if not qids_list:
            print("No qids were collected for GRPO training. Check your --qids-path and --dataset-root settings.")
            print(f"Found dataset files: {paths if 'paths' in locals() else 'none'}")
            print(f"queries_by_dataset keys: {list(queries_by_dataset.keys())}")
            sys.exit(1)

        # propagate CLI verbose into evaluation layer so retrieval prints more info
        # === 여기서부터 offline GRPO 호출 ===
        run_offline_grpo(
            adapter_dir=args.adapter_dir,
            base_model=args.base_model,
            qids=qids_list,
            queries_by_dataset=queries_by_dataset,
            steps=args.grpo_steps,
            lr=args.lr,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            save_name=getattr(args, "save_name", None),
            wandb_enable=getattr(args, "wandb", False),
            wandb_project=getattr(args, "wandb_project", None),
            wandb_run_name=getattr(args, "wandb_run_name", None),
            use_task_tokens=args.task_tokens,
            bf16=not args.no_bf16,
            retrieval_type=args.retrieval_type,
            zero_ndcg_threshold=args.zero_ndcg_threshold,
            reranker=getattr(args, "reranker", None),
            clip_eps=args.clip_eps,
            logrho_clip=args.logrho_clip,
            kl_beta=args.kl_beta,
            length_norm=args.length_norm,
            max_cands_per_qid=args.rollouts,
            cand_select_mode=args.rollout_subset_mode,
            surrogate_mode=args.surrogate,
            reward_transform=args.reward_transform,
        )
