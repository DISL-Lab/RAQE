# Environment setup

Reference environment (the one the reported numbers were produced on):

| | |
|---|---|
| OS | Ubuntu 22.04 (Linux 5.15) |
| GPUs | 2 × NVIDIA RTX PRO 6000 (96 GB) |
| CUDA driver | 580.x (CUDA 12.8 wheels) |
| Python | 3.11 |
| JDK | OpenJDK **21** (required by Pyserini/Anserini) |

Both training stages use LoRA + bf16 on an 8B backbone, so ~40 GB of VRAM per
process is enough; two processes reproduce the batch sizes in the paper.

---

## 1. Python environment

```bash
conda create -n raqe python=3.11 -y
conda activate raqe

# Match the CUDA build to your driver; cu128 is what the paper used.
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

`environment.yml` is a full `conda env export` of the development machine. It
pins everything, including packages RAQE does not use, and is only meant as a
last-resort reference when a version conflict needs to be traced. Prefer
`requirements.txt`.

### Hugging Face access

`meta-llama/Llama-3.1-8B-Instruct` and `meta-llama/Llama-3.2-3B-Instruct` are
gated. Request access on the Hub, then:

```bash
cp .env.example .env     # .env is gitignored
export HF_TOKEN=hf_...   # or: huggingface-cli login
```

No API keys are needed for the RAQE training or evaluation path itself; the
entries in `.env.example` are only for optional hosted baselines.

---

## 2. Java 21 for Pyserini

Pyserini's Lucene backend needs the `jdk.incubator.vector` module, which exists
only in JDK 17+ builds that ship incubator modules. A JRE without it aborts the
process at import time with:

```
Error occurred during initialization of boot layer
java.lang.module.FindException: Module jdk.incubator.vector not found
```

This bites in particular when a conda environment contains its own `java` — for
example a JetBrains "nomod" runtime — which then shadows the system JDK. Install
a real JDK 21 and point the JVM variables at it:

```bash
sudo apt install openjdk-21-jdk          # or: conda install -c conda-forge openjdk=21

export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export JVM_PATH=$JAVA_HOME/lib/server/libjvm.so
```

Verify:

```bash
python -c "from pyserini.search.lucene import LuceneSearcher; print('pyserini ok')"
# WARNING: Using incubator modules: jdk.incubator.vector
# pyserini ok
```

Every entry point that touches Pyserini imports `raqe_jvm` first, which probes
the candidate JDKs with `java --add-modules jdk.incubator.vector -version` and
exports `JVM_PATH`/`JAVA_HOME` for the first one that passes — system JDKs before
conda ones, since a conda env is where the broken runtime usually lives. So the
exports above are a safety net rather than a requirement; an explicit `JVM_PATH`
is always respected. What you do have to provide is a JDK 21 that exists
somewhere on the machine.

---

## 3. Accelerate

Both training stages launch through `accelerate`. The scripts pass
`--num_processes` and `--mixed_precision bf16` explicitly, so a bare default
config is enough:

```bash
accelerate config default
```

Pick the GPUs with `CUDA_VISIBLE_DEVICES`:

```bash
CUDA_VISIBLE_DEVICES=0,1 NUM_PROCESSES=2 bash scripts/train_sft.sh
```

`NUM_PROCESSES` changes the effective batch size and therefore the number of
optimizer updates. The paper's step counts (1,010 for Stage 1 and 1,730 for
Stage 2) assume **two** processes; see `configs/paper_defaults.md`.

---

## 4. Smoke test

```bash
python - <<'PY'
import raqe_jvm                       # resolves the JDK before pyserini loads
import torch, transformers, peft, accelerate
from pyserini.search.lucene import LuceneSearcher
import pytrec_eval
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count())
print("transformers", transformers.__version__, "peft", peft.__version__, "accelerate", accelerate.__version__)
print("pyserini + pytrec_eval ok")
PY
```

Expected on the reference machine:

```
torch 2.8.0+cu128 cuda True 2
transformers 4.57.3 peft 0.17.1 accelerate 1.12.0
pyserini + pytrec_eval ok
```

---

## 5. Known version constraints

- **transformers ≥ 4.46** changed gradient-accumulation loss scaling. Stage 1
  uses the stock `Trainer`, so it follows whatever the installed version does;
  pin the version if you need to match a previous run exactly.
- **trl** is imported by `grpo/step1/test_grpo_offline_real.py` but the offline
  GRPO loop is hand-written — any recent version works.
- **numpy < 2** is required by the pinned `faiss`/`pyserini` builds.
- **pyserini 1.2.0** is what the prebuilt index names in this repo were verified
  against; older versions use different index identifiers.
