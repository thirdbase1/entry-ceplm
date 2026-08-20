# Entry CEPLM

**Contract-Echo Path Language Modeling** — train small specialist LMs (~3M–50M params) on [Entry](https://github.com/thirdbase1/entry-agents) agent-harness code and related public production repos.

Designed to run on **CPU**, including **Termux on Android** (16GB RAM phones).

## What is CEPLM?

Not plain next-token training. Each batch mixes:

1. **Path conditioning** — sequences start with `PATH:<repo-relative-path>`
2. **Contract echo** — synthetic Q↔A about harness contracts (run state machine, permissions, tool metadata, verification, gateway fallback)
3. **Flip tasks** — description→contract and contract→description
4. **Surface code LM** — next-character prediction on path-conditioned source

Char-level ASCII (vocab 128) — no tokenizer training required.

## Model sizes

| Config | Params | Layers | Emb | Context | Phone RAM |
|--------|--------|--------|-----|---------|-----------|
| **3M** | ~3.16M | 7 | 192 | 192 | any (2GB+) |
| **50M** | ~49.4M | 10 | 640 | 256 | 8–16GB recommended |

## Quick start

```bash
git clone https://github.com/thirdbase1/entry-ceplm.git
cd entry-ceplm
pip install -r requirements.txt
bash scripts/build_corpus.sh
bash scripts/train_3m.sh
# or: STEPS=5000 ACCUM=8 bash scripts/train_50m.sh
```

---

## Termux on Android (16GB / 256GB)

### 1. Install Termux

Install **Termux** from [F-Droid](https://f-droid.org/packages/com.termux/) (not Play Store).

### 2. Packages

```bash
pkg update -y && pkg upgrade -y
pkg install -y git python clang make libffi openssl rust binutils python-pip curl
```

### 3. Clone

```bash
cd ~
git clone https://github.com/thirdbase1/entry-ceplm.git
cd entry-ceplm
```

### 4. PyTorch (CPU)

```bash
pip install --upgrade pip numpy
pip install torch --index-url https://download.pytorch.org/whl/cpu
# if that fails:
#   pkg install -y python-torch
python3 -c "import torch; print(torch.__version__)"
export OMP_NUM_THREADS=4
```

### 5. Build corpus

```bash
bash scripts/build_corpus.sh
```

Pulls `entry-agents`, `entry-gateway`, `vercel-labs/open-agents` → `data/corpus_train.txt`.

### 6. Train

```bash
# 3M first (reliable)
STEPS=5000 BATCH=8 bash scripts/train_3m.sh

# 50M on 16GB phone
pkg install -y termux-api && termux-wake-lock
STEPS=8000 ACCUM=8 bash scripts/train_50m.sh
```

Plug in power. If OOM, lower ACCUM or stick to 3M.

### 7. Sample

```bash
python3 - <<'PY'
import torch
from src.ceplm_train import TinyGPT, Config, encode, decode
cfg = Config()
model = TinyGPT(cfg)
ckpt = torch.load("checkpoints_3m/ceplm_step3000.pt", map_location="cpu")
model.load_state_dict(ckpt["model"]); model.eval()
prompt = "PATH:docs/contracts\nQ: List the three Entry permission modes.\nA:"
out = model.generate(torch.tensor([encode(prompt)]), max_new=120, temperature=0.6)
print(decode(out[0].tolist()))
PY
```

## Layout

```
src/ceplm_train.py      # ~3M
src/ceplm_50m_train.py  # ~50M
scripts/build_corpus.sh
scripts/train_3m.sh
scripts/train_50m.sh
```

Upstream repos keep their own licenses.
