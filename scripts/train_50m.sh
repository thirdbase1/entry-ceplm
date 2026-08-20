#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
# ~50M model — needs ~6–10GB free RAM; use low batch + accum on phone
python3 src/ceplm_50m_train.py \
  --corpus data/corpus_train.txt \
  --out checkpoints_50m \
  --steps "${STEPS:-5000}" \
  --batch 1 \
  --accum "${ACCUM:-8}" \
  --lr 2e-4 \
  --eval_every 250 \
  --save_every 500
