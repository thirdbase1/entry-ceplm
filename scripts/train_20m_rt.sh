#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
# CEPLM-RT ~21M — merged CEPLM + Rollout Trace Modeling
python3 src/ceplm_rt_20m_train.py \
  --corpus data/corpus_train.txt \
  --out checkpoints_20m_rt \
  --steps "${STEPS:-5000}" \
  --batch 1 \
  --accum "${ACCUM:-8}" \
  --lr 2.5e-4 \
  --eval_every 500 \
  --save_every 1000
