#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
# 3M model — comfortable on any phone
python3 src/ceplm_train.py \
  --corpus data/corpus_train.txt \
  --out checkpoints_3m \
  --steps "${STEPS:-3000}" \
  --batch "${BATCH:-8}" \
  --lr 3e-4 \
  --eval_every 200
