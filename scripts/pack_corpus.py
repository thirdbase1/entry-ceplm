#!/usr/bin/env python3
"""Pack .ts/.tsx/.js/.md from repos/ into data/corpus_train.txt"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOS = ROOT / "repos"
OUT = ROOT / "data" / "corpus_train.txt"
EXTS = {".ts", ".tsx", ".js", ".jsx", ".md", ".mjs"}
SKIP = {"node_modules", ".git", "dist", "build", ".next", "coverage"}

parts = []
for path in sorted(REPOS.rglob("*")):
    if not path.is_file():
        continue
    if path.suffix.lower() not in EXTS:
        continue
    if any(s in path.parts for s in SKIP):
        continue
    if path.stat().st_size > 400_000:
        continue
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        continue
    rel = path.relative_to(REPOS).as_posix()
    parts.append(f"===== FILE: {rel} =====\n{text}\n")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(parts), encoding="utf-8")
print(f"Wrote {OUT} ({OUT.stat().st_size:,} bytes, {len(parts)} files)")
