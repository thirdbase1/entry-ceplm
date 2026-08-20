#!/data/data/com.termux/files/usr/bin/bash
# Build training corpus from public repos (run on device or any machine)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/data" "$ROOT/repos"
cd "$ROOT/repos"

download() {
  local name="$1" url="$2"
  if [ -d "$name" ]; then
    echo "exists: $name"
    return
  fi
  echo "Downloading $name ..."
  curl -sL --max-time 180 "$url" -o "${name}.tgz"
  mkdir -p "$name"
  tar --no-same-owner -xzf "${name}.tgz" -C "$name" --strip-components=1 2>/dev/null || \
    tar -xzf "${name}.tgz" -C "$name" --strip-components=1
  rm -f "${name}.tgz"
  echo "ok: $name"
}

download entry-agents "https://codeload.github.com/thirdbase1/entry-agents/tar.gz/main"
download entry-gateway "https://codeload.github.com/thirdbase1/entry-gateway/tar.gz/main"
download open-agents "https://codeload.github.com/vercel-labs/open-agents/tar.gz/main"

python3 "$ROOT/scripts/pack_corpus.py"
echo "Corpus ready: $ROOT/data/corpus_train.txt"
