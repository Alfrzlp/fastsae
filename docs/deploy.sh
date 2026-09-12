#!/bin/bash
# ──────────────────────────────────────────────────────────────
# deploy.sh — Deploy to ridsonap.github.io/fastsae
#
# Setup (once):
#   git clone https://github.com/ridsonap/ridsonap.github.io.git
#   cd ridsonap.github.io
#   mkdir -p fastsae
#
# Run each time:
#   bash /path/to/this/deploy.sh
# ──────────────────────────────────────────────────────────────

set -e

GHPAGES="$HOME/ridsonap.github.io"  # adjust if your clone is elsewhere
SRC="$(dirname "$0")"  # use local docs folder

if [ ! -d "$GHPAGES" ]; then
  echo "ERROR: GitHub Pages repo not found at $GHPAGES"
  echo "Clone it first:"
  echo "  git clone https://github.com/ridsonap/ridsonap.github.io.git $GHPAGES"
  echo "  cd $GHPAGES && mkdir -p fastsae"
  exit 1
fi

TARGET="$GHPAGES/fastsae"
mkdir -p "$TARGET"

echo "Deploying to https://ridsonap.github.io/fastsae"
cp "$SRC/index.html" "$TARGET/index.html"
cp "$SRC/benchmark.html" "$TARGET/benchmark.html"
[ -f "$SRC/benchmark-data.json" ] && cp "$SRC/benchmark-data.json" "$TARGET/benchmark-data.json"

cd "$GHPAGES"
git add fastsae/
git commit -m "Update fastsae docs - $(date '+%Y-%m-%d %H:%M')" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
git push origin main

echo "Done → https://ridsonap.github.io/fastsae"
