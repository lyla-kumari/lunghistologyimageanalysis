#!/usr/bin/env bash
set -euo pipefail

# Activate venv if present
if [ -d "venv" ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

OUT_DIR=out
mkdir -p "$OUT_DIR"

python testlh.py "path/to/image.png" --out_dir "$OUT_DIR" --save_images --alveoli_overlay

echo "Outputs written to $OUT_DIR"
