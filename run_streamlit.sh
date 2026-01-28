#!/usr/bin/env bash
set -euo pipefail

# Activate venv if present
if [ -d "venv" ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

echo "Starting Streamlit app..."
streamlit run testlh.py
