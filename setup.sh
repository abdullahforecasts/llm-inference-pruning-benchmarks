#!/usr/bin/env bash
# One-shot local setup: venv + CPU-only torch + the rest of requirements.txt.
# Run with: bash setup.sh
# (then activate the venv yourself afterward: source .venv/bin/activate)
set -e

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install torch --index-url https://download.pytorch.org/whl/cpu -q
pip install -r requirements.txt -q

echo ""
echo "setup done. activate the venv in your shell with:"
echo "  source .venv/bin/activate"
