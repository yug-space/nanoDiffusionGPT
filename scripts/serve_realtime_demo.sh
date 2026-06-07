#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${OUT_DIR:-out-diffusion-shakespeare-char-100k}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
DEVICE="${DEVICE:-auto}"

if [[ ! -x .venv/bin/python ]]; then
  echo ".venv is missing. Run ./scripts/train_shakespeare_diffusion_100k.sh first." >&2
  exit 1
fi

.venv/bin/python web_demo.py \
  "--out_dir=$OUT_DIR" \
  "--host=$HOST" \
  "--port=$PORT" \
  "--device=$DEVICE"
