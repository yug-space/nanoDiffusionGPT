#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${OUT_DIR:-out-diffusion-shakespeare-char-100k}"
CKPT="$OUT_DIR/ckpt.pt"

if [[ ! -f "$CKPT" ]]; then
  echo "No checkpoint found at $CKPT" >&2
  echo "Start a new run first:" >&2
  echo "  ./scripts/train_shakespeare_diffusion_100k.sh" >&2
  exit 1
fi

echo "Resuming NanoDiffusionGPT from $CKPT"

OUT_DIR="$OUT_DIR" INIT_FROM=resume "$SCRIPT_DIR/train_shakespeare_diffusion_100k.sh"
