#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_FILE="${CONFIG_FILE:-config/train_shakespeare_char_100k.py}"
OUT_DIR="${OUT_DIR:-out-diffusion-shakespeare-char-100k}"
SAMPLE_AFTER="${SAMPLE_AFTER:-1}"

if [[ ! -x .venv/bin/python ]]; then
  if command -v python3.12 >/dev/null 2>&1; then
    BASE_PY="$(command -v python3.12)"
  elif command -v python3 >/dev/null 2>&1; then
    BASE_PY="$(command -v python3)"
  else
    echo "Could not find python3.12 or python3." >&2
    exit 1
  fi
  "$BASE_PY" -m venv .venv
fi

PY=".venv/bin/python"

if ! "$PY" -c "import torch, numpy" >/dev/null 2>&1; then
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install -r requirements.txt
fi

if [[ ! -f data/shakespeare_char/input.txt ]]; then
  mkdir -p data/shakespeare_char
  curl -L --fail \
    https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt \
    -o data/shakespeare_char/input.txt
fi

if [[ ! -f data/shakespeare_char/train.bin || ! -f data/shakespeare_char/val.bin || ! -f data/shakespeare_char/meta.pkl ]]; then
  "$PY" data/shakespeare_char/prepare.py
fi

DEVICE="${DEVICE:-$("$PY" - <<'PY'
import torch
if torch.cuda.is_available():
    print("cuda")
elif torch.backends.mps.is_available():
    print("mps")
else:
    print("cpu")
PY
)}"

DTYPE="${DTYPE:-$("$PY" - <<PY
import torch
device = "$DEVICE"
if device == "cuda":
    print("bfloat16" if torch.cuda.is_bf16_supported() else "float16")
else:
    print("float32")
PY
)}"

if [[ -z "${COMPILE:-}" ]]; then
  if [[ "$DEVICE" == cuda* ]]; then
    COMPILE=True
  else
    COMPILE=False
  fi
fi

INIT_FROM="${INIT_FROM:-scratch}"
if [[ -f "$OUT_DIR/ckpt.pt" && "${RESUME:-auto}" != "never" ]]; then
  INIT_FROM="resume"
fi

mkdir -p "$OUT_DIR"
LOG_FILE="${LOG_FILE:-$OUT_DIR/train.log}"

TRAIN_ARGS=(
  train.py
  "$CONFIG_FILE"
  "--out_dir=$OUT_DIR"
  "--init_from=$INIT_FROM"
  "--device=$DEVICE"
  "--dtype=$DTYPE"
  "--compile=$COMPILE"
)

add_override() {
  local env_name="$1"
  local config_name="$2"
  local value="${!env_name:-}"
  if [[ -n "$value" ]]; then
    TRAIN_ARGS+=("--$config_name=$value")
  fi
}

add_override MAX_ITERS max_iters
if [[ -n "${MAX_ITERS:-}" && -z "${LR_DECAY_ITERS:-}" ]]; then
  TRAIN_ARGS+=("--lr_decay_iters=$MAX_ITERS")
fi
add_override LR_DECAY_ITERS lr_decay_iters
add_override WARMUP_ITERS warmup_iters
add_override LEARNING_RATE learning_rate
add_override MIN_LR min_lr
add_override EVAL_INTERVAL eval_interval
add_override EVAL_ITERS eval_iters
add_override LOG_INTERVAL log_interval
add_override BATCH_SIZE batch_size
add_override BLOCK_SIZE block_size
add_override GRAD_ACCUM gradient_accumulation_steps
add_override N_LAYER n_layer
add_override N_HEAD n_head
add_override N_EMBD n_embd
add_override DROPOUT dropout

echo "NanoDiffusionGPT tiny Shakespeare"
echo "config=$CONFIG_FILE"
echo "out_dir=$OUT_DIR init_from=$INIT_FROM"
echo "device=$DEVICE dtype=$DTYPE compile=$COMPILE"
echo "log=$LOG_FILE"

"$PY" -u "${TRAIN_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"

if [[ "$SAMPLE_AFTER" == "1" ]]; then
  "$PY" -u sample.py \
    "--out_dir=$OUT_DIR" \
    "--device=$DEVICE" \
    "--dtype=$DTYPE" \
    "--compile=False" \
    "--max_new_tokens=${SAMPLE_TOKENS:-160}" \
    "--steps=${SAMPLE_STEPS:-128}" \
    "--num_samples=${SAMPLE_COUNT:-5}" \
    "--start=${SAMPLE_START:-To be}" \
    "--temperature=${SAMPLE_TEMP:-0.8}" \
    "--top_k=${SAMPLE_TOP_K:-30}"
fi
