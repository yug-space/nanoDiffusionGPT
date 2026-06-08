"""
Sample from a trained NanoDiffusionGPT checkpoint.

Unlike GPT sampling, this does not append one token at a time. It creates a
fixed number of [MASK] tokens after the prompt and denoises them in parallel.
"""

import os
import pickle
from contextlib import nullcontext

import torch

from model import NanoDiffusionGPT, NanoDiffusionGPTConfig

# -----------------------------------------------------------------------------
out_dir = "out-diffusion"
start = "\n"
num_samples = 5
max_new_tokens = 300
steps = 128
temperature = 0.8
top_k = 20
seed = 1337
device = "cuda"
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
compile = False
if os.path.exists("configurator.py"):
    exec(open("configurator.py").read())
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

ckpt_path = os.path.join(out_dir, "ckpt.pt")
checkpoint = torch.load(ckpt_path, map_location=device)
model = NanoDiffusionGPT(NanoDiffusionGPTConfig(**checkpoint["model_args"]))
state_dict = checkpoint["model"]
unwanted_prefix = "_orig_mod."
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)
if compile:
    model = torch.compile(model)
raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

dataset = checkpoint["config"]["dataset"]
meta_path = os.path.join("data", dataset, "meta.pkl")
if not os.path.exists(meta_path):
    meta_path = os.path.join(out_dir, "meta.pkl")
if not os.path.exists(meta_path):
    meta_path = "meta.pkl"
print(f"Loading meta from {meta_path}...")
with open(meta_path, "rb") as f:
    meta = pickle.load(f)
stoi, itos = meta["stoi"], meta["itos"]


def encode(s):
    return [stoi[c] for c in s if c in stoi]


def decode(ids):
    return "".join(itos[i] for i in ids if i in itos)


if start.startswith("FILE:"):
    with open(start[5:], "r", encoding="utf-8") as f:
        start = f.read()
start_ids = encode(start)
if not start_ids:
    raise ValueError("prompt has no characters that exist in the dataset vocabulary")
if len(start_ids) + max_new_tokens > raw_model.config.block_size:
    raise ValueError(
        f"prompt length ({len(start_ids)}) + max_new_tokens ({max_new_tokens}) "
        f"exceeds block_size ({raw_model.config.block_size})"
    )
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(
                x,
                max_new_tokens=max_new_tokens,
                steps=steps,
                temperature=temperature,
                top_k=top_k,
            )
            print(decode(y[0].tolist()))
            print("---------------")
