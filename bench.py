"""
A short NanoDiffusionGPT benchmark.
"""

import os
import pickle
import time
from contextlib import nullcontext

import numpy as np
import torch

from model import NanoDiffusionGPT, NanoDiffusionGPTConfig

# -----------------------------------------------------------------------------
dataset = "shakespeare_char"
batch_size = 12
block_size = 256
n_layer = 6
n_head = 6
n_embd = 384
bias = False
real_data = True
seed = 1337
device = "cuda"
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
compile = True
min_mask_prob = 0.05
max_mask_prob = 0.95
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

if real_data:
    data_dir = os.path.join("data", dataset)
    with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    base_vocab_size = meta["vocab_size"]
    train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")

    def get_batch():
        ix = torch.randint(len(train_data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((train_data[i : i + block_size]).astype(np.int64)) for i in ix])
        if device_type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
        return x

else:
    base_vocab_size = 65
    x = torch.randint(base_vocab_size, (batch_size, block_size), device=device)
    get_batch = lambda: x

mask_token_id = base_vocab_size
vocab_size = base_vocab_size + 1


def corrupt_batch(clean_ids):
    B, T = clean_ids.shape
    noise = torch.rand(B, 1, device=clean_ids.device)
    noise = min_mask_prob + (max_mask_prob - min_mask_prob) * noise
    loss_mask = torch.rand(B, T, device=clean_ids.device) < noise
    no_mask_rows = loss_mask.sum(dim=1) == 0
    if no_mask_rows.any():
        forced_pos = torch.randint(0, T, (int(no_mask_rows.sum().item()),), device=clean_ids.device)
        loss_mask[no_mask_rows, forced_pos] = True
    noisy_ids = clean_ids.clone()
    noisy_ids[loss_mask] = mask_token_id
    return noisy_ids, clean_ids, loss_mask


model = NanoDiffusionGPT(
    NanoDiffusionGPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=0.0,
        bias=bias,
        mask_token_id=mask_token_id,
    )
)
model.to(device)
optimizer = model.configure_optimizers(weight_decay=1e-2, learning_rate=1e-4, betas=(0.9, 0.95), device_type=device_type)

if compile:
    print("Compiling model...")
    model = torch.compile(model)

if device_type == "cuda":
    torch.cuda.synchronize()

for stage, num_steps in enumerate([10, 20]):
    t0 = time.time()
    clean = get_batch()
    for k in range(num_steps):
        noisy, targets, loss_mask = corrupt_batch(clean)
        with ctx:
            logits, loss = model(noisy, targets, loss_mask)
        clean = get_batch()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        print(f"{k}/{num_steps} diffusion loss: {loss.item():.4f}")
    if device_type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    mfu = raw_model.estimate_mfu(batch_size * num_steps, dt)
    if stage == 1:
        print(f"time per iteration: {dt/num_steps*1000:.4f}ms, MFU: {mfu*100:.2f}%")
