# NanoDiffusionGPT

A small diffusion language model in the style of nanoGPT.

It is diffusion-only:

- bidirectional Transformer attention
- random `[MASK]` corruption during training
- loss only on masked positions
- generation by iterative denoising
- no GPT-2 loading or next-token training path

## Install

```sh
pip install -r requirements.txt
```

The training scripts create and use `.venv` automatically if you prefer not to
install into your global Python.

## Quick Start

Prepare tiny Shakespeare:

```sh
python data/shakespeare_char/prepare.py
```

Run a small local training job:

```sh
python train.py config/train_shakespeare_char.py
```

Sample:

```sh
python sample.py --out_dir=out-diffusion-shakespeare-char --start="To be"
```

## 100k Run

Final checkpoint on Hugging Face:

```text
https://huggingface.co/Yugg9826/nanoDiffusionGPT-shakespeare-char
```

Start or auto-resume a 100,000-iteration run:

```sh
./scripts/train_shakespeare_diffusion_100k.sh
```

Resume only if a checkpoint already exists:

```sh
./scripts/resume_shakespeare_diffusion_100k.sh
```

Realtime browser demo:

```sh
./scripts/serve_realtime_demo.sh
```

Then open:

```text
http://127.0.0.1:8000
```

Default output:

```text
out-diffusion-shakespeare-char-100k/ckpt.pt
out-diffusion-shakespeare-char-100k/train.log
```

Common overrides:

```sh
N_LAYER=2 N_HEAD=2 N_EMBD=128 BATCH_SIZE=16 \
./scripts/train_shakespeare_diffusion_100k.sh
```

```sh
DEVICE=cpu COMPILE=False SAMPLE_AFTER=0 \
./scripts/train_shakespeare_diffusion_100k.sh
```

## Core Idea

GPT learns next-token prediction:

```text
input:  To be or no
target: o be or not
```

NanoDiffusionGPT learns masked denoising:

```text
clean:  To be or not
noisy:  T[MASK] be [MASK] not
target: To be or not
```

During training, `train.py`:

1. loads a clean text block
2. randomly replaces some characters with `[MASK]`
3. runs the bidirectional Transformer
4. computes loss only where tokens were masked

During sampling, `sample.py`:

1. encodes the prompt
2. appends `[MASK]` tokens
3. predicts all masked positions
4. fills the most confident masks
5. repeats for several denoising steps

Each denoising step is one model forward pass. For example,
`--steps=128 --max_new_tokens=180` usually runs `128` forward passes per sample.

With the default 100k config, training uses one train forward/backward pass per
iteration because `gradient_accumulation_steps = 1`. Evaluation adds
`2 * eval_iters` forward-only passes every `eval_interval`.

## Repository Layout

```text
model.py                         diffusion Transformer
train.py                         training loop
sample.py                        denoising sampler
bench.py                         short benchmark
config/train_shakespeare_char.py small demo config
config/train_shakespeare_char_100k.py 100k run config
data/shakespeare_char/prepare.py tiny Shakespeare character data prep
scripts/train_shakespeare_diffusion_100k.sh setup + train wrapper
scripts/resume_shakespeare_diffusion_100k.sh resume-only wrapper
scripts/serve_realtime_demo.sh      realtime browser demo
web_demo.py                         local web server for live denoising
requirements.txt                 Python dependencies
```

Generated files are ignored by git:

```text
.venv/
out*/
*.pt
*.bin
*.pkl
*.log
input.txt
__pycache__/
```

## Notes

This is an educational small model. A 100k tiny Shakespeare run should show the
denoising pipeline improving, but it is not comparable to large diffusion LLMs
such as Dream or LLaDA.
