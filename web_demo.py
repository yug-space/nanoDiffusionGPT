"""
Realtime NanoDiffusionGPT generation demo.

Run:
    python web_demo.py --out_dir=out-diffusion-shakespeare-char-100k

Then open:
    http://127.0.0.1:8000
"""

import argparse
import html
import json
import math
import os
import pickle
import time
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import torch
from torch.nn import functional as F

from model import NanoDiffusionGPT, NanoDiffusionGPTConfig


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>NanoDiffusionGPT Realtime</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101114;
      --panel: #17191f;
      --panel-2: #1d2028;
      --text: #f2f3f5;
      --muted: #a8adba;
      --line: #2b303b;
      --accent: #4fb477;
      --accent-2: #79a8ff;
      --warn: #e7b85d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 16px 20px;
      border-bottom: 1px solid var(--line);
      background: #12141a;
    }
    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 650;
      letter-spacing: 0;
    }
    main {
      display: grid;
      grid-template-columns: 340px 1fr;
      gap: 16px;
      padding: 16px;
      max-width: 1280px;
      margin: 0 auto;
    }
    section, aside {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    aside {
      padding: 14px;
    }
    .status {
      color: var(--muted);
      font-size: 13px;
    }
    label {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin: 14px 0 6px;
    }
    input, textarea, button {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel-2);
      color: var(--text);
      font: inherit;
    }
    input, textarea {
      padding: 9px 10px;
    }
    textarea {
      min-height: 80px;
      resize: vertical;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    button {
      margin-top: 16px;
      padding: 10px 12px;
      cursor: pointer;
      background: var(--accent);
      border-color: #3f9864;
      color: #06130b;
      font-weight: 700;
    }
    button:disabled {
      cursor: not-allowed;
      opacity: 0.6;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      border-bottom: 1px solid var(--line);
    }
    .metric {
      padding: 14px;
      border-right: 1px solid var(--line);
    }
    .metric:last-child { border-right: 0; }
    .metric .value {
      display: block;
      font-size: 24px;
      font-weight: 700;
    }
    .metric .label {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-top: 2px;
    }
    .output {
      padding: 16px;
    }
    pre {
      min-height: 420px;
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font: 16px/1.55 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    }
    .mask {
      color: var(--warn);
    }
    .bar {
      height: 8px;
      background: #0b0c0f;
      border-radius: 999px;
      overflow: hidden;
      margin-top: 12px;
      border: 1px solid var(--line);
    }
    .fill {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, var(--accent), var(--accent-2));
    }
    .note {
      margin-top: 14px;
      color: var(--muted);
      font-size: 12px;
    }
    @media (max-width: 860px) {
      main { grid-template-columns: 1fr; }
      .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .metric:nth-child(2) { border-right: 0; }
      .metric:nth-child(-n+2) { border-bottom: 1px solid var(--line); }
    }
  </style>
</head>
<body>
  <header>
    <h1>NanoDiffusionGPT Realtime Forward Passes</h1>
    <div class="status" id="serverStatus">checkpoint loading...</div>
  </header>
  <main>
    <aside>
      <label for="prompt">Prompt</label>
      <textarea id="prompt">To be</textarea>

      <div class="row">
        <div>
          <label for="tokens">New Tokens</label>
          <input id="tokens" type="number" min="1" max="240" value="180" />
        </div>
        <div>
          <label for="steps">Denoise Steps</label>
          <input id="steps" type="number" min="1" max="256" value="128" />
        </div>
      </div>

      <div class="row">
        <div>
          <label for="temperature">Temperature</label>
          <input id="temperature" type="number" min="0.1" max="2" step="0.05" value="0.95" />
        </div>
        <div>
          <label for="topK">Top K</label>
          <input id="topK" type="number" min="0" max="100" value="65" />
        </div>
      </div>

      <button id="run">Generate</button>
      <div class="note" id="note">
        Each denoising step runs one Transformer forward pass. Fewer masks than steps means it can stop early.
      </div>
    </aside>

    <section>
      <div class="metrics">
        <div class="metric"><span class="value" id="step">0</span><span class="label">step</span></div>
        <div class="metric"><span class="value" id="passes">0</span><span class="label">forward passes</span></div>
        <div class="metric"><span class="value" id="remaining">0</span><span class="label">masks left</span></div>
        <div class="metric"><span class="value" id="elapsed">0.0s</span><span class="label">elapsed</span></div>
      </div>
      <div class="output">
        <div class="bar"><div class="fill" id="fill"></div></div>
        <pre id="text"></pre>
      </div>
    </section>
  </main>

  <script>
    const el = id => document.getElementById(id);
    let source = null;

    async function loadStatus() {
      const res = await fetch("/status");
      const data = await res.json();
      el("serverStatus").textContent =
        `${data.checkpoint} | ${data.device} | ${data.params_m.toFixed(2)}M params | block ${data.block_size}`;
    }

    function setRunning(running) {
      el("run").disabled = running;
      el("run").textContent = running ? "Generating..." : "Generate";
    }

    function renderText(value) {
      const escaped = value
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
      el("text").innerHTML = escaped.replaceAll("[MASK]", '<span class="mask">[MASK]</span>');
    }

    function run() {
      if (source) source.close();
      setRunning(true);
      renderText("");
      el("step").textContent = "0";
      el("passes").textContent = "0";
      el("remaining").textContent = "0";
      el("elapsed").textContent = "0.0s";
      el("fill").style.width = "0%";

      const params = new URLSearchParams({
        prompt: el("prompt").value,
        max_new_tokens: el("tokens").value,
        steps: el("steps").value,
        temperature: el("temperature").value,
        top_k: el("topK").value
      });
      source = new EventSource(`/generate?${params.toString()}`);

      source.addEventListener("update", event => {
        const data = JSON.parse(event.data);
        el("step").textContent = data.step;
        el("passes").textContent = data.forward_passes;
        el("remaining").textContent = data.remaining;
        el("elapsed").textContent = `${data.elapsed.toFixed(1)}s`;
        el("fill").style.width = `${data.progress * 100}%`;
        renderText(data.text);
      });

      source.addEventListener("done", event => {
        const data = JSON.parse(event.data);
        el("step").textContent = data.step;
        el("passes").textContent = data.forward_passes;
        el("remaining").textContent = data.remaining;
        el("elapsed").textContent = `${data.elapsed.toFixed(1)}s`;
        el("fill").style.width = "100%";
        renderText(data.text);
        setRunning(false);
        source.close();
      });

      source.addEventListener("error", () => {
        setRunning(false);
        if (source) source.close();
      });
    }

    el("run").addEventListener("click", run);
    loadStatus().catch(err => {
      el("serverStatus").textContent = String(err);
    });
  </script>
</body>
</html>
"""


def pick_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device):
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


class DemoState:
    def __init__(self, out_dir, device):
        self.out_dir = out_dir
        self.device = pick_device(device)
        self.dtype = pick_dtype(self.device)
        ckpt_path = os.path.join(out_dir, "ckpt.pt")
        self.checkpoint_name = ckpt_path
        checkpoint = torch.load(ckpt_path, map_location=self.device)

        self.model = NanoDiffusionGPT(NanoDiffusionGPTConfig(**checkpoint["model_args"]))
        state_dict = checkpoint["model"]
        for key in list(state_dict.keys()):
            if key.startswith("_orig_mod."):
                state_dict[key[len("_orig_mod.") :]] = state_dict.pop(key)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        meta_path = os.path.join("data", checkpoint["config"]["dataset"], "meta.pkl")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self.stoi = meta["stoi"]
        self.itos = meta["itos"]

    def encode(self, text):
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids, show_masks=False):
        out = []
        for token_id in ids:
            token_id = int(token_id)
            if token_id == self.model.config.mask_token_id:
                out.append("[MASK]" if show_masks else "")
            elif token_id in self.itos:
                out.append(self.itos[token_id])
        return "".join(out)

    def status(self):
        return {
            "checkpoint": self.checkpoint_name,
            "device": self.device,
            "params_m": self.model.get_num_params() / 1e6,
            "block_size": self.model.config.block_size,
        }


def stream_generation(state, prompt, max_new_tokens, steps, temperature, top_k):
    prompt_ids = state.encode(prompt)
    if not prompt_ids:
        raise ValueError("prompt has no characters from the dataset vocabulary")
    total_len = len(prompt_ids) + max_new_tokens
    if total_len > state.model.config.block_size:
        raise ValueError(f"prompt + new tokens is {total_len}, block size is {state.model.config.block_size}")

    device = state.device
    x = torch.full(
        (1, total_len),
        state.model.config.mask_token_id,
        dtype=torch.long,
        device=device,
    )
    x[:, : len(prompt_ids)] = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    prompt_mask = torch.zeros_like(x, dtype=torch.bool)
    prompt_mask[:, : len(prompt_ids)] = True

    start = time.time()
    forward_passes = 0
    steps = max(1, steps)

    initial_remaining = int(((x == state.model.config.mask_token_id) & (~prompt_mask)).sum().item())
    yield {
        "step": 0,
        "forward_passes": 0,
        "remaining": initial_remaining,
        "elapsed": 0.0,
        "progress": 0.0,
        "text": state.decode(x[0].tolist(), show_masks=True),
    }

    with torch.no_grad():
        for step in range(steps):
            remaining = (x == state.model.config.mask_token_id) & (~prompt_mask)
            remaining_count = int(remaining.sum().item())
            if remaining_count == 0:
                break

            ctx = torch.amp.autocast(device_type="cuda", dtype=state.dtype) if device == "cuda" else nullcontext()
            with ctx:
                logits, _ = state.model(x)
            forward_passes += 1

            logits[..., state.model.config.mask_token_id] = -float("inf")
            logits = logits / max(temperature, 1e-6)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[..., [-1]], -float("inf"))

            probs = F.softmax(logits, dim=-1)
            pred = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(x.shape)
            confidence = probs.gather(-1, pred.unsqueeze(-1)).squeeze(-1)
            confidence = confidence.masked_fill(~remaining, -1.0)

            steps_left = steps - step
            fill_count = max(1, math.ceil(remaining_count / steps_left))
            selected = torch.topk(confidence.view(-1), fill_count).indices
            flat_x = x.view(-1)
            flat_pred = pred.view(-1)
            flat_x[selected] = flat_pred[selected]

            new_remaining = int(((x == state.model.config.mask_token_id) & (~prompt_mask)).sum().item())
            yield {
                "step": step + 1,
                "forward_passes": forward_passes,
                "remaining": new_remaining,
                "elapsed": time.time() - start,
                "progress": 1.0 - (new_remaining / max(1, initial_remaining)),
                "text": state.decode(x[0].tolist(), show_masks=True),
            }


class Handler(BaseHTTPRequestHandler):
    state = None

    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args), flush=True)

    def send_json(self, obj):
        payload = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            payload = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if parsed.path == "/status":
            self.send_json(self.state.status())
            return

        if parsed.path == "/generate":
            self.handle_generate(parsed)
            return

        if parsed.path == "/generate_once":
            self.handle_generate_once(parsed)
            return

        self.send_response(404)
        self.end_headers()

    def handle_generate(self, parsed):
        query = parse_qs(parsed.query)
        prompt = query.get("prompt", ["To be"])[0]
        max_new_tokens = max(1, min(240, int(query.get("max_new_tokens", ["180"])[0])))
        steps = max(1, min(256, int(query.get("steps", ["128"])[0])))
        temperature = max(0.1, min(2.0, float(query.get("temperature", ["0.95"])[0])))
        top_k = max(0, min(100, int(query.get("top_k", ["65"])[0])))

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        try:
            last = None
            for update in stream_generation(self.state, prompt, max_new_tokens, steps, temperature, top_k):
                last = update
                event = "done" if update["remaining"] == 0 else "update"
                payload = json.dumps(update)
                self.wfile.write(f"event: {event}\ndata: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
            if last is not None and last["remaining"] != 0:
                payload = json.dumps(last)
                self.wfile.write(f"event: done\ndata: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
        except Exception as exc:
            payload = json.dumps({"error": str(exc), "text": html.escape(str(exc))})
            self.wfile.write(f"event: done\ndata: {payload}\n\n".encode("utf-8"))
            self.wfile.flush()

    def handle_generate_once(self, parsed):
        query = parse_qs(parsed.query)
        prompt = query.get("prompt", ["To be"])[0]
        max_new_tokens = max(1, min(240, int(query.get("max_new_tokens", ["32"])[0])))
        steps = max(1, min(256, int(query.get("steps", ["16"])[0])))
        temperature = max(0.1, min(2.0, float(query.get("temperature", ["0.95"])[0])))
        top_k = max(0, min(100, int(query.get("top_k", ["65"])[0])))
        last = None
        for update in stream_generation(self.state, prompt, max_new_tokens, steps, temperature, top_k):
            last = update
        self.send_json(last or {"error": "no update"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="out-diffusion-shakespeare-char-100k")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    Handler.state = DemoState(args.out_dir, args.device)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"serving http://{args.host}:{args.port}")
    print(f"checkpoint {os.path.join(args.out_dir, 'ckpt.pt')}")
    server.serve_forever()


if __name__ == "__main__":
    main()
