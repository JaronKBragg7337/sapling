"""Training-speed benchmark for the v4 core (same model, batch and precision as Stage A), on random tokens.
Measures tokens/second so the MSI's RTX 4060 (~35,500 tok/s in Stage A) can be compared with a rented GPU. No data needed.

  python tools/bench_speed.py [--seconds 150] [--micro 16] [--notify https://ntfy.sh/<topic>]

Prints one JSON line; with --notify it also POSTs that line (a one-off topic, nothing secret in it)."""
import argparse
import json
import os
import sys
import time
import urllib.request

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sapling.model import V4_CONFIG, SaplingCore, autocast_ctx  # noqa: E402

VOCAB = 17661
BATCH = 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=150)
    ap.add_argument("--micro", type=int, default=16)
    ap.add_argument("--notify", default="")
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    model = SaplingCore(vocab_size=VOCAB, **V4_CONFIG).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4)
    T = V4_CONFIG["block_size"]
    steps = tok = 0
    t0 = None
    start = time.time()
    while True:
        opt.zero_grad(set_to_none=True)
        for _ in range(BATCH // a.micro):
            x = torch.randint(0, VOCAB, (a.micro, T), device=dev)
            with autocast_ctx(dev):
                _, loss = model(x, x)
            (loss / (BATCH // a.micro)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        steps += 1
        if steps == 20:                       # warm-up done: start the clock
            torch.cuda.synchronize(); t0 = time.time(); tok = 0
        elif t0 is not None:
            tok += BATCH * T
            if time.time() - t0 > a.seconds:
                break
        if time.time() - start > a.seconds + 300:
            break
    torch.cuda.synchronize()
    el = time.time() - t0
    row = {"gpu": torch.cuda.get_device_name(0), "params": sum(p.numel() for p in model.parameters()),
           "micro": a.micro, "batch": BATCH, "seconds": round(el, 1), "tok_per_s": round(tok / el),
           "hours_for_1_2B": round(1.2e9 / (tok / el) / 3600, 2), "torch": torch.__version__}
    line = json.dumps(row)
    print(line, flush=True)
    if a.notify:
        urllib.request.urlopen(urllib.request.Request(a.notify, data=line.encode(), method="POST"), timeout=30)


if __name__ == "__main__":
    main()
