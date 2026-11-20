"""Train one fresh ~350M SaplingCore for the bounded 800M-token pilot."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import traceback
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, ROOT)

CONFIG = dict(d_model=1024, n_head=16, n_layer=26, block_size=512, rope=True)
TARGET_TOKENS = 800_000_000
REQUIRED = ("reading.npy", "lesson.npz", "talk.npz", "school.npy", "test.jsonl")
SEED = 7337


def reading_batch(reading, rng, count: int):
    import numpy as np
    rows = [np.asarray(reading[rng.randrange(len(reading))]) for _ in range(count)]
    return np.stack(rows).astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", default="/workspace/out")
    ap.add_argument("--bundle", default="/workspace/pilot350_bundle")
    ap.add_argument("--target-tokens", type=int, default=TARGET_TOKENS)
    ap.add_argument("--minutes", type=float, default=105)
    ap.add_argument("--micro", type=int, default=16)
    ap.add_argument("--no-compile", action="store_true")
    a = ap.parse_args()
    if a.target_tokens != TARGET_TOKENS:
        raise RuntimeError(f"the pilot target is fixed at {TARGET_TOKENS:,} tokens")
    if min(a.target_tokens, a.micro) <= 0 or a.minutes <= 0:
        raise RuntimeError("target, micro-batch and minute limit must be positive")
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.bundle, exist_ok=True)
    st_path = os.path.join(a.out, "status.json")
    state = {"state": "downloading", "mode": "fresh-350M-pilot", "since": time.time()}

    def status(row=None, **kw):
        if row:
            state["last"] = row
        state.update(kw)
        state["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        tmp = st_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=1)
        os.replace(tmp, st_path)

    try:
        status()
        meta = os.path.join(a.bundle, "bundle.json")
        urllib.request.urlretrieve(f"{a.base}/bundle.json", meta)
        with open(meta, encoding="utf-8") as f:
            info = json.load(f)
        if str(info.get("status", "")).upper().startswith("INCOMPLETE"):
            raise RuntimeError("incomplete clean bundle")
        core_name = info.get("core_file")
        if not core_name:
            raise RuntimeError("bundle has no current core/tokenizer")
        if info.get("urls"):
            raise RuntimeError("pilot assets must come only from this clean release")
        names = [core_name, *REQUIRED]
        missing = [name for name in names if name not in info.get("files", {})]
        if missing:
            raise RuntimeError(f"bundle is missing required assets: {missing}")
        t0 = time.time()
        for name in names:
            if os.path.basename(name) != name:
                raise RuntimeError(f"unsafe bundle asset name: {name}")
            dest = os.path.join(a.bundle, name)
            expected = int(info["files"][name])
            for attempt in range(5):
                try:
                    urllib.request.urlretrieve(f"{a.base}/{name}", dest)
                    if os.path.getsize(dest) == expected:
                        break
                except Exception as e:
                    status(error=f"download {name} try {attempt + 1}: {e}")
                time.sleep(10)
            if not os.path.exists(dest) or os.path.getsize(dest) != expected:
                raise RuntimeError(f"{name}: download did not match {expected} bytes")
        status(download_s=round(time.time() - t0, 1))

        import numpy as np
        import torch
        from sapling.longread_lesson import Ragged, heldout_accuracy
        from sapling.model import SaplingCore, autocast_ctx
        from sapling.train import load_checkpoint, save_checkpoint
        if not torch.cuda.is_available():
            raise RuntimeError("the pilot requires a CUDA H100")
        status(gpu=torch.cuda.get_device_name(0))
        old, tok, old_ck = load_checkpoint(device="cpu", path=os.path.join(a.bundle, core_name))
        tokenizer_source = {"generation": int(old_ck.get("generation") or 0),
                            "vocab_size": len(tok.itos), "source_config": old_ck.get("config")}
        del old, old_ck
        torch.manual_seed(SEED)
        model = SaplingCore(len(tok.itos), **CONFIG).to("cuda")
        params = model.n_params()
        if not 300_000_000 <= params <= 400_000_000:
            raise RuntimeError(f"fresh core is not approximately 350M parameters: {params:,}")
        train_model = model if a.no_compile else torch.compile(model)
        batch, block, micro = 64, model.block_size, a.micro
        if batch % micro:
            raise RuntimeError("micro-batch must divide batch 64")
        opt = torch.optim.AdamW([
            {"params": [p for p in model.parameters() if p.dim() >= 2], "weight_decay": 0.1},
            {"params": [p for p in model.parameters() if p.dim() < 2], "weight_decay": 0.0},
        ], lr=3e-4, betas=(0.9, 0.95), fused=True)
        lesson = Ragged(os.path.join(a.bundle, "lesson.npz"))
        talk = Ragged(os.path.join(a.bundle, "talk.npz"))
        school = torch.from_numpy(np.load(os.path.join(a.bundle, "school.npy")).astype(np.int64))
        reading = np.load(os.path.join(a.bundle, "reading.npy"), mmap_mode="r")
        if len(reading) <= block + 1 or len(school) <= block + 1:
            raise RuntimeError("reading.npy and school.npy must each exceed one context block")
        with open(os.path.join(a.bundle, "test.jsonl"), encoding="utf-8") as f:
            test = [json.loads(line) for line in f]
        shares = {"lesson": 16, "reading": 32, "talk": 8, "school": 8}
        target = a.target_tokens
        total_steps = int(math.ceil(target / (batch * block)))
        rng = random.Random(SEED)
        tokens = step = 0
        losses = []
        evals = []
        started = t_train = time.time()
        log_path = os.path.join(a.out, "log.jsonl")

        def log(row):
            row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **row}
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            status(row)

        before = heldout_accuracy(model, tok, test)
        evals.append({"step": 0, **before})
        log({"kind": "start", "fresh": True, "gpu": torch.cuda.get_device_name(0), "params": params,
             "config": model.config(), "target_tokens": target, "tokens_per_param": round(target / params, 3),
             "total_steps": total_steps, "batch": batch, "micro": micro, "compiled": not a.no_compile,
             "shares": shares, "minutes_cap": a.minutes, "lesson_examples": len(lesson),
             "talk_examples": len(talk), "reading_tokens_available": int(len(reading)),
             "school_tokens_available": int(len(school)), "heldout_before": before})
        ctx = autocast_ctx("cuda")
        while step < total_steps:
            xs, ys = [], []
            x, y = lesson.batch([rng.randrange(len(lesson)) for _ in range(shares["lesson"])], block, tok.stoi["<pad>"])
            xs.append(x)
            ys.append(y)
            r = torch.from_numpy(reading_batch(reading, rng, shares["reading"]))
            xs.append(r[:, :-1])
            ys.append(r[:, 1:])
            x, y = talk.batch([rng.randrange(len(talk)) for _ in range(shares["talk"])], block, tok.stoi["<pad>"])
            xs.append(x)
            ys.append(y)
            starts = [rng.randrange(0, len(school) - block) for _ in range(shares["school"])]
            s = torch.stack([school[o:o + block + 1] for o in starts])
            xs.append(s[:, :-1])
            ys.append(s[:, 1:])
            X = torch.cat(xs).to("cuda", non_blocking=True)
            Y = torch.cat(ys).to("cuda", non_blocking=True)
            lr = 3e-4 * (step + 1) / 300 if step < 300 else 3e-5 + 0.5 * (3e-4 - 3e-5) * (
                1 + math.cos(math.pi * min(1.0, (step - 300) / max(1, total_steps - 300))))
            for group in opt.param_groups:
                group["lr"] = lr
            opt.zero_grad(set_to_none=True)
            perm = torch.randperm(len(X), device="cuda")
            total_loss = 0.0
            for i in range(0, len(X), micro):
                j = perm[i:i + micro]
                with ctx:
                    _, loss = train_model(X[j], Y[j])
                (loss * len(j) / len(X)).backward()
                total_loss += float(loss.detach()) * len(j) / len(X)
            if not math.isfinite(total_loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(total_loss)
            step += 1
            tokens += len(X) * block
            if step % 200 == 0 or step == total_steps:
                elapsed = time.time() - t_train
                rate = tokens / elapsed
                log({"kind": "step", "step": step, "of": total_steps, "tokens": tokens,
                     "tokens_per_param": round(tokens / params, 3),
                     "loss": round(sum(losses[-200:]) / min(200, len(losses)), 4), "lr": round(lr, 7),
                     "tok_per_s": round(rate), "eta_minutes": round((target - tokens) / rate / 60, 1)})
            if time.time() - started > a.minutes * 60 - 180:
                log({"kind": "time budget reached", "step": step, "tokens": tokens})
                break
        after = heldout_accuracy(model, tok, test, n=min(1000, len(test)))
        elapsed = time.time() - t_train
        run = {"kind": "fresh 350M bounded pilot", "fresh": True, "steps": step, "tokens": tokens,
               "target_tokens": target, "target_reached": tokens >= target,
               "tokens_per_param": round(tokens / params, 3), "params": params, "config": model.config(),
               "batch": batch, "micro": micro, "compiled": not a.no_compile, "shares": shares, "lr": 3e-4,
               "minutes": round((time.time() - started) / 60, 1), "tok_per_s": round(tokens / elapsed),
               "heldout_before": before, "heldout_after": after, "evals": evals,
               "loss_last": round(sum(losses[-200:]) / max(1, len(losses[-200:])), 4)}
        benchmark = {"gpu": torch.cuda.get_device_name(0), "params": params, "config": model.config(),
                     "micro": micro, "batch": batch, "compiled": not a.no_compile,
                     "training_seconds": round(elapsed, 1), "tok_per_s": run["tok_per_s"],
                     "target_tokens": target, "tokens": tokens, "target_reached": run["target_reached"],
                     "torch": torch.__version__}
        with open(os.path.join(a.out, "benchmark.json"), "w", encoding="utf-8") as f:
            json.dump(benchmark, f, indent=1)
        model.eval()
        metadata = {"seed": SEED, "generation": 0, "core_version": 5.0, "staged": True,
                    "label": "v5 fresh ~350M clean 800M-token pilot",
                    "note": "Fresh weights; only the tokenizer was copied from the current core. Never auto-installed.",
                    "tokenizer_source": tokenizer_source, "reading": run, "longread": run}
        save_checkpoint(model, tok, metadata, path=os.path.join(a.out, "core_staged.pt"))
        log({"kind": "done", **{k: run[k] for k in ("steps", "tokens", "target_reached", "tokens_per_param", "minutes", "loss_last")},
             "heldout_after": after})
        status(state="done", run={k: run[k] for k in ("steps", "tokens", "target_reached", "tokens_per_param", "minutes", "heldout_after")},
               benchmark=benchmark)
    except Exception as e:
        status(state="failed", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-3000:])
        raise


if __name__ == "__main__":
    main()
