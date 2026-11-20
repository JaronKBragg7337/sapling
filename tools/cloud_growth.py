"""Benchmark or train the fresh growth core on one rented GPU."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import traceback
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["bench", "train"])
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", default="/workspace/out")
    ap.add_argument("--bundle", default="/workspace/growth_bundle")
    ap.add_argument("--bench-seconds", type=float, default=45)
    ap.add_argument("--minutes", type=float, default=180)
    ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--compile", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.bundle, exist_ok=True)
    st_path = os.path.join(a.out, "status.json")
    state = {"state": "downloading", "mode": a.mode, "since": time.time()}

    def status(row=None, **kw):
        if row:
            state["last"] = row
        state.update(kw)
        state["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        tmp = st_path + ".tmp"
        json.dump(state, open(tmp, "w"), indent=1)
        os.replace(tmp, st_path)

    try:
        status()
        meta = os.path.join(a.bundle, "bundle.json")
        urllib.request.urlretrieve(f"{a.base}/bundle.json", meta)
        info = json.load(open(meta, encoding="utf-8"))
        names = [info["core_file"]] if a.mode == "bench" else list(info["files"])
        t0 = time.time()
        for name in names:
            dest = os.path.join(a.bundle, name)
            if name == "reading_index.json" or name == info["core_file"] or name in ("lesson.npz", "talk.npz", "school.npy", "test.jsonl") or name.endswith(".npy"):
                expected = info["files"][name]
                url = info.get("urls", {}).get(name, f"{a.base}/{name}")
                for attempt in range(5):
                    try:
                        urllib.request.urlretrieve(url, dest)
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
        from sapling.model import SaplingCore, autocast_ctx
        from sapling.train import load_checkpoint
        status(gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
        old, tok, old_ck = load_checkpoint(device="cpu", path=os.path.join(a.bundle, info["core_file"]))
        config = dict(d_model=768, n_head=12, n_layer=12, block_size=512, rope=True)
        device = "cuda"
        torch.manual_seed(7337)
        model = SaplingCore(len(tok.itos), **config).to(device)
        train_model = torch.compile(model) if a.compile else model
        params = model.n_params()
        batch, block, micro = 64, model.block_size, a.micro
        opt = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), fused=True)
        if a.mode == "bench":
            model.train()
            steps = tokens = 0
            started = time.time()
            t0 = None
            while True:
                opt.zero_grad(set_to_none=True)
                for _ in range(batch // micro):
                    x = torch.randint(0, len(tok.itos), (micro, block), device=device)
                    with autocast_ctx(device):
                        _, loss = train_model(x, x)
                    (loss / (batch // micro)).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                steps += 1
                if steps == 20:
                    torch.cuda.synchronize()
                    t0 = time.time()
                    tokens = 0
                elif t0 is not None:
                    tokens += batch * block
                    if time.time() - t0 >= a.bench_seconds:
                        break
                if time.time() - started > a.bench_seconds + 240:
                    raise RuntimeError("benchmark did not reach its measurement window")
            torch.cuda.synchronize()
            elapsed = time.time() - t0
            result = {"gpu": torch.cuda.get_device_name(0), "params": params, "config": model.config(),
                      "micro": micro, "batch": batch, "compiled": a.compile, "seconds": round(elapsed, 1),
                      "tok_per_s": round(tokens / elapsed), "target_tokens": int(math.ceil(params * 20)),
                      "hours_for_20x": round(params * 20 / (tokens / elapsed) / 3600, 3),
                      "torch": torch.__version__}
            json.dump(result, open(os.path.join(a.out, "benchmark.json"), "w"), indent=1)
            status(state="done", benchmark=result)
            print(json.dumps(result), flush=True)
            return

        from sapling.longread_lesson import NOTE, Ragged, heldout_accuracy
        from sapling.train import save_checkpoint
        del old
        model.train()
        model = model.to(device)
        opt = torch.optim.AdamW([
            {"params": [p for p in model.parameters() if p.dim() >= 2], "weight_decay": 0.1},
            {"params": [p for p in model.parameters() if p.dim() < 2], "weight_decay": 0.0},
        ], lr=6e-4, betas=(0.9, 0.95), fused=True)
        lesson = Ragged(os.path.join(a.bundle, "lesson.npz"))
        talk = Ragged(os.path.join(a.bundle, "talk.npz"))
        school = torch.from_numpy(np.load(os.path.join(a.bundle, "school.npy")).astype(np.int64))
        test = [json.loads(line) for line in open(os.path.join(a.bundle, "test.jsonl"), encoding="utf-8")]
        index = json.load(open(os.path.join(a.bundle, "reading_index.json"), encoding="utf-8"))
        source_names = ("stories", "adult", "web")
        shards = {name: [np.load(os.path.join(a.bundle, s["file"]), mmap_mode="r")
                         for s in index["sources"][name]["shards"]] for name in source_names}
        shares = {"lesson": 16, "stories": 12, "adult": 4, "web": 16, "talk": 8, "school": 8}
        assert sum(shares.values()) == batch
        target = int(math.ceil(params * 20))
        total_steps = int(math.ceil(target / (batch * block)))
        rng = random.Random(7337)
        losses = []
        tokens = 0
        step = 0
        started = time.time()
        t_train = time.time()
        next_eval = time.time() + 20 * 60
        next_save = time.time() + 30 * 60
        evals = []
        log_path = os.path.join(a.out, "log.jsonl")

        def log(row):
            row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **row}
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            status(row)

        def metadata(run):
            return {"seed": 7337, "corpus_sha": old_ck.get("corpus_sha"), "generation": int(old_ck.get("generation") or 0),
                    "core_version": 5.0, "staged": True,
                    "label": "v5 growth: fresh 98.6M core with long-reading interleaved from step one",
                    "note": f"growth: core v{float(old_ck.get('core_version') or 4):g} -> v5.0 (fresh d768/12L growth)",
                    "grown_from": {k: old_ck.get(k) for k in ("generation", "params", "core_version", "config")},
                    "story_stats": old_ck.get("story_stats"), "docs_stats": old_ck.get("docs_stats"),
                    "school": old_ck.get("school"), "talk": old_ck.get("talk"), "reading": run, "longread": run}

        before = heldout_accuracy(model, tok, test)
        evals.append({"step": 0, **before})
        log({"kind": "start", "device": device, "gpu": torch.cuda.get_device_name(0), "params": params,
             "config": model.config(), "target_tokens": target, "tokens_per_param": 20, "total_steps": total_steps,
             "batch": batch, "micro": micro, "shares": shares, "minutes_cap": a.minutes,
             "lesson_examples": len(lesson), "talk_examples": len(talk), "heldout_before": before,
             "reading_tokens_available": {k: index["sources"][k]["tokens"] for k in source_names}})
        ctx = autocast_ctx(device)
        while step < total_steps:
            xs, ys = [], []
            x, y = lesson.batch([rng.randrange(len(lesson)) for _ in range(shares["lesson"])], block, tok.stoi["<pad>"])
            xs.append(x); ys.append(y)
            for name in source_names:
                rows = []
                for _ in range(shares[name]):
                    arr = shards[name][rng.randrange(len(shards[name]))]
                    off = rng.randrange(0, len(arr) - block - 1)
                    rows.append(np.asarray(arr[off:off + block + 1]))
                w = torch.from_numpy(np.stack(rows).astype(np.int64))
                xs.append(w[:, :-1]); ys.append(w[:, 1:])
            x, y = talk.batch([rng.randrange(len(talk)) for _ in range(shares["talk"])], block, tok.stoi["<pad>"])
            xs.append(x); ys.append(y)
            starts = [rng.randrange(0, len(school) - block - 1) for _ in range(shares["school"])]
            s = torch.stack([school[o:o + block + 1] for o in starts])
            xs.append(s[:, :-1]); ys.append(s[:, 1:])
            X, Y = torch.cat(xs).to(device, non_blocking=True), torch.cat(ys).to(device, non_blocking=True)
            cur = 6e-4 * (step + 1) / 300 if step < 300 else 6e-5 + 0.5 * (6e-4 - 6e-5) * (
                1 + math.cos(math.pi * min(1.0, (step - 300) / max(1, total_steps - 300))))
            for g in opt.param_groups:
                g["lr"] = cur
            opt.zero_grad(set_to_none=True)
            perm = torch.randperm(len(X), device=device)
            total_loss = 0.0
            for i in range(0, len(X), micro):
                j = perm[i:i + micro]
                with ctx:
                    _, loss = train_model(X[j], Y[j])
                (loss * len(j) / len(X)).backward()
                total_loss += float(loss.detach()) * len(j) / len(X)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(total_loss)
            step += 1
            tokens += len(X) * block
            if step % 200 == 0:
                elapsed = time.time() - t_train
                rate = tokens / elapsed
                log({"kind": "step", "step": step, "of": total_steps, "tokens": tokens,
                     "tokens_per_param": round(tokens / params, 3), "loss": round(sum(losses[-200:]) / min(200, len(losses)), 4),
                     "lr": round(cur, 7), "tok_per_s": round(rate), "eta_minutes": round((target - tokens) / rate / 60, 1)})
            if time.time() >= next_eval:
                ev = heldout_accuracy(model, tok, test)
                evals.append({"step": step, **ev})
                log({"kind": "heldout", "step": step, **ev})
                next_eval = time.time() + 20 * 60
            if time.time() >= next_save:
                partial = {"steps": step, "tokens": tokens, "target_tokens": target, "target_reached": False}
                save_checkpoint(model, tok, metadata(partial), path=os.path.join(a.out, "latest.pt"))
                next_save = time.time() + 30 * 60
            if time.time() - started > a.minutes * 60 - 240:
                log({"kind": "time budget reached", "step": step, "tokens": tokens})
                break
        after = heldout_accuracy(model, tok, test, n=min(1000, len(test)))
        run = {"kind": "fresh growth with interleaved long-reading", "steps": step, "tokens": tokens,
               "target_tokens": target, "target_reached": tokens >= target, "tokens_per_param": round(tokens / params, 3),
               "params": params, "config": model.config(), "batch": batch, "micro": micro, "shares": shares,
               "lr": 6e-4, "minutes": round((time.time() - started) / 60, 1),
               "tok_per_s": round(tokens / (time.time() - t_train)), "heldout_before": before, "heldout_after": after,
               "evals": evals, "loss_last": round(sum(losses[-200:]) / max(1, len(losses[-200:])), 4), "note": NOTE}
        model.eval()
        save_checkpoint(model, tok, metadata(run), path=os.path.join(a.out, "core_staged.pt"))
        log({"kind": "done", **{k: run[k] for k in ("steps", "tokens", "target_reached", "tokens_per_param", "minutes", "loss_last")},
             "heldout_after": after})
        status(state="done", run={k: run[k] for k in ("steps", "tokens", "target_reached", "tokens_per_param", "minutes", "heldout_after")})
    except Exception as e:
        status(state="failed", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-3000:])
        raise


if __name__ == "__main__":
    main()
