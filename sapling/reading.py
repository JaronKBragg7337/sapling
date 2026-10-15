"""The reading stage - v3.2 "much more reading" (Jaron 10/14 9:40 PM, lab-style order: more reading first,
watch for the copying ability to switch on, then tutor lessons, then his documents).

Why: a rough compute-optimal guide is ~20 tokens read per parameter; the 17.6M core had read a few million.
Its miss on E7L (copying a sentence out of a cold passage, 0/40) looks like the missing "induction head"
circuit, which in the research literature switches on suddenly once enough text has been read. So: read a
lot, and measure that switch while it happens (E10, the induction probe, and E7L at every checkpoint).

What this is NOT: a new core. It is the same weights (grown in place, DESIGN-CHOICES D17/D20), the same
vocabulary (17,660; words it merely reads are spelled from pieces, D16), the same memory.db and hash chain
(school never writes the ledger; the swap adds ONE `growth` row, D15), the generation keeps counting.

Text (license-checked live at fetch, tools/fetch_reading.py -> data/reading/manifest.json):
- TinyStories full GPT-4 train split (CDLA-Sharing-1.0): ~2.7M children's stories
- Simple English Wikipedia (CC BY-SA 3.0 + GFDL, attribution in README): ~242k articles
- Project Gutenberg books, public domain in the US (each header checked; header/footer/licence stripped)

The text is tokenized ONCE into uint16 shards (data/reading/shards/, gitignored) and read in order, one
pass: every batch = 25 story sequences + 4 adult-text sequences + 3 rehearsal sequences (its whole school,
levels 1-3, and its own record) so it does not forget what it already knows. GPU sharing: when blender.exe
is running (a Phobos render) or results/reading/pause exists, it saves, frees the card, waits, and resumes -
every pause is a log row. Every ~20M tokens read: a probe (E1-E10, no ledger writes) + a checkpoint, so the
switch is visible as a chart in the control room.

    python tools/fetch_reading.py                 # the text (once)
    python -m sapling.reading prepare             # tokenize -> shards (once, CPU, ~15 min)
    python -m sapling.reading stage --tokens 400e6   # GPU; resumable; -> results/core_staged.pt
    python -m sapling.heartbeat stop && python -m sapling.grow swap && python -m sapling.heartbeat start
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import time
from array import array

import numpy as np
import torch

from .curriculum3 import DOCS_HOLDOUT, chunk_passages, doc_text, gutenberg_title, strip_gutenberg
from .exams import run_exams
from .model import autocast_ctx
from .text import normalize
from .train import CHECKPOINT, RESULTS_DIR, Tokenizer, get_batch, load_checkpoint, save_checkpoint, sentences_to_tensor

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "reading")
RAW_DIR = os.path.join(DATA_DIR, "raw")
SHARD_DIR = os.path.join(DATA_DIR, "shards")
OUT_DIR = os.path.join(RESULTS_DIR, "reading")
LOG = os.path.join(OUT_DIR, "log.jsonl")
STATE = os.path.join(OUT_DIR, "state.json")
LATEST = os.path.join(OUT_DIR, "latest.pt")
PAUSE_FILE = os.path.join(OUT_DIR, "pause")
STAGED = os.path.join(RESULTS_DIR, "core_staged.pt")
SHARD_TOKENS = 50_000_000
SOURCES = ("stories", "adult")
# batch composition (of 32): stories / adult text / rehearsal of everything it already knows
SHARES = {"stories": 25, "adult": 4, "rehearsal": 3}
NOTE = ("Reading stage text: TinyStories GPT-4 train split (CDLA-Sharing-1.0); Simple English Wikipedia "
        "(wikimedia/wikipedia 20231101.simple, CC BY-SA 3.0 + GFDL, Wikipedia contributors); Project Gutenberg "
        "public-domain books (headers checked at fetch, licence text stripped). The held-out essay pg1022 is never read.")


# ---------------------------------------------------------------------------
# the text, document by document (already normalised = spaced, lower case)
# ---------------------------------------------------------------------------

def iter_stories(path: str):
    """The TinyStories train file, one story at a time, streamed (2.2 GB)."""
    buf = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if "<|endoftext|>" in line:
                before, _, after = line.partition("<|endoftext|>")
                buf.append(before)
                s = "".join(buf).strip()
                if s:
                    yield normalize(s)
                buf = [after]
            else:
                buf.append(line)
    s = "".join(buf).strip()
    if s:
        yield normalize(s)


def iter_wiki(path: str):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["title", "text"])
        for title, text in zip(t.column("title").to_pylist(), t.column("text").to_pylist()):
            if text and len(text.split()) >= 20:
                yield doc_text(f"{title}.\n\n{text}")


def iter_gutenberg(gdir: str):
    """Every kept book as passages (the reader's unit), the held-out essay never."""
    if not os.path.isdir(gdir):
        return
    for name in sorted(os.listdir(gdir)):
        if not name.endswith(".txt") or name.startswith(DOCS_HOLDOUT + "."):
            continue
        raw = open(os.path.join(gdir, name), encoding="utf-8", errors="replace").read()
        body = strip_gutenberg(raw) if "Project Gutenberg" in raw[:2000] else raw
        for p in chunk_passages(doc_text(body)):
            yield p


def iter_adult(raw_dir: str, wiki_share: float = 0.8):
    """Wikipedia articles and book passages interleaved by token count (wiki ~4:1), so a
    50M-token shard holds both and neither ends up last."""
    wiki = iter_wiki(os.path.join(raw_dir, "simplewiki.parquet")) if os.path.exists(os.path.join(raw_dir, "simplewiki.parquet")) else iter(())
    books = iter_gutenberg(os.path.join(raw_dir, "gutenberg"))
    nw = nb = 0
    w_done = b_done = False
    while not (w_done and b_done):
        want_wiki = (not w_done) and (b_done or nw * (1 - wiki_share) <= nb * wiki_share)
        it = wiki if want_wiki else books
        try:
            d = next(it)
        except StopIteration:
            if want_wiki:
                w_done = True
            else:
                b_done = True
            continue
        if want_wiki:
            nw += len(d.split())
        else:
            nb += len(d.split())
        yield d


# ---------------------------------------------------------------------------
# shards
# ---------------------------------------------------------------------------

def vocab_sha(tok: Tokenizer, n: int | None = None) -> str:
    """Hash of the first n tokens (all by default). The heartbeat only ever APPENDS tokens (ids never move),
    so shards made under 17,660 tokens are still exact under 17,661: compare the prefix."""
    return hashlib.sha256("\n".join(tok.itos[:n] if n else tok.itos).encode()).hexdigest()[:16]


class FastEncoder:
    """tok.encode with a word -> ids cache (the same ids as Tokenizer, ~10x faster on 500M tokens)."""

    def __init__(self, tok: Tokenizer):
        self.tok = tok
        self.cache: dict[str, tuple[int, ...]] = {}
        self.bos, self.eos = tok.stoi["<bos>"], tok.stoi["<eos>"]

    def ids(self, text: str) -> list[int]:
        out = [self.bos]
        c = self.cache
        for w in text.split():
            t = c.get(w)
            if t is None:
                t = c[w] = tuple(self.tok.stoi[p] for p in self.tok.tokens(w))
            out.extend(t)
        out.append(self.eos)
        return out


def prepare(tok: Tokenizer, raw_dir: str = RAW_DIR, shard_dir: str = SHARD_DIR, shard_tokens: int = SHARD_TOKENS,
            sources: dict | None = None, verbose: bool = True) -> dict:
    """Tokenize every source once into uint16 shards + index.json. `sources` = {name: iterator of
    normalised documents} (the default = stories + adult from raw_dir)."""
    assert len(tok.itos) < 65535, "uint16 shards need a vocabulary under 65,535"
    os.makedirs(shard_dir, exist_ok=True)
    enc = FastEncoder(tok)
    if sources is None:
        sources = {"stories": iter_stories(os.path.join(raw_dir, "TinyStoriesV2-GPT4-train.txt")),
                   "adult": iter_adult(raw_dir)}
    index = {"vocab_size": len(tok.itos), "vocab_sha": vocab_sha(tok), "shard_tokens": shard_tokens,
             "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "sources": {}}
    idx_path = os.path.join(shard_dir, "index.json")
    if os.path.exists(idx_path):                      # re-tokenize one source, keep the others' shards
        try:
            old = json.load(open(idx_path, encoding="utf-8"))
            if old.get("vocab_sha") == vocab_sha(tok, int(old.get("vocab_size") or 0)):
                index["sources"] = {k: v for k, v in old["sources"].items() if k not in sources}
        except Exception:
            pass
    for name, it in sources.items():
        t0 = time.time()
        buf, shards, n_tok, n_doc, k = array("H"), [], 0, 0, 0

        def flush():
            nonlocal buf, k
            if not len(buf):
                return
            fn = f"{name}_{k:03d}.npy"
            np.save(os.path.join(shard_dir, fn), np.frombuffer(buf.tobytes(), dtype=np.uint16))
            shards.append({"file": fn, "tokens": len(buf)})
            if verbose:
                print(f"[reading] {fn}: {len(buf):,} tokens ({n_doc:,} docs so far, {time.time() - t0:.0f}s)", flush=True)
            buf, k = array("H"), k + 1

        for doc in it:
            ids = enc.ids(doc)
            buf.extend(ids)
            n_tok += len(ids); n_doc += 1
            if len(buf) >= shard_tokens:
                flush()
        flush()
        index["sources"][name] = {"shards": shards, "tokens": n_tok, "docs": n_doc, "seconds": round(time.time() - t0, 1)}
    json.dump(index, open(idx_path, "w", encoding="utf-8"), indent=1)
    return index


class Stream:
    """One source's shards, read in order in fixed windows; wraps to a new epoch at the end."""

    def __init__(self, shard_dir: str, shards: list[dict], block: int, shard: int = 0, offset: int = 0, epoch: int = 0):
        self.dir, self.shards, self.block = shard_dir, shards, block
        self.shard, self.offset, self.epoch = shard, offset, epoch
        self._arr = None
        self._loaded = -1

    def _data(self):
        if self._loaded != self.shard:
            self._arr = np.load(os.path.join(self.dir, self.shards[self.shard]["file"]), mmap_mode="r")
            self._loaded = self.shard
        return self._arr

    def next(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        while len(xs) < n:
            a = self._data()
            if self.offset + self.block + 1 > len(a):
                self.shard, self.offset = (self.shard + 1) % len(self.shards), 0
                if self.shard == 0:
                    self.epoch += 1
                continue
            w = torch.from_numpy(a[self.offset: self.offset + self.block + 1].astype(np.int64))
            xs.append(w[:-1]); ys.append(w[1:])
            self.offset += self.block
        return torch.stack(xs), torch.stack(ys)

    def state(self) -> dict:
        return {"shard": self.shard, "offset": self.offset, "epoch": self.epoch}


# ---------------------------------------------------------------------------
# GPU sharing
# ---------------------------------------------------------------------------

def gpu_busy(pause_file: str = PAUSE_FILE) -> str | None:
    """Why training should hold: 'blender' (a render is running) or 'pause file' (the owner's hand)."""
    if os.path.exists(pause_file):
        return "pause file"
    try:
        import psutil
        for p in psutil.process_iter(["name"]):
            if (p.info.get("name") or "").lower() == "blender.exe":
                return "blender"
        return None
    except Exception:
        pass
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", "Get-Process blender -ErrorAction SilentlyContinue"],
                             capture_output=True, text=True, timeout=20).stdout
        return "blender" if "blender" in out.lower() else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# the stage
# ---------------------------------------------------------------------------

def _log(row: dict, path: str = LOG):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **row}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def _lr_at(step: int, total_steps: int, lr: float, min_lr: float, warmup: int) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * p))


def exam_scores(card: dict) -> dict:
    return {e["exam"]: {"score": e["score"], "accuracy": e["accuracy"]} for e in card["exams"]}


def rehearsal_sentences(seed: int, store=None, verbose: bool = True) -> list[str]:
    """Everything it already knows: the whole school (levels 1-3, the 'all' mix) + its own record."""
    from .grow import LESSONS, LESSONS2, LESSONS3, build_groups, stage_mix
    rng = random.Random(seed + 11)
    groups, _, _, _ = build_groups(seed, store, stories=True, max_stories=20000)
    order = [n for n in LESSONS + LESSONS2 + LESSONS3 + ["record"] if groups.get(n)] + ["all"]
    mix = stage_mix(groups, order, len(order) - 1, rng, earlier=order[:-1])
    if verbose:
        print(f"[reading] rehearsal: {len(mix):,} sentences from {len(order) - 1} lessons + record", flush=True)
    return mix


def stage(target_tokens: float = 400e6, batch: int = 32, shares: dict | None = None, lr: float = 3e-4,
          min_lr: float = 3e-5, warmup: int = 300, probe_every: float = 20e6, seed: int = 7337,
          device: str | None = None, old_path: str = CHECKPOINT, staged: str = STAGED, out_dir: str = OUT_DIR,
          shard_dir: str = SHARD_DIR, rehearsal: list[str] | None = None, store=None, resume: bool = True,
          wait_for_gpu: bool = True, poll_s: float = 60.0, step_log_every: int = 100, verbose: bool = True,
          label: str = "much more reading", to_version: float = 3.2, keep_probe_checkpoints: bool = True,
          busy_fn=None) -> dict:
    """Read `target_tokens` of new text (one pass, in order) into the live core; probe + checkpoint every
    `probe_every` tokens; pause while the GPU is busy; write the staged core for `grow swap`."""
    t_all = time.time()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    shares = {**SHARES, **(shares or {})}
    assert sum(shares.values()) == batch, f"shares {shares} must add up to the batch {batch}"
    os.makedirs(out_dir, exist_ok=True)
    log_path, state_path, latest = os.path.join(out_dir, "log.jsonl"), os.path.join(out_dir, "state.json"), os.path.join(out_dir, "latest.pt")
    pause_file = os.path.join(out_dir, "pause")
    busy = busy_fn or (lambda: gpu_busy(pause_file))      # tests inject their own (a real render must not stall them)
    index = json.load(open(os.path.join(shard_dir, "index.json"), encoding="utf-8"))
    n_new_per_step = (batch - shares["rehearsal"])
    rng = random.Random(seed + 2)
    torch.manual_seed(seed + 2)

    # the core: resume from latest.pt (model + optimizer) or start from the live checkpoint
    st = {}
    if resume and os.path.exists(state_path) and os.path.exists(latest):
        st = json.load(open(state_path, encoding="utf-8"))
        model, tok, ck = load_checkpoint(device=device, path=latest)
        if verbose:
            print(f"[reading] resuming at step {st['step']} ({st['tokens_new'] / 1e6:.1f}M tokens read)", flush=True)
    else:
        model, tok, ck = load_checkpoint(device=device, path=old_path)
        st = {"step": 0, "tokens_new": 0, "tokens_total": 0, "streams": {}, "pauses": [], "probes": [],
              "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds_training": 0.0,
              "from": {"generation": int(ck.get("generation", 0)), "params": model.n_params(), "vocab_size": len(tok.itos),
                       "core_version": float(ck.get("core_version", 3)), "config": model.config()}}
        _log({"kind": "start", "target_tokens": target_tokens, "batch": batch, "shares": shares, "lr": lr, "min_lr": min_lr,
              "warmup": warmup, "probe_every": probe_every, "device": device, "from": st["from"],
              "sources": {k: {"tokens": v["tokens"], "docs": v["docs"], "shards": len(v["shards"])} for k, v in index["sources"].items()}}, log_path)
    if index["vocab_sha"] != vocab_sha(tok, int(index["vocab_size"])):
        raise RuntimeError("shards were tokenized with a different vocabulary - run `prepare` again")
    from_info = st["from"]
    block = model.block_size
    total_steps = int(math.ceil(target_tokens / (n_new_per_step * block)))
    streams = {name: Stream(shard_dir, index["sources"][name]["shards"], block, **st["streams"].get(name, {}))
               for name in SOURCES if index["sources"].get(name, {}).get("shards")}
    if rehearsal is None:
        rehearsal = rehearsal_sentences(seed, store, verbose=verbose)
    reh = sentences_to_tensor(rehearsal, tok, block)
    if len(reh) < block + 2:
        reh = torch.cat([reh] * (block * 2 // max(1, len(reh)) + 2))

    def make_opt(m):
        o = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
        opt_path = latest + ".opt"
        if resume and os.path.exists(opt_path) and st["step"] > 0:
            try:
                o.load_state_dict(torch.load(opt_path, map_location=device, weights_only=False))
            except Exception as e:
                _log({"kind": "note", "note": f"optimizer state not loaded: {e}"}, log_path)
        return o

    def save_latest(m, o):
        meta = {"seed": seed, "generation": from_info["generation"], "core_version": to_version, "staged": True,
                "reading_step": st["step"], "reading_tokens": st["tokens_new"], "corpus_sha": ck.get("corpus_sha")}
        save_checkpoint(m, tok, meta, path=latest)
        torch.save(o.state_dict(), latest + ".opt")
        st["streams"] = {k: s.state() for k, s in streams.items()}
        json.dump(st, open(state_path, "w", encoding="utf-8"), indent=1)

    def probe(m, why: str):
        m.eval()
        t0 = time.time()
        card = run_exams(seed=seed, verbose=False, model=m, tok=tok, meta={"generation": from_info["generation"], "note": "reading probe"},
                         store=None, write=False, trigger=f"reading probe {st['tokens_new'] / 1e6:.0f}M")
        scores = exam_scores(card)
        row = {"kind": "probe", "why": why, "step": st["step"], "tokens_new": st["tokens_new"], "tokens_total": st["tokens_total"],
               "tokens_m": round(st["tokens_new"] / 1e6, 1), "loss": st.get("loss_recent"), "lr": st.get("lr"),
               "seconds_training": round(st["seconds_training"], 1), "exams": scores,
               "writing": [s["text"] for s in next((e["detail"] for e in card["exams"] if e["exam"] == "E8_writing"), [])],
               "probe_seconds": round(time.time() - t0, 1)}
        _log(row, log_path)
        st["probes"].append({k: row[k] for k in ("ts", "step", "tokens_m", "loss") if k in row} | {"tokens_m": row["tokens_m"], "step": row["step"], "loss": row["loss"],
                            "E10": scores.get("E10_induction", {}).get("accuracy"), "E7L": scores.get("E7L_longread", {}).get("accuracy"),
                            "E7": scores.get("E7_reading", {}).get("accuracy"), "E1": scores.get("E1_composition", {}).get("accuracy"),
                            "E6": scores.get("E6_grammar", {}).get("accuracy")})
        if keep_probe_checkpoints:
            save_checkpoint(m, tok, {"seed": seed, "generation": from_info["generation"], "core_version": to_version, "staged": True,
                                     "reading_step": st["step"], "reading_tokens": st["tokens_new"]},
                            path=os.path.join(out_dir, f"probe_{int(st['tokens_new'] / 1e6):04d}M.pt"))
        if verbose:
            e = scores
            print(f"[reading] probe @ {st['tokens_new'] / 1e6:.0f}M tokens: E10 {e.get('E10_induction', {}).get('score')} "
                  f"E7L {e.get('E7L_longread', {}).get('score')} E7 {e.get('E7_reading', {}).get('score')} "
                  f"E1 {e.get('E1_composition', {}).get('score')} E6 {e.get('E6_grammar', {}).get('score')} ({row['probe_seconds']}s)", flush=True)
        m.train()
        return row

    def wait_gpu(m, o):
        """Blender (or the owner's pause file) wants the card: save, free it, wait, reload."""
        why = busy()
        if not why:
            return m, o
        save_latest(m, o)
        t0 = time.time()
        if verbose:
            print(f"[reading] pause: {why} - saved, freeing the GPU, polling every {poll_s:.0f}s", flush=True)
        del o
        m.cpu(); del m
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        clear = 0
        while clear < 2:
            time.sleep(poll_s)
            clear = clear + 1 if not busy() else 0
        m, _, _ = load_checkpoint(device=device, path=latest)
        m.train()
        o = make_opt(m)
        row = {"why": why, "seconds": round(time.time() - t0, 1), "at_step": st["step"], "tokens_new": st["tokens_new"]}
        st["pauses"].append(row)
        _log({"kind": "pause", **row}, log_path)
        if verbose:
            print(f"[reading] resumed after {row['seconds']:.0f}s", flush=True)
        return m, o

    model.train()
    opt = make_opt(model)
    if wait_for_gpu:
        model, opt = wait_gpu(model, opt)
    ctx = autocast_ctx(device)
    losses: list[float] = []
    next_probe = (int(st["tokens_new"] // probe_every) + 1) * probe_every if probe_every > 0 else float("inf")
    if st["step"] == 0 and not st["probes"]:
        probe(model, "before")
    t_seg = time.time()
    while st["tokens_new"] < target_tokens:
        if wait_for_gpu and st["step"] % 20 == 0 and busy():
            st["seconds_training"] += time.time() - t_seg
            model, opt = wait_gpu(model, opt)
            t_seg = time.time()
        xs, ys = [], []
        for name, s in streams.items():
            x, y = s.next(shares[name]); xs.append(x); ys.append(y)
        x, y = get_batch(reh, block, shares["rehearsal"], rng)
        xs.append(x); ys.append(y)
        x, y = torch.cat(xs).to(device), torch.cat(ys).to(device)
        cur_lr = _lr_at(st["step"], total_steps, lr, min_lr, warmup)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        try:
            with ctx:
                _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        except torch.cuda.OutOfMemoryError if device.startswith("cuda") else MemoryError:
            # the heartbeat or a render took the room: free what we can, wait, try the same batch again (data)
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            _log({"kind": "oom", "step": st["step"], "tokens_new": st["tokens_new"]}, log_path)
            if verbose:
                print("[reading] CUDA out of memory - waiting 30s and retrying", flush=True)
            time.sleep(30)
            continue
        losses.append(loss.item())
        st["step"] += 1
        st["tokens_new"] += n_new_per_step * block
        st["tokens_total"] += batch * block
        st["lr"] = round(cur_lr, 7)
        if st["step"] % step_log_every == 0:
            recent = sum(losses[-step_log_every:]) / min(step_log_every, len(losses))
            st["loss_recent"] = round(recent, 4)
            st["seconds_training"] += time.time() - t_seg; t_seg = time.time()
            rate = n_new_per_step * block * step_log_every / max(1e-6, time.time() - st.get("_t_last", time.time() - 1)) if "_t_last" in st else None
            st["_t_last"] = time.time()
            _log({"kind": "step", "step": st["step"], "tokens_new": st["tokens_new"], "tokens_m": round(st["tokens_new"] / 1e6, 1),
                  "loss": st["loss_recent"], "lr": st["lr"], "tok_per_s": round(rate) if rate else None,
                  "streams": {k: s.state() for k, s in streams.items()}}, log_path)
            if verbose:
                print(f"[reading] step {st['step']} {st['tokens_new'] / 1e6:.1f}M/{target_tokens / 1e6:.0f}M loss {recent:.3f} lr {cur_lr:.2e}"
                      f"{' ' + str(round(rate / 1000)) + 'k tok/s' if rate else ''}", flush=True)
        if st["tokens_new"] >= next_probe:
            st["seconds_training"] += time.time() - t_seg
            probe(model, "checkpoint")
            save_latest(model, opt)
            next_probe += probe_every
            t_seg = time.time()
    st["seconds_training"] += time.time() - t_seg
    if not st["probes"] or st["probes"][-1]["step"] != st["step"]:
        probe(model, "end")
    save_latest(model, opt)
    model.eval()
    run = {"kind": "reading stage", "label": label, "started": st["started"], "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "steps": st["step"], "tokens_new": st["tokens_new"], "tokens_total": st["tokens_total"], "batch": batch, "shares": shares,
           "lr": lr, "min_lr": min_lr, "warmup": warmup, "seconds_training": round(st["seconds_training"], 1),
           "seconds_wall_this_call": round(time.time() - t_all, 1), "pauses": st["pauses"], "probes": st["probes"],
           "streams": {k: s.state() for k, s in streams.items()}, "loss_last": st.get("loss_recent"),
           "sources": {k: {"tokens": v["tokens"], "docs": v["docs"]} for k, v in index["sources"].items()}, "note": NOTE}
    meta = {"seed": seed, "corpus_sha": ck.get("corpus_sha"), "generation": from_info["generation"] + 1, "core_version": to_version,
            "staged": True, "label": label,
            "note": f"growth: core v{from_info['core_version']:g} -> v{to_version:g} (in place - {label}); grown from gen {from_info['generation']}",
            "grown_from": from_info, "story_stats": ck.get("story_stats"), "docs_stats": ck.get("docs_stats"), "school": ck.get("school"),
            "reading": run}
    save_checkpoint(model, tok, meta, path=staged)
    _log({"kind": "done", **{k: run[k] for k in ("steps", "tokens_new", "tokens_total", "seconds_training")}, "staged": staged}, log_path)
    if verbose:
        print(f"[reading] done: {run['tokens_new'] / 1e6:.1f}M new tokens in {run['steps']} steps, "
              f"{run['seconds_training'] / 60:.0f} min of training, {len(run['pauses'])} pauses -> {staged}", flush=True)
    return {"model": model, "tok": tok, "meta": meta, "run": run, "staged": staged}


# ---------------------------------------------------------------------------
# what the control room shows
# ---------------------------------------------------------------------------

def summary(out_dir: str = OUT_DIR, max_rows: int = 60) -> dict | None:
    """The reading stage for the page: probe rows (tokens read vs E10 / E7L / E7 / E1 / E6), status, pauses."""
    log_path, state_path = os.path.join(out_dir, "log.jsonl"), os.path.join(out_dir, "state.json")
    if not os.path.exists(log_path):
        return None
    rows, start, last_step, pauses, done = [], None, None, [], None
    try:
        for line in open(log_path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            k = r.get("kind")
            if k == "probe":
                e = r.get("exams", {})
                rows.append({"ts": r.get("ts"), "tokens_m": r.get("tokens_m"), "loss": r.get("loss"), "step": r.get("step"),
                             "E10": e.get("E10_induction", {}).get("accuracy"), "E10s": e.get("E10_induction", {}).get("score"),
                             "E7L": e.get("E7L_longread", {}).get("accuracy"), "E7Ls": e.get("E7L_longread", {}).get("score"),
                             "E7": e.get("E7_reading", {}).get("accuracy"), "E1": e.get("E1_composition", {}).get("accuracy"),
                             "E6": e.get("E6_grammar", {}).get("accuracy"), "E1b": e.get("E1b_bridge", {}).get("accuracy")})
            elif k == "start":
                start = {"ts": r.get("ts"), "target_tokens": r.get("target_tokens"), "from": r.get("from"), "sources": r.get("sources")}
            elif k == "step":
                last_step = {"ts": r.get("ts"), "tokens_m": r.get("tokens_m"), "loss": r.get("loss"), "tok_per_s": r.get("tok_per_s"), "step": r.get("step")}
            elif k == "pause":
                pauses.append({k2: r.get(k2) for k2 in ("ts", "why", "seconds", "tokens_new")})
            elif k == "done":
                done = {"ts": r.get("ts"), "tokens_new": r.get("tokens_new"), "steps": r.get("steps"), "seconds_training": r.get("seconds_training")}
    except Exception:
        return None
    paused = None
    if os.path.exists(os.path.join(out_dir, "pause")):
        paused = "pause file"
    return {"start": start, "last_step": last_step, "probes": rows[-max_rows:], "pauses": pauses[-10:], "done": done,
            "paused": paused, "note": NOTE}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "stage", "summary"])
    ap.add_argument("--tokens", type=float, default=400e6)
    ap.add_argument("--probe-every", type=float, default=20e6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=7337)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--no-wait", action="store_true", help="do not pause for blender / the pause file")
    ap.add_argument("--only", choices=list(SOURCES), help="prepare: re-tokenize one source, keep the other's shards")
    args = ap.parse_args(argv)
    if args.cmd == "prepare":
        _, tok, _ = load_checkpoint(device="cpu")
        src = None
        if args.only == "stories":
            src = {"stories": iter_stories(os.path.join(RAW_DIR, "TinyStoriesV2-GPT4-train.txt"))}
        elif args.only == "adult":
            src = {"adult": iter_adult(RAW_DIR)}
        idx = prepare(tok, sources=src)
        print(json.dumps({k: {"tokens": v["tokens"], "docs": v["docs"], "shards": len(v["shards"])} for k, v in idx["sources"].items()}, indent=1))
    elif args.cmd == "stage":
        stage(target_tokens=args.tokens, probe_every=args.probe_every, lr=args.lr, seed=args.seed,
              resume=not args.no_resume, wait_for_gpu=not args.no_wait)
    else:
        print(json.dumps(summary(), indent=1, default=str))


if __name__ == "__main__":
    main()
