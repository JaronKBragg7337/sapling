"""Conversation lessons - Stage B of the 10/15 teaching plan (Claude as teacher; Jaron: "better communication...
build it as an AI like any other AI first").

What it learns here is the shape of a conversation, in the format the chat box already uses:

    <bos> q : <what you said> a : <its answer> <eos> q : <next thing you said> a : <its answer> <eos> ...

A question that comes with a text keeps the reading frame it was taught at school:

    <bos> passage : <text> q : <question> a : <answer> <eos>

Loss is on its own answers only (and the <eos> that ends each one) - it learns to answer, not to imitate the person.
Every step also replays reading windows and its school, so the reading and the school stay (the v3.2.1 lesson).

Text (licences checked live 10/15, recorded in data/reading/manifest.json by tools/fetch_reading.py):
- OpenAssistant oasst1 (Apache-2.0): English conversation trees; each path through the best-ranked answers
- databricks-dolly-15k (CC BY-SA 3.0): instructions with answers, some with a passage
- its own lessons: who it is and what it is, from its real state (name, age, size, what it read) - several
  wordings of each question, one true answer

Cleaning (DESIGN-CHOICES D26): code blocks and non-English trees are left out (the tokenizer has no symbols for
code); an answer that refuses, or that calls itself another assistant ("as an ai language model", "open assistant"),
ends the conversation before that turn - that is another system's caution and another system's name, not Sapling's.

    python -m sapling.talk build                  # -> data/talk/lessons.jsonl (CPU, ~1 min)
    python -m sapling.talk train --core results/core_v4_staged.pt   # GPU -> results/core_v4_talk_staged.pt
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import re
import time

import torch

from .model import autocast_ctx
from .text import normalize
from .train import RESULTS_DIR, load_checkpoint, save_checkpoint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "reading", "raw")
TALK_DIR = os.path.join(ROOT, "data", "talk")
LESSONS = os.path.join(TALK_DIR, "lessons.jsonl")
OUT_DIR = os.path.join(RESULTS_DIR, "talk")
STAGED = os.path.join(RESULTS_DIR, "core_v4_talk_staged.pt")

DROP = re.compile(r"as an ai|language model|open ?assistant|openassistant|laion|i'm sorry, but|i am sorry, but|"
                  r"i cannot|i can't help|i can not|i'm not able to|i am not able to|chatgpt|openai", re.I)


def _clean(text: str) -> str | None:
    if "```" in text or text.count("    ") > 6:
        return None
    t = normalize(text)
    return t if len(t.split()) >= 1 else None


# ---------------------------------------------------------------------------
# sources -> conversations [{"passage": str|None, "turns": [[user, answer], ...], "src": ...}]
# ---------------------------------------------------------------------------

def oasst_conversations(path: str = os.path.join(RAW, "oasst1_ready.trees.jsonl.gz"), max_paths: int = 3) -> list[dict]:
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            root = t["prompt"]
            if root.get("lang") != "en":
                continue
            paths = []

            def walk(node, turns):
                # node is a prompter message; its answers are ranked (rank 0 = best)
                u = _clean(node["text"])
                if u is None:
                    if turns:
                        paths.append(turns)
                    return
                answers = sorted([r for r in node.get("replies", []) if r.get("rank") is not None and r["rank"] <= 1
                                  and not r.get("deleted")], key=lambda r: r["rank"])
                if not answers:
                    if turns:
                        paths.append(turns)
                    return
                for a in answers:
                    if len(paths) >= max_paths:
                        return
                    if DROP.search(a["text"]):
                        if turns:
                            paths.append(turns)
                        continue
                    ta = _clean(a["text"])
                    if ta is None:
                        if turns:
                            paths.append(turns)
                        continue
                    nxt = turns + [[u, ta]]
                    follow = [r for r in a.get("replies", []) if r.get("lang") == "en" and not r.get("deleted")]
                    if not follow:
                        paths.append(nxt)
                    else:
                        walk(follow[0], nxt)

            walk(root, [])
            seen = set()
            for p in paths:
                key = json.dumps(p)
                if p and key not in seen:
                    seen.add(key)
                    out.append({"passage": None, "turns": p, "src": "oasst1"})
    return out


def dolly_conversations(path: str = os.path.join(RAW, "databricks-dolly-15k.jsonl")) -> list[dict]:
    out = []
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        if DROP.search(d["response"]):
            continue
        q, a = _clean(d["instruction"]), _clean(d["response"])
        if not q or not a:
            continue
        ctx = _clean(d.get("context") or "") if d.get("context") else None
        out.append({"passage": ctx, "turns": [[q, a]], "src": "dolly15k"})
    return out


def self_lessons(facts: dict, rng: random.Random, copies: int = 6) -> list[dict]:
    """Who it is, from its real state. Several wordings of each question (the same knowledge met more than one way),
    one true answer each. `facts` comes from its own record at build time."""
    qa = [
        (["what is your name ?", "who are you ?", "what are you called ?", "tell me your name .", "hello ! who are you ?",
          "hi , who am i talking to ?"],
         f"i am sapling , a small ai that grows as it learns . {facts['maker']} built me ."),
        (["who made you ?", "who built you ?", "who created you ?", "where do you come from ?"],
         f"{facts['maker']} built me , from scratch . my brain , my memory and my heartbeat are my own , not borrowed from another ai ."),
        (["what are you ?", "are you an ai ?", "what kind of thing are you ?", "are you a person ?"],
         "i am an ai , a young one . i read , i remember what people teach me , and i practise while my heartbeat runs ."),
        (["how big is your brain ?", "how many parameters do you have ?", "how large is your core ?"],
         f"my core has about {facts['params_m']} million parameters . it can hold {facts['context']} tokens in mind at once ."),
        (["what have you read ?", "what did you learn from ?", "what do you read ?", "what books have you read ?"],
         f"i have read {facts['read']} . that was {facts['tokens']} tokens of text ."),
        (["how old are you ?", "when were you born ?", "when did you start ?"],
         f"my clock started on {facts['born']} . my heartbeat keeps counting from there ."),
        (["what can you do ?", "how can you help me ?", "what are you good at ?"],
         "i can talk with you , answer questions , tell short stories , and remember what you teach me . "
         "teach me a word with / teach and i will keep it ."),
        (["do you remember things ?", "can you remember what i tell you ?", "do you have a memory ?"],
         "yes . what you teach me goes into my memory first , and my heartbeat trains it into my brain later ."),
        (["what is a heartbeat for you ?", "what does your heartbeat do ?", "are you always running ?"],
         "my heartbeat is a loop that keeps going on its own . on each beat i rest , practise , explore or train what i learned ."),
        (["what would you like to learn next ?", "what do you want to learn ?"],
         "more about the world , and how to talk well with people . i would like to read more and to be taught ."),
    ]
    out = []
    for qs, a in qa:
        for _ in range(copies):
            for q in qs:
                out.append({"passage": None, "turns": [[q, a]], "src": "self"})
    rng.shuffle(out)
    return out


def self_facts(core_path: str | None = None) -> dict:
    """Its real state, read from its own files (nothing invented)."""
    from .memory import MemoryStore
    facts = {"maker": "jaron", "params_m": 60, "context": 512, "tokens": "more than a billion",
             "read": "children's stories , simple english wikipedia , 116 old books from project gutenberg , and educational web pages",
             "born": "october 14 , 2026"}
    try:
        st = MemoryStore()
        row = st.db.execute("SELECT ts FROM ledger WHERE kind='birth' ORDER BY id LIMIT 1").fetchone()
        if row:
            t = time.localtime(float(row[0]))
            facts["born"] = normalize(time.strftime("%B %d , %Y", t).replace(" 0", " "))
    except Exception:
        pass
    if core_path and os.path.exists(core_path):
        ck = torch.load(core_path, map_location="cpu", weights_only=False)
        n = sum(v.numel() for k, v in ck["model_state"].items() if k != "head.weight")
        facts["params_m"] = round(n / 1e6)
        facts["context"] = ck["config"]["block_size"]
        r = ck.get("reading") or {}
        if r.get("tokens_new"):
            n = r["tokens_new"]
            facts["tokens"] = "more than a billion" if n >= 1e9 else f"about {round(n / 1e6)} million"
    return facts


def build(seed: int = 7337, core_path: str | None = None, verbose: bool = True) -> dict:
    rng = random.Random(seed)
    convs = oasst_conversations() + dolly_conversations()
    facts = self_facts(core_path)
    convs += self_lessons(facts, rng)
    rng.shuffle(convs)
    os.makedirs(TALK_DIR, exist_ok=True)
    with open(LESSONS, "w", encoding="utf-8") as f:
        for c in convs:
            f.write(json.dumps(c) + "\n")
    by = {}
    for c in convs:
        by[c["src"]] = by.get(c["src"], 0) + 1
    info = {"conversations": len(convs), "by_source": by, "facts": facts, "file": os.path.relpath(LESSONS, ROOT)}
    if verbose:
        print(json.dumps(info, indent=1))
    return info


# ---------------------------------------------------------------------------
# encoding (the exact chat-box format; session.py builds its prompts with the same function)
# ---------------------------------------------------------------------------

def encode_turns(tok, turns: list, passage: str | None = None, answer_last: bool = True) -> tuple[list[int], list[int]]:
    """-> (ids, targets): ids = <bos> [passage : P] q : U a : A <eos> q : U a : A <eos> ...; targets = -1 except on the
    answers and their <eos>. With answer_last=False the last turn's answer is left open (a chat prompt)."""
    bos, eos = tok.stoi["<bos>"], tok.stoi["<eos>"]
    ids, tg = [bos], [-1]
    for i, (u, a) in enumerate(turns):
        head = (f"passage : {passage} " if (passage and i == 0) else "") + f"q : {u} a :"
        h = tok.encode(head, bos=False, eos=False)
        ids += h
        tg += [-1] * len(h)
        if i == len(turns) - 1 and not answer_last:
            break
        b = tok.encode(a, bos=False, eos=False) + [eos]
        ids += b
        tg += b
    return ids, tg


def to_example(tok, conv: dict, block: int) -> tuple[list[int], list[int]] | None:
    """Fit a conversation into one window: drop turns from the end until it fits; a single turn that is too long
    keeps its question and the first part of its answer (no <eos>, so it never learns to stop mid-thought)."""
    turns = conv["turns"]
    while turns:
        ids, tg = encode_turns(tok, turns, conv.get("passage"))
        if len(ids) <= block + 1:
            return ids, tg
        if len(turns) > 1:
            turns = turns[:-1]
            continue
        if conv.get("passage"):
            conv = {**conv, "passage": " ".join(conv["passage"].split()[:200])}
            ids, tg = encode_turns(tok, turns, conv["passage"])
            if len(ids) <= block + 1:
                return ids, tg
        ids, tg = ids[:block + 1], tg[:block + 1]
        if sum(1 for t in tg if t != -1) < 8:
            return None
        return ids, tg
    return None


def load_examples(tok, block: int, path: str = LESSONS) -> list[tuple[list[int], list[int]]]:
    ex = []
    for line in open(path, encoding="utf-8"):
        e = to_example(tok, json.loads(line), block)
        if e:
            ex.append(e)
    return ex


def batch_examples(examples, idx: list[int], pad: int, block: int):
    T = block + 1
    X = torch.full((len(idx), T), pad, dtype=torch.long)
    Y = torch.full((len(idx), T), -1, dtype=torch.long)
    for r, i in enumerate(idx):
        ids, tg = examples[i]
        X[r, :len(ids)] = torch.tensor(ids)
        Y[r, :len(tg)] = torch.tensor(tg)
    return X[:, :-1], Y[:, 1:]


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def train(core: str, staged: str = STAGED, epochs: float = 2.0, conv_per_step: int = 16, read_per_step: int = 12,
          school_per_step: int = 4, micro: int = 16, lr: float = 1e-4, min_lr: float = 1e-5, warmup: int = 200,
          seed: int = 7337, device: str | None = None, out_dir: str = OUT_DIR, store=None, wait_for_gpu: bool = True,
          verbose: bool = True, max_steps: int | None = None) -> dict:
    from .consolidate import reading_replay
    from .reading import eval_windows, gpu_busy, rehearsal_sentences, window_losses, SHARD_DIR
    from .train import get_batch, sentences_to_tensor
    t0 = time.time()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)
    log = os.path.join(out_dir, "log.jsonl")

    def _log(row):
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **row}, default=str) + "\n")

    rng = random.Random(seed + 5)
    torch.manual_seed(seed + 5)
    model, tok, ck = load_checkpoint(device=device, path=core)
    block = model.block_size
    examples = load_examples(tok, block)
    replay = reading_replay(block, rng)
    reh = sentences_to_tensor(rehearsal_sentences(seed, store, verbose=verbose), tok, block)
    windows = eval_windows(SHARD_DIR, block)
    total = int(math.ceil(epochs * len(examples) / conv_per_step))
    if max_steps:
        total = min(total, max_steps)
    batch = conv_per_step + read_per_step + school_per_step
    _log({"kind": "start", "core": core, "examples": len(examples), "steps": total, "batch": batch, "lr": lr,
          "window_loss": window_losses(model, windows)})
    if verbose:
        print(f"[talk] {len(examples):,} conversation examples, {total:,} steps of {batch} "
              f"({conv_per_step} conversation / {read_per_step} reading / {school_per_step} school)", flush=True)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.95))
    ctx = autocast_ctx(device)
    pad = tok.stoi["<pad>"]
    order = []
    losses = []
    model.train()
    for step in range(total):
        if wait_for_gpu and step % 50 == 0:
            while gpu_busy(os.path.join(out_dir, "pause")):
                _log({"kind": "pause", "step": step})
                time.sleep(60)
        if len(order) < conv_per_step:
            more = list(range(len(examples)))
            rng.shuffle(more)
            order += more
        idx, order = order[:conv_per_step], order[conv_per_step:]
        xc, yc = batch_examples(examples, idx, pad, block)
        xs, ys = [xc], [yc]
        if replay and read_per_step:
            xr, yr = replay(read_per_step)
            xs.append(xr); ys.append(yr)
        if school_per_step:
            xh, yh = get_batch(reh, block, school_per_step, rng)
            xs.append(xh); ys.append(yh)
        x, y = torch.cat(xs).to(device), torch.cat(ys).to(device)
        p = min(1.0, max(0, step - warmup) / max(1, total - warmup))
        cur = lr * (step + 1) / warmup if step < warmup else min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * p))
        for g in opt.param_groups:
            g["lr"] = cur
        opt.zero_grad(set_to_none=True)
        perm = torch.randperm(len(x), device=x.device)
        tot = 0.0
        for i in range(0, len(x), micro):
            j = perm[i:i + micro]
            with ctx:
                _, l = model(x[j], y[j])
            (l * len(j) / len(x)).backward()
            tot += float(l.detach()) * len(j) / len(x)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(tot)
        if (step + 1) % 100 == 0:
            _log({"kind": "step", "step": step + 1, "loss": round(sum(losses[-100:]) / 100, 4), "lr": round(cur, 7)})
            if verbose:
                print(f"[talk] step {step + 1}/{total} loss {sum(losses[-100:]) / 100:.3f} lr {cur:.2e}", flush=True)
    model.eval()
    wl = window_losses(model, windows)
    run = {"kind": "conversation lessons", "examples": len(examples), "steps": total, "batch": batch, "lr": lr,
           "epochs": epochs, "seconds": round(time.time() - t0, 1), "loss_last": round(sum(losses[-100:]) / max(1, len(losses[-100:])), 4),
           "window_loss_after": wl, "lessons": os.path.relpath(LESSONS, ROOT)}
    meta = {k: ck.get(k) for k in ("seed", "corpus_sha", "generation", "core_version", "grown_from", "story_stats",
                                    "docs_stats", "school", "reading")}
    meta.update({"staged": True, "label": (ck.get("label") or "v4") + " + conversation lessons", "talk": run,
                 "note": f"growth: core -> v{float(ck.get('core_version', 4)):g} + conversation lessons"})
    save_checkpoint(model, tok, meta, path=staged)
    _log({"kind": "done", **run, "staged": staged})
    if verbose:
        print(f"[talk] done in {run['seconds'] / 60:.0f} min, loss {run['loss_last']}, windows {wl} -> {staged}", flush=True)
    return {"model": model, "tok": tok, "run": run, "staged": staged}


def teacher_summary(max_labels: int = 6) -> dict:
    """For the control room: the same measurements and the same ten chat prompts at every point of the teaching
    plan (before / after Stage A / after Stage B / after 5+ consolidations), oldest first; and the run log tail."""
    tdir = os.path.join(RESULTS_DIR, "teacher")
    meas, samples, run = [], [], []
    try:
        for line in open(os.path.join(tdir, "measurements.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            meas.append({k: r.get(k) for k in ("ts", "label", "generation", "params", "window_loss", "exams")})
    except (OSError, ValueError):
        pass
    sdir = os.path.join(RESULTS_DIR, "chat_samples")
    if os.path.isdir(sdir):
        by_label = {}
        for name in sorted(os.listdir(sdir)):
            try:
                d = json.load(open(os.path.join(sdir, name), encoding="utf-8"))
                by_label[d["label"]] = {"label": d["label"], "ts": d["ts"], "params": d.get("params"),
                                        "single": [[x["prompt"], x["answer"]] for x in d["single"]],
                                        "conversation": [[x["prompt"], x["answer"]] for x in d.get("conversation", [])]}
            except Exception:
                continue
        samples = sorted(by_label.values(), key=lambda d: d["ts"])[-max_labels:]
    try:
        run = [json.loads(l) for l in open(os.path.join(tdir, "run.jsonl"), encoding="utf-8").read().splitlines()[-8:]]
        run = [{k: r.get(k) for k in ("ts", "step", "label", "generation", "n", "error")} for r in run]
    except (OSError, ValueError):
        run = []
    return {"measurements": meas[-max_labels:], "samples": samples, "run": run}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "train"])
    ap.add_argument("--core", default=os.path.join(RESULTS_DIR, "core_v4_staged.pt"))
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build(core_path=a.core)
    else:
        train(a.core, epochs=a.epochs, max_steps=a.max_steps)


if __name__ == "__main__":
    main()
