"""Consolidation — what it has met goes into its weights.

Jaron 10/14 (design restraints 2 + 3): "Facts never go into its brain ->
CRITICAL FIX. It must be able to become what it learns." and "Learning gated
by owner approval -> REMOVE." So: no gate. The heartbeat chooses when to
consolidate; this module fine-tunes the live core on its own record.

What it trains on (fed-not-filled: prefer what resolved, don't block the rest):
- taught / found / self facts from memory, weighted: confirmed x3,
  recalled-at-least-twice x2, otherwise x1; invalidated rows are left out
  (an invalidation is a verdict, not a fade)
- practice items for its open goals (weak parts), training material only
- a rehearsal slice of the original school corpus (so it does not forget
  the language while learning the facts) - catastrophic forgetting is the
  known risk; the exams after each consolidation are how it is watched.

Every consolidation = one checkpoint kept (results/checkpoints/gen_N.pt),
one ledger row, one exam run appended to the report card. Observable, not
gated.
"""

from __future__ import annotations

import os
import random
import time

import torch

from .curriculum import LESSONS, build_corpus
from .exams import run_exams
from .model import autocast_ctx
from .goals import pick_practice
from .train import CHECKPOINT, CHECKPOINT_DIR, get_batch, save_checkpoint, sentences_to_tensor

FACT_FORMS = ("{content} .", "q : what does {word} mean ? a : {content} .")


def consolidated_ids(store) -> set[int]:
    ids: set[int] = set()
    for row in store.ledger_rows("consolidation", limit=100000):
        ids.update(int(i) for i in row.get("memory_ids", []))
    return ids


def pending_facts(store) -> list[dict]:
    done = consolidated_ids(store)
    return [f for f in store.facts() if f["id"] not in done and f["state"] != "invalidated"]


def _fact_sentences(fact: dict) -> list[str]:
    content = fact["content"].strip().rstrip(".").strip()
    toks = content.lower().split()
    out = [content.lower() + " ."]
    if len(toks) >= 4 and toks[0] == "the" and toks[1] == "word" and toks[3] == "means":
        out.append(f"q : what does {toks[2]} mean ? a : {content.lower()} .")
    return out


FACT_SHARE = 0.30   # facts are ~30% of the sentences it trains on (at ~3% they blended into one)
# v3 retune (DESIGN-CHOICES D14): the 12M core takes a gentler step than the 0.82M one did, and rehearses
# BOTH levels so a consolidation does not cost it its sentences. First try (lr 3e-4, 300 level-2 sentences,
# 60 stories): gen 153 fit its mix to loss 0.045 in 200 steps and E7 fell 27 -> 20/40, E6 14 -> 13/48 - the
# big core memorises a small mix fast. Second setting (below): half the step, twice the level-2 rehearsal.
LR_V3 = 1.5e-4
# D34 (10/16): a core past 50M params folds at half that step (tuned on the 60.5M v4 core; D14's 1.5e-4 was for 12-17M)
LR_BIG = 7.5e-5
BIG_PARAMS = 50_000_000
REHEARSAL2 = 600     # level-2 generated sentences per consolidation
REHEARSAL_STORIES = 150
# v3.1: level 3 rehearsal (copy / bridge / document reading) so reading skills survive consolidation too
REHEARSAL3 = {"copy": 200, "bridge": 300, "docs": 80, "longread": 40, "prose": 60}   # bridge 100 -> 300 (D33: E1b 84 -> 56 in 27 folds)
_CORPUS2 = {}
_CORPUS3 = {}


def record_sentences(store) -> list[str]:
    """Everything it was taught or said, as training sentences, weighted the way
    consolidation weights them (confirmed x3, recalled-twice x2, else x1).
    Shared by consolidate() and grow() so the grown core carries the same record."""
    out: list[str] = []
    for f in store.facts():
        if f["state"] == "invalidated":
            continue
        w = 3 if f["state"] == "confirmed" else (2 if f["recalls"] >= 2 else 1)
        for s in _fact_sentences(f):
            out += [s] * w
    return out


def _corpus2(seed: int):
    if seed not in _CORPUS2:
        from .curriculum2 import build_corpus2
        _CORPUS2[seed] = build_corpus2(seed, stories=True, max_stories=2000)
    return _CORPUS2[seed]


def _corpus3(seed: int, tok=None):
    if seed not in _CORPUS3:
        from .curriculum3 import build_corpus3
        c2 = _corpus2(seed)
        words = sorted({t for s in c2["lessons"]["stories"][:2000] for t in s.split() if t.isalpha()})
        _CORPUS3[seed] = build_corpus3(seed, words=words, story_sents=c2["lessons"]["stories"][:500],
                                       sizes={"copy": 800, "prose": 400, "docs": 400, "longread": 200})
    return _CORPUS3[seed]


# v3.2.1 (10/15, Claude as teacher): every consolidation also rehearses the v3.2 reading. Without it, 200 steps on
# school sentences alone wiped the 400M-token reading in ~9 generations (adult text loss 3.82 -> 8.12 by gen 250,
# 9.60 by gen 286; stories 1.47 -> 4.13). Same idea as the reading stage's own rehearsal, the other way round.
READ_REPLAY_SHARE = 0.75          # share of each consolidation batch that is reading windows
# v4.0.3 (10/15 7:50 PM, found by Vex review #47): v4 reads FineWeb-Edu as source "web" (14 of every 32 sequences); the
# replay left it out, so folding would have worn the web reading down. Replay now follows the v4 reading mix (11/4/14).
# A source with no shards (pre-v4 index) is skipped, so the v3 split still works.
READ_REPLAY_SPLIT = {"stories": 11 / 29, "adult": 4 / 29, "web": 14 / 29}


def reading_replay(block: int, rng: random.Random):
    """-> fn(n) giving (x, y) of n random windows from the v3.2 reading shards, or None if none exist."""
    import json
    import numpy as np
    from .reading import SHARD_DIR
    try:
        idx = json.load(open(os.path.join(SHARD_DIR, "index.json"), encoding="utf-8"))
    except (OSError, ValueError):
        return None
    arrs = {}
    for name in READ_REPLAY_SPLIT:
        files = [s["file"] for s in idx.get("sources", {}).get(name, {}).get("shards", [])]
        if files:
            arrs[name] = [np.load(os.path.join(SHARD_DIR, f), mmap_mode="r") for f in files]
    if not arrs:
        return None

    def fn(n: int):
        xs = []
        names = list(arrs)
        weights = [READ_REPLAY_SPLIT[k] for k in names]
        for _ in range(n):
            a = rng.choice(arrs[rng.choices(names, weights)[0]])
            off = rng.randrange(0, len(a) - block - 1)
            xs.append(torch.from_numpy(np.asarray(a[off:off + block + 1]).astype("int64")))
        t = torch.stack(xs)
        return t[:, :-1], t[:, 1:]
    return fn


# v4 (10/15): a core that had conversation lessons also replays them in every consolidation, so the way it talks
# survives the heartbeat the same way the reading does (the v3.2.1 lesson, applied before it could happen).
TALK_REPLAY_SHARE = 0.25          # of each batch; reading then 0.5, school/facts 0.25
MICRO = 16                        # micro-batch: a 60M core at 32 x 512 in one pass fills the 8 GB card


_TALK_CACHE: dict = {}


# v4.1.1 (10/16, D33): a core that had the long-reading lesson replays it in every consolidation too. Without it,
# 27 heartbeat folds took E7L 17 -> 8/40 (the v3.2.1 lesson a third time: what a consolidation does not replay, it erases).
# First try (10/16 08:30): batch 32 = lesson 8 / reading 10 / talk 6 / school 8 -> after 5 folds E7L held (15/40) but the
# text-window losses rose ~0.15-0.25 and E10 fell 131 -> 44: taking reading away to make room was the wrong trade.
# Now (D33): the lesson is ADDED, nothing taken away - batch 40 = lesson 8 / reading 16 / talk 8 / school + facts 8.
LESSON_REPLAY = 8
SHARES_WITH_LESSON = {"reading": 16, "talk": 8}
BATCH_WITH_LESSON = 40
_LESSON_CACHE: dict = {}


def had_longread(brain) -> bool:
    m = brain.meta or {}
    return bool(m.get("longread")) or "long-reading lesson" in str(m.get("label") or "")


def lesson_replay(brain, rng: random.Random):
    """-> fn(n) giving (x, y) of n long-reading lesson examples (answer-only loss), or None for a core that never had
    the lesson. For a core that had it, a missing lesson file RAISES (a logged tick error), never a quiet skip."""
    if not had_longread(brain):
        return None
    from .longread_lesson import BUNDLE, Ragged
    path = os.path.join(BUNDLE, "lesson.npz")
    if not os.path.exists(path):
        raise RuntimeError(f"lesson replay: {path} is missing - run `python -m sapling.longread_lesson build`")
    key = (path, os.path.getmtime(path))
    if key not in _LESSON_CACHE:
        _LESSON_CACHE.clear()
        _LESSON_CACHE[key] = Ragged(path)
    lesson = _LESSON_CACHE[key]
    pad = brain.tok.stoi["<pad>"]

    def fn(n: int):
        return lesson.batch([rng.randrange(len(lesson)) for _ in range(n)], brain.model.block_size, pad)
    return fn


def talk_replay(brain, rng: random.Random):
    """-> fn(n) giving (x, y) of n conversation lessons (loss on its answers only), or None for a core that never
    had conversation lessons. For a core that talks, a failure RAISES (the tick logs it as an error) instead of
    quietly consolidating without its conversation lessons (Grok Build review 10/15)."""
    from .talk import LESSONS, batch_examples, load_examples
    if not brain.talks():
        return None
    if not os.path.exists(LESSONS):
        raise RuntimeError(f"talk replay: {LESSONS} is missing - run `python -m sapling.talk build`")
    try:
        key = (len(brain.tok.itos), brain.model.block_size, os.path.getmtime(LESSONS))
        if key not in _TALK_CACHE:
            _TALK_CACHE.clear()
            _TALK_CACHE[key] = load_examples(brain.tok, brain.model.block_size)
        ex = _TALK_CACHE[key]
    except Exception as e:
        raise RuntimeError(f"talk replay failed: {type(e).__name__}: {e}") from e
    if not ex:
        raise RuntimeError("talk replay: the lesson file gave no examples")
    pad = brain.tok.stoi["<pad>"]

    def fn(n: int):
        return batch_examples(ex, [rng.randrange(len(ex)) for _ in range(n)], pad, brain.model.block_size)
    return fn


def consolidate(brain, store, goals: list[dict], steps: int = 200, lr: float | None = None,
                batch_size: int = 32, seed: int = 7337, verbose: bool = True,
                rehearsal: int = 400) -> dict:
    t0 = time.time()
    rng = random.Random(seed + brain.generation + 1)
    torch.manual_seed(seed + brain.generation + 1)
    model, tok = brain.model, brain.tok
    v3 = float((brain.meta or {}).get("core_version", 2)) >= 3
    v31 = float((brain.meta or {}).get("core_version", 2)) >= 3.1
    if lr is None:
        lr = (LR_BIG if model.n_params() > BIG_PARAMS else LR_V3) if v3 else 5e-4

    facts = [f for f in store.facts() if f["state"] != "invalidated"]
    pending = pending_facts(store)
    pending_ids = {f["id"] for f in pending}
    fact_mix: list[str] = []
    for f in facts:
        w = 3 if f["state"] == "confirmed" else (2 if f["recalls"] >= 2 else 1)
        if f["id"] not in pending_ids:
            w = max(1, w - 1)          # already in weights: lighter rehearsal
        for s in _fact_sentences(f):
            fact_mix += [s] * w
    other: list[str] = []
    for it in pick_practice(goals, rng, batch=24, seed=seed):
        other += [f"{it['q']} {it['a']}"] * 2
    corpus = build_corpus(seed)
    school = [s for name in LESSONS for s in corpus["lessons"][name]]
    other += rng.sample(school, min(rehearsal, len(school)))
    if v3:
        c2 = _corpus2(seed)
        gen2 = [s for n in ("simple", "grammar", "paragraphs", "comprehension") for s in c2["lessons"][n]]
        other += rng.sample(gen2, min(int(rehearsal * REHEARSAL2 / 400), len(gen2)))
        if c2["lessons"]["stories"]:
            other += rng.sample(c2["lessons"]["stories"], min(int(rehearsal * REHEARSAL_STORIES / 400), len(c2["lessons"]["stories"])))
    if v31:
        c3 = _corpus3(seed)
        r3 = REHEARSAL3
        if had_longread(brain):
            # D36: the old docs/longread school rehearsal (40 + 80 sentences in a different answer style) wears
            # the cloud long-reading lesson out of the weights - fold tests S1-S4 (10/16): E7L 12 -> 16/40 in 5
            # folds without it, "which sentence" held 5/16 instead of 1/16, E1b 81, E5 6/7. Cores that never had
            # the lesson keep the old mix.
            r3 = {k: v for k, v in r3.items() if k not in ("longread", "docs")}
        for name, n in r3.items():
            L = c3["lessons"].get(name) or []
            if L:
                other += rng.sample(L, min(int(rehearsal * n / 400), len(L)))
    # facts (incl. passages it has read) are ~FACT_SHARE of the mix in BOTH directions: repeated when few,
    # sampled down when a document brings hundreds of passages (D14 lesson: the big core memorises fast)
    cap = int(len(other) * FACT_SHARE / (1 - FACT_SHARE))
    if fact_mix and len(fact_mix) > cap > 0:
        fact_mix = rng.sample(fact_mix, cap)
    rep = max(1, int(cap / max(1, len(fact_mix)))) if fact_mix else 0
    mix = other + fact_mix * rep
    rng.shuffle(mix)

    added: list[str] = []
    if not brain.tok.has_pieces():
        for s in mix:
            added += brain.grow_vocab_for(s, why="consolidation")
    else:
        # v3.1: words it was TAUGHT become tokens of their own; words it merely read are spelled from pieces
        for f in facts:
            if f.get("provenance") == "user":
                for s in _fact_sentences(f):
                    added += brain.grow_vocab_for(s, why="consolidation")

    with brain.lock:
        model.train()
        data = sentences_to_tensor(mix, tok, model.block_size).to(brain.device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        losses = []
        ctx = autocast_ctx(str(brain.device))
        replay = reading_replay(model.block_size, rng)
        talk = talk_replay(brain, rng)
        lesson = lesson_replay(brain, rng)
        if lesson:
            batch_size = max(batch_size, BATCH_WITH_LESSON)
            n_lesson = LESSON_REPLAY
            n_talk = SHARES_WITH_LESSON["talk"] if talk else 0
            n_read = SHARES_WITH_LESSON["reading"] if replay else 0
        else:
            n_lesson = 0
            n_talk = int(batch_size * TALK_REPLAY_SHARE) if talk else 0
            n_read = int(batch_size * (READ_REPLAY_SHARE - (TALK_REPLAY_SHARE if talk else 0))) if replay else 0
        for _ in range(steps):
            x, y = get_batch(data, model.block_size, batch_size - n_read - n_talk - n_lesson, rng)
            if n_lesson:
                lx, ly = lesson(n_lesson)
                x, y = torch.cat([x, lx.to(x.device)]), torch.cat([y, ly.to(y.device)])
            if n_read:
                rx, ry = replay(n_read)
                x, y = torch.cat([x, rx.to(x.device)]), torch.cat([y, ry.to(y.device)])
            if n_talk:
                tx, ty = talk(n_talk)
                x, y = torch.cat([x, tx.to(x.device)]), torch.cat([y, ty.to(y.device)])
            opt.zero_grad(set_to_none=True)
            tot = 0.0
            for i in range(0, len(x), MICRO):
                xb, yb = x[i:i + MICRO], y[i:i + MICRO]
                with ctx:
                    _, lb = model(xb, yb)
                (lb * len(xb) / len(x)).backward()
                tot += float(lb.detach()) * len(xb) / len(x)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(tot)
        model.eval()
        brain.generation += 1
        meta = {"seed": seed, "corpus_sha": (brain.meta or {}).get("corpus_sha"),
                "generation": brain.generation, "core_version": (brain.meta or {}).get("core_version", 2),
                "grown_from": (brain.meta or {}).get("grown_from"),
                # v4: these travel with every generation (talk = it had conversation lessons -> the chat box uses them)
                "talk": (brain.meta or {}).get("talk"), "label": (brain.meta or {}).get("label"),
                "longread": (brain.meta or {}).get("longread"),
                "note": f"consolidation {brain.generation}: {len(pending)} new facts, {len(facts)} total"}
        brain.meta = {**(brain.meta or {}), **meta}
        kept = save_checkpoint(model, tok, meta,
                               path=os.path.join(CHECKPOINT_DIR, f"gen_{brain.generation:04d}.pt"), half=True)
        save_checkpoint(model, tok, meta, path=CHECKPOINT)

    # Jaron 10/14 2:59 PM (P2): self-made memories DO train its brain - try it and
    # observe. Record which of them went in, so exam/E5 drift with vs without
    # self rows can be compared later (tools/self_memory_drift.py).
    self_ids = sorted(f["id"] for f in facts if f.get("provenance") == "self")
    row = {
        "generation": brain.generation, "memory_ids": sorted(pending_ids),
        "self_ids": self_ids, "self_new": sorted(pending_ids & set(self_ids)),
        "facts_self": len(self_ids), "facts_user": len([f for f in facts if f.get("provenance") == "user"]),
        "facts_total": len(facts), "facts_new": len(pending), "sentences": len(mix), "fact_sentences": len(fact_mix) * rep,
        "tokens": int(data.numel()), "steps": steps, "lr": lr,
        "talk_share": round(n_talk / batch_size, 2), "read_share": round(n_read / batch_size, 2),
        "lesson_share": round(n_lesson / batch_size, 2),
        "loss_first": round(losses[0], 4), "loss_last": round(sum(losses[-10:]) / min(10, len(losses)), 4),
        "vocab_added": added, "vocab_size": len(tok.itos),
        "checkpoint": os.path.relpath(kept, os.path.dirname(CHECKPOINT_DIR)),
        "seconds": round(time.time() - t0, 1),
    }
    store.meter("consolidation", row)
    if verbose:
        print(f"[consolidate] gen {row['generation']} facts new={row['facts_new']} "
              f"loss {row['loss_first']} -> {row['loss_last']} vocab {row['vocab_size']} "
              f"({row['seconds']}s) -> {row['checkpoint']}")

    card = run_exams(seed=seed, verbose=verbose, model=model, tok=tok, meta=brain.meta,
                     store=store, consolidated_facts=[f["content"] for f in facts],
                     trigger=f"consolidation gen {brain.generation}")
    row["exams"] = [{k: e[k] for k in ("exam", "score", "accuracy", "verdict")} for e in card["exams"]]
    return row
