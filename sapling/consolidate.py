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


def consolidate(brain, store, goals: list[dict], steps: int = 200, lr: float = 5e-4,
                batch_size: int = 32, seed: int = 7337, verbose: bool = True,
                rehearsal: int = 400) -> dict:
    t0 = time.time()
    rng = random.Random(seed + brain.generation + 1)
    torch.manual_seed(seed + brain.generation + 1)

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
    rep = max(1, int(len(other) * FACT_SHARE / (1 - FACT_SHARE) / max(1, len(fact_mix)))) if fact_mix else 0
    mix = other + fact_mix * rep
    rng.shuffle(mix)

    added: list[str] = []
    for s in mix:
        added += brain.grow_vocab_for(s, why="consolidation")

    model, tok = brain.model, brain.tok
    with brain.lock:
        model.train()
        data = sentences_to_tensor(mix, tok, model.block_size).to(brain.device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        losses = []
        for _ in range(steps):
            x, y = get_batch(data, model.block_size, batch_size, rng)
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        model.eval()
        brain.generation += 1
        meta = {"seed": seed, "corpus_sha": (brain.meta or {}).get("corpus_sha"),
                "generation": brain.generation,
                "note": f"consolidation {brain.generation}: {len(pending)} new facts, {len(facts)} total"}
        brain.meta = {**(brain.meta or {}), **meta}
        kept = save_checkpoint(model, tok, meta,
                               path=os.path.join(CHECKPOINT_DIR, f"gen_{brain.generation:04d}.pt"))
        save_checkpoint(model, tok, meta, path=CHECKPOINT)

    row = {
        "generation": brain.generation, "memory_ids": sorted(pending_ids),
        "facts_total": len(facts), "facts_new": len(pending), "sentences": len(mix), "fact_sentences": len(fact_mix) * rep,
        "tokens": int(data.numel()), "steps": steps, "lr": lr,
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
