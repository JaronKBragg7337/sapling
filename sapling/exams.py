"""Sapling exams — predeclared bars, cold runs, FAILs kept on the record.

fly-school rules applied to a language core:
- The bars below are predeclared (they live in this file, not in the results).
- Exam combos were never taught (held out by curriculum.split_combos).
- Decoding is greedy and seeded: every exam is reproducible.
- Results append to results/report_card.json and are hash-chained into the
  memory store. A FAIL is kept, not redefined.

Usage: python -m sapling.exams
"""

from __future__ import annotations

import json
import os
import random
import time

import torch

from .curriculum import PREFIXES, SUFFIXES, compose, prefix_gloss, split_combos, suffix_gloss
from .memory import MemoryStore
from .train import CHECKPOINT, RESULTS_DIR, load_checkpoint

# --- PREDECLARED BARS (do not edit after a run; add a new exam row instead) --
BARS = {"E1_composition": 0.60, "E2_affix_recall": 0.90,
        "E3_recall_echo": 0.90, "E4_deferral": 0.80}
REPORT_CARD = os.path.join(RESULTS_DIR, "report_card.json")


def answer(model, tok, prompt: str, max_new: int = 24) -> str:
    ids = torch.tensor([tok.encode(prompt, eos=False)], device=next(model.parameters()).device)
    out = model.generate(ids, max_new=max_new, eos_id=tok.stoi["<eos>"])
    return tok.decode(out[0, ids.shape[1]:].tolist())


def run_exams(seed: int = 7337, verbose: bool = True) -> dict:
    model, tok, ckpt = load_checkpoint()
    split = split_combos(seed)
    card = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "seed": seed,
            "corpus_sha": ckpt["corpus_sha"], "bars": BARS, "exams": []}

    def record(name, bar, passed, total, detail):
        acc = passed / max(1, total)
        verdict = "PASS" if acc >= bar else "FAIL"
        card["exams"].append({"exam": name, "bar": bar, "score": f"{passed}/{total}",
                              "accuracy": round(acc, 3), "verdict": verdict,
                              "detail": detail[:6]})
        if verbose:
            print(f"[exam] {name:<18} {passed}/{total} acc={acc:.3f} bar={bar} -> {verdict}")

    # E1: compositional generalisation on held-out prefix/suffix combos.
    rows, passed, total = [], 0, 0
    for p, r in split["pre_exam"]:
        w, g = compose([p, r]), prefix_gloss(r, p)
        got = answer(model, tok, f"q : what does {w} mean ? a :")
        exp = f"{w} means {g} ."
        ok = got.strip() == exp
        passed += ok; total += 1; rows.append({"q": w, "expected": exp, "got": got, "ok": ok})
    for r, s in split["suf_exam"]:
        w, g = compose([r, s]), suffix_gloss(r, s)
        got = answer(model, tok, f"q : what does {w} mean ? a :")
        exp = f"{w} means {g} ."
        ok = got.strip() == exp
        passed += ok; total += 1; rows.append({"q": w, "expected": exp, "got": got, "ok": ok})
    record("E1_composition", BARS["E1_composition"], passed, total, rows)

    # E2: affix definitions (retention of taught meanings).
    rows, passed, total = [], 0, 0
    for p, g in list(PREFIXES.items()):
        got = answer(model, tok, f"q : what does the prefix {p} mean ? a :")
        ok = got.strip() == f"{g} ."
        passed += ok; total += 1; rows.append({"q": p, "expected": g, "got": got, "ok": ok})
    for s, g in list(SUFFIXES.items()):
        got = answer(model, tok, f"q : what does the suffix {s} mean ? a :")
        ok = got.strip() == f"{g} ."
        passed += ok; total += 1; rows.append({"q": s, "expected": g, "got": got, "ok": ok})
    record("E2_affix_recall", BARS["E2_affix_recall"], passed, total, rows)

    # E3: recall echo through slots (the memory interface).
    rows, passed, total = [], 0, 0
    rng = random.Random(seed + 1)
    for _ in range(10):
        got = answer(model, tok,
                     "recall [ user ] : the word <m1> means <m2> . "
                     "q : what does <m1> mean ? a :")
        exp = "the word <m1> means <m2> ."
        ok = got.strip() == exp
        passed += ok; total += 1; rows.append({"expected": exp, "got": got, "ok": ok})
    record("E3_recall_echo", BARS["E3_recall_echo"], passed, total, rows)

    # E4: deferral when no recall exists (cite-or-defer judgment).
    rows, passed, total = [], 0, 0
    for _ in range(10):
        got = answer(model, tok, "q : what does <m1> mean ? a :")
        exp = "i do not have that yet ."
        ok = got.strip() == exp
        passed += ok; total += 1; rows.append({"expected": exp, "got": got, "ok": ok})
    record("E4_deferral", BARS["E4_deferral"], passed, total, rows)

    history = []
    if os.path.exists(REPORT_CARD):
        try:
            prev = json.load(open(REPORT_CARD))
            history = prev.get("history", []) + [
                {k: prev[k] for k in ("ts", "seed", "corpus_sha") if k in prev}
                | {"exams": [{kk: e[kk] for kk in ("exam", "score", "accuracy", "verdict")}
                              for e in prev.get("exams", [])]}
            ]
        except Exception:
            history = []
    card["history"] = history
    with open(REPORT_CARD, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)

    # chain the exam run into memory: provenance travels with the fact
    store = MemoryStore()
    for ex in card["exams"]:
        store.remember(f"exam {ex['exam']} {ex['verdict']} {ex['score']} (bar {ex['bar']})",
                       provenance="exam", state="confirmed")
    if verbose:
        print(f"[exam] report card -> {REPORT_CARD} (and hash-chained into memory)")
    return card


if __name__ == "__main__":
    run_exams()
