"""Sapling exams — predeclared bars, cold runs, FAILs kept on the record.

fly-school rules applied to a language core:
- The bars below are predeclared (they live in this file, not in the results).
- Exam combos were never taught (held out by curriculum.split_combos) and are
  never practiced (goals.py works on parts, from training material only).
- Decoding is greedy and seeded: every exam is reproducible.
- Results append to results/report_card.json and are hash-chained into the
  memory store. A FAIL is kept, not redefined.

v2: run_exams() can take an already-loaded model (the heartbeat re-runs the
exams after every consolidation - Jaron 10/14: observable, "data, not a
gate"). E5_facts_in_weights is NOT an outside exam: it asks, with no recall
context, about facts it consolidated from its own record, to show whether a
fact actually moved into the weights. It has no bar; it is information.

Usage: python -m sapling.exams
"""

from __future__ import annotations

import json
import os
import random
import re
import time

import torch

from .curriculum import PREFIXES, SUFFIXES, compose, prefix_gloss, split_combos, suffix_gloss
from .curriculum2 import exam_items_grammar, exam_items_reading
from .goals import parts_of_question
from .memory import MemoryStore
from .train import RESULTS_DIR, load_checkpoint

# --- PREDECLARED BARS (do not edit after a run; add a new exam row instead) --
# E6/E7 added 10/14 for level 2 (Jaron: "school level 2: real sentences"); bars set BEFORE the first run.
BARS = {"E1_composition": 0.60, "E2_affix_recall": 0.90,
        "E3_recall_echo": 0.90, "E4_deferral": 0.80,
        "E6_grammar": 0.60, "E7_reading": 0.50}
REPORT_CARD = os.path.join(RESULTS_DIR, "report_card.json")
WRITING_DIR = os.path.join(RESULTS_DIR, "writing_samples")
WORD_MEANS = re.compile(r"^the word (\S+) means (.+?)\s*\.?$", re.I)
# E8 (info only): free writing, saved for Jaron to read. Seeded sampling = reproducible.
WRITING_PROMPTS = ["once upon a time ,", "tom has a red ball .", "the little dog",
                   "q : what do you like to do ? a :"]


def answer(model, tok, prompt: str, max_new: int = 24, temperature: float = 0.0, seed: int | None = None) -> str:
    ids = torch.tensor([tok.encode(prompt, eos=False)], device=next(model.parameters()).device)
    out = model.generate(ids, max_new=max_new, eos_id=tok.stoi["<eos>"], temperature=temperature, seed=seed)
    return tok.decode(out[0, ids.shape[1]:].tolist())


def run_exams(seed: int = 7337, verbose: bool = True, model=None, tok=None,
              meta: dict | None = None, store: MemoryStore | None = None,
              consolidated_facts: list[str] | None = None, trigger: str = "manual") -> dict:
    if model is None:
        model, tok, ckpt = load_checkpoint()
        meta = {k: ckpt.get(k) for k in ("corpus_sha", "generation", "note")}
    meta = meta or {}
    was_training = model.training
    model.eval()
    split = split_combos(seed)
    card = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "seed": seed, "trigger": trigger,
            "corpus_sha": meta.get("corpus_sha"), "generation": meta.get("generation", 0),
            "vocab_size": len(tok.itos), "bars": BARS, "exams": []}
    detail_rows = []   # for the failure record (parts), E1/E2 only

    def record(name, bar, passed, total, detail):
        acc = passed / max(1, total)
        verdict = ("PASS" if acc >= bar else "FAIL") if bar is not None else "INFO"
        card["exams"].append({"exam": name, "bar": bar, "score": f"{passed}/{total}",
                              "accuracy": round(acc, 3), "verdict": verdict,
                              "detail": detail[:8]})
        if verbose:
            print(f"[exam] {name:<20} {passed}/{total} acc={acc:.3f} bar={bar} -> {verdict}")

    # E1: compositional generalisation on held-out prefix/suffix combos.
    rows, passed, total = [], 0, 0
    for p, r in split["pre_exam"]:
        w, g = compose([p, r]), prefix_gloss(r, p)
        q = f"q : what does {w} mean ? a :"
        got = answer(model, tok, q)
        exp = f"{w} means {g} ."
        ok = got.strip() == exp
        passed += ok; total += 1; rows.append({"q": w, "expected": exp, "got": got, "ok": ok})
        detail_rows.append({"q": w, "ok": ok, "parts": parts_of_question(q)})
    for r, s in split["suf_exam"]:
        w, g = compose([r, s]), suffix_gloss(r, s)
        q = f"q : what does {w} mean ? a :"
        got = answer(model, tok, q)
        exp = f"{w} means {g} ."
        ok = got.strip() == exp
        passed += ok; total += 1; rows.append({"q": w, "expected": exp, "got": got, "ok": ok})
        detail_rows.append({"q": w, "ok": ok, "parts": parts_of_question(q)})
    rows.sort(key=lambda r: r["ok"])   # misses first on the card
    record("E1_composition", BARS["E1_composition"], passed, total, rows)

    # E2: affix definitions (retention of taught meanings).
    rows, passed, total = [], 0, 0
    for p, g in list(PREFIXES.items()):
        got = answer(model, tok, f"q : what does the prefix {p} mean ? a :")
        ok = got.strip() == f"{g} ."
        passed += ok; total += 1; rows.append({"q": p, "expected": g, "got": got, "ok": ok})
        detail_rows.append({"q": p, "ok": ok, "parts": [f"prefix:{p}"]})
    for s, g in list(SUFFIXES.items()):
        got = answer(model, tok, f"q : what does the suffix {s} mean ? a :")
        ok = got.strip() == f"{g} ."
        passed += ok; total += 1; rows.append({"q": s, "expected": g, "got": got, "ok": ok})
        detail_rows.append({"q": s, "ok": ok, "parts": [f"suffix:{s}"]})
    rows.sort(key=lambda r: r["ok"])
    record("E2_affix_recall", BARS["E2_affix_recall"], passed, total, rows)

    # E3: recall echo through slots (the memory interface), 1..4 recalls.
    rows, passed, total = [], 0, 0
    rng = random.Random(seed + 1)
    pairs = [("<m1>", "<m2>"), ("<m3>", "<m4>"), ("<m5>", "<m6>"), ("<m7>", "<m8>")]
    for i in range(12):
        n = 1 + (i % 4)
        chosen = pairs[:n]
        order = list(chosen); rng.shuffle(order)
        asked = rng.choice(chosen)
        ctx = " ".join(f"recall [ user ] : the word {w} means {g} ." for w, g in order)
        got = answer(model, tok, f"{ctx} q : what does {asked[0]} mean ? a :")
        exp = f"the word {asked[0]} means {asked[1]} ."
        ok = got.strip() == exp
        passed += ok; total += 1; rows.append({"n_recalls": n, "expected": exp, "got": got, "ok": ok})
    record("E3_recall_echo", BARS["E3_recall_echo"], passed, total, rows)

    # E4: deferral when no recall exists (cite-or-defer judgment).
    rows, passed, total = [], 0, 0
    for w, _g in pairs + pairs[:1]:
        got = answer(model, tok, f"q : what does {w} mean ? a :")
        exp = "i do not have that yet ."
        ok = got.strip() == exp
        passed += ok; total += 1; rows.append({"expected": exp, "got": got, "ok": ok})
    for _ in range(5):
        got = answer(model, tok, "q : what does <m1> mean ? a :")
        ok = got.strip() == "i do not have that yet ."
        passed += ok; total += 1; rows.append({"expected": "i do not have that yet .", "got": got, "ok": ok})
    record("E4_deferral", BARS["E4_deferral"], passed, total, rows)

    # E6: grammar - fill the right word / tense on held-out (verb, object) pairs (level 2).
    rows, passed, total = [], 0, 0
    for it in exam_items_grammar(seed):
        got = answer(model, tok, it["q"], max_new=20)
        ok = got.strip() == it["a"]
        passed += ok; total += 1; rows.append({"kind": it["kind"], "q": it["q"], "expected": it["a"], "got": got, "ok": ok})
    rows.sort(key=lambda r: r["ok"])
    record("E6_grammar", BARS["E6_grammar"], passed, total, rows)

    # E7: reading - answer a question about a 2-4 sentence passage built only from held-out facts.
    rows, passed, total = [], 0, 0
    for it in exam_items_reading(seed):
        got = answer(model, tok, it["q"], max_new=12)
        ok = got.strip() == it["a"]
        passed += ok; total += 1; rows.append({"q": it["q"], "expected": it["a"], "got": got, "ok": ok})
    rows.sort(key=lambda r: r["ok"])
    record("E7_reading", BARS["E7_reading"], passed, total, rows)

    # E8 (INFO): free writing samples, saved for the owner to read; no bar.
    samples = []
    for i, p in enumerate(WRITING_PROMPTS):
        got = answer(model, tok, p, max_new=80, temperature=0.8, seed=seed + i)
        samples.append({"prompt": p, "text": got})
    try:
        os.makedirs(WRITING_DIR, exist_ok=True)
        fn = os.path.join(WRITING_DIR, f"{card['ts'].replace(':', '')}_gen{card['generation']}.txt")
        with open(fn, "w", encoding="utf-8") as f:
            f.write(f"# writing sample - gen {card['generation']} - {card['ts']} - {trigger}\n\n")
            for s in samples:
                f.write(f"> {s['prompt']}\n{s['text']}\n\n")
    except Exception:
        fn = None
    card["exams"].append({"exam": "E8_writing", "bar": None, "score": f"{len(samples)} samples", "accuracy": None,
                          "verdict": "INFO", "detail": samples, "file": os.path.basename(fn) if fn else None})
    if verbose:
        print(f"[exam] E8_writing          {len(samples)} samples -> {fn}")

    # E5 (INFO, internal): did consolidated facts move into the weights?
    if consolidated_facts:
        rows, passed, total = [], 0, 0
        for fact in consolidated_facts:
            m = WORD_MEANS.match(fact.strip())
            if not m:
                continue
            word, gloss = m.group(1).lower(), m.group(2).strip()
            got = answer(model, tok, f"q : what does {word} mean ? a :")
            exp = f"the word {word} means {gloss} ."
            ok = got.strip() == exp
            passed += ok; total += 1; rows.append({"q": word, "expected": exp, "got": got, "ok": ok})
        if total:
            record("E5_facts_in_weights", None, passed, total, rows)

    history = []
    if os.path.exists(REPORT_CARD):
        try:
            prev = json.load(open(REPORT_CARD, encoding="utf-8"))
            history = prev.get("history", []) + [
                {k: prev.get(k) for k in ("ts", "seed", "corpus_sha", "generation", "trigger", "vocab_size")}
                | {"exams": [{kk: e[kk] for kk in ("exam", "score", "accuracy", "verdict")}
                              for e in prev.get("exams", [])]}
            ]
        except Exception:
            history = []
    card["history"] = history
    with open(REPORT_CARD, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)

    # chain the exam run into memory: provenance travels with the fact
    store = store or MemoryStore()
    for ex in card["exams"]:
        store.remember(f"exam {ex['exam']} {ex['verdict']} {ex['score']} (bar {ex['bar']}) gen {card['generation']}",
                       provenance="exam", state="confirmed")
    store.meter("exam", {"trigger": trigger, "generation": card["generation"],
                         "vocab_size": card["vocab_size"],
                         "results": [{k: e[k] for k in ("exam", "score", "accuracy", "verdict")}
                                     for e in card["exams"]]})
    store.meter("exam_detail", {"generation": card["generation"], "items": detail_rows})
    if was_training:
        model.train()
    if verbose:
        print(f"[exam] report card -> {REPORT_CARD} (and hash-chained into memory)")
    return card


if __name__ == "__main__":
    run_exams()
