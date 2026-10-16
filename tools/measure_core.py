"""Measure one core the same way at every point of the 10/15 teaching plan (Claude as teacher): exams E1-E10 (score
only - no ledger or report-card writes), loss on the fixed text windows (reading.eval_windows), saved as one JSON row
in results/teacher/measurements.jsonl. The chat samples are tools/chat_samples.py.

    python tools/measure_core.py --label "before - v3.2 gen 287" --core results/checkpoints/gen_0287.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sapling.exams import run_exams  # noqa: E402
from sapling.memory import MemoryStore  # noqa: E402
from sapling.reading import eval_windows, exam_scores, window_losses  # noqa: E402
from sapling.train import CHECKPOINT, RESULTS_DIR, load_checkpoint  # noqa: E402

OUT = os.path.join(RESULTS_DIR, "teacher", "measurements.jsonl")


def e7l_kind(q: str) -> str:
    q = q.split(" q : ")[-1]
    for k in ("which sentence", "word comes after", "word comes before", "copy the first sentence of the passage with",
              "copy the first sentence"):
        if k in q:
            return k
    return "other"


def e7l_by_kind(model, tok) -> dict:
    """E7L broken down by question kind and by one / two passages (the exam's own items, same greedy answers)."""
    from sapling.curriculum3 import exam_items_longread
    from sapling.exams import answer
    out = {}
    for it in exam_items_longread(7337):
        if len(tok.encode(it["q"], eos=False)) + 44 > model.block_size:
            continue
        ok = answer(model, tok, it["q"], max_new=44).strip() == it["a"]
        for key in (e7l_kind(it["q"]), f"{e7l_kind(it['q'])} / {it['kind']}"):
            c = out.setdefault(key, [0, 0]); c[0] += ok; c[1] += 1
    return {k: f"{a}/{b}" for k, (a, b) in sorted(out.items())}


def measure(core: str, label: str) -> dict:
    if os.environ.get("SAPLING_DEVICE", "cuda") != "cpu":
        from sapling.reading import wait_until_gpu_free
        wait_until_gpu_free(who="measure_core")
    model, tok, ck = load_checkpoint(path=core)
    user_facts = [f["content"] for f in MemoryStore().facts() if f["provenance"] == "user"]
    card = run_exams(verbose=False, model=model, tok=tok, meta={"generation": ck.get("generation")}, store=None,
                     consolidated_facts=user_facts, write=False, trigger=f"teacher measure: {label}")
    row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "label": label, "core": os.path.relpath(core, ROOT),
           "generation": ck.get("generation"), "params": model.n_params(), "config": model.config(),
           "window_loss": window_losses(model, eval_windows(block=model.block_size)),
           "exams": {k: v["score"] for k, v in exam_scores(card).items()}, "e7l_by_kind": e7l_by_kind(model, tok)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", default=CHECKPOINT)
    ap.add_argument("--label", required=True)
    a = ap.parse_args()
    print(json.dumps(measure(a.core, a.label), indent=1))
