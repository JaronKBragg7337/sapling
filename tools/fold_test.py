"""The teacher's retention measurement (10/16): N consolidations forced in a controlled, ISOLATED run - a copy of
memory.db, checkpoints / report card / writing samples into a scratch folder - so the live core, the ledger and
the heartbeat's drives are untouched. Same code path as the heartbeat (consolidate.consolidate), so it measures what
the heartbeat would do; then tools/measure_core.py on the last generation.

    python tools/fold_test.py --core results/cloud/20261016-0122/core_staged.pt --label "lr 7.5e-5" [--lr 7.5e-5]
        [--lesson 12 --batch 44] [--folds 5]
"""
import argparse
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--lesson", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    a = ap.parse_args()
    from sapling import consolidate as C, exams as E, membrane
    from sapling.goals import GoalTracker
    from sapling.memory import MemoryStore
    from sapling.session import Brain
    work = tempfile.mkdtemp(prefix="sapling_folds_")
    with sqlite3.connect(os.path.join(ROOT, "results", "memory.db")) as src, sqlite3.connect(os.path.join(work, "memory.db")) as dst:
        src.backup(dst)
    os.makedirs(os.path.join(work, "ck"))
    C.CHECKPOINT_DIR, C.CHECKPOINT = os.path.join(work, "ck"), os.path.join(work, "live.pt")
    E.REPORT_CARD, E.WRITING_DIR = os.path.join(work, "card.json"), os.path.join(work, "writing")
    if a.lesson is not None:
        C.LESSON_REPLAY = a.lesson
    if a.batch is not None:
        C.BATCH_WITH_LESSON = a.batch
    store = MemoryStore(os.path.join(work, "memory.db"))
    brain = Brain(store=store, path=a.core)
    steps = int(membrane.settings()["consolidate_steps"])
    rows = []
    t0 = time.time()
    for i in range(a.folds):
        goals = GoalTracker(store).update()
        r = C.consolidate(brain, store, goals, steps=steps, lr=a.lr, verbose=False)
        rows.append({k: r.get(k) for k in ("generation", "lr", "loss_first", "loss_last", "lesson_share", "read_share", "talk_share", "seconds")})
        print(json.dumps(rows[-1]), flush=True)
    last = os.path.join(work, "ck", f"gen_{brain.generation:04d}.pt")
    from measure_core import measure
    m = measure(last, f"fold test: {a.label} ({a.folds} folds from {os.path.basename(a.core)})")
    out = {"label": a.label, "folds": rows, "seconds": round(time.time() - t0), "measure": {k: m[k] for k in ("window_loss", "exams")}}
    os.makedirs(os.path.join(ROOT, "results", "teacher"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "teacher", "fold_tests.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(out) + "\n")
    print(json.dumps(out["measure"]), flush=True)
    store.db.close()
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
