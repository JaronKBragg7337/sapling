"""self_memory_drift.py — D2 in DESIGN-CHOICES.md: self-made memories train its brain.
Print, per consolidation, how many self rows were in the mix (and how many were new)
next to the exam scores that followed, so "with vs without" can be read off one table.

Usage: python tools/self_memory_drift.py [--json]
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sapling.memory import MemoryStore  # noqa: E402

EXAMS = ("E1_composition", "E2_affix_recall", "E3_recall_echo", "E4_deferral", "E5_facts_in_weights")


def table() -> list[dict]:
    store = MemoryStore()
    rows = sorted(store.ledger_rows("consolidation", limit=100000), key=lambda r: r.get("generation", 0))
    out = []
    for r in rows:
        ex = {e["exam"]: e["score"] for e in r.get("exams", []) or []}
        out.append({"gen": r.get("generation"), "facts_total": r.get("facts_total"),
                    "self": r.get("facts_self"), "self_new": len(r.get("self_new") or []),
                    "user": r.get("facts_user"), "loss_last": r.get("loss_last"),
                    **{k.split("_")[0]: ex.get(k, "-") for k in EXAMS}})
    store.db.close()
    return out


def main():
    rows = table()
    if "--json" in sys.argv:
        print(json.dumps(rows, indent=1)); return
    print(f"{'gen':>4} {'facts':>5} {'self':>5} {'+new':>4} {'user':>4} {'loss':>6}  E1    E2    E3    E4    E5")
    for r in rows:
        print(f"{r['gen']:>4} {str(r['facts_total']):>5} {str(r['self']):>5} {r['self_new']:>4} {str(r['user']):>4} "
              f"{str(r['loss_last']):>6}  {r['E1']:<5} {r['E2']:<5} {r['E3']:<5} {r['E4']:<5} {r['E5']}")
    with_new = [r for r in rows if r["self_new"]]
    without = [r for r in rows if r["self"] is not None and not r["self_new"]]
    print(f"\nconsolidations with new self rows: {len(with_new)}; with self count logged but none new: {len(without)}; "
          f"older rows (before self logging, 10/14 3 PM): {len([r for r in rows if r['self'] is None])}")


if __name__ == "__main__":
    main()
