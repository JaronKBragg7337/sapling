"""Sapling goals — practice goals it sets for itself from its own failure record.

Jaron 10/14 (design restraint 6, "exams set from outside -> asked for more
explanation"): the outside exams stay as the check that cannot be faked. This
module is the other half: it reads its own misses (practice ticks + exam
misses), finds the weak PARTS (a prefix, a suffix, a root), opens a goal per
weak part, practices it, and closes the goal when the recent tries are clean.

Exam hygiene: a goal is about a PART, never about an exam question. Practice
material comes only from the training split, so held-out exam combos are
never studied - the exam stays cold.
"""

from __future__ import annotations

import random

from .curriculum import PREFIXES, ROOTS, SUFFIXES, compose, prefix_gloss, split_combos, suffix_gloss

RECENT_TRIES = 10          # a goal looks at the last N tries of its part
OPEN_IF_MISSES_GE = 1      # any miss in the recent window keeps the goal open


def part_items(split: dict) -> dict[str, list[tuple[str, str]]]:
    """part -> [(prompt, expected)] from TRAINING material only."""
    items: dict[str, list[tuple[str, str]]] = {}
    for p, r in split["pre_train"]:
        w, g = compose([p, r]), prefix_gloss(r, p)
        q, a = f"q : what does {w} mean ? a :", f"{w} means {g} ."
        items.setdefault(f"prefix:{p}", []).append((q, a))
        items.setdefault(f"root:{r}", []).append((q, a))
    for r, s in split["suf_train"]:
        w, g = compose([r, s]), suffix_gloss(r, s)
        q, a = f"q : what does {w} mean ? a :", f"{w} means {g} ."
        items.setdefault(f"suffix:{s}", []).append((q, a))
        items.setdefault(f"root:{r}", []).append((q, a))
    for p, g in PREFIXES.items():
        items.setdefault(f"prefix:{p}", []).append((f"q : what does the prefix {p} mean ? a :", f"{g} ."))
    for s, g in SUFFIXES.items():
        items.setdefault(f"suffix:{s}", []).append((f"q : what does the suffix {s} mean ? a :", f"{g} ."))
    for r, g in ROOTS.items():
        items.setdefault(f"root:{r}", []).append((f"q : what does the root {r} mean ? a :", f"the root {r} means {g} ."))
    return items


def parts_of_question(q: str) -> list[str]:
    """Which parts a question exercises (by the word parts it contains)."""
    toks = q.replace("?", " ").replace(":", " ").split()
    out = []
    for t in toks:
        if t in PREFIXES:
            out.append(f"prefix:{t}")
        if t in SUFFIXES:
            out.append(f"suffix:{t}")
        if t in ROOTS:
            out.append(f"root:{t}")
    return out


def failure_record(store, limit_rows: int = 400) -> dict[str, list[bool]]:
    """part -> [ok, ok, ...] oldest..newest, from practice ticks AND exam misses."""
    rec: dict[str, list[bool]] = {}
    rows = store.ledger_rows("practice", limit=limit_rows)
    rows += store.ledger_rows("exam_detail", limit=limit_rows)
    rows.sort(key=lambda r: r["id"])
    for row in rows:
        for it in row.get("items", []):
            for part in it.get("parts", []):
                rec.setdefault(part, []).append(bool(it.get("ok")))
    return rec


def derive_goals(store) -> list[dict]:
    rec = failure_record(store)
    goals = []
    for part, hist in sorted(rec.items()):
        recent = hist[-RECENT_TRIES:]
        misses = sum(1 for ok in recent if not ok)
        status = "open" if misses >= OPEN_IF_MISSES_GE else "met"
        goals.append({"part": part, "status": status, "misses_recent": misses,
                      "tries_recent": len(recent), "tries_total": len(hist),
                      "target": f"0 misses in the last {RECENT_TRIES} tries"})
    goals.sort(key=lambda g: (g["status"] != "open", -g["misses_recent"]))
    return goals


class GoalTracker:
    """Logs only transitions (goal opened / goal met) to the ledger."""

    def __init__(self, store):
        self.store = store
        self._last: dict[str, str] = {}
        for row in reversed(store.ledger_rows("goal", limit=5000)):
            self._last[row["part"]] = row["status"]

    def update(self) -> list[dict]:
        goals = derive_goals(self.store)
        for g in goals:
            prev = self._last.get(g["part"])
            if prev != g["status"]:
                self.store.meter("goal", {"part": g["part"], "status": g["status"],
                                          "misses_recent": g["misses_recent"],
                                          "tries_recent": g["tries_recent"], "target": g["target"]})
                self._last[g["part"]] = g["status"]
        return goals

    def open_goals(self) -> list[dict]:
        return [g for g in derive_goals(self.store) if g["status"] == "open"]


def pick_practice(goals: list[dict], rng: random.Random, batch: int = 6,
                  seed: int = 7337) -> list[dict]:
    """Weak parts first (weighted by misses), then random training material."""
    split = split_combos(seed)
    items = part_items(split)
    out = []
    open_goals = [g for g in goals if g["status"] == "open"]
    weights = [1 + g["misses_recent"] for g in open_goals]
    n_focus = min(batch, len(open_goals) and batch * 2 // 3 or 0)
    for _ in range(n_focus):
        g = rng.choices(open_goals, weights=weights, k=1)[0]
        q, a = rng.choice(items[g["part"]])
        out.append({"q": q, "a": a, "parts": parts_of_question(q), "focus": g["part"]})
    all_items = [it for lst in items.values() for it in lst]
    while len(out) < batch:
        q, a = rng.choice(all_items)
        out.append({"q": q, "a": a, "parts": parts_of_question(q), "focus": None})
    return out
