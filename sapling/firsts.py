"""Small "notice this" detectors. They read Sapling's records and never gate or train it."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIRSTS_TOOL = Path(os.environ.get("SAPLING_FIRSTS_TOOL", str(ROOT.parent.parent / "AI-Shared" / "tools" / "firsts.py")))
FIRSTS_FILE = FIRSTS_TOOL.resolve().parents[1] / "state" / "firsts" / "sapling.jsonl"
DETECTOR_LOG = ROOT / "results" / "firsts_detector.log"
MIN_NEW_BEST_ACCURACY_GAIN = 0.05
MIN_NEW_BEST_ANSWER_GAIN = 2


def detector_state_path(anchor: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(anchor)), "firsts_detector.json")


def log_error(message: str) -> None:
    try:
        DETECTOR_LOG.parent.mkdir(parents=True, exist_ok=True)
        with DETECTOR_LOG.open("a", encoding="utf-8") as f:
            f.write(message + "\n")
    except Exception:
        pass


def emit(event: dict) -> bool:
    args = [sys.executable, str(FIRSTS_TOOL), "add", "--project", "sapling",
            "--what", event["what"], "--why", event["why"], "--proof", event["proof"], "--by", "kimi"]
    if event.get("checked"):
        args.append("--checked")
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=90, stdin=subprocess.DEVNULL)
        if result.returncode:
            log_error(f"firsts tool exited {result.returncode}: {(result.stderr or result.stdout).strip()[:500]}")
            return False
        return True
    except Exception as e:
        log_error(f"firsts tool failed: {type(e).__name__}: {e}")
        return False


def read_firsts(limit: int = 50) -> list[dict]:
    try:
        lines = FIRSTS_FILE.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return []
    out = []
    for line in reversed(lines):
        try:
            row = json.loads(line)
            if row.get("project") == "sapling":
                out.append({k: row.get(k) for k in ("at", "what", "why", "proof", "checked", "by")})
        except (TypeError, ValueError):
            continue
        if len(out) >= limit:
            break
    return out


def _load_state(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        return state if isinstance(state, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_state(path: str, state: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _score(exam: dict) -> tuple[int, int, float] | None:
    match = re.fullmatch(r"(\d+)/(\d+)", str(exam.get("score", "")))
    if not match:
        return None
    passed, total = map(int, match.groups())
    return passed, total, passed / total if total else 0.0


def _exam_history(card: dict) -> list[dict]:
    current = {k: card.get(k) for k in ("ts", "trigger", "generation")}
    current["exams"] = card.get("exams", [])
    return list(card.get("history", [])) + [current]


def _exam_key(card: dict) -> str:
    return f"{card.get('ts')}|{card.get('generation')}"


def _exam_reference(card: dict, exam: dict) -> str:
    return (f"results/report_card.json {card.get('ts')} gen {card.get('generation')}: "
            f"{exam.get('exam')} {exam.get('score')} {exam.get('verdict')}")


def scan_exams(card: dict, state_path: str) -> list[dict]:
    state = _load_state(state_path)
    key = _exam_key(card)
    if state.get("exam_last") == key:
        return []
    rows = _exam_history(card)
    if not state.get("exam_initialized"):
        best = {}
        passed_exams = set()
        for row in rows:
            for exam in row.get("exams", []):
                score = _score(exam)
                if score:
                    passed, total, accuracy = score
                    best_key = f"{exam.get('exam')}|{total}"
                    if best_key not in best or accuracy > best[best_key]["accuracy"]:
                        best[best_key] = {"passed": passed, "total": total, "accuracy": accuracy,
                                          "score": exam.get("score"), "ts": row.get("ts"), "generation": row.get("generation")}
                if exam.get("verdict") == "PASS":
                    passed_exams.add(str(exam.get("exam")))
        state.update({"exam_initialized": True, "exam_best": best,
                      "exam_passed": sorted(passed_exams), "exam_last": key})
        _save_state(state_path, state)
        return []

    events = []
    best = state.setdefault("exam_best", {})
    passed_exams = set(state.get("exam_passed", []))
    pending_rows = [rows[-1]]
    for i, row in enumerate(rows):
        if _exam_key(row) == state.get("exam_last"):
            pending_rows = rows[i + 1:]
            break
    for row in pending_rows:
        for exam in row.get("exams", []):
            name = str(exam.get("exam"))
            score = _score(exam)
            first_pass = exam.get("verdict") == "PASS" and name not in passed_exams
            clear_best = False
            previous = None
            if score:
                passed, total, accuracy = score
                best_key = f"{name}|{total}"
                previous = best.get(best_key)
                clear_best = bool(previous and passed - previous["passed"] >= MIN_NEW_BEST_ANSWER_GAIN
                                  and accuracy - previous["accuracy"] + 1e-12 >= MIN_NEW_BEST_ACCURACY_GAIN)
                if previous is None or accuracy > previous["accuracy"]:
                    best[best_key] = {"passed": passed, "total": total, "accuracy": accuracy,
                                      "score": exam.get("score"), "ts": row.get("ts"), "generation": row.get("generation")}
            if first_pass:
                bar = exam.get("bar")
                bar_text = f" at the predeclared {float(bar):.0%} bar" if isinstance(bar, (int, float)) else ""
                events.append({"what": f"Sapling passed {name} for the first time: {exam.get('score')}",
                               "why": f"It crossed{bar_text} after no earlier PASS on this exam.",
                               "proof": _exam_reference(row, exam), "checked": False})
            elif clear_best:
                gain = (accuracy - previous["accuracy"]) * 100
                events.append({"what": f"Sapling set a clear new best on {name}: {previous['score']} -> {exam.get('score')}",
                               "why": f"Same exam and question count: +{gain:.1f} percentage points and +{passed - previous['passed']} correct answers, above the noise guard.",
                               "proof": (f"{_exam_reference(row, exam)}; previous best {previous['score']} "
                                         f"at {previous.get('ts')} gen {previous.get('generation')}"), "checked": False})
            if exam.get("verdict") == "PASS":
                passed_exams.add(name)
    state["exam_passed"] = sorted(passed_exams)
    state["exam_last"] = key
    _save_state(state_path, state)
    for event in events:
        emit(event)
    return events


def _confirmed_self_memory(store) -> dict | None:
    row = store.db.execute(
        "SELECT m.id, m.ts, m.content FROM memories m WHERE m.provenance='self' "
        "AND EXISTS (SELECT 1 FROM memories v WHERE v.provenance='verdict' AND v.content=('confirm #' || m.id)) "
        "AND NOT EXISTS (SELECT 1 FROM memories v WHERE v.provenance='verdict' AND v.content=('invalidate #' || m.id)) "
        "ORDER BY m.id LIMIT 1"
    ).fetchone()
    return {"id": row[0], "ts": row[1], "content": row[2]} if row else None


def scan_store(store, state_path: str, include_own_words: bool = True) -> list[dict]:
    state = _load_state(state_path)
    events = []
    goal_rows = store.ledger_rows("goal", limit=1000000 if not state.get("goals_initialized") else 1000,
                                  since_id=int(state.get("goal_last_id", 0)))
    goal_types = set(state.get("goal_types", []))
    for row in reversed(goal_rows):
        part = str(row.get("part", ""))
        goal_type = part.split(":", 1)[0] if part else ""
        if not goal_type:
            continue
        if state.get("goals_initialized") and row.get("status") == "open" and goal_type not in goal_types:
            events.append({"what": f"Sapling set its first {goal_type} goal: {part}",
                           "why": "No earlier open goal in its ledger used this goal type.",
                           "proof": (f"results/memory.db ledger row {row.get('id')}: {part} {row.get('status')}; "
                                     f"{row.get('misses_recent')} misses in {row.get('tries_recent')} tries"), "checked": False})
        goal_types.add(goal_type)
    if goal_rows:
        state["goal_last_id"] = max(int(row["id"]) for row in goal_rows)
    state["goal_types"] = sorted(goal_types)
    state["goals_initialized"] = True

    tick_rows = store.ledger_rows("tick", limit=1000000 if not state.get("ticks_initialized") else 1000,
                                  since_id=int(state.get("tick_last_id", 0)))
    behaviors = set(state.get("behaviors", []))
    self_stop_seen = bool(state.get("self_stop_seen"))
    for row in reversed(tick_rows):
        action = str(row.get("action", ""))
        result = row.get("result") or {}
        if state.get("ticks_initialized") and action == "stop" and not self_stop_seen:
            report = result.get("report")
            events.append({"what": f"Sapling chose to stop for the first time at tick {row.get('tick')}",
                           "why": "The heartbeat's own stop drive won the draw; this was not an owner stop.",
                           "proof": f"results/memory.db ledger row {row.get('id')}; results/stop_reports/{report}", "checked": False})
        self_stop_seen = self_stop_seen or action == "stop"
        kind = result.get("kind")
        if kind:
            behavior = f"{action}:{kind}"
            if state.get("ticks_initialized") and behavior not in behaviors:
                events.append({"what": f"Sapling showed a new tick behavior: {behavior}",
                               "why": "This typed outcome had never appeared in its earlier tick record; the notice does not judge its content correct.",
                               "proof": f"results/memory.db ledger row {row.get('id')}, tick {row.get('tick')}: {json.dumps(result, default=str)[:300]}",
                               "checked": False})
            behaviors.add(behavior)
    if tick_rows:
        state["tick_last_id"] = max(int(row["id"]) for row in tick_rows)
    state["behaviors"] = sorted(behaviors)
    state["self_stop_seen"] = self_stop_seen
    state["ticks_initialized"] = True

    own_words_seen = bool(state.get("own_words_seen"))
    check_own_words = include_own_words or not state.get("own_words_initialized")
    own = _confirmed_self_memory(store) if check_own_words and not own_words_seen else None
    if state.get("own_words_initialized") and own and not own_words_seen:
        events.append({"what": f"Sapling had its own words confirmed for the first time (memory #{own['id']})",
                       "why": "A self-written memory crossed the existing confirmation path and has not been invalidated.",
                       "proof": f"results/memory.db memory #{own['id']}: {own['content'][:200]}", "checked": False})
    if check_own_words:
        state["own_words_seen"] = own_words_seen or bool(own)
        state["own_words_initialized"] = True
    _save_state(state_path, state)
    for event in events:
        emit(event)
    return events
