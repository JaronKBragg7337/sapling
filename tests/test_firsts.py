import json

import pytest

from sapling import firsts
from sapling.memory import MemoryStore


@pytest.fixture()
def sent(monkeypatch):
    events = []
    monkeypatch.setattr(firsts, "emit", lambda event: events.append(event) or True)
    return events


def exam_card(ts, generation, score, verdict="INFO", history=None):
    return {"ts": ts, "generation": generation, "history": history or [],
            "exams": [{"exam": "E_test", "score": score, "verdict": verdict, "bar": 0.6}]}


def test_exam_first_scan_is_only_a_baseline(tmp_path, sent):
    state = str(tmp_path / "state.json")
    old = exam_card("2026-01-01", 1, "50/100", "FAIL")
    assert firsts.scan_exams(exam_card("2026-01-02", 2, "80/100", "PASS", [old]), state) == []
    assert sent == []
    saved = json.loads(open(state, encoding="utf-8").read())
    assert saved["exam_best"]["E_test|100"]["score"] == "80/100"
    assert saved["exam_passed"] == ["E_test"]


def test_exam_new_best_needs_five_points_and_two_answers(tmp_path, sent):
    state = str(tmp_path / "state.json")
    firsts.scan_exams(exam_card("2026-01-01", 1, "50/100"), state)
    firsts.scan_exams(exam_card("2026-01-02", 2, "52/100"), state)
    firsts.scan_exams(exam_card("2026-01-03", 3, "55/100"), state)
    assert sent == []
    events = firsts.scan_exams(exam_card("2026-01-04", 4, "60/100"), state)
    assert len(events) == len(sent) == 1
    assert "+5.0 percentage points" in events[0]["why"]
    assert "50/100" not in events[0]["proof"] and "55/100 -> 60/100" in events[0]["what"]
    assert events[0]["checked"] is False


def test_exam_best_does_not_compare_different_question_counts(tmp_path, sent):
    state = str(tmp_path / "state.json")
    firsts.scan_exams(exam_card("2026-01-01", 1, "5/10"), state)
    assert firsts.scan_exams(exam_card("2026-01-02", 2, "20/20"), state) == []
    assert sent == []


def test_exam_first_pass_fires_once_and_deduplicates_repeat_card(tmp_path, sent):
    state = str(tmp_path / "state.json")
    firsts.scan_exams(exam_card("2026-01-01", 1, "50/100", "FAIL"), state)
    passed = exam_card("2026-01-02", 2, "70/100", "PASS")
    events = firsts.scan_exams(passed, state)
    assert len(events) == 1 and "first time" in events[0]["what"]
    assert "60% bar" in events[0]["why"]
    assert firsts.scan_exams(passed, state) == []
    events = firsts.scan_exams(exam_card("2026-01-03", 3, "80/100", "PASS"), state)
    assert len(events) == 1 and "new best" in events[0]["what"]
    assert len(sent) == 2 and sum("first time" in e["what"] for e in sent) == 1


def test_exam_backlog_is_processed_in_order_without_false_new_best(tmp_path, sent):
    state = str(tmp_path / "state.json")
    first = exam_card("2026-01-01", 1, "50/100")
    firsts.scan_exams(first, state)
    missed = exam_card("2026-01-02", 2, "90/100")
    current = exam_card("2026-01-03", 3, "80/100", history=[first, missed])
    events = firsts.scan_exams(current, state)
    assert len(events) == 1 and "50/100 -> 90/100" in events[0]["what"]
    assert "gen 2" in events[0]["proof"] and "80/100" not in events[0]["proof"]
    saved = json.loads(open(state, encoding="utf-8").read())
    assert saved["exam_best"]["E_test|100"]["score"] == "90/100"


def test_new_goal_type_and_tick_behavior_use_ledger_proof(tmp_path, sent):
    store = MemoryStore(str(tmp_path / "memory.db"))
    state = str(tmp_path / "state.json")
    try:
        store.meter("goal", {"part": "prefix:re", "status": "open", "misses_recent": 2, "tries_recent": 10})
        store.meter("tick", {"tick": 1, "action": "explore", "result": {"kind": "generate"}})
        assert firsts.scan_store(store, state) == []
        store.meter("goal", {"part": "reasoning:bridge", "status": "open", "misses_recent": 3, "tries_recent": 10})
        store.meter("tick", {"tick": 2, "action": "explore", "result": {"kind": "invent", "said": "new"}})
        events = firsts.scan_store(store, state)
        assert [e["what"] for e in events] == [
            "Sapling set its first reasoning goal: reasoning:bridge",
            "Sapling showed a new tick behavior: explore:invent",
        ]
        assert all("memory.db ledger row" in e["proof"] and not e["checked"] for e in events)
        assert firsts.scan_store(store, state) == []
        assert len(sent) == 2
    finally:
        store.db.close()


def test_first_self_stop_is_distinct_from_owner_stop(tmp_path, sent):
    store = MemoryStore(str(tmp_path / "memory.db"))
    state = str(tmp_path / "state.json")
    try:
        store.meter("tick", {"tick": 1, "action": "rest", "result": {}})
        firsts.scan_store(store, state)
        store.meter("tick", {"tick": 2, "action": "stop", "result": {"report": "self.json"}})
        events = firsts.scan_store(store, state)
        assert len(events) == 1 and "chose to stop" in events[0]["what"]
        assert "stop_reports/self.json" in events[0]["proof"]
    finally:
        store.db.close()


def test_first_confirmed_own_words_and_invalidation_guard(tmp_path, sent):
    store = MemoryStore(str(tmp_path / "memory.db"))
    state = str(tmp_path / "state.json")
    try:
        firsts.scan_store(store, state)
        mid = store.remember("i found a pattern", provenance="self")
        firsts.scan_store(store, state)
        assert sent == []
        store.remember(f"confirm #{mid}", provenance="verdict", state="confirmed")
        events = firsts.scan_store(store, state)
        assert len(events) == 1 and f"memory #{mid}" in events[0]["what"]
        firsts.scan_store(store, state)
        assert len(sent) == 1
    finally:
        store.db.close()

    other = MemoryStore(str(tmp_path / "other.db"))
    try:
        firsts.scan_store(other, str(tmp_path / "other-state.json"))
        mid = other.remember("not trusted", provenance="self")
        other.remember(f"confirm #{mid}", provenance="verdict", state="confirmed")
        other.remember(f"invalidate #{mid}", provenance="verdict", state="confirmed")
        assert firsts.scan_store(other, str(tmp_path / "other-state.json")) == []
    finally:
        other.db.close()


def test_read_firsts_is_newest_first_and_keeps_check_status(tmp_path, monkeypatch):
    path = tmp_path / "sapling.jsonl"
    rows = [{"at": "2026-01-01T01:00:00", "project": "sapling", "what": "old", "why": "one", "proof": "1", "checked": False, "by": "kimi"},
            {"at": "2026-01-02T01:00:00", "project": "sapling", "what": "new", "why": "two", "proof": "2", "checked": True, "by": "kimi"}]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n", encoding="utf-8")
    monkeypatch.setattr(firsts, "FIRSTS_FILE", path)
    result = firsts.read_firsts()
    assert [r["what"] for r in result] == ["new", "old"]
    assert [r["checked"] for r in result] == [True, False]
    assert "project" not in result[0]


def test_owner_relay_adds_firsts_even_when_heartbeat_state_has_none(monkeypatch):
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
    import owner_relay
    rows = [{"at": "2026-01-02T01:00:00", "what": "new", "why": "two", "proof": "2", "checked": True, "by": "kimi"}]
    monkeypatch.setattr(owner_relay, "hb_get", lambda path: {"running": True})
    monkeypatch.setattr(firsts, "read_firsts", lambda: rows)
    state = owner_relay.full_state()
    assert state["running"] is True and state["firsts"] == rows
