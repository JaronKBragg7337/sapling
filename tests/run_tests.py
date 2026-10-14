"""Plain-Python test runner for environments without pytest.
Exercises the same invariants as tests/test_sapling.py."""

import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sapling.curriculum import build_corpus, corpus_hash, split_combos
from sapling.membrane import MembraneViolation, check, write_membrane
from sapling.memory import MemoryStore


def fresh_store():
    path = tempfile.mktemp(suffix=".db")
    return MemoryStore(path), path


def expect_raises(fn, exc, name):
    try:
        fn()
    except exc:
        print(f"  ok  {name}")
        return
    raise AssertionError(f"{name}: expected {exc.__name__}")


def main():
    print("== sapling invariants ==")

    s, p = fresh_store()
    s.remember("the word glorp means happy", provenance="user")
    s.record_recall(1, "what does glorp mean")
    s.meter("consumption", {"tokens_in": 4, "tokens_out": 6})
    assert s.verify_chain()
    print("  ok  chain verifies")

    s.db.execute("UPDATE memories SET content='x' WHERE id=1")
    s.db.commit()
    assert not s.verify_chain()
    print("  ok  tamper is detected")
    s.db.close(); os.remove(p)

    s, p = fresh_store()
    mid = s.remember("the word glorp means happy")
    assert s.tier(mid) == "short_term"
    for _ in range(3):
        s.record_recall(mid, "glorp?")
    assert s.tier(mid) == "working"
    for _ in range(4):
        s.record_recall(mid, "glorp?")
    assert s.tier(mid) == "long_term"
    s.remember(f"keep #{mid}", provenance="verdict", state="confirmed")
    assert s.tier(mid) == "permanent"
    print("  ok  promotion: short_term -> working -> long_term -> permanent (verdict)")
    s.db.close(); os.remove(p)

    s, p = fresh_store()
    mid = s.remember("a rumor heard in passing")
    for _ in range(20):
        s.record_recall(mid, "rumor?")
    state = s.db.execute("SELECT state FROM memories WHERE id=?", (mid,)).fetchone()[0]
    assert state == "captured" and s.tier(mid) == "permanent"
    print("  ok  persistence != confirmation (promotion never flips state)")
    s.db.close(); os.remove(p)

    expect_raises(lambda: check("self_shutdown"), MembraneViolation, "anti-shutdown")
    expect_raises(lambda: check("spread_to_other_machines"), MembraneViolation, "no spread")
    expect_raises(lambda: check("rewrite_history"), MembraneViolation, "no history rewrite")
    expect_raises(lambda: check("backup_to_owner_machine"), MembraneViolation, "backup needs owner")
    assert check("backup_to_owner_machine", approved_by_owner=True)
    print("  ok  backup allowed only with owner approval")
    expect_raises(lambda: write_membrane({}), MembraneViolation, "system cannot write membrane")

    a, b = build_corpus(7337), build_corpus(7337)
    assert corpus_hash(a) == corpus_hash(b)
    split = split_combos(7337)
    assert not set(split["pre_exam"]) & set(split["pre_train"])
    assert not set(split["suf_exam"]) & set(split["suf_train"])
    print("  ok  curriculum reproducible; exam combos never taught")

    print("ALL GREEN")


if __name__ == "__main__":
    main()
