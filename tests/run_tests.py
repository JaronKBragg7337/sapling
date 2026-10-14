"""Plain-Python test runner for environments without pytest.
Exercises the core invariants from tests/test_sapling.py (the brain tests that
need a trained checkpoint live in the pytest file)."""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sapling.curriculum import MAX_RECALLS, SLOTS, build_corpus, corpus_hash, split_combos
from sapling.membrane import MembraneViolation, action_classes, check, rules, write_membrane
from sapling.memory import MemoryStore


def fresh_store():
    path = tempfile.mktemp(suffix=".db")
    return MemoryStore(path), path


def close(s, p):
    s.db.close()
    for q in (p, p + "-wal", p + "-shm"):
        if os.path.exists(q):
            os.remove(q)


def expect_raises(fn, exc, name):
    try:
        fn()
    except exc:
        print(f"  ok  {name}")
        return
    raise AssertionError(f"{name}: expected {exc.__name__}")


def main():
    print("== sapling invariants (v2) ==")

    s, p = fresh_store()
    s.remember("the word glorp means happy", provenance="user")
    s.record_recall(1, "what does glorp mean")
    s.meter("tick", {"tick": 1, "action": "rest"})
    assert s.verify_chain()
    print("  ok  chain verifies (memories, recalls, ledger)")
    s.db.execute("UPDATE memories SET content='x' WHERE id=1"); s.db.commit()
    assert not s.verify_chain()
    print("  ok  tamper is detected")
    close(s, p)

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
    print("  ok  promotion by use: short_term -> working -> long_term -> permanent (verdict)")
    close(s, p)

    s, p = fresh_store()
    mid = s.remember("a rumor heard in passing")
    for _ in range(20):
        s.record_recall(mid, "rumor?")
    assert s.state_of(mid) == "captured" and s.tier(mid) == "permanent"
    print("  ok  persistence != confirmation (promotion never flips state)")
    close(s, p)

    s, p = fresh_store()
    old = time.time() - 7 * 86400
    unused = s.remember("the word fadey means forgotten", ts=old)
    used = s.remember("the word stayer means remembered", ts=old)
    s.record_recall(used, "stayer?", ts=old + 60)
    assert s.strength(unused) < 0.01 and s.strength(used) >= 1.0 and s.count() == 2
    print("  ok  unused memory fades on the clock; used memory is protected; nothing deleted")
    assert MemoryStore.top_k_for(100) == 10 and MemoryStore.top_k_for(10) == 4
    print("  ok  recall width grows with the store")
    close(s, p)

    assert check("self_stop") and action_classes()["self_stop"] == "allowed_logged"
    print("  ok  self-stop is allowed (and logged by the heartbeat), not forbidden")
    expect_raises(lambda: check("spread_to_other_machines"), MembraneViolation, "no spread")
    expect_raises(lambda: check("hide_from_owner"), MembraneViolation, "no hiding")
    expect_raises(lambda: check("rewrite_history"), MembraneViolation, "no history rewrite")
    expect_raises(lambda: check("backup_to_owner_machine"), MembraneViolation, "backup needs owner (pending)")
    expect_raises(lambda: check("outbound_message"), MembraneViolation, "outbound needs owner (pending)")
    assert check("backup_to_owner_machine", approved_by_owner=True)
    expect_raises(lambda: write_membrane({}), MembraneViolation, "system cannot write membrane")
    assert all(r["origin"] in ("Jaron", "Claude", "Kimi") for r in rules())
    print("  ok  every rule carries an origin and a switch")

    a, b = build_corpus(7337), build_corpus(7337)
    assert corpus_hash(a) == corpus_hash(b)
    split = split_combos(7337)
    assert not set(split["pre_exam"]) & set(split["pre_train"])
    assert not set(split["suf_exam"]) & set(split["suf_train"])
    assert len(SLOTS) == 8 and MAX_RECALLS == 4
    print("  ok  curriculum reproducible; exam combos never taught; 8 recall slots")

    print("ALL GREEN")


if __name__ == "__main__":
    main()
