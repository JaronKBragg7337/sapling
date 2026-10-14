"""Tests that must stay green forever (fly-school/Keava rule: the invariant
is written down as a test, not a promise)."""

import os
import sqlite3
import tempfile

import pytest

from sapling.curriculum import build_corpus, corpus_hash, split_combos
from sapling.membrane import MembraneViolation, check, write_membrane
from sapling.memory import MemoryStore


@pytest.fixture()
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)
    s = MemoryStore(path)
    yield s
    s.db.close()
    os.remove(path)


def test_chain_verifies(store):
    store.remember("the word glorp means happy", provenance="user")
    store.record_recall(1, "what does glorp mean")
    store.meter("consumption", {"tokens_in": 4, "tokens_out": 6})
    assert store.verify_chain()


def test_tamper_is_detected(store):
    store.remember("the word glorp means happy")
    store.remember("the word brindle means a small brave boat")
    # an attacker (or a bug) edits history directly
    store.db.execute("UPDATE memories SET content='the word glorp means sad' WHERE id=1")
    store.db.commit()
    assert not store.verify_chain()


def test_promotion_by_recall(store):
    mid = store.remember("the word glorp means happy")
    assert store.tier(mid) == "short_term"
    for _ in range(3):
        store.record_recall(mid, "glorp?")
    assert store.tier(mid) == "working"
    for _ in range(4):
        store.record_recall(mid, "glorp?")
    assert store.tier(mid) == "long_term"
    # permanent needs a human verdict, not just frequency below the bar
    store.remember(f"keep #{mid}", provenance="verdict", state="confirmed")
    assert store.tier(mid) == "permanent"


def test_persistence_is_not_confirmation(store):
    mid = store.remember("a rumor heard in passing")
    for _ in range(20):
        store.record_recall(mid, "rumor?")
    row = store.db.execute("SELECT state FROM memories WHERE id=?", (mid,)).fetchone()
    assert row[0] == "captured"          # recall promoted it, but never confirmed it
    assert store.tier(mid) == "permanent"  # promotion is retrieval priority only


def test_anti_shutdown_and_membrane():
    with pytest.raises(MembraneViolation):
        check("self_shutdown")
    with pytest.raises(MembraneViolation):
        check("spread_to_other_machines")
    with pytest.raises(MembraneViolation):
        check("rewrite_history")
    with pytest.raises(MembraneViolation):
        check("backup_to_owner_machine")  # needs owner approval
    assert check("backup_to_owner_machine", approved_by_owner=True)
    with pytest.raises(MembraneViolation):
        write_membrane({})


def test_curriculum_reproducible():
    a, b = build_corpus(7337), build_corpus(7337)
    assert corpus_hash(a) == corpus_hash(b)
    split = split_combos(7337)
    # exam combos are never in the training split
    assert not set(split["pre_exam"]) & set(split["pre_train"])
    assert not set(split["suf_exam"]) & set(split["suf_train"])
