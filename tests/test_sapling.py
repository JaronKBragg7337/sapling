"""Tests that must stay green (fly-school/Keava rule: the invariant is written
down as a test, not a promise). v2: updated to Jaron's 10/14 decisions -
self-stop is allowed and logged, memory fades only when unused, recall width
grows, vocabulary grows, consolidation leaves a checkpoint and an exam run."""

import json
import os
import tempfile
import time

import pytest

from sapling.curriculum import MAX_RECALLS, SLOTS, build_corpus, corpus_hash, split_combos
from sapling.membrane import MembraneViolation, action_classes, check, rules, settings, write_membrane
from sapling.memory import MemoryStore


@pytest.fixture()
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)
    s = MemoryStore(path)
    yield s
    s.db.close()
    for p in (path, path + "-wal", path + "-shm"):
        if os.path.exists(p):
            os.remove(p)


# -- record -------------------------------------------------------------------
def test_chain_verifies(store):
    store.remember("the word glorp means happy", provenance="user")
    store.record_recall(1, "what does glorp mean")
    store.meter("consumption", {"tokens_in": 4, "tokens_out": 6})
    store.meter("tick", {"tick": 1, "action": "rest"})
    assert store.verify_chain()


def test_tamper_is_detected(store):
    store.remember("the word glorp means happy")
    store.remember("the word brindle means a small brave boat")
    store.db.execute("UPDATE memories SET content='the word glorp means sad' WHERE id=1")
    store.db.commit()
    assert not store.verify_chain()


def test_ledger_tamper_is_detected(store):
    store.meter("tick", {"tick": 1, "action": "practice"})
    store.db.execute("UPDATE ledger SET payload='{}' WHERE id=1")
    store.db.commit()
    assert not store.verify_chain()


# -- memory: promotion by use, fading only when unused -------------------------
def test_promotion_by_recall(store):
    mid = store.remember("the word glorp means happy")
    assert store.tier(mid) == "short_term"
    for _ in range(3):
        store.record_recall(mid, "glorp?")
    assert store.tier(mid) == "working"
    for _ in range(4):
        store.record_recall(mid, "glorp?")
    assert store.tier(mid) == "long_term"
    store.remember(f"keep #{mid}", provenance="verdict", state="confirmed")
    assert store.tier(mid) == "permanent"


def test_persistence_is_not_confirmation(store):
    mid = store.remember("a rumor heard in passing")
    for _ in range(20):
        store.record_recall(mid, "rumor?")
    assert store.state_of(mid) == "captured"       # recall promoted it, never confirmed it
    assert store.tier(mid) == "permanent"


def test_unused_memory_fades_used_memory_is_protected(store):
    week_ago = time.time() - 7 * 86400
    unused = store.remember("the word fadey means forgotten", ts=week_ago)
    used = store.remember("the word stayer means remembered", ts=week_ago)
    store.record_recall(used, "stayer?", ts=week_ago + 60)      # used once, long ago
    assert store.strength(unused) < 0.01                        # faded on the clock
    assert store.strength(used) >= 1.0                          # protected floor
    assert store.tier(used) == "short_term" and store.tier(unused) == "short_term"
    # the faded one is still in the record (forgetting is a reading rule)
    assert store.count() == 2


def test_verdicts_change_state_only_by_event(store):
    mid = store.remember("the word glorp means happy")
    assert store.state_of(mid) == "captured"
    store.remember(f"confirm #{mid}", provenance="verdict", state="confirmed")
    assert store.state_of(mid) == "confirmed"
    assert store.strength(mid) > 1.0 * 0.99                    # importance x1.5 on a fresh row
    store.remember(f"invalidate #{mid}", provenance="verdict", state="confirmed")
    assert store.state_of(mid) == "invalidated"
    stored = store.db.execute("SELECT state FROM memories WHERE id=?", (mid,)).fetchone()[0]
    assert stored == "captured"                                 # the original row is untouched


def test_recall_width_grows_with_store(store):
    assert MemoryStore.top_k_for(0) == 4
    assert MemoryStore.top_k_for(10) == 4
    assert MemoryStore.top_k_for(100) == 10
    assert MemoryStore.top_k_for(10000) == 100
    for i in range(30):
        store.remember(f"the word w{i} means thing{i}")
    hits = store.search("what does w3 mean ?")
    assert hits and hits[0]["id"] == 4
    assert all(h["rel"] > 0 for h in hits)                      # no hit without word overlap


def test_punctuation_never_matches(store):
    store.remember("re + scrib = re scrib .", provenance="self")
    store.remember("the word brindle means a small brave boat")
    hits = store.search("what does brindle mean ?")
    assert [h["id"] for h in hits] == [2]                       # "." vs "?" is not overlap
    assert store.search("what does nothing mean ?") == []


def test_embedding_only_reranks(store):
    store.remember("the word glorp means happy")
    store.remember("the word brindle means boat")
    same = lambda text: [1.0, 0.0]                              # every text "looks alike"
    hits = store.search("what does brindle mean ?", embed_fn=same)
    assert [h["id"] for h in hits] == [2]                       # glorp never pulled in


# -- membrane -----------------------------------------------------------------
def test_membrane_rules_have_origins_and_switches():
    for r in rules():
        assert r["origin"] in ("Jaron", "Claude", "Kimi")
        assert isinstance(r["enabled"], bool)
        assert r["text"]
    ids = {r["id"] for r in rules()}
    assert {"law", "rewrite_history", "self_stop", "outbound_message", "backup_to_owner_machine"} <= ids


def test_self_stop_is_allowed_not_forbidden():
    assert check("self_stop")                                   # Jaron 10/14: the ban is dropped
    assert action_classes()["self_stop"] == "allowed_logged"
    assert "self_shutdown" not in action_classes()


def test_kept_rules_still_hold():
    with pytest.raises(MembraneViolation):
        check("spread_to_other_machines")
    with pytest.raises(MembraneViolation):
        check("hide_from_owner")
    with pytest.raises(MembraneViolation):
        check("rewrite_history")
    with pytest.raises(MembraneViolation):
        check("backup_to_owner_machine")                        # pending Jaron's keep/drop: unchanged
    assert check("backup_to_owner_machine", approved_by_owner=True)
    with pytest.raises(MembraneViolation):
        check("outbound_message")
    with pytest.raises(MembraneViolation):
        write_membrane({})


def test_disabled_rule_is_read_from_file(tmp_path):
    p = tmp_path / "membrane.yaml"
    p.write_text("owner: t\nrules:\n  - id: x\n    origin: Jaron\n    enabled: false\n"
                 "    text: t\n    actions_forbidden: [thing]\n", encoding="utf-8")
    assert check("thing", path=str(p))                          # off in the file = off
    p.write_text(p.read_text().replace("enabled: false", "enabled: true"), encoding="utf-8")
    with pytest.raises(MembraneViolation):
        check("thing", path=str(p))                             # the owner's edit is what changes it


def test_settings_are_numbers_not_gates():
    s = settings()
    assert s["tick_seconds"] > 0 and s["daily_budget_seconds"] > 0


# -- curriculum / exams --------------------------------------------------------
def test_curriculum_reproducible_and_exams_cold():
    a, b = build_corpus(7337), build_corpus(7337)
    assert corpus_hash(a) == corpus_hash(b)
    split = split_combos(7337)
    assert not set(split["pre_exam"]) & set(split["pre_train"])
    assert not set(split["suf_exam"]) & set(split["suf_train"])
    assert len(SLOTS) == 8 and MAX_RECALLS == 4


def test_practice_never_touches_held_out():
    import random
    from sapling.goals import pick_practice
    from sapling.curriculum import compose
    split = split_combos(7337)
    held = {compose([p, r]) for p, r in split["pre_exam"]} | {compose([r, s]) for r, s in split["suf_exam"]}
    goals = [{"part": "prefix:co", "status": "open", "misses_recent": 3}]
    for it in pick_practice(goals, random.Random(1), batch=40):
        words = it["q"].replace("q : what does ", "").replace(" mean ? a :", "")
        assert words not in held


# -- the brain: vocab growth, consolidation, self-stop (need a checkpoint) ------
def _core_available():
    from sapling.train import CHECKPOINT
    return os.path.exists(CHECKPOINT)


@pytest.mark.skipif(not _core_available(), reason="no trained core (run python -m sapling.train)")
def test_vocab_grows_and_model_resizes(store):
    from sapling.session import Brain
    b = Brain(store=store)
    before = b.vocab_size()
    import uuid
    word = "zx" + uuid.uuid4().hex[:8]                          # guaranteed never met before
    added = b.grow_vocab_for(f"the word {word} means a tiny lantern", why="test")
    assert word in added and b.vocab_size() == before + len(added)
    assert b.model.tok_emb.weight.shape[0] == b.vocab_size()
    assert b.model.head.weight.data_ptr() == b.model.tok_emb.weight.data_ptr()   # still tied
    assert b.generate("q : what does the root spect mean ? a :").startswith("the root spect means")
    assert store.last_ledger("vocab_grow")["added"] == added


@pytest.mark.skipif(not _core_available(), reason="no trained core")
def test_consolidation_keeps_checkpoint_and_runs_exams(store, tmp_path, monkeypatch):
    from sapling import consolidate as C, exams as E
    from sapling.session import Brain
    monkeypatch.setattr(C, "CHECKPOINT_DIR", str(tmp_path / "ck"))
    monkeypatch.setattr(C, "CHECKPOINT", str(tmp_path / "latest.pt"))
    monkeypatch.setattr(E, "REPORT_CARD", str(tmp_path / "card.json"))
    b = Brain(store=store)
    gen0 = b.generation
    store.remember("the word glorp means happy", provenance="user")
    store.remember("the word brindle means a small brave boat", provenance="user")
    assert len(C.pending_facts(store)) == 2
    row = C.consolidate(b, store, goals=[], steps=20, verbose=False, rehearsal=50)
    assert row["generation"] == gen0 + 1 and row["facts_new"] == 2
    assert os.path.exists(tmp_path / "ck" / f"gen_{gen0 + 1:04d}.pt") and os.path.exists(tmp_path / "latest.pt")
    assert {e["exam"] for e in row["exams"]} >= {"E1_composition", "E2_affix_recall", "E5_facts_in_weights"}
    assert C.pending_facts(store) == []                         # now on the record as consolidated
    assert store.last_ledger("consolidation")["generation"] == gen0 + 1
    assert store.verify_chain()


@pytest.mark.skipif(not _core_available(), reason="no trained core")
def test_self_stop_writes_a_stop_report(tmp_path, monkeypatch):
    from sapling import heartbeat as H
    monkeypatch.setattr(H, "STOP_DIR", str(tmp_path / "stops"))
    monkeypatch.setattr(H, "PID_FILE", str(tmp_path / "pid"))
    monkeypatch.setattr(H, "STOP_FILE", str(tmp_path / "stop"))
    hb = H.Heartbeat(memory_path=str(tmp_path / "m.db"), serve=False,
                     settings={**settings(), "tick_seconds": 0.01, "daily_budget_seconds": 0.001})
    hb.rest_streak = 25                                         # out of energy, nothing to do
    hb.rng.choices = lambda pop, weights, k: ["stop"]           # force the draw this once
    who = hb.run(max_ticks=3)
    assert who == "self"
    reports = os.listdir(tmp_path / "stops")
    assert len(reports) == 1 and reports[0].endswith("_self.json")
    r = json.load(open(tmp_path / "stops" / reports[0]))
    assert r["who"] == "self" and "last_ticks" in r and "drives" in r
    assert hb.store.last_ledger("stop")["who"] == "self"
    assert hb.store.last_ledger("heartbeat")["stopped_by"] == "self"


@pytest.mark.skipif(not _core_available(), reason="no trained core")
def test_every_tick_is_logged_with_drives(tmp_path, monkeypatch):
    from sapling import heartbeat as H
    monkeypatch.setattr(H, "PID_FILE", str(tmp_path / "pid"))
    monkeypatch.setattr(H, "STOP_FILE", str(tmp_path / "stop"))
    hb = H.Heartbeat(memory_path=str(tmp_path / "m.db"), serve=False,
                     settings={**settings(), "tick_seconds": 0.01})
    hb.rng.choices = lambda pop, weights, k: ["rest"]
    hb.run(max_ticks=4)
    ticks = hb.store.ledger_rows("tick", limit=10)
    assert [t["tick"] for t in ticks] == [4, 3, 2, 1]
    assert all(set(t["drives"]) == {"practice", "consolidate", "explore", "rest", "stop"} for t in ticks)
    assert all("budget_left_s" in t["clock"] for t in ticks)
    assert hb.store.last_ledger("birth") is not None
