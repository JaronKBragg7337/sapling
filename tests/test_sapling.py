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
    monkeypatch.setattr(E, "WRITING_DIR", str(tmp_path / "writing"))   # E8 samples: never into results/ from a test
    b = Brain(store=store)
    gen0 = b.generation
    store.remember("the word glorp means happy", provenance="user")
    store.remember("the word brindle means a small brave boat", provenance="user")
    assert len(C.pending_facts(store)) == 2
    row = C.consolidate(b, store, goals=[], steps=20, verbose=False, rehearsal=50)
    assert row["generation"] == gen0 + 1 and row["facts_new"] == 2
    assert os.path.exists(tmp_path / "ck" / f"gen_{gen0 + 1:04d}.pt") and os.path.exists(tmp_path / "latest.pt")
    assert {e["exam"] for e in row["exams"]} >= {"E1_composition", "E2_affix_recall", "E5_facts_in_weights", "E7L_longread", "E9_owner_exam"}
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


# -- 10/14 2:59 PM decisions: owner's hand, archive, self-made memories on record ---
def test_owner_rules_flips_one_switch_and_keeps_comments(tmp_path):
    import shutil
    sys_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
    import importlib, sys
    sys.path.insert(0, sys_path)
    owner_rules = importlib.import_module("owner_rules")
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "membrane.yaml")
    p = tmp_path / "membrane.yaml"
    shutil.copy(src, p)
    before = p.read_text(encoding="utf-8")
    s = MemoryStore(str(tmp_path / "m.db"))
    row = owner_rules.set_enabled("outbound_message", False, by="test@owner", why="test",
                                  path=str(p), log=str(tmp_path / "owner.jsonl"), store=s)
    after = p.read_text(encoding="utf-8")
    assert row["before"] is True and row["after"] is False
    assert {r["id"]: r["enabled"] for r in rules(str(p))}["outbound_message"] is False
    assert {r["id"]: r["enabled"] for r in rules(str(p))}["law"] is True          # only that one switch
    assert after.count("#") == before.count("#")                                   # comments survive
    assert s.last_ledger("owner_action")["rule"] == "outbound_message" and s.verify_chain()
    assert (tmp_path / "owner.jsonl").read_text().count("\n") == 1
    s.db.close()


def test_ai_package_has_no_write_path_to_membrane():
    """The owner's hand (tools/owner_rules.py, tools/owner_relay.py) lives OUTSIDE
    the sapling package; nothing in sapling/ imports it or opens membrane.yaml for writing."""
    pkg = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sapling")
    for name in os.listdir(pkg):
        if name.endswith(".py"):
            src = open(os.path.join(pkg, name), encoding="utf-8").read()
            assert "owner_rules" not in src and "owner_relay" not in src, name
            assert "MEMBRANE_PATH, \"w\"" not in src and "MEMBRANE_PATH,\"w\"" not in src, name


def test_archive_plan_keeps_last_10_and_one_per_day():
    from sapling.archive import plan
    local = [{"gen": g, "path": f"gen_{g}.pt", "size": 10, "mtime": 0.0,
              "day": "2026-10-13" if g <= 5 else "2026-10-14"} for g in range(1, 28)]
    p = plan(local, keep_last=10)
    assert p["keep"] == [5] + list(range(18, 28))             # gen 5 = last of 10-13; 18..27 = last 10
    assert [m["gen"] for m in p["move"]] == [1, 2, 3, 4] + list(range(6, 18))
    assert "not the last checkpoint of 2026-10-13" in p["why"][1]


def test_archive_step_without_credentials_moves_nothing(store, monkeypatch, tmp_path):
    from sapling import archive as A
    ck = tmp_path / "ck"; ck.mkdir()
    for g in range(1, 15):
        (ck / f"gen_{g:04d}.pt").write_bytes(b"x" * 10)
    monkeypatch.setattr(A, "CHECKPOINT_DIR", str(ck))
    monkeypatch.setattr(A, "RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(A, "creds", lambda: None)
    row = A.archive_step(store)
    assert row["moved"] == [] and row["errors"] and len(os.listdir(ck)) == 14   # never delete without a verified cloud copy
    assert store.last_ledger("archive")["cloud"] is False and store.verify_chain()


@pytest.mark.skipif(not _core_available(), reason="no trained core")
def test_consolidation_records_self_made_memories(store, tmp_path, monkeypatch):
    from sapling import consolidate as C, exams as E
    from sapling.session import Brain
    monkeypatch.setattr(C, "CHECKPOINT_DIR", str(tmp_path / "ck"))
    monkeypatch.setattr(C, "CHECKPOINT", str(tmp_path / "latest.pt"))
    monkeypatch.setattr(E, "REPORT_CARD", str(tmp_path / "card.json"))
    monkeypatch.setattr(E, "WRITING_DIR", str(tmp_path / "writing"))   # E8 samples: never into results/ from a test
    b = Brain(store=store)
    store.remember("the word glorp means happy", provenance="user")
    sid = store.remember("the root zan zan ly", provenance="self", state="captured")
    row = C.consolidate(b, store, goals=[], steps=10, verbose=False, rehearsal=30)
    assert row["self_ids"] == [sid] and row["self_new"] == [sid] and row["facts_self"] == 1 and row["facts_user"] == 1
    assert store.last_ledger("consolidation")["self_ids"] == [sid]


def test_recall_count_handles_missing_row(store, monkeypatch):
    """10/14 crash: fetchone() came back None -> TypeError in recall_count. Now 0."""
    mid = store.remember("a fact", provenance="user")
    class _NoRow:
        def fetchone(self): return None
    real = store.db
    class _DB:
        def execute(self, *a, **k): return _NoRow()
        def __getattr__(self, n): return getattr(real, n)
    monkeypatch.setattr(store, "db", _DB())
    assert store.recall_count(mid) == 0


def test_store_is_safe_across_threads(store):
    import threading
    ids = [store.remember(f"fact {i}", provenance="user") for i in range(5)]
    errs = []
    def reader():
        try:
            for _ in range(300):
                store.facts()
        except Exception as e:      # noqa: BLE001
            errs.append(repr(e))
    ts = [threading.Thread(target=reader) for _ in range(6)]
    [t.start() for t in ts]
    for i in range(100):
        store.record_recall(ids[i % 5], "q")
    [t.join() for t in ts]
    assert not errs, errs[:2]


@pytest.mark.skipif(not _core_available(), reason="no trained core")
def test_crash_writes_a_crash_report(tmp_path, monkeypatch):
    from sapling import heartbeat as H
    monkeypatch.setattr(H, "STOP_DIR", str(tmp_path / "stops"))
    monkeypatch.setattr(H, "PID_FILE", str(tmp_path / "pid"))
    monkeypatch.setattr(H, "STOP_FILE", str(tmp_path / "stop"))
    hb = H.Heartbeat(memory_path=str(tmp_path / "m.db"), serve=False,
                     settings={**settings(), "tick_seconds": 0.01})
    def boom(): raise TypeError("boom")
    hb.choose = boom
    with pytest.raises(TypeError):
        hb.run(max_ticks=2)
    names = os.listdir(tmp_path / "stops")
    assert len(names) == 1 and names[0].endswith("_crash.json")
    r = json.load(open(tmp_path / "stops" / names[0]))
    assert r["kind"] == "crash" and "boom" in r["drives"]["traceback"] and "last_ticks" in r


# -- v3 (10/14 7 PM, "school level 2: real sentences") ---------------------------------------
def test_normalize_spaces_text_like_the_school():
    from sapling.text import normalize, sentences
    assert normalize('Tom said, "I don\'t like it!" What?') == "tom said , i don't like it ! what ?"
    assert normalize("what does hello mean?") == "what does hello mean ?"
    assert sentences("a b . c d ? e") == ["a b .", "c d ?", "e"]


def test_level2_is_reproducible_and_exams_are_cold():
    from sapling.curriculum2 import build_corpus2, exam_items_grammar, exam_items_reading, split_level2, VERBS
    a = build_corpus2(7337, stories=False)
    b = build_corpus2(7337, stories=False)
    assert a["lessons"] == b["lessons"]
    sp = split_level2(7337)
    assert not set(sp["vo_train"]) & set(sp["vo_exam"])
    assert not set(sp["fact_train"]) & set(sp["fact_exam"])
    # no held-out (verb, object) pair, in any form, inside any generated lesson
    text = " " + " ".join(s for L in a["lessons"].values() for s in L) + " "
    for v, o in sp["vo_exam"]:
        head, _, rest = v.partition(" ")
        for form in (head,) + VERBS[head]:
            assert f" {form}{(' ' + rest) if rest else ''} {o} " not in text
    assert exam_items_grammar(7337) == exam_items_grammar(7337) and len(exam_items_grammar(7337)) == 48
    r = exam_items_reading(7337)
    assert len(r) == 40 and all(it["q"].startswith("passage :") and it["q"].endswith("a :") for it in r)
    # reading passages use held-out facts only: a held-out (name, noun, colour) never appears in a lesson
    for nm, noun, col in sp["fact_exam"][:50]:
        assert f"{nm} has a {col} {noun} ." not in text


def test_v3_core_is_bigger_and_old_checkpoints_still_load():
    from sapling.model import SaplingCore, V3_CONFIG
    m = SaplingCore(3200, **V3_CONFIG)
    assert 10_000_000 <= m.n_params() <= 20_000_000
    assert m.config()["block_size"] == 256 and m.head.weight.data_ptr() == m.tok_emb.weight.data_ptr()
    small = SaplingCore(178, d_model=128, n_head=4, n_layer=4, block_size=64)   # the v2 shape
    assert small.n_params() < 1_000_000
    import torch
    x = torch.randint(0, 178, (2, 10))
    logits, loss = small(x, x)
    assert logits.shape == (2, 10, 178) and loss.item() < 8.0          # small init: starts near ln(vocab)


def test_exam_bars_for_level2_are_predeclared():
    from sapling.exams import BARS
    assert BARS["E6_grammar"] == 0.60 and BARS["E7_reading"] == 0.50
    assert BARS["E1_composition"] == 0.60                                # old bars untouched


def test_growth_smoke_keeps_chain_and_logs_growth_rows():
    """The whole pipeline on CPU in a temp dir, twice: a fresh tiny v3-shaped core (level 1 + 2 + 3 + the
    record), then the SAME core grown in place to v3.1 (new vocab rows + pieces, context 64 -> 128).
    Two `growth` ledger rows on one chain, every old token kept, E1-E9 on the card."""
    import shutil
    from sapling.grow import smoke
    out = smoke(verbose=False)
    try:
        st = out["store"]
        assert st.verify_chain()
        rows = list(reversed(st.ledger_rows("growth", limit=10)))      # oldest first
        assert len(rows) == 2
        assert rows[0]["event"] == "growth: core v- -> v3" and rows[0]["to"]["core_version"] == 3
        assert rows[1]["event"] == "growth: core v3 -> v3.1" and rows[1]["to"]["core_version"] == 3.1
        assert rows[1]["from"]["generation"] == rows[0]["to"]["generation"]
        assert rows[1]["to"]["generation"] == rows[1]["from"]["generation"] + 1
        assert rows[1]["to"]["vocab_size"] == len(out["tok"].itos) > rows[1]["from"]["vocab_size"]
        assert rows[1]["to"]["config"]["block_size"] == 128 and rows[1]["from"]["config"]["block_size"] == 64
        assert rows[1]["to"]["pieces"] is True
        assert set(e["exam"] for e in rows[1]["from"]["exams"]) >= {"E6_grammar", "E7_reading"}
        names = {e["exam"] for e in rows[1]["to"]["exams"]}
        assert {"E1_composition", "E6_grammar", "E7_reading", "E7L_longread", "E1b_bridge", "E9_owner_exam", "E8_writing"} <= names
        e9 = next(e for e in rows[1]["to"]["exams"] if e["exam"] == "E9_owner_exam")
        assert e9["verdict"] == "INFO" and e9["score"] == "0 items"       # the slot is there; Jaron has not filled it
        tok = out["tok"]
        assert "hello" in tok.stoi and "once" in tok.stoi               # record + stories both in the vocabulary
        assert tok.has_pieces() and tok.tokens("report") == ["re", "##port"]   # the bridge split
        assert "<unk>" not in tok.tokens("archaeologists digitised the necklace")   # adult words spell from pieces
        assert tok.decode(tok.encode("archaeologists digitised the necklace")) == "archaeologists digitised the necklace"
        assert os.path.isdir(os.path.join(out["tmp"], "writing")) and os.listdir(os.path.join(out["tmp"], "writing"))
        st.db.close()
    finally:
        shutil.rmtree(out["tmp"], ignore_errors=True)


# -- v3.1 (10/14 night, "prepare it to read my documents") -----------------------------------------
def test_pieces_spell_unknown_words_and_old_tokenizers_are_unchanged():
    from sapling.pieces import base_pieces, learn_pieces, split_word, join_pieces
    from sapling.train import Tokenizer
    base = ["<pad>", "<bos>", "<eos>", "<unk>", "re", "port", "the", "word", "means", ".", "?"]
    old = Tokenizer({t: i for i, t in enumerate(base)}, list(base))
    assert old.tokens("re port zap") == ["re", "port", "<unk>"] and not old.has_pieces()
    learned = learn_pieces({"archaeology": 5, "biology": 8, "geology": 3, "ology": 2}, n_merges=30)
    assert learned and learned == learn_pieces({"archaeology": 5, "biology": 8, "geology": 3, "ology": 2}, n_merges=30)
    itos = base + base_pieces() + ["##port"] + learned
    tok = Tokenizer({t: i for i, t in enumerate(itos)}, itos, splits={"report": ["re", "##port"]})
    assert tok.has_pieces()
    assert tok.tokens("the word report means archaeology .") == ["the", "word", "re", "##port", "means"] + split_word("archaeology", tok.stoi) + ["."]
    assert "<unk>" not in tok.tokens("zebra 42 don't")
    assert tok.decode(tok.encode("the word report means archaeology zqx .")) == "the word report means archaeology zqx ."
    assert join_pieces(["arch", "##ae", "##ology", "."]) == ["archaeology", "."]
    assert tok.unknown_tokens("report zqx the") == ["zqx"]                 # a forced split is never "unknown"


def test_level3_lessons_are_cold_for_E6_and_E1():
    from sapling.curriculum import compose, split_combos
    from sapling.curriculum2 import VERBS, split_level2
    from sapling.curriculum3 import build_corpus3, composed_words, exam_items_bridge
    kw = dict(docs=False, words=["lantern", "river", "engine"], story_sents=["tom has a red ball ."], sizes={"copy": 300})
    c = build_corpus3(7337, **kw)
    assert c["lessons"]["copy"] == build_corpus3(7337, **kw)["lessons"]["copy"]
    text = " " + " ".join(c["lessons"]["copy"]) + " "
    sp = split_level2(7337)
    for v, o in sp["vo_exam"]:                       # E6's held-out (verb, object) pairs never appear, in any form
        head, _, rest = v.partition(" ")
        for form in (head,) + VERBS[head]:
            assert f" {form}{(' ' + rest) if rest else ''} {o} " not in text
    assert any("q : copy :" in s for s in c["lessons"]["copy"]) and any("q : fill :" in s for s in c["lessons"]["copy"])
    bridge = " " + " ".join(c["lessons"]["bridge"]) + " "
    s1 = split_combos(7337)
    for p, r in s1["pre_exam"]:                      # E1's held-out combos: neither spelling in the bridge lesson
        assert f" {p + r} " not in bridge and f" {compose([p, r])} " not in bridge
    for r, s in s1["suf_exam"]:
        assert f" {r + s} " not in bridge and f" {compose([r, s])} " not in bridge
    assert len(exam_items_bridge(7337)) == 86 and composed_words()["report"] == ["re", "##port"]


def test_memory_schema_keeps_old_chain_and_chains_documents(tmp_path):
    import sqlite3, hashlib, json as J
    # a pre-v3.1 database written by the old code (no doc/passage columns, no documents table)
    p = str(tmp_path / "old.db")
    con = sqlite3.connect(p)
    con.executescript("""CREATE TABLE memories (id INTEGER PRIMARY KEY, ts REAL NOT NULL, content TEXT NOT NULL,
        provenance TEXT NOT NULL, state TEXT NOT NULL, prev_hash TEXT NOT NULL, hash TEXT NOT NULL);
        CREATE TABLE recalls (id INTEGER PRIMARY KEY, ts REAL NOT NULL, memory_id INTEGER NOT NULL, query TEXT NOT NULL,
        prev_hash TEXT NOT NULL, hash TEXT NOT NULL);
        CREATE TABLE ledger (id INTEGER PRIMARY KEY, ts REAL NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL,
        prev_hash TEXT NOT NULL, hash TEXT NOT NULL);""")
    prev = "sapling-genesis"
    for i, c in enumerate(("the word glorp means happy", "the word zint means cold")):
        ts = 1000.0 + i
        payload = J.dumps({"ts": ts, "content": c, "provenance": "user", "state": "captured"}, sort_keys=True)
        h = hashlib.sha256((prev + "|" + payload).encode()).hexdigest()
        con.execute("INSERT INTO memories (ts, content, provenance, state, prev_hash, hash) VALUES (?,?,?,?,?,?)",
                    (ts, c, "user", "captured", prev, h))
        prev = h
    con.commit(); con.close()
    s = MemoryStore(p)                               # opening migrates: columns added, old rows untouched
    assert s.verify_chain() and s.count() == 2
    d = s.add_document("Walking", "chat", "abc123", 500, 2)
    a = s.remember("first passage .", provenance="retrieval", doc=d, passage=0)
    b = s.remember("second passage .", provenance="retrieval", doc=d, passage=1)
    assert s.verify_chain()
    assert [x["passage"] for x in s.passages(d)] == [0, 1] and s.documents()[0]["title"] == "Walking"
    assert s.search("first passage", record=False) == []            # passages are the reader's, not the fact search's
    assert all(f["doc"] == d for f in s.facts() if f["id"] in (a, b))
    s.db.execute("UPDATE memories SET passage=7 WHERE id=?", (b,)); s.db.commit()
    assert not s.verify_chain()                                     # provenance is inside the hash
    s.db.close()


def test_reader_chunks_and_retrieves_with_provenance(store):
    from sapling import reader
    text = ("The lantern was made in Sweden in 1891. It burned whale oil. " * 3 +
            "The necklace was found in a cave near the river. The cave was dry. " * 3 +
            "Franklin printed newspapers in Philadelphia. He flew a kite in a storm. " * 3)
    r = reader.read_document(store, text=text, title="notes")
    assert r["new"] and r["passages"] >= 1 and store.verify_chain()
    again = reader.read_document(store, text=text, title="notes")
    assert again["new"] is False and store.count("documents") == 1    # same sha256 = nothing added twice
    hits = reader.retrieve(store, "where was the necklace found ?", k=2)
    assert hits and "necklace" in hits[0]["content"] and hits[0]["doc"] == r["doc"]
    assert store.recall_count(hits[0]["id"]) == 1                       # a pick is a recorded recall
    assert reader.retrieve(store, "what is the capital of france ?", k=2) == []
    assert store.last_ledger("read_document")["title"] == "notes"


def test_model_context_grows_in_place():
    import torch
    from sapling.model import SaplingCore
    m = SaplingCore(50, d_model=32, n_head=4, n_layer=2, block_size=16)
    before = m.pos_emb.weight[:16].clone()
    assert m.grow_context(32) == 16 and m.block_size == 32 and m.grow_context(32) == 0
    assert torch.equal(m.pos_emb.weight[:16], before)                  # the learned rows stay
    logits, _ = m(torch.randint(0, 50, (1, 30)))
    assert logits.shape == (1, 30, 50) and m.config()["block_size"] == 32


def test_owner_exam_file_is_read_fresh(tmp_path):
    from sapling.exams import owner_exam
    p = tmp_path / "owner_exam.json"
    p.write_text('{"bar": 0.7, "items": [{"q": "who made the lantern ?", "a": "Sweden", "match": "contains"}]}', encoding="utf-8")
    e = owner_exam(str(p))
    assert e["bar"] == 0.7 and e["items"][0]["q"] == "who made the lantern ?"
    p.write_text('{"items": []}', encoding="utf-8")
    assert owner_exam(str(p))["items"] == [] and owner_exam(str(tmp_path / "missing.json"))["items"] == []


# -- v3.2 (10/14 night, "much more reading") ---------------------------------------------------
def _tiny_core(block: int = 64):
    from sapling.curriculum import build_corpus, build_vocab
    from sapling.model import SaplingCore
    from sapling.train import Tokenizer
    import torch
    torch.manual_seed(1)
    corpus = build_corpus(7337)
    v = build_vocab(corpus)
    tok = Tokenizer(v["stoi"], v["itos"])
    return SaplingCore(len(tok.itos), d_model=32, n_head=4, n_layer=2, block_size=block), tok


def test_induction_probe_is_seeded_and_an_untrained_core_is_at_chance():
    from sapling.exams import induction_probe
    m, tok = _tiny_core(64)
    a = induction_probe(m, tok, seed=7337, n_seq=4, lengths=(8, 24))
    b = induction_probe(m, tok, seed=7337, n_seq=4, lengths=(8, 24))
    assert a == b                                                      # reproducible from its seed
    assert a["total"] == 4 * (8 - 1) + 4 * (24 - 1) and [r["length"] for r in a["by_length"]] == [8, 24]
    assert a["accuracy"] < 0.1                                         # no circuit yet: chance is ~1/vocab
    assert m.training                                                  # the mode it found is the mode it leaves


def test_reading_shards_roundtrip_and_stream_walks_in_order(tmp_path):
    import numpy as np
    from sapling.reading import FastEncoder, Stream, prepare, vocab_sha
    m, tok = _tiny_core(16)
    docs = ["re + spect = re spect .", "the word spect means look .", "she will re spect the bright lantern ."]
    idx = prepare(tok, shard_dir=str(tmp_path), shard_tokens=12, sources={"stories": iter(docs)}, verbose=False)
    src = idx["sources"]["stories"]
    assert idx["vocab_sha"] == vocab_sha(tok) and src["docs"] == 3 and len(src["shards"]) >= 2
    ids = np.concatenate([np.load(tmp_path / s["file"]) for s in src["shards"]]).tolist()
    assert ids == sum((FastEncoder(tok).ids(d) for d in docs), [])    # the same ids, in order, nothing lost
    assert tok.decode(ids[:8]).startswith("re + spect")
    st = Stream(str(tmp_path), src["shards"], block=4)
    x, y = st.next(2)
    assert x.shape == (2, 4) and y[0, 0].item() == x[0, 1].item()      # y is x shifted by one
    assert st.state()["offset"] == 8 and st.state()["epoch"] == 0
    for _ in range(20):
        st.next(2)
    assert st.state()["epoch"] >= 1                                    # wraps, counting epochs


def test_reading_stage_smoke_probes_pauses_and_stages_the_core(tmp_path):
    """CPU, seconds: tiny core, tiny shards; probes (E1-E10, no ledger writes) at every N tokens, a pause file
    is honoured, resume continues from latest.pt, and the staged core carries the reading record (v3.2)."""
    import json
    import os
    import threading
    import time
    from sapling.reading import prepare, stage, summary
    from sapling.train import load_checkpoint, save_checkpoint
    m, tok = _tiny_core(16)
    old = tmp_path / "core.pt"
    save_checkpoint(m, tok, {"seed": 7337, "generation": 5, "core_version": 3.1, "corpus_sha": "x"}, path=str(old))
    docs = [f"the word w{i} means thing {i} ." for i in range(80)]
    sh = tmp_path / "shards"
    prepare(tok, shard_dir=str(sh), shard_tokens=200, sources={"stories": iter(docs), "adult": iter(docs[:20])}, verbose=False)
    out = tmp_path / "reading"
    reh = ["re + spect = re spect ."] * 40
    kw = dict(batch=4, shares={"stories": 2, "adult": 1, "rehearsal": 1}, probe_every=16 * 3 * 4, warmup=2, device="cpu",
              old_path=str(old), staged=str(tmp_path / "staged.pt"), out_dir=str(out), shard_dir=str(sh), rehearsal=reh,
              poll_s=0.2, step_log_every=2, verbose=False, keep_probe_checkpoints=False,
              busy_fn=lambda: "pause file" if (out / "pause").exists() else None)   # never the real blender
    r = stage(target_tokens=16 * 3 * 8, **kw)                          # 8 steps, probes at 0 / 4 / 8
    rows = [json.loads(l) for l in open(out / "log.jsonl", encoding="utf-8")]
    probes = [x for x in rows if x["kind"] == "probe"]
    assert [p["step"] for p in probes] == [0, 4, 8] and probes[0]["why"] == "before"   # no duplicate "end" probe
    assert "E10_induction" in probes[-1]["exams"] and "E7_reading" in probes[-1]["exams"] and probes[-1]["writing"]
    assert not (tmp_path / "report_card.json").exists()               # probes never write the card or the ledger
    assert r["meta"]["core_version"] == 3.2 and r["meta"]["generation"] == 6 and r["meta"]["reading"]["steps"] == 8
    _, _, ck = load_checkpoint(device="cpu", path=str(tmp_path / "staged.pt"))
    assert ck["reading"]["tokens_new"] == 16 * 3 * 8 and ck["grown_from"]["generation"] == 5
    s = summary(str(out))
    assert len(s["probes"]) == 3 and s["done"]["steps"] == 8 and s["probes"][-1]["E10"] is not None
    # the pause file: training holds (saved, GPU released), then goes on when the file is gone
    (out / "pause").write_text("owner")
    threading.Timer(1.0, lambda: os.remove(out / "pause")).start()
    t0 = time.time()
    r2 = stage(target_tokens=16 * 3 * 12, **kw)                        # resumes at step 8, pauses, ends at 12
    assert time.time() - t0 >= 1.0 and r2["run"]["steps"] == 12 and r2["run"]["pauses"] and r2["run"]["pauses"][0]["why"] == "pause file"
    assert summary(str(out))["pauses"][0]["why"] == "pause file"


def test_exams_write_false_leaves_no_trace(tmp_path, monkeypatch):
    from sapling import exams as E
    m, tok = _tiny_core(64)
    monkeypatch.setattr(E, "REPORT_CARD", str(tmp_path / "card.json"))
    monkeypatch.setattr(E, "WRITING_DIR", str(tmp_path / "writing"))
    card = E.run_exams(seed=7337, verbose=False, model=m, tok=tok, meta={"generation": 1}, store=None, write=False, trigger="probe")
    names = {e["exam"] for e in card["exams"]}
    assert {"E1_composition", "E7L_longread", "E10_induction", "E8_writing"} <= names
    assert not (tmp_path / "card.json").exists() and not (tmp_path / "writing").exists()
