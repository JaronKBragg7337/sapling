"""v4 (10/15, Claude as teacher): rotary core, Unreal GPU sharing, conversation lessons, chat display."""
import torch

from sapling.model import SaplingCore
from sapling.session import pretty
from sapling.train import Tokenizer


def _tok():
    itos = ["<pad>", "<bos>", "<eos>", "<unk>", "q", ":", "a", "passage", "hi", "hello", "?", ".", "how", "are", "you", "fine"]
    return Tokenizer({t: i for i, t in enumerate(itos)}, itos)


def test_rope_core_trains_saves_and_has_no_position_table():
    m = SaplingCore(16, d_model=32, n_head=4, n_layer=2, block_size=64, rope=True)
    assert m.pos_emb is None and m.config()["rope"] is True
    m2 = SaplingCore(**m.config())
    m2.load_state_dict(m.state_dict())
    x = torch.randint(0, 16, (2, 20))
    l1, _ = m(x)
    l2, _ = m2(x)
    assert torch.allclose(l1, l2)
    # rotary = relative: the same short sequence later in the window gives the same next-token logits for its tokens
    _, loss = m(x, x)
    loss.backward()
    assert m.tok_emb.weight.grad is not None
    assert m.grow_context(128) == 64 and m.block_size == 128


def test_old_config_has_no_rope_key():
    m = SaplingCore(16, d_model=32, n_head=4, n_layer=2, block_size=64)
    assert "rope" not in m.config() and m.pos_emb is not None


def test_repeat_penalty_changes_nothing_at_one():
    torch.manual_seed(0)
    m = SaplingCore(16, d_model=32, n_head=4, n_layer=2, block_size=64, rope=True).eval()
    x = torch.tensor([[1, 8, 9]])
    a = m.generate(x, max_new=8, eos_id=2)
    b = m.generate(x, max_new=8, eos_id=2, repeat_penalty=1.0)
    assert torch.equal(a, b)


def test_gpu_busy_sees_unreal(monkeypatch, tmp_path):
    import psutil
    from sapling import reading

    class P:
        def __init__(self, n):
            self.info = {"name": n}
    monkeypatch.setattr(psutil, "process_iter", lambda attrs=None: [P("python.exe"), P("UnrealEditor.exe")])
    assert reading.gpu_busy(str(tmp_path / "nope")) == "unreal"
    monkeypatch.setattr(psutil, "process_iter", lambda attrs=None: [P("python.exe")])
    assert reading.gpu_busy(str(tmp_path / "nope")) is None


def test_conversation_format_masks_everything_but_its_answers():
    from sapling.talk import encode_turns
    tok = _tok()
    ids, tg = encode_turns(tok, [["hi", "hello ."], ["how are you ?", "fine ."]])
    words = [tok.itos[i] for i in ids]
    assert words[:6] == ["<bos>", "q", ":", "hi", "a", ":"]
    learned = [tok.itos[t] for t in tg if t != -1]
    assert learned == ["hello", ".", "<eos>", "fine", ".", "<eos>"]
    # a chat prompt leaves the last answer open
    ids2, _ = encode_turns(tok, [["hi", "hello ."], ["how are you ?", ""]], answer_last=False)
    assert [tok.itos[i] for i in ids2[-2:]] == ["a", ":"]


def test_passage_frame_for_questions_with_a_text():
    from sapling.talk import encode_turns
    tok = _tok()
    ids, _ = encode_turns(tok, [["how are you ?", "fine ."]], passage="hello .")
    assert [tok.itos[i] for i in ids[1:4]] == ["passage", ":", "hello"]


def test_refusals_and_other_names_are_left_out():
    from sapling.talk import DROP
    assert DROP.search("As an AI language model, I cannot")
    assert DROP.search("I am Open Assistant")
    assert not DROP.search("Rain falls because water drops get heavy.")


def test_pretty_is_display_only_casing():
    assert pretty("i'm sapling . jaron built me , i think .") == "I'm Sapling. Jaron built me, I think."


def test_no_self_lesson_without_an_owner_file(tmp_path):
    import random
    from sapling.talk import owner_answers, self_lessons
    missing = str(tmp_path / "none.json")
    assert owner_answers(missing) == {}
    assert self_lessons(random.Random(1), owner=owner_answers(missing)) == []


def test_self_lessons_are_the_owners_words_only(tmp_path):
    import random
    from sapling.talk import SELF_QUESTIONS, owner_answers, save_owner_answer, self_lessons
    path = str(tmp_path / "own.json")
    save_owner_answer("name", "I'm Sapling, Jaron's AI!", "test", path=path)
    save_owner_answer("size", "   ", "test", path=path)          # empty = skipped
    own = owner_answers(path)
    assert set(own) == {"name"}
    ls = self_lessons(random.Random(1), copies=1, owner=own)
    assert len(ls) == len(SELF_QUESTIONS["name"])
    assert {c["turns"][0][1] for c in ls} == {"i'm sapling , jaron's ai !"}      # normalised like every lesson


def test_relay_saves_self_words_as_an_owner_action(monkeypatch):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
    import owner_relay
    import sapling.talk as talk
    seen = {}
    monkeypatch.setattr(talk, "save_owner_answer", lambda k, t, by: seen.update(k=k, t=t, by=by) or {"key": k, "chars": len(t)})
    monkeypatch.setattr(owner_relay.owner_rules, "log_owner_action", lambda row, store=None: row)
    monkeypatch.setattr(owner_relay, "MemoryStore", lambda: type("S", (), {"db": type("D", (), {"close": lambda self: None})()})())
    r = owner_relay.do_owner("self_words", {"key": "maker", "text": "Jaron made me."}, by="owner@test")
    assert r["ok"] and r["row"]["action"] == "self_words" and seen == {"k": "maker", "t": "Jaron made me.", "by": "owner@test"}


# -- Grok Build review fixes (10/15 10:18 PM) ------------------------------------------------------------------------
def test_driver_tolerates_torn_log_lines_and_old_done_rows(tmp_path, monkeypatch):
    import json, os, sys, time
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
    import teach_v4 as T
    log, staged = tmp_path / "log.jsonl", tmp_path / "staged.pt"
    monkeypatch.setattr(T, "V4_LOG", str(log))
    monkeypatch.setattr(T, "A_CORE", str(staged))
    monkeypatch.setattr(T, "STARTED", time.time())
    log.write_text('{"kind": "step"}\n\n{"kind": "pro')                        # torn last line
    assert T.stage_a_done() is False
    old = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - 3600))
    staged.write_bytes(b"x")
    log.write_text(json.dumps({"kind": "done", "ts": old, "staged": str(staged)}) + "\n")
    assert T.stage_a_done() is False                                            # a done row from before this driver
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    log.write_text(json.dumps({"kind": "done", "ts": now, "staged": str(staged)}) + "\n{\"kind\"")
    assert T.stage_a_done() is True


def test_checkpoint_write_is_atomic(tmp_path):
    import os
    from sapling.model import SaplingCore
    from sapling.train import Tokenizer, save_checkpoint
    itos = ["<pad>", "<bos>", "<eos>", "<unk>"]
    p = str(tmp_path / "c.pt")
    save_checkpoint(SaplingCore(4, 16, 2, 1, 8), Tokenizer({t: i for i, t in enumerate(itos)}, itos), {}, path=p)
    assert os.path.exists(p) and not os.path.exists(p + ".tmp")


def test_talk_replay_failure_is_loud_for_a_talking_core(monkeypatch, tmp_path):
    import random
    import pytest
    from sapling import consolidate as C, talk
    class B:
        def talks(self): return True
    monkeypatch.setattr(talk, "LESSONS", str(tmp_path / "missing.jsonl"))
    with pytest.raises(RuntimeError, match="talk replay"):
        C.talk_replay(B(), random.Random(0))
    class Quiet:
        def talks(self): return False
    assert C.talk_replay(Quiet(), random.Random(0)) is None


def test_heartbeat_skips_gpu_work_while_a_render_holds_the_card_and_owes_consolidations_after_growth(tmp_path, monkeypatch):
    from sapling import heartbeat as H, reading, membrane
    monkeypatch.setattr(H, "STOP_DIR", str(tmp_path / "stops"))
    monkeypatch.setattr(H, "PID_FILE", str(tmp_path / "pid"))
    monkeypatch.setattr(H, "STOP_FILE", str(tmp_path / "stop"))
    hb = H.Heartbeat(memory_path=str(tmp_path / "m.db"), serve=False, settings={**membrane.settings(), "tick_seconds": 0.01})
    monkeypatch.setattr(reading, "gpu_busy", lambda *a: "unreal")
    hb.rng.choices = lambda pop, weights, k: ["practice"]
    row = hb.tick()
    assert row["action"] == "rest" and row["result"]["gpu_busy"] == "unreal"
    assert hb.consolidations_owed_after_growth() == 0
    hb.store.meter("growth", {"event": "test growth"})
    assert hb.consolidations_owed_after_growth() == 5
    hb.store.meter("consolidation", {"generation": 1})
    assert hb.consolidations_owed_after_growth() == 4
