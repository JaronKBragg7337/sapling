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
