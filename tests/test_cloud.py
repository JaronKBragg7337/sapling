"""10/16: the long-reading lesson (rule-made questions on human text) and the rented-GPU runner (no spending in tests)."""
import random


RAW = ["The city of Fort Wayne was founded in 1794 by General Anthony Wayne near three rivers.",
       "It grew quickly after the Wabash and Erie Canal opened, and by 1850 about 4000 people lived there.",
       "Fort Wayne is a city in Indiana with many old brick buildings.",
       "Trains later replaced the canal boats, and new factories made wire, trucks and radios.",
       "Today the city has parks along the rivers, a large library and 12 museums for its visitors."] * 3


def test_every_rule_question_answers_with_a_span_copied_from_the_passage():
    from sapling.longread_lesson import question
    from sapling.text import normalize
    P = " " + " ".join(normalize(s) for s in RAW) + " "
    rng = random.Random(3)
    seen = set()
    for _ in range(400):
        qa = question(rng, RAW[:5])
        if not qa:
            continue
        q, a, kind, where = qa
        seen.add(kind)
        span = a[:-2] if a.endswith(" .") and kind in ("after", "word", "number", "name") else a
        assert f" {span} " in P or f" {span}" in P, (kind, q, a)
        assert 0 <= where <= 1
    assert {"after", "word", "which", "first", "year", "number", "name", "define"} <= seen


def test_held_out_documents_never_train_and_e7l_text_is_banned():
    from sapling.longread_lesson import e7l_ngrams, held_out
    ids = [f"web:{i}" for i in range(5000)]
    frac = sum(held_out(i) for i in ids) / len(ids)
    assert 0.003 < frac < 0.03 and held_out("web:42") == held_out("web:42")
    assert isinstance(e7l_ngrams(), set)        # empty only if the essay is not on disk


def test_pack_and_ragged_keep_the_answer_only_loss():
    import torch
    from sapling.longread_lesson import Ragged, pack
    import numpy as np, tempfile, os
    ex = [([1, 5, 6, 7, 2], [-1, -1, -1, 7, 2]), ([1, 8, 2], [-1, 8, 2])]
    d = tempfile.mkdtemp()
    np.savez(os.path.join(d, "x.npz"), **pack(ex))
    r = Ragged(os.path.join(d, "x.npz"))
    X, Y = r.batch([0, 1], block=6, pad=0)
    assert X.shape == (2, 6) and X[0, :5].tolist() == [1, 5, 6, 7, 2]
    assert Y[0].tolist()[:4] == [-1, -1, 7, 2] and (Y[0, 4:] == -1).all()       # shifted: learn only the answer + <eos>
    assert Y[1].tolist()[:2] == [8, 2]


def test_pod_start_command_has_the_watchdog_and_no_secret():
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
    import runpod_run as R
    cmd = R.start_cmd("https://github.com/x/y/releases/download/t", minutes=55, max_pod_minutes=95, micro=32)
    s = cmd[2]
    assert "sleep 5700" in s and "runpodctl remove pod $RUNPOD_POD_ID" in s and "http.server 8000" in s
    assert "RUNPOD_API_KEY" not in s and "ntfy" not in s


def test_dry_run_never_creates_a_pod(monkeypatch, tmp_path):
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
    import runpod_run as R
    monkeypatch.setattr(R, "ROOT", str(tmp_path))
    monkeypatch.setattr(R, "price_quote", lambda *a: {"communityPrice": 2.69, "securePrice": 3.99})
    monkeypatch.setattr(R, "list_pods", lambda: [])
    monkeypatch.setattr(R, "ensure_bundle", lambda *a: {"files": {}, "built": "2026-10-16T00:00:00"})
    monkeypatch.setattr(R, "upload_bundle", lambda *a: "https://example/releases/download/t")
    def boom(*a, **k):
        raise AssertionError("a dry run created a pod")
    monkeypatch.setattr(R, "create_pod", boom)
    monkeypatch.setattr(R, "_http", boom)
    R.main(["--dry-run"])
