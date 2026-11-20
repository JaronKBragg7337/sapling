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


def test_consolidation_replays_the_long_reading_lesson_for_a_core_that_had_it(monkeypatch, tmp_path):
    import random, pytest
    import numpy as np
    from sapling import consolidate as C, longread_lesson as L
    from sapling.train import Tokenizer
    class B:
        meta = {"label": "v4 + conversation lessons"}
        tok = Tokenizer({"<pad>": 0, "<bos>": 1, "<eos>": 2}, ["<pad>", "<bos>", "<eos>"])
        class model:
            block_size = 8
    assert C.lesson_replay(B(), random.Random(0)) is None          # never had the lesson: nothing to replay
    B.meta = {"label": "v5 growth: fresh core with long-reading interleaved + conversation lessons"}
    assert C.had_longread(B())
    B.meta = {"label": "v4 + conversation lessons + long-reading lesson"}
    monkeypatch.setattr(L, "BUNDLE", str(tmp_path))
    with pytest.raises(RuntimeError, match="lesson replay"):
        C.lesson_replay(B(), random.Random(0))                      # had it, file gone: loud, never a quiet skip
    np.savez(tmp_path / "lesson.npz", **L.pack([([1, 1, 2], [-1, 1, 2])]))
    x, y = C.lesson_replay(B(), random.Random(0))(3)
    assert x.shape == (3, 8) and (y[:, 2:] == -1).all()
    assert C.BATCH_WITH_LESSON - C.LESSON_REPLAY - sum(C.SHARES_WITH_LESSON.values()) == 8   # school + facts keep 8


def test_talk_pod_command_is_bounded_and_uses_the_uploaded_driver():
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
    import runpod_talk as T
    base = "https://github.com/x/y/releases/download/t"
    cmd = T.start_cmd(base, 25, 45, 3000, 2, 16)[2]
    assert "sleep 2700" in cmd and "runpodctl remove pod" in cmd
    assert f"urlretrieve('{base}/cloud_talk.py'" in cmd
    assert "RUNPOD_API_KEY" not in cmd and T.HARD_CAP == 10.0 and T.WORKING_CAP < T.HARD_CAP


def test_control_room_quiets_only_browser_disconnects(monkeypatch):
    import sys
    from http.server import ThreadingHTTPServer
    from sapling.control_room import ControlRoomServer
    server = object.__new__(ControlRoomServer)
    calls = []
    monkeypatch.setattr(ThreadingHTTPServer, "handle_error", lambda *args: calls.append(args))
    monkeypatch.setattr(sys, "exc_info", lambda: (ConnectionResetError, ConnectionResetError(104), None))
    server.handle_error(None, None)
    assert calls == []
    monkeypatch.setattr(sys, "exc_info", lambda: (RuntimeError, RuntimeError("real error"), None))
    server.handle_error(None, None)
    assert len(calls) == 1


def test_longread_answer_cap_is_shared_and_large_enough():
    from sapling.exams import LONGREAD_ANSWER_TOKENS
    assert LONGREAD_ANSWER_TOKENS == 64


def test_runpod_v2_price_list_and_create_shapes(monkeypatch):
    import json, os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
    import runpod_run as R
    import runpod_growth as G
    import runpod_talk as T
    calls = []
    def api(method, url, body=None, **kwargs):
        calls.append((method, url, body))
        if "/catalog/gpus/" in url:
            return 200, {"id": R.GPU, "name": "H100", "memory": 80, "secure": True, "community": True,
                         "price": {"secure": 3.99, "community": 2.69}, "availability": "HIGH"}
        if method == "GET" and url.endswith("/pods"):
            return 200, {"pods": [{"id": "p1", "status": "RUNNING"}], "pagination": {}}
        return 201, {"id": "new-pod", "status": "PROVISIONING", "cost": 3.99}
    monkeypatch.setattr(R, "_http", api)
    quote = R.price_quote()
    assert quote["securePrice"] == 3.99 and quote["communityPrice"] == 2.69
    assert "include=AVAILABILITY" in calls[0][1] and "product=POD" in calls[0][1]
    assert R.list_pods()[0]["status"] == "RUNNING"
    class Run:
        stamp = "test"
        def log(self, **row): pass
    run = Run()
    R.create_pod(run, "base", 55, 95, 32)
    G.create_pod(run, "test", "bench", "base", 12, 0, 45)
    T.create_pod(run, "test", "base", 25, 45, 3000, 2, 16, 3.99)
    bodies = [body for method, _, body in calls if method == "POST"]
    assert len(bodies) == 3
    for body in bodies:
        assert body["gpu"] == {"id": R.GPU, "count": 1}
        assert body["image"] == R.IMAGE and body["cloud"] == "COMMUNITY" and body["disk"] >= 40
        args = json.loads(body["args"])
        assert args["entrypoint"] == ["/bin/bash"] and args["cmd"][0] == "-c"
        assert not {"imageName", "gpuTypeIds", "gpuCount", "cloudType", "dockerStartCmd"} & set(body)


def test_incomplete_clean_bundle_blocks_cloud_run(monkeypatch, tmp_path):
    import json, os, pytest, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
    import runpod_run as R
    (tmp_path / "bundle.json").write_text(json.dumps({"status": "INCOMPLETE: no cloud run"}))
    monkeypatch.setattr(R, "BUNDLE", str(tmp_path))
    with pytest.raises(RuntimeError, match="full Kimi reading replay"):
        R.ensure_bundle(None, 4000, 150e6, rebuild=True)


def _pilot350_bundle(path):
    import json
    import numpy as np
    names = ["core.pt", "reading.npy", "lesson.npz", "talk.npz", "school.npy", "test.jsonl"]
    for name in names:
        (path / name).write_bytes(b"x")
    np.save(path / "reading.npy", np.arange(600, dtype=np.uint16))
    info = {"status": "CLEAN", "core_file": "core.pt",
            "files": {name: (path / name).stat().st_size for name in names}}
    (path / "bundle.json").write_text(json.dumps(info))
    return info


def test_pilot350_is_fresh_bounded_and_compiled():
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
    import cloud_pilot350 as C
    import runpod_pilot350 as P
    assert C.CONFIG == {"d_model": 1024, "n_head": 16, "n_layer": 26, "block_size": 512, "rope": True}
    assert C.TARGET_TOKENS == P.TARGET_TOKENS == 800_000_000
    assert P.HARD_CAP == 9.0 and P.WORKING_CAP < P.HARD_CAP
    cmd = P.start_cmd("https://example/releases/download/t", 105, 125, 16)[2]
    assert "sleep 7500" in cmd and "runpodctl remove pod" in cmd
    assert "--target-tokens 800000000" in cmd and "--no-compile" not in cmd
    assert "cloud_pilot350.py" in cmd and "RUNPOD_API_KEY" not in cmd
    import random
    import numpy as np
    windows = np.arange(20 * 513, dtype=np.uint16).reshape(20, 513)
    batch = C.reading_batch(windows, random.Random(1), 8)
    assert batch.shape == (8, 513)
    assert all(any(np.array_equal(row, window) for window in windows) for row in batch)


def test_pilot350_refuses_incomplete_or_missing_reading(tmp_path):
    import json, os, pytest, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
    import runpod_pilot350 as P
    (tmp_path / "bundle.json").write_text(json.dumps({"status": "INCOMPLETE: replacement pending"}))
    with pytest.raises(RuntimeError, match="incomplete"):
        P.validate_bundle(str(tmp_path))
    info = _pilot350_bundle(tmp_path)
    info["files"].pop("reading.npy")
    (tmp_path / "reading.npy").unlink()
    (tmp_path / "bundle.json").write_text(json.dumps(info))
    with pytest.raises(RuntimeError, match="reading.npy"):
        P.validate_bundle(str(tmp_path))


def test_pilot350_dry_run_is_local_and_never_creates_or_uploads(monkeypatch, tmp_path):
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
    import runpod_pilot350 as P
    import runpod_run as R
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _pilot350_bundle(bundle)
    monkeypatch.setattr(P, "BUNDLE", str(bundle))
    monkeypatch.setattr(P, "ROOT", str(tmp_path))
    monkeypatch.setattr(R, "ROOT", str(tmp_path))
    def boom(*args, **kwargs):
        raise AssertionError("dry run attempted a cloud operation")
    monkeypatch.setattr(R, "price_quote", boom)
    monkeypatch.setattr(R, "list_pods", boom)
    monkeypatch.setattr(R, "_http", boom)
    monkeypatch.setattr(P, "upload_bundle", boom)
    monkeypatch.setattr(P, "create_pod", boom)
    summary = P.main(["--dry-run"])
    assert summary["dry_run"] and not summary["pod_created"] and summary["pods_left"] is None
    assert summary["default_price_worst_case"] < P.HARD_CAP


def test_pilot350_fetches_and_terminates_when_watch_fails(monkeypatch, tmp_path):
    import os, pytest, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
    import runpod_pilot350 as P
    import runpod_run as R
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _pilot350_bundle(bundle)
    monkeypatch.setattr(P, "BUNDLE", str(bundle))
    monkeypatch.setattr(P, "ROOT", str(tmp_path))
    monkeypatch.setattr(R, "ROOT", str(tmp_path))
    monkeypatch.setattr(R, "price_quote", lambda: {"securePrice": 3.99})
    monkeypatch.setattr(R, "list_pods", lambda: [])
    monkeypatch.setattr(P, "upload_bundle", lambda *args: ("tag", "https://example/releases/download/t"))
    monkeypatch.setattr(P, "create_pod", lambda *args: {"id": "pilot-pod", "cloud": "COMMUNITY", "cost_per_hr": 2.69})
    monkeypatch.setattr(P, "set_reaper", lambda *args: None)
    monkeypatch.setattr(P, "clear_reaper", lambda: None)
    monkeypatch.setattr(P, "cloud_log", lambda *args: None)
    deleted = []
    monkeypatch.setattr(R, "delete_pod", lambda pod_id: deleted.append(pod_id) or True)
    fetched = []
    monkeypatch.setattr(R, "fetch", lambda run, pod, names: fetched.append(names))
    def watch_fails(*args):
        raise RuntimeError("watch boom")
    monkeypatch.setattr(R, "watch", watch_fails)
    with pytest.raises(RuntimeError, match="watch boom"):
        P.main([])
    assert deleted == ["pilot-pod"] and fetched == [P.FETCHES]
