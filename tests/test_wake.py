import json
import os
import sys

from sapling.memory import MemoryStore

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import wake


def test_wake_records_one_vanished_exit_on_the_chain(tmp_path, monkeypatch):
    monkeypatch.setattr(wake, "STOP_DIR", str(tmp_path / "stops"))
    store = MemoryStore(str(tmp_path / "memory.db"))
    try:
        store.meter("heartbeat", {"event": "process_start", "pid": 12345, "generation": 7})
        name = wake.record_vanished(store)
        assert name and name.endswith("_vanished.json")
        report = json.load(open(tmp_path / "stops" / name, encoding="utf-8"))
        assert report["pid"] == 12345 and report["kind"] == "vanished"
        assert store.last_ledger("vanished")["pid"] == 12345
        assert wake.record_vanished(store) is None
        assert store.verify_chain()
        store.meter("heartbeat", {"event": "process_end", "stopped_by": "owner"})
        assert wake.record_vanished(store) is None
    finally:
        store.db.close()


def test_wake_recovers_vanished_process_through_normal_launcher(tmp_path, monkeypatch):
    monkeypatch.setattr(wake, "STOP_DIR", str(tmp_path / "stops"))
    monkeypatch.setattr(wake, "PID_FILE", str(tmp_path / "missing.pid"))
    monkeypatch.setattr(wake, "log", lambda message: None)
    calls = []
    result = type("Result", (), {"returncode": 0, "stdout": "scheduled", "stderr": ""})()
    monkeypatch.setattr(wake.subprocess, "run", lambda *args, **kwargs: calls.append(args[0]) or result)
    store = MemoryStore(str(tmp_path / "memory.db"))
    store.meter("heartbeat", {"event": "process_start", "pid": 12345})
    store.db.close()
    monkeypatch.setattr(wake, "MemoryStore", lambda: MemoryStore(str(tmp_path / "memory.db")))
    wake.main()
    assert calls and calls[0][-3:] == ["-m", "sapling.heartbeat", "start"]


def test_windows_start_delegates_to_scheduled_task(tmp_path, monkeypatch, capsys):
    from sapling import heartbeat
    monkeypatch.setattr(heartbeat, "PID_FILE", str(tmp_path / "missing.pid"))
    monkeypatch.setattr(heartbeat.os, "name", "nt")
    calls = []
    result = type("Result", (), {"returncode": 0, "stdout": "started", "stderr": ""})()
    monkeypatch.setattr(heartbeat.subprocess, "run", lambda *args, **kwargs: calls.append(args[0]) or result)
    heartbeat.main(["start"])
    assert calls == [["schtasks", "/Run", "/TN", "Sapling Heartbeat"]]
    assert "scheduled task" in capsys.readouterr().out
