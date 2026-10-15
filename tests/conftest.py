"""Shared test guard (10/15): a Heartbeat built in a test must never run the REAL archive step - it would upload
real checkpoints to the GitHub release and remove local files (found when a test run raced the archive migration).
The archive's own tests call sapling.archive directly with every path patched."""
import pytest


@pytest.fixture(autouse=True)
def _no_real_archive_from_heartbeat(monkeypatch):
    import sapling.heartbeat as H
    monkeypatch.setattr(H, "archive_step", lambda store, *a, **k: {"moved": [], "errors": [], "note": "archive off in tests"})
    yield
