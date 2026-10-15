"""The 10/15 teaching plan, run end to end without anyone at the keyboard (Claude as teacher).

Stage A (already running: `python -m sapling.reading stage --v4 ...`) -> this script waits for it, then:
  1. measure the Stage A core (exams + loss windows) and its 10 chat samples       -> "after Stage A (reading)"
  2. conversation lessons (sapling.talk build + train)                              -> "after Stage B (conversation)"
  3. measure + chat samples for that core
  4. the swap (heartbeat is stopped; one `growth` row on the chain; old core kept)
  5. heartbeat restarted (gpu_yield start) and left to do its USUAL consolidations; when 5 new ones are on the
     chain, the newest generation is measured again (on the CPU, beside the running heartbeat)  -> "after 5 consolidations"
Every step is a row in results/teacher/run.jsonl; measurements in results/teacher/measurements.jsonl; chat samples in
results/chat_samples/. If a step fails the error is logged and the script stops there (the heartbeat is restarted
either way, so Sapling is never left down by this script).

    python tools/teach_v4.py            # started detached right after Stage A was launched
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

PY = sys.executable
RUN = os.path.join(ROOT, "results", "teacher", "run.jsonl")
V4_LOG = os.path.join(ROOT, "results", "reading_v4", "log.jsonl")
A_CORE = os.path.join(ROOT, "results", "core_v4_staged.pt")
B_CORE = os.path.join(ROOT, "results", "core_v4_talk_staged.pt")
WHO = "Claude (teacher) - v4 teaching plan, 10/15"


def log(row: dict):
    os.makedirs(os.path.dirname(RUN), exist_ok=True)
    row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **row}
    with open(RUN, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")
    print(json.dumps(row, default=str), flush=True)


def sh(args: list[str], env: dict | None = None, timeout: float | None = None) -> str:
    p = subprocess.run([PY] + args, cwd=ROOT, capture_output=True, text=True, timeout=timeout,
                       env={**os.environ, "PYTHONUNBUFFERED": "1", **(env or {})})
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} failed: {p.stderr[-1500:]}")
    return p.stdout


def stage_a_done() -> bool:
    try:
        return any(json.loads(l).get("kind") == "done" for l in open(V4_LOG, encoding="utf-8"))
    except OSError:
        return False


def measure(core: str, label: str, cpu: bool = False):
    env = {"SAPLING_DEVICE": "cpu"} if cpu else None
    sh(["tools/measure_core.py", "--core", core, "--label", label], env=env)
    sh(["tools/chat_samples.py", "--core", core, "--label", label], env=env)
    log({"step": "measured", "label": label, "core": os.path.relpath(core, ROOT)})


def consolidations_since(row_id: int) -> list[dict]:
    from sapling.memory import MemoryStore
    st = MemoryStore()
    rows = st.db.execute("SELECT id, payload FROM ledger WHERE kind='consolidation' AND id > ? ORDER BY id", (row_id,)).fetchall()
    return [{"id": r[0], **json.loads(r[1])} for r in rows]


def last_ledger_id() -> int:
    from sapling.memory import MemoryStore
    return int(MemoryStore().db.execute("SELECT COALESCE(MAX(id), 0) FROM ledger").fetchone()[0])


def main():
    log({"step": "waiting for Stage A"})
    while not stage_a_done():
        time.sleep(120)
    log({"step": "Stage A done"})
    try:
        measure(A_CORE, "after Stage A - v4 reading")
        sh(["-m", "sapling.talk", "build", "--core", A_CORE])
        sh(["-m", "sapling.talk", "train", "--core", A_CORE])
        log({"step": "Stage B done"})
        measure(B_CORE, "after Stage B - conversation lessons")
        # the swap: the heartbeat is already stopped for the training (gpu_yield stop); make sure
        from owner_relay import heartbeat_alive
        if heartbeat_alive():
            sh(["tools/gpu_yield.py", "stop", WHO])
        out = sh(["-c", "from sapling.grow import swap; import sys; "
                  f"r = swap(staged={B_CORE!r}); print(r['event']['to']['generation'])"])
        gen = int(out.strip().splitlines()[-1])
        mark = last_ledger_id()
        log({"step": "swapped", "generation": gen, "ledger_id": mark})
    except Exception as e:
        log({"step": "error", "error": str(e)[-2000:], "trace": traceback.format_exc()[-2000:]})
        sh(["tools/gpu_yield.py", "start", WHO])
        return
    sh(["tools/gpu_yield.py", "start", WHO])
    log({"step": "heartbeat restarted - waiting for 5 of its own consolidations"})
    t0 = time.time()
    rows = []
    while time.time() - t0 < 10 * 3600:
        rows = consolidations_since(mark)
        if len(rows) >= 5:
            break
        time.sleep(120)
    if not rows:
        log({"step": "no consolidation in 10 h - nothing to re-measure"})
        return
    g = rows[min(4, len(rows) - 1)]["generation"]
    core = os.path.join(ROOT, "results", "checkpoints", f"gen_{g:04d}.pt")
    log({"step": "consolidations seen", "n": len(rows), "generation": g,
         "summary": [{k: r.get(k) for k in ("generation", "loss_first", "loss_last", "talk_share", "read_share", "seconds")} for r in rows]})
    try:
        measure(core, f"after {len(rows)} consolidations - gen {g}", cpu=True)
    except Exception as e:
        log({"step": "error", "error": str(e)[-2000:]})
    log({"step": "finished"})


if __name__ == "__main__":
    main()
