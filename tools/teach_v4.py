"""The 10/15 teaching plan, run end to end without anyone at the keyboard (Claude as teacher).

Stage A (already running: `python -m sapling.reading stage --v4 ...`) -> this script waits for it, then:
  1. measure the Stage A core (exams + loss windows) and its 10 chat samples       -> "after Stage A (reading)"
  2. conversation lessons (sapling.talk build + train)                              -> "after Stage B (conversation)"
  3. measure + chat samples for that core
  4. the swap (heartbeat is stopped; one `growth` row on the chain; old core kept)
  5. heartbeat restarted (gpu_yield start, then verified alive) and left to do its USUAL consolidations (D30: the
     first 5 after a growth may happen with any pending count); when 5 new ones are on the chain, that generation is
     measured again (on the CPU, beside the running heartbeat)  -> "after 5 consolidations". A heartbeat that goes
     down meanwhile gets the parole re-armed (--if-ours) instead of a 10 h wait.
Grok Build review 10/15 10:18 PM: the heartbeat is stopped AND gone before Stage B and before the swap; the whole run
sits in try/finally so the heartbeat is always asked back; a torn log line no longer crashes the wait.
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


STARTED = time.time()


def stage_a_done() -> bool:
    """A complete `done` line, written after this driver started, whose staged core exists. Torn or blank lines
    (Stage A is still appending) count as 'not yet' (Grok Build review 10/15)."""
    try:
        lines = open(V4_LOG, encoding="utf-8").read().splitlines()
    except OSError:
        return False
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("kind") != "done":
            continue
        staged = r.get("staged") or A_CORE
        try:
            t = time.mktime(time.strptime(r.get("ts", ""), "%Y-%m-%dT%H:%M:%S"))
        except ValueError:
            return False
        return os.path.exists(staged) and t >= STARTED - 60 and os.path.getmtime(staged) >= STARTED - 60
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


def heartbeat_gone(why: str):
    """gpu_yield stop WAITS until the process is gone (up to 15 min); refuse to go on while it is still alive."""
    from owner_relay import heartbeat_alive
    if heartbeat_alive():
        out = sh(["tools/gpu_yield.py", "stop", f"{WHO} - {why}"]).strip()
        log({"step": "heartbeat stop", "why": why, "result": out})
    if heartbeat_alive():
        raise RuntimeError(f"heartbeat still alive before {why}")


def heartbeat_up(why: str, wait_s: float = 8 * 3600) -> bool:
    """Arm the parole (it waits for renders / training itself) and wait until the heartbeat is really running."""
    from owner_relay import heartbeat_alive
    out = sh(["tools/gpu_yield.py", "start", f"{WHO} - {why}"]).strip()
    t0 = time.time()
    while not heartbeat_alive() and time.time() - t0 < wait_s:
        time.sleep(30)
    alive = heartbeat_alive()
    log({"step": "heartbeat start", "why": why, "result": out, "alive": alive, "waited_s": round(time.time() - t0)})
    return alive


def main():
    log({"step": "waiting for Stage A", "driver_pid": os.getpid()})
    swapped = False
    try:
        while not stage_a_done():
            time.sleep(120)
        log({"step": "Stage A done"})
        heartbeat_gone("Stage B")
        measure(A_CORE, "after Stage A - v4 reading")
        sh(["-m", "sapling.talk", "build", "--core", A_CORE])
        sh(["-m", "sapling.talk", "train", "--core", A_CORE])
        log({"step": "Stage B done"})
        measure(B_CORE, "after Stage B - conversation lessons")
        heartbeat_gone("the swap")
        out = sh(["-c", "from sapling.grow import swap; import sys; "
                  f"r = swap(staged={B_CORE!r}); print(r['event']['to']['generation'])"])
        gen = int(out.strip().splitlines()[-1])
        mark = last_ledger_id()
        swapped = True
        log({"step": "swapped", "generation": gen, "ledger_id": mark})
    except Exception as e:
        log({"step": "error", "error": str(e)[-2000:], "trace": traceback.format_exc()[-2000:]})
    finally:
        try:
            heartbeat_up("after the teaching run" if swapped else "teaching run stopped early")
        except Exception as e:
            log({"step": "error", "error": f"restart: {e}"})
    if not swapped:
        return
    log({"step": "waiting for 5 of its own consolidations"})
    from owner_relay import heartbeat_alive
    t0, rows, paroles = time.time(), [], 0
    while time.time() - t0 < 10 * 3600:
        rows = consolidations_since(mark)
        if len(rows) >= 5:
            break
        if not heartbeat_alive():
            # down (a render's stop, a crash, or a self-stop): ask for the parole instead of waiting 10 h on a dead pid.
            # --if-ours leaves a self-stop to the owner; the teacher's run stopped it, so that one is ours to restart.
            paroles += 1
            out = sh(["tools/gpu_yield.py", "start", f"{WHO} - 5-consolidation check", "--if-ours"]).strip()
            log({"step": "heartbeat down while waiting", "parole": out, "n": paroles})
            time.sleep(600)
            continue
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
