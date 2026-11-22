"""Sapling wakes after eligible self-sleeps and recovers crashes or unexplained exits.
Owner and external stops stay stopped. The Windows heartbeat launcher delegates to Task Scheduler.
"""
import json, os, sys, time, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from sapling.heartbeat import PID_FILE, STOP_DIR, _pid_alive
from sapling.memory import MemoryStore


def log(m):
    open(os.path.join(ROOT, "results", "wake.log"), "a", encoding="utf-8").write(time.strftime("%Y-%m-%dT%H:%M:%S ") + m + "\n")


def record_vanished(store):
    rows = store.ledger_rows("heartbeat", limit=100)
    start = next((r for r in rows if r.get("event") == "process_start"), None)
    end = next((r for r in rows if r.get("event") == "process_end"), None)
    if not start or (end and end["id"] > start["id"]):
        return None
    marked = store.last_ledger("vanished")
    if marked and marked["id"] > start["id"]:
        return None
    last_tick = store.last_ledger("tick")
    consolidation = store.last_ledger("consolidation")
    report = {"kind": "vanished", "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "who": "vanished",
              "reason": f"heartbeat pid {start.get('pid')} disappeared without process_end; recorded by Sapling Wake",
              "tick": last_tick.get("tick") if last_tick else None, "pid": start.get("pid"),
              "process_start_ts": start.get("ts"), "memories": store.count(),
              "generation": consolidation.get("generation") if consolidation else None}
    os.makedirs(STOP_DIR, exist_ok=True)
    name = f"{report['ts'].replace(':', '')}_vanished.json"
    with open(os.path.join(STOP_DIR, name), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    store.meter("vanished", report)
    return name


def main():
    alive = os.path.exists(PID_FILE) and _pid_alive(int(open(PID_FILE).read().strip() or 0))
    if not alive:
        store = MemoryStore()
        try:
            vanished = record_vanished(store)
            if vanished:
                log(f"recorded unexplained exit: {vanished}")
        finally:
            store.db.close()
    reports = sorted(f for f in os.listdir(STOP_DIR) if f.endswith(".json"))
    last = reports[-1] if reports else ""
    self_sleep_due = last.endswith("_self.json") and last[:10] < time.strftime("%Y-%m-%d")
    recoverable_exit = not last or last.endswith("_crash.json") or last.endswith("_vanished.json")
    if not alive and (self_sleep_due or recoverable_exit):
        r = subprocess.run([sys.executable, "-m", "sapling.heartbeat", "start"], cwd=ROOT, capture_output=True, text=True)
        log(f"started after {last or 'no stop record'}: {(r.stdout or r.stderr).strip()[:300]}")
        if r.returncode:
            raise RuntimeError(f"heartbeat start exited {r.returncode}")


if __name__ == "__main__":
    main()
