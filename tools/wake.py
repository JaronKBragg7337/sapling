"""Sapling wakes on its own (Jaron 10/26: "A sleep time yes but sleep means wake up on own too").
Run by the "Sapling Wake" scheduled task every 30 min. If the heartbeat isn't running AND its last stop was its OWN
choice (budget spent, a stop report named *_self.json) from an earlier day, the daily budget has refilled - start it.
An owner stop (*_owner.json) stays stopped until the owner restarts it. Log: results/wake.log."""
import os, sys, time, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from sapling.heartbeat import PID_FILE, STOP_DIR, _pid_alive

def log(m):
    open(os.path.join(ROOT, "results", "wake.log"), "a", encoding="utf-8").write(time.strftime("%Y-%m-%dT%H:%M:%S ") + m + "\n")

alive = os.path.exists(PID_FILE) and _pid_alive(int(open(PID_FILE).read().strip() or 0))
reports = sorted(f for f in os.listdir(STOP_DIR) if f.endswith(".json"))
last = reports[-1] if reports else ""
if not alive and last.endswith("_self.json") and last[:10] < time.strftime("%Y-%m-%d"):
    r = subprocess.run([sys.executable, "-m", "sapling.heartbeat", "start"], cwd=ROOT, capture_output=True, text=True)
    log("woke up after sleeping (%s): %s" % (last, (r.stdout or r.stderr).strip()[:200]))
