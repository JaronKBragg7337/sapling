"""Sapling gives the GPU to another job, logged (Jaron 10/15: heartbeat + Unreal on one 8 GB card froze both).

  python tools/gpu_yield.py stop  "<who>"            -> asks the heartbeat to stop, WAITS until its process is gone
                                                        (up to 15 min: a consolidation + exams must finish first), logs
                                                        an owner_stop; prints "stopped" / "not running" / "stop TIMEOUT"
  python tools/gpu_yield.py start "<who>"            -> runs scheduled_parole.py (waits for UnrealEditor and any core
                                                        training to leave the GPU, then restarts + logs)
  python tools/gpu_yield.py start "<who>" --if-ours  -> the same, but only if the heartbeat is down because a GPU job
                                                        stopped it (its last stop was an owner stop logged by this tool);
                                                        a self-stop or crash is left for the owner's parole (D6)
Called by the news render (AI-Shared/projects/news-show/pipeline/run_rendered_news.ps1) and tools/teach_v4.py.
Grok Build review 10/15 10:18 PM: `stop` used to return before the process was gone."""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from owner_relay import heartbeat_alive  # noqa: E402
import owner_rules  # noqa: E402
from sapling.memory import MemoryStore  # noqa: E402

WHY = "GPU handed to another job"
STOP_WAIT_S = 15 * 60


def wait_gone(timeout: float = STOP_WAIT_S, poll: float = 2.0) -> bool:
    t0 = time.time()
    while heartbeat_alive():
        if time.time() - t0 > timeout:
            return False
        time.sleep(poll)
    return True


def down_because_of_a_gpu_job() -> bool:
    """True if the heartbeat's last stop was an owner stop and the newest owner_stop row came from this tool."""
    stop_dir = os.path.join(ROOT, "results", "stop_reports")
    try:
        last = sorted(os.listdir(stop_dir))[-1]
        who = json.load(open(os.path.join(stop_dir, last), encoding="utf-8")).get("who")
    except Exception:
        return False
    if who != "owner":
        return False
    try:
        rows = [json.loads(l) for l in open(owner_rules.LOG, encoding="utf-8").read().splitlines() if l.strip()]
        stops = [r for r in rows if r.get("action") == "owner_stop"]
        return bool(stops) and stops[-1].get("why") == WHY
    except Exception:
        return False


def stop(who: str) -> str:
    if not heartbeat_alive():
        return "not running"
    subprocess.run([sys.executable, "-m", "sapling.heartbeat", "stop"], cwd=ROOT, timeout=60)
    gone = wait_gone()
    owner_rules.log_owner_action({"action": "owner_stop", "by": who, "why": WHY, "gone": gone}, store=MemoryStore())
    return "stopped" if gone else "stop TIMEOUT (heartbeat still alive after 15 min)"


def start(who: str, if_ours: bool = False) -> str:
    if heartbeat_alive():
        return "already running"
    if if_ours and not down_because_of_a_gpu_job():
        return "not armed (down for another reason - the owner's parole)"
    subprocess.Popen([sys.executable, os.path.join(ROOT, "tools", "scheduled_parole.py"), "--by", who], cwd=ROOT)
    return "parole armed"


def main():
    cmd, who = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "another GPU job")
    if cmd == "stop":
        print(stop(who))
    elif cmd == "start":
        print(start(who, if_ours="--if-ours" in sys.argv))


if __name__ == "__main__":
    main()
