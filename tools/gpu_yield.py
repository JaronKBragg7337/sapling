"""Sapling gives the GPU to another job, logged (Jaron 10/15: heartbeat + Unreal on one 8 GB card froze both).

  python tools/gpu_yield.py stop  "<who>"   -> stops the heartbeat if it is running, logs an owner_stop
  python tools/gpu_yield.py start "<who>"   -> runs scheduled_parole.py (waits for UnrealEditor to exit, then restarts + logs)
Called by the news render (AI-Shared/projects/news-show/pipeline/run_rendered_news.ps1)."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from owner_relay import heartbeat_alive  # noqa: E402
import owner_rules  # noqa: E402
from sapling.memory import MemoryStore  # noqa: E402


def main():
    cmd, who = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "another GPU job")
    if cmd == "stop":
        if not heartbeat_alive():
            print("not running")
            return
        subprocess.run([sys.executable, "-m", "sapling.heartbeat", "stop"], cwd=ROOT, timeout=60)
        owner_rules.log_owner_action({"action": "owner_stop", "by": who, "why": "GPU handed to another job"},
                                     store=MemoryStore())
        print("stopped")
    elif cmd == "start":
        subprocess.Popen([sys.executable, os.path.join(ROOT, "tools", "scheduled_parole.py"), "--by", who], cwd=ROOT)
        print("parole armed")


if __name__ == "__main__":
    main()
