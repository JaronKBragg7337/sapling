"""One-shot parole on the owner's behalf (Jaron asked 10/15 7:31 AM: "restart at 10:30 please").

Waits for the daily news render (UnrealEditor) to leave the GPU, then starts the heartbeat if it is
down and logs the parole like the control room's Restart button does. Run by the Windows task
"Sapling Parole 10:30" (one-shot)."""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from owner_relay import down_state, heartbeat_alive  # noqa: E402
import owner_rules  # noqa: E402
from sapling.memory import MemoryStore  # noqa: E402

BY = "Jaron (asked in chat 10/15 7:31 AM; scheduled by Claude)"


def unreal_running() -> bool:
    out = subprocess.run(["tasklist"], capture_output=True, text=True).stdout.lower()
    return "unrealeditor" in out


def main():
    waited = 0
    while unreal_running() and waited < 2 * 3600:
        time.sleep(60)
        waited += 60
    if heartbeat_alive():
        print("already running")
        return
    last_stop = (down_state().get("last_stop") or {})
    p = subprocess.run([sys.executable, "-m", "sapling.heartbeat", "start"], cwd=ROOT,
                       capture_output=True, text=True, timeout=120)
    row = owner_rules.log_owner_action({"action": "parole_start", "by": BY, "after_stop": last_stop,
                                        "waited_for_render_s": waited,
                                        "start_output": (p.stdout + p.stderr).strip()[:300]}, store=MemoryStore())
    print(row)


if __name__ == "__main__":
    main()
