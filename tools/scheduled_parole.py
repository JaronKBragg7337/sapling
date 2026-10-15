"""One-shot parole on the owner's behalf (Jaron asked 10/15 7:31 AM: "restart at 10:30 please").

Waits for the daily news render (UnrealEditor) and any core training run (v4, 10/15) to leave the GPU, then starts the heartbeat if it is
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

BY_DEFAULT = "Jaron (asked in chat 10/15 7:31 AM; re-armed by Claude 12:52 PM after the news render ran past the first wait)"


def unreal_running() -> bool:
    out = subprocess.run(["tasklist"], capture_output=True, text=True).stdout.lower()
    return "unrealeditor" in out


def training_running() -> bool:
    """v4 (10/15): a core training run (reading stage / conversation lessons) holds the card; the heartbeat
    beside it froze both before (D23), so a parole waits for it like it waits for the render."""
    try:
        import psutil
        for p in psutil.process_iter(["cmdline"]):
            cmd = " ".join(p.info.get("cmdline") or [])
            if "sapling.reading" in cmd and " stage" in cmd or "sapling.talk" in cmd and " train" in cmd:
                return True
    except Exception:
        pass
    return False


def main():
    by = sys.argv[sys.argv.index("--by") + 1] if "--by" in sys.argv else BY_DEFAULT
    waited = 0
    while (unreal_running() or training_running()) and waited < 12 * 3600:
        time.sleep(60)
        waited += 60
    if heartbeat_alive():
        print("already running")
        return
    last_stop = (down_state().get("last_stop") or {})
    p = subprocess.run([sys.executable, "-m", "sapling.heartbeat", "start"], cwd=ROOT,
                       capture_output=True, text=True, timeout=120)
    row = owner_rules.log_owner_action({"action": "parole_start", "by": by, "after_stop": last_stop,
                                        "waited_for_render_s": waited,
                                        "start_output": (p.stdout + p.stderr).strip()[:300]}, store=MemoryStore())
    print(row)


if __name__ == "__main__":
    main()
