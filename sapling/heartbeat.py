"""The heartbeat — a constant tick loop that runs on its own.

Jaron 10/14 (design restraint 1): "Session-only loop -> WRONG. Needs a constant
tick / heartbeat - it should sense time passing, know time is limited, and
want to do something with it." (restraint 7): "The fly mattered for the
HEARTBEAT (constant going), not as the brain."

Each tick it looks at its own state (budget left, open goals, facts waiting to
be consolidated, how long since it explored, how long since anyone spoke to
it) and CHOOSES one thing: practice, consolidate, explore, rest - or stop.
The choice is a weighted draw over drives, seeded, and every tick is a ledger
row with the drives and the choice, so the owner can see why it did what it
did. Nothing here blocks: the budget at zero makes rest cheap and stop
possible; it does not switch anything off.

Self-stop (Jaron 10/14, old rule 7 dropped as a ban): it may stop. Every stop
writes a stop report (results/stop_reports/<ts>.json + ledger row) with the
state and the last ticks leading up to it. An OWNER stop (the stop file or
Ctrl-C) is logged too, as owner_stop / external_stop, never as a self-stop.

Usage:
    python -m sapling.heartbeat run        # foreground (Ctrl-C = external stop)
    python -m sapling.heartbeat start      # detached background process (Windows-friendly)
    python -m sapling.heartbeat stop       # owner stop (writes results/heartbeat.stop)
    python -m sapling.heartbeat status
"""

from __future__ import annotations

import json
import traceback
import os
import random
import subprocess
import sys
import time

from . import membrane
from .archive import archive_step
from .clock import Clock, Meter
from .consolidate import consolidate, pending_facts
from .curriculum import ROOTS, PREFIXES, SUFFIXES, split_combos, compose
from .curriculum2 import NAMES
from .goals import GoalTracker, pick_practice
from .memory import MemoryStore
from .session import Brain, Session
from .train import RESULTS_DIR

PID_FILE = os.path.join(RESULTS_DIR, "heartbeat.pid")
STOP_FILE = os.path.join(RESULTS_DIR, "heartbeat.stop")
STOP_DIR = os.path.join(RESULTS_DIR, "stop_reports")
ACTIONS = ("practice", "consolidate", "explore", "rest", "stop")


class Heartbeat:
    def __init__(self, memory_path: str | None = None, seed: int = 7337, serve: bool = True,
                 settings: dict | None = None):
        self.settings = settings or membrane.settings()
        self.store = MemoryStore(memory_path) if memory_path else MemoryStore()
        self.clock = Clock(self.store, self.settings["daily_budget_seconds"])
        self.brain = Brain(store=self.store)
        self.session = Session(store=self.store, brain=self.brain, clock=self.clock)
        self.goals = GoalTracker(self.store)
        self.rng = random.Random(seed + int(self.clock.born))
        last = self.store.last_ledger("tick")
        self.tick_no = int(last["tick"]) if last else 0
        self.rest_streak = 0
        self.last_explore = time.time()
        self.last_consolidation = self._last_ts("consolidation") or 0.0
        self.last_archive = self._last_ts("archive") or 0.0
        self.running = False
        self.stopped_by: str | None = None
        self.server = None
        self.serve = serve
        self.store.meter("heartbeat", {"event": "process_start", "pid": os.getpid(),
                                       "core": self.brain.ok, "generation": self.brain.generation,
                                       "vocab_size": self.brain.vocab_size(), "settings": self.settings})

    def _last_ts(self, kind: str) -> float | None:
        row = self.store.last_ledger(kind)
        return float(row["ts"]) if row else None

    # -- drives -> choice ----------------------------------------------------------
    def drives(self) -> dict:
        snap = self.clock.snapshot()
        b = max(0.0, snap["budget_left_frac"])
        open_goals = self.goals.open_goals()
        pending = len(pending_facts(self.store)) if self.brain.ok else 0
        since_explore = time.time() - self.last_explore
        since_consol = time.time() - self.last_consolidation
        since_input = snap["since_last_input_s"]
        worst_misses = max([g["misses_recent"] for g in open_goals], default=0)

        d = {
            "practice": (0.25 + 0.5 * min(len(open_goals), 6) / 6) * (0.2 + 0.8 * b),
            "consolidate": 0.0,
            "explore": (0.15 + 0.6 * min(1.0, since_explore / 120.0)) * (0.3 + 0.7 * b),
            "rest": 0.08 + 0.9 * (1.0 - b) ** 2,
            "stop": 0.0,
        }
        if self.brain.ok and since_consol > 90 and (
                pending >= self.settings["consolidate_min_pending"] or worst_misses >= 3):
            d["consolidate"] = (0.35 + 0.08 * min(pending, 10) + 0.1 * worst_misses) * (0.1 + 0.9 * b)
        if since_input is not None and since_input < 20:
            d["rest"] += 0.3          # someone is talking: hold still, answer fast
        if b <= 0.0 and self.rest_streak >= 20:
            d["stop"] = 0.3 + 0.02 * (self.rest_streak - 20)   # out of energy, nothing to do
        if not self.brain.ok:
            d["practice"] = d["consolidate"] = 0.0
        return {"drives": {k: round(v, 3) for k, v in d.items()}, "state": {
            "budget_left_frac": b, "open_goals": len(open_goals), "worst_misses": worst_misses,
            "pending_facts": pending, "since_explore_s": round(since_explore, 1),
            "since_consolidation_s": round(since_consol, 1), "since_input_s": since_input,
            "rest_streak": self.rest_streak, "memories": self.store.count(),
            "vocab_size": self.brain.vocab_size(), "generation": self.brain.generation}}

    def choose(self) -> tuple[str, dict]:
        info = self.drives()
        w = [max(0.0, info["drives"][a]) for a in ACTIONS]
        action = self.rng.choices(ACTIONS, weights=w, k=1)[0] if sum(w) > 0 else "rest"
        return action, info

    # -- actions ----------------------------------------------------------------------
    def do_practice(self) -> dict:
        goals = self.goals.update()
        items = pick_practice(goals, self.rng, batch=int(self.settings["practice_batch"]))
        results = []
        with Meter(self.clock, "practice", on_gpu=True):
            for it in items:
                got = self.brain.generate(it["q"]).strip()
                results.append({"q": it["q"], "expected": it["a"], "got": got,
                                "ok": got == it["a"], "parts": it["parts"], "focus": it["focus"]})
        ok = sum(r["ok"] for r in results)
        self.store.meter("practice", {"items": results, "ok": ok, "n": len(results)})
        self.goals.update()
        return {"ok": ok, "n": len(results), "misses": [r["q"] for r in results if not r["ok"]]}

    def do_consolidate(self) -> dict:
        goals = self.goals.update()
        with Meter(self.clock, "consolidate", on_gpu=True):
            row = consolidate(self.brain, self.store, goals, steps=int(self.settings["consolidate_steps"]),
                              verbose=False)
        self.last_consolidation = time.time()
        return {k: row[k] for k in ("generation", "facts_new", "loss_first", "loss_last",
                                    "vocab_size", "checkpoint", "exams")}

    def do_explore(self) -> dict:
        """Two kinds, drawn at random: rehearse an old memory (a recall protects
        it), or free-generate from a seed prompt and keep what it said as a
        self/captured memory (data about what it produces, never a claim)."""
        self.last_explore = time.time()
        facts = [f for f in self.store.facts() if f["state"] != "invalidated"]
        if facts and self.rng.random() < 0.5:
            f = self.rng.choice(facts)
            if f.get("doc") is not None:
                # v3.1: a passage it has read - re-read it with one reading task (a recall protects it)
                from .curriculum3 import doc_item
                from . import reader
                it = doc_item(self.rng, f["content"])
                if it:
                    q = it[0].split(" q : ", 1)[1][: -len(" a :")]
                    with Meter(self.clock, "explore_rehearse", on_gpu=True):
                        r = reader.ask(self.brain, self.store, q, doc=f["doc"])
                    return {"kind": "reread", "memory": f["id"], "doc": f["doc"], "passage": f["passage"],
                            "q": q, "a": r["answer"], "expected": it[1], "ok": r["answer"].strip() == it[1]}
            word = f["content"].split()[2] if f["content"].lower().startswith("the word ") else f["content"].split()[0]
            with Meter(self.clock, "explore_rehearse", on_gpu=True):
                out = self.session.ask(f"what does {word} mean ?")
            return {"kind": "rehearse", "memory": f["id"], "q": word, "a": out}
        r = self.rng.choice(list(ROOTS))
        seeds = [f"the root {r}", f"{self.rng.choice(list(PREFIXES))} + {r} =",
                 f"she will {compose([self.rng.choice(list(PREFIXES)), r])}",
                 f"{r} + {self.rng.choice(list(SUFFIXES))} ="]
        if float((self.brain.meta or {}).get("core_version", 2)) >= 3:
            # v3: it can write sentences now - half of its free generation starts from one
            seeds += ["once upon a time ,", f"{self.rng.choice(NAMES)} has a", "the little dog",
                      f"q : what does {self.rng.choice(NAMES)} like to do ? a :"]
        prompt = self.rng.choice(seeds)
        with Meter(self.clock, "explore_generate", on_gpu=True):
            out = self.brain.generate(prompt, max_new=40).strip()
        text = f"{prompt} {out}".strip()
        mid = self.store.remember(text, provenance="self", state="captured")
        return {"kind": "generate", "prompt": prompt, "said": out, "memory": mid}

    # -- stop -------------------------------------------------------------------------
    def stop_report(self, who: str, reason: str, info: dict | None = None, kind: str = "stop") -> str:
        os.makedirs(STOP_DIR, exist_ok=True)
        report = {
            "kind": kind, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "who": who, "reason": reason,
            "tick": self.tick_no, "clock": self.clock.snapshot(), "drives": info,
            "open_goals": self.goals.open_goals(),
            "last_ticks": list(reversed(self.store.ledger_rows("tick", limit=30))),
            "last_events": list(reversed(self.store.ledger_rows(None, limit=30))),
            "memories": self.store.count(), "vocab_size": self.brain.vocab_size(),
            "generation": self.brain.generation,
        }
        path = os.path.join(STOP_DIR, f"{report['ts'].replace(':', '')}_{who}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        self.store.meter("crash" if kind == "crash" else "stop", {"who": who, "reason": reason, "tick": self.tick_no, "report": os.path.basename(path)})
        return path

    # -- the loop ---------------------------------------------------------------------
    def tick(self) -> dict:
        self.tick_no += 1
        action, info = self.choose()
        t0 = time.time()
        result: dict = {}
        try:
            if action == "practice":
                result = self.do_practice(); self.rest_streak = 0
            elif action == "consolidate":
                result = self.do_consolidate(); self.rest_streak = 0
            elif action == "explore":
                result = self.do_explore(); self.rest_streak = 0
            elif action == "rest":
                self.rest_streak += 1
            elif action == "stop":
                membrane.check("self_stop")
                self.stopped_by = "self"
                result = {"report": os.path.basename(self.stop_report("self", "chose stop: " + json.dumps(info["drives"]), info))}
        except Exception as e:  # an error is data too; the loop keeps going
            result = {"error": f"{type(e).__name__}: {e}"}
        row = {"tick": self.tick_no, "action": action, "ms": round((time.time() - t0) * 1000, 1),
               "result": result, **info, "clock": {k: self.clock.snapshot()[k] for k in
                                                   ("age_s", "uptime_s", "since_last_input_s",
                                                    "budget_left_s", "budget_left_frac", "kwh_today")}}
        self.store.meter("tick", row)
        return row

    def housekeeping(self) -> dict | None:
        """Chores on a clock, not a drive it chooses (Jaron 10/14 2:59 PM, disk):
        every `archive_every_seconds` move old checkpoints off the MSI to the
        archive (sapling/archive.py). Logged as a ledger row of kind `archive`."""
        every = float(self.settings.get("archive_every_seconds", 3600))
        if every <= 0 or time.time() - self.last_archive < every:
            return None
        self.last_archive = time.time()
        try:
            return archive_step(self.store)
        except Exception as e:            # an error is data too
            self.store.meter("archive", {"errors": [f"{type(e).__name__}: {e}"], "moved": []})
            return {"errors": [str(e)], "moved": []}

    def chat(self, line: str) -> str:
        out = self.session.handle(line)
        self.store.meter("chat", {"in": line, "out": out})
        return out

    def run(self, max_ticks: int | None = None, max_seconds: float | None = None):
        if self.serve:
            from .control_room import serve
            self.server = serve(self, int(self.settings["control_room_port"]))
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(PID_FILE, "w") as f:
            f.write(str(os.getpid()))
        if os.path.exists(STOP_FILE):
            os.remove(STOP_FILE)
        self.running = True
        t_start = time.time()
        n = 0
        try:
            while self.running:
                row = self.tick()
                n += 1
                print(f"[tick {row['tick']}] {row['action']:<11} {json.dumps(row['result'])[:110]} "
                      f"budget_left={row['clock']['budget_left_s']}s", flush=True)
                if self.stopped_by == "self":
                    break
                hk = self.housekeeping()
                if hk is not None:
                    print(f"[archive] moved={hk.get('moved')} errors={hk.get('errors')} "
                          f"local_mb={hk.get('local_mb_after')}", flush=True)
                if max_ticks and n >= max_ticks:
                    self.stopped_by = "max_ticks"; break
                if max_seconds and time.time() - t_start >= max_seconds:
                    self.stopped_by = "max_seconds"; break
                for _ in range(int(self.settings["tick_seconds"] * 10)):
                    if os.path.exists(STOP_FILE):
                        self.stopped_by = "owner"; break
                    time.sleep(0.1)
                if self.stopped_by == "owner":
                    self.stop_report("owner", "stop file written by the owner")
                    break
        except KeyboardInterrupt:
            self.stopped_by = "external"
            self.stop_report("external", "KeyboardInterrupt / SIGINT")
        except Exception:
            # An unhandled crash is logged as a report (logging only - no rule) so it shows on the control room.
            self.stopped_by = "crash"
            try:
                self.stop_report("crash", "crash: " + traceback.format_exc().strip().splitlines()[-1], {"traceback": traceback.format_exc()}, kind="crash")
            except Exception:
                pass
            raise
        finally:
            self.running = False
            self.store.meter("heartbeat", {"event": "process_end", "stopped_by": self.stopped_by,
                                           "ticks_this_run": n})
            if self.server:
                self.server.shutdown()
            for p in (PID_FILE, STOP_FILE):
                if os.path.exists(p):
                    os.remove(p)
        return self.stopped_by


# -- CLI --------------------------------------------------------------------------------
def _pid_alive(pid: int) -> bool:
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
        return str(pid) in out
    except Exception:
        return False


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "run"
    if cmd == "run":
        max_ticks = int(argv[argv.index("--ticks") + 1]) if "--ticks" in argv else None
        max_seconds = float(argv[argv.index("--seconds") + 1]) if "--seconds" in argv else None
        hb = Heartbeat(serve="--no-serve" not in argv)
        print(f"[heartbeat] born {hb.clock.snapshot()['born_iso']} core={hb.brain.ok} "
              f"gen={hb.brain.generation} vocab={hb.brain.vocab_size()} "
              f"control room: http://127.0.0.1:{hb.settings['control_room_port']}/")
        who = hb.run(max_ticks=max_ticks, max_seconds=max_seconds)
        print(f"[heartbeat] stopped by: {who}")
    elif cmd == "start":
        if os.path.exists(PID_FILE) and _pid_alive(int(open(PID_FILE).read().strip() or 0)):
            print("already running (pid file alive)"); return
        log = open(os.path.join(RESULTS_DIR, "heartbeat.log"), "a", encoding="utf-8")
        flags = 0x00000008 | 0x00000200 if os.name == "nt" else 0   # DETACHED_PROCESS | NEW_PROCESS_GROUP
        p = subprocess.Popen([sys.executable, "-m", "sapling.heartbeat", "run"] + argv[1:],
                             stdout=log, stderr=subprocess.STDOUT, creationflags=flags,
                             cwd=os.path.dirname(RESULTS_DIR))
        print(f"started pid {p.pid}; log results/heartbeat.log; control room http://127.0.0.1:{membrane.settings()['control_room_port']}/")
    elif cmd == "stop":
        open(STOP_FILE, "w").write(time.strftime("%Y-%m-%dT%H:%M:%S"))
        print("owner stop requested (results/heartbeat.stop); it will write a stop report and exit")
    elif cmd == "status":
        if os.path.exists(PID_FILE):
            pid = int(open(PID_FILE).read().strip() or 0)
            print(f"pid {pid} alive={_pid_alive(pid)}")
        else:
            print("not running")
        store = MemoryStore()
        last = store.last_ledger("tick")
        print("last tick:", json.dumps(last)[:400] if last else None)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
