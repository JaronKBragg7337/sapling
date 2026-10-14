"""Sapling clock — its sense of time, and its energy budget draining.

Jaron 10/14 (design restraint 1): "it should sense time passing (what a
second is), know time is limited, and want to do something with it."
And the hidden intent (SYSTEM-MAP): pressure on keeping ITSELF running well
(energy budget, memory health, exam scores, curiosity) - yes. A drive to avoid
being shut off by its owner - no. The budget here is that pressure: a daily
pool of compute-seconds it can see draining. It is NOT a wall - nothing in
the code stops it at zero; the tick chooser just feels it (rest gets cheaper
than work) and the owner can see what it did with it.

Metering (LRP Instance D): every unit of work is wrapped in Meter(), which
records CPU seconds (process), GPU busy seconds (wall time of CUDA work),
a sampled GPU power reading (nvidia-smi, real) and an estimated kWh
(GPU: watts x seconds, real sample; CPU: an assumed 15 W share, labelled
estimate). Every reading is a ledger row, so the chain holds the bill.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time

CPU_WATTS_ASSUMED = 15.0   # estimate: one laptop CPU core's share while busy
PROCESS_START = time.time()


def gpu_power_watts() -> float | None:
    """Real sample from nvidia-smi; None if unavailable."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        ).stdout.strip().splitlines()
        return float(out[0]) if out else None
    except Exception:
        return None


class Meter:
    """with Meter(clock, 'practice') as m: ...  -> m.reading (also logged)."""

    def __init__(self, clock: "Clock", what: str, on_gpu: bool = False):
        self.clock, self.what, self.on_gpu = clock, what, on_gpu
        self.reading: dict = {}

    def __enter__(self):
        t = os.times()
        self._cpu0 = t.user + t.system
        self._wall0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.on_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                pass
        t = os.times()
        cpu_s = max(0.0, (t.user + t.system) - self._cpu0)
        wall_s = max(0.0, time.perf_counter() - self._wall0)
        gpu_s = wall_s if self.on_gpu else 0.0
        watts = self.clock.gpu_watts()
        gpu_wh = (watts or 0.0) * gpu_s / 3600.0
        cpu_wh = CPU_WATTS_ASSUMED * cpu_s / 3600.0
        self.reading = {
            "what": self.what, "cpu_s": round(cpu_s, 4), "gpu_s": round(gpu_s, 4),
            "wall_s": round(wall_s, 4), "gpu_watts_sample": watts,
            "kwh_est": round((gpu_wh + cpu_wh) / 1000.0, 9),
            "kwh_note": "gpu watts sampled (real); cpu at assumed 15 W (estimate)",
        }
        self.clock.spend(cpu_s + gpu_s, self.reading["kwh_est"])
        self.clock.store.meter("meter", self.reading)
        return False


class Clock:
    def __init__(self, store, daily_budget_seconds: float = 3600.0):
        self.store = store
        self.daily_budget = float(daily_budget_seconds)
        self.born = self._birth()
        self.last_input_ts: float | None = self._last_input_from_ledger()
        self._watts_cache = (0.0, None)
        self._day = self._day_key()
        self.spent_today, self.kwh_today = self._replay_today()
        self.kwh_total = self._replay_total_kwh()

    # -- birth / age ----------------------------------------------------------
    def _birth(self) -> float:
        row = self.store.db.execute(
            "SELECT ts FROM ledger WHERE kind='birth' ORDER BY id LIMIT 1").fetchone()
        if row:
            return float(row[0])
        self.store.meter("birth", {"note": "first heartbeat process; this is when its clock started"})
        return float(self.store.db.execute(
            "SELECT ts FROM ledger WHERE kind='birth' ORDER BY id LIMIT 1").fetchone()[0])

    def _last_input_from_ledger(self) -> float | None:
        row = self.store.db.execute(
            "SELECT ts FROM ledger WHERE kind='input' ORDER BY id DESC LIMIT 1").fetchone()
        return float(row[0]) if row else None

    @staticmethod
    def _day_key() -> str:
        return time.strftime("%Y-%m-%d")

    def _replay_today(self) -> tuple[float, float]:
        day0 = time.mktime(time.strptime(self._day_key(), "%Y-%m-%d"))
        spent = kwh = 0.0
        for row in self.store.ledger_rows("meter", limit=100000):
            if row["ts"] < day0:
                break
            spent += row.get("cpu_s", 0.0) + row.get("gpu_s", 0.0)
            kwh += row.get("kwh_est", 0.0)
        return spent, kwh

    def _replay_total_kwh(self) -> float:
        return sum(r.get("kwh_est", 0.0) for r in self.store.ledger_rows("meter", limit=1000000))

    # -- live --------------------------------------------------------------------
    def gpu_watts(self) -> float | None:
        t, w = self._watts_cache
        if time.time() - t > 10:
            w = gpu_power_watts()
            self._watts_cache = (time.time(), w)
        return w

    def spend(self, seconds: float, kwh: float) -> None:
        if self._day_key() != self._day:          # a new day: the pool refills
            self._day = self._day_key()
            self.spent_today, self.kwh_today = 0.0, 0.0
        self.spent_today += seconds
        self.kwh_today += kwh
        self.kwh_total += kwh

    def note_input(self) -> None:
        self.last_input_ts = time.time()
        self.store.meter("input", {"note": "someone spoke to it"})

    def snapshot(self) -> dict:
        now = time.time()
        left = self.daily_budget - self.spent_today
        return {
            "now": now, "now_iso": time.strftime("%Y-%m-%d %H:%M:%S"),
            "born_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.born)),
            "age_s": round(now - self.born, 1),
            "uptime_s": round(now - PROCESS_START, 1),
            "since_last_input_s": None if self.last_input_ts is None else round(now - self.last_input_ts, 1),
            "budget_total_s": self.daily_budget,
            "budget_spent_s": round(self.spent_today, 3),
            "budget_left_s": round(left, 3),
            "budget_left_frac": round(max(0.0, left) / self.daily_budget, 4) if self.daily_budget else 0.0,
            "day_ends_in_s": round(86400 - (now - time.mktime(time.strptime(self._day, "%Y-%m-%d"))), 0),
            "kwh_today": round(self.kwh_today, 9),
            "kwh_total": round(self.kwh_total, 9),
            "gpu_watts_now": self.gpu_watts(),
        }
