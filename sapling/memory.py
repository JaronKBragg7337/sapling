"""Sapling memory — append-only, hash-chained, promotion by recall, decay by clock.

Lineage, stated plainly:
- Append-only hash chain: Atomic Chronicle / prediction-research-system.
- Forgetting on the real clock (half-life): fly-school (Tully & Quinn 1985).
- Provenance label travels with the fact; "could not see" is an entry: watch.
- Persistence != confirmation: watch rule 5. Promotion changes retrieval
  priority, never truth status.
- Relational over linear: memory-as-spatial-arrangement (sim-tested ~3x on
  recall). The packing-geometry strategy is an honest empty slot (see
  arrangement_packing_geometry() below): the math is new, the application
  is unbuilt, and nothing here pretends otherwise.

Two tables, both append-only. Nothing is ever UPDATEd or DELETEd:
- memories: content + provenance + state (captured / confirmed / invalidated)
- recalls:  one row per recall event. Strength and tier are *derived* views,
  recomputed from events — so history is never rewritten, only re-read.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time

DEFAULT_DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "memory.db"
)

GENESIS = "sapling-genesis"

# Wall-clock half-lives in days, per derived tier (fly-school: forgetting
# runs on the real clock, so spaced practice is what survives).
HALF_LIFE_DAYS = {"short_term": 0.25, "working": 1.0, "long_term": 7.0, "permanent": None}

# Promotion thresholds on *decayed recall evidence* (each recall contributes
# 0.5^(age/half_life), so recent repeated recall promotes; old recall fades).
PROMOTE_WORKING = 2.0
PROMOTE_LONG_TERM = 5.0
# permanent is never reached by frequency alone: it needs a human "keep"
# verdict OR sustained recall evidence above this bar.
PROMOTE_PERMANENT = 12.0


def _now() -> float:
    return time.time()


class MemoryStore:
    def __init__(self, path: str = DEFAULT_DB):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self):
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY,
                ts REAL NOT NULL,
                content TEXT NOT NULL,
                provenance TEXT NOT NULL,      -- user / retrieval / self / exam
                state TEXT NOT NULL,           -- captured / confirmed / invalidated
                prev_hash TEXT NOT NULL,
                hash TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS recalls (
                id INTEGER PRIMARY KEY,
                ts REAL NOT NULL,
                memory_id INTEGER NOT NULL,
                query TEXT NOT NULL,
                prev_hash TEXT NOT NULL,
                hash TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS ledger (
                id INTEGER PRIMARY KEY,
                ts REAL NOT NULL,
                kind TEXT NOT NULL,            -- consumption / contribution
                payload TEXT NOT NULL,
                prev_hash TEXT NOT NULL,
                hash TEXT NOT NULL
            );
            """
        )
        self.db.commit()

    # -- hash chain ----------------------------------------------------------
    def _last_hash(self, table: str) -> str:
        row = self.db.execute(f"SELECT hash FROM {table} ORDER BY id DESC LIMIT 1").fetchone()
        return row[0] if row else GENESIS

    @staticmethod
    def _hash(payload: str, prev_hash: str) -> str:
        return hashlib.sha256((prev_hash + "|" + payload).encode()).hexdigest()

    # -- writes (append only; there is deliberately no update/delete API) ----
    def remember(self, content: str, provenance: str = "user", state: str = "captured") -> int:
        ts = _now()
        payload = json.dumps(
            {"ts": ts, "content": content, "provenance": provenance, "state": state},
            sort_keys=True,
        )
        prev = self._last_hash("memories")
        cur = self.db.execute(
            "INSERT INTO memories (ts, content, provenance, state, prev_hash, hash)"
            " VALUES (?,?,?,?,?,?)",
            (ts, content, provenance, state, prev, self._hash(payload, prev)),
        )
        self.db.commit()
        return int(cur.lastrowid)

    def record_recall(self, memory_id: int, query: str) -> None:
        ts = _now()
        payload = json.dumps({"ts": ts, "memory_id": memory_id, "query": query}, sort_keys=True)
        prev = self._last_hash("recalls")
        self.db.execute(
            "INSERT INTO recalls (ts, memory_id, query, prev_hash, hash) VALUES (?,?,?,?,?)",
            (ts, memory_id, query, prev, self._hash(payload, prev)),
        )
        self.db.commit()

    def meter(self, kind: str, payload: dict) -> None:
        """Consumption/contribution ledger entry (ECIH: meter what you consume;
        Keava: contribution is recorded-not-scored — verdicts come from people)."""
        ts = _now()
        body = json.dumps({"ts": ts, "kind": kind, **payload}, sort_keys=True)
        prev = self._last_hash("ledger")
        self.db.execute(
            "INSERT INTO ledger (ts, kind, payload, prev_hash, hash) VALUES (?,?,?,?,?)",
            (ts, kind, body, prev, self._hash(body, prev)),
        )
        self.db.commit()

    # -- derived views -------------------------------------------------------
    def _recall_events(self, memory_id: int) -> list[float]:
        rows = self.db.execute(
            "SELECT ts FROM recalls WHERE memory_id=? ORDER BY ts", (memory_id,)
        ).fetchall()
        return [r[0] for r in rows]

    def _verdict_keep(self, memory_id: int) -> bool:
        """A human 'keep' verdict is itself a memory row referencing the id."""
        rows = self.db.execute(
            "SELECT 1 FROM memories WHERE provenance='verdict' AND content LIKE ?",
            (f"keep #{memory_id}%",),
        ).fetchall()
        return bool(rows)

    def recall_evidence(self, memory_id: int, half_life_days: float = 1.0) -> float:
        now = _now()
        ev = 0.0
        for ts in self._recall_events(memory_id):
            age_days = max(0.0, (now - ts) / 86400.0)
            ev += 0.5 ** (age_days / half_life_days)
        return ev

    def tier(self, memory_id: int) -> str:
        if self._verdict_keep(memory_id):
            return "permanent"
        ev = self.recall_evidence(memory_id)
        if ev >= PROMOTE_PERMANENT:
            return "permanent"
        if ev >= PROMOTE_LONG_TERM:
            return "long_term"
        if ev >= PROMOTE_WORKING:
            return "working"
        return "short_term"

    def strength(self, memory_id: int) -> float:
        """Retrieval weight: creation boost + decayed recalls, tier half-life."""
        t = self.tier(memory_id)
        hl = HALF_LIFE_DAYS[t]
        if hl is None:
            return float("inf")
        row = self.db.execute("SELECT ts FROM memories WHERE id=?", (memory_id,)).fetchone()
        if not row:
            return 0.0
        age_days = max(0.0, (_now() - row[0]) / 86400.0)
        base = 0.5 ** (age_days / hl)          # the creation event itself decays
        return base + self.recall_evidence(memory_id, half_life_days=hl)

    # -- retrieval ------------------------------------------------------------
    # stopwords carry no retrieval signal ("the" alone must never pull a memory)
    STOPWORDS = {"the", "a", "an", "what", "does", "do", "mean", "means",
                 "is", "of", "?", "word"}

    @classmethod
    def _overlap(cls, query: str, content: str) -> float:
        q = {w.strip(".,?:!") for w in query.lower().split()} - cls.STOPWORDS
        c = {w.strip(".,?:!") for w in content.lower().split()} - cls.STOPWORDS
        if not q or not c:
            return 0.0
        return len(q & c) / len(q | c)

    def search(self, query: str, limit: int = 3, min_score: float = 0.08) -> list[dict]:
        rows = self.db.execute(
            "SELECT id, ts, content, provenance, state FROM memories"
        ).fetchall()
        scored = []
        for mid, ts, content, prov, state in rows:
            rel = self._overlap(query, content)
            if rel <= 0:
                continue
            st = self.strength(mid)
            score = rel * st
            if score >= min_score:
                scored.append(
                    {"id": mid, "content": content, "provenance": prov, "state": state,
                     "tier": self.tier(mid), "rel": round(rel, 3),
                     "strength": round(st, 3), "score": score}
                )
        scored.sort(key=lambda r: r["score"], reverse=True)
        for hit in scored[:limit]:
            self.record_recall(hit["id"], query)  # recall is itself recorded
        return scored[:limit]

    def report(self) -> list[dict]:
        rows = self.db.execute(
            "SELECT id, ts, content, provenance, state FROM memories ORDER BY id"
        ).fetchall()
        out = []
        for mid, ts, content, prov, state in rows:
            n = len(self._recall_events(mid))
            out.append(
                {"id": mid, "age_h": round((_now() - ts) / 3600, 2), "content": content,
                 "provenance": prov, "state": state, "recalls": n,
                 "tier": self.tier(mid), "strength": round(self.strength(mid), 3)}
            )
        return out

    # -- integrity -------------------------------------------------------------
    def verify_chain(self) -> bool:
        ok = True
        for table, cols in (
            ("memories", ("ts", "content", "provenance", "state")),
            ("recalls", ("ts", "memory_id", "query")),
        ):
            prev = GENESIS
            rows = self.db.execute(
                f"SELECT id, {', '.join(cols)}, prev_hash, hash FROM {table} ORDER BY id"
            ).fetchall()
            for row in rows:
                rid = row[0]
                vals = row[1: 1 + len(cols)]
                prev_hash, stored = row[-2], row[-1]
                payload = json.dumps(dict(zip(cols, vals)), sort_keys=True)
                if prev_hash != prev or self._hash(payload, prev_hash) != stored:
                    print(f"[memory] CHAIN BREAK at {table}#{rid}")
                    ok = False
                prev = stored
        prev = GENESIS
        for rid, _ts, _kind, payload, prev_hash, stored in self.db.execute(
            "SELECT id, ts, kind, payload, prev_hash, hash FROM ledger ORDER BY id"
        ).fetchall():
            if prev_hash != prev or self._hash(payload, prev_hash) != stored:
                print(f"[memory] CHAIN BREAK at ledger#{rid}")
                ok = False
            prev = stored
        return ok

    # -- the honest empty slot -------------------------------------------------
    def arrangement_packing_geometry(self):
        """Open Problem, kept open on purpose (Keava-Owent precedent).

        1. The reaching toward a non-linear memory medium is OLD.
        2. Relational memory is OLD and already in production (used above).
        3. The packing math is NEW (planar unit-distance result, May 2026).
        4. Applying that packing geometry to memory arrangement is UNBUILT.
        """
        raise NotImplementedError(
            "packing-geometry arrangement is an open problem; relational "
            "arrangement is what runs. See memory-as-spatial-arrangement."
        )


if __name__ == "__main__":
    import sys

    store = MemoryStore(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB)
    print(f"chain ok: {store.verify_chain()}")
    for row in store.report():
        print(f"  #{row['id']:<3} [{row['tier']:<10}] recalls={row['recalls']:<3} "
              f"strength={row['strength']:<7} ({row['provenance']}/{row['state']}) "
              f"{row['content']}")
