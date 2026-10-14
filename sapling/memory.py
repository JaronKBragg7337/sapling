"""Sapling memory — append-only, hash-chained, promotion by use, fading only for the unused.

Lineage, stated plainly:
- Append-only hash chain: Atomic Chronicle / prediction-research-system.
- Provenance label travels with the fact; "could not see" is an entry: watch.
- Persistence != confirmation: watch rule 5. Promotion changes retrieval
  priority, never truth status.
- Relational over linear: memory-as-spatial-arrangement. The packing-geometry
  strategy stays an honest empty slot (see arrangement_packing_geometry()).

v2 changes (Jaron 10/14, SAPLING-DECISIONS):
- "Memory fades by clock -> FIX. What matters / gets used should stay."
  A memory that was NEVER recalled fades on the real clock. A memory that
  has been recalled is protected: each use raises a floor its strength cannot
  drop below and doubles its half-life (spaced repetition). A human `keep`
  verdict or a `confirm` verdict counts as importance and strengthens it.
- "4 recall slots -> increase and let it grow." search() returns a top-k
  that grows with the store (k = max(4, sqrt(n))) and can blend the core's
  own token embeddings into the score (its vocabulary grows, so its sense of
  similarity grows with it).

Three tables, all append-only. Nothing is ever UPDATEd or DELETEd:
- memories: content + provenance + state (captured / confirmed / invalidated)
- recalls:  one row per recall event. Strength and tier are *derived* views,
  recomputed from events — so history is never rewritten, only re-read.
- ledger:   every tick, meter reading, consolidation, goal, stop report.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import threading
import time

DEFAULT_DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "memory.db"
)

GENESIS = "sapling-genesis"

# Fading applies ONLY to memories that were never used (Jaron 10/14).
UNUSED_HALF_LIFE_DAYS = 1.0
# Each recall doubles the half-life of the recall evidence (spaced repetition)
BASE_RECALL_HALF_LIFE_DAYS = 1.0
# A used memory never drops below this floor (per recall, capped)
FLOOR_PER_RECALL = 0.25
FLOOR_CAP = 3.0
# importance multipliers
IMPORTANCE_CONFIRMED = 1.5

# Tier by recall COUNT (never demoted by the clock; use is what promotes)
PROMOTE_WORKING = 2
PROMOTE_LONG_TERM = 5
PROMOTE_PERMANENT = 12


def _now() -> float:
    return time.time()


class _Rows:
    """Rows already fetched under the lock, so no thread ever reads a half-used cursor."""
    def __init__(self, rows, lastrowid, rowcount):
        self._rows, self.lastrowid, self.rowcount = list(rows), lastrowid, rowcount

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def __iter__(self):
        return iter(self._rows)


class _LockedDB:
    """One sqlite connection shared by the heartbeat thread and the control-room HTTP threads.
    Every statement runs and is fully fetched inside one lock (10/14 crash: a shared cursor was
    read by two threads at once -> InterfaceError / fetchone() returned None)."""
    def __init__(self, conn, lock):
        self._c, self._lock = conn, lock

    def execute(self, *a, **k):
        with self._lock:
            cur = self._c.execute(*a, **k)
            return _Rows(cur.fetchall() if cur.description else [], cur.lastrowid, cur.rowcount)

    def executescript(self, *a, **k):
        with self._lock:
            return self._c.executescript(*a, **k)

    def commit(self):
        with self._lock:
            return self._c.commit()

    def __getattr__(self, name):
        return getattr(self._c, name)


class MemoryStore:
    def __init__(self, path: str = DEFAULT_DB):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.lock = threading.RLock()
        self.db = _LockedDB(sqlite3.connect(path, check_same_thread=False), self.lock)
        self.db.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self):
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY,
                ts REAL NOT NULL,
                content TEXT NOT NULL,
                provenance TEXT NOT NULL,      -- user / retrieval / self / exam / verdict
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
                kind TEXT NOT NULL,            -- consumption / tick / meter / consolidation / goal / stop / ...
                payload TEXT NOT NULL,
                prev_hash TEXT NOT NULL,
                hash TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS recalls_mid ON recalls(memory_id);
            CREATE INDEX IF NOT EXISTS ledger_kind ON ledger(kind, id);
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
    def remember(self, content: str, provenance: str = "user", state: str = "captured",
                 ts: float | None = None) -> int:
        """`ts` exists so tests can plant an old memory; it is still appended, never edited."""
        with self.lock:
            ts = _now() if ts is None else ts
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

    def record_recall(self, memory_id: int, query: str, ts: float | None = None) -> None:
        with self.lock:
            ts = _now() if ts is None else ts
            payload = json.dumps({"ts": ts, "memory_id": memory_id, "query": query}, sort_keys=True)
            prev = self._last_hash("recalls")
            self.db.execute(
                "INSERT INTO recalls (ts, memory_id, query, prev_hash, hash) VALUES (?,?,?,?,?)",
                (ts, memory_id, query, prev, self._hash(payload, prev)),
            )
            self.db.commit()

    def meter(self, kind: str, payload: dict) -> int:
        """Ledger entry (ECIH: meter what you consume; Keava: contribution is
        recorded-not-scored). v2: ticks, meter readings, consolidations, goals
        and stop reports all go here too — one chain for everything it did."""
        with self.lock:
            ts = _now()
            body = json.dumps({"ts": ts, "kind": kind, **payload}, sort_keys=True, default=str)
            prev = self._last_hash("ledger")
            cur = self.db.execute(
                "INSERT INTO ledger (ts, kind, payload, prev_hash, hash) VALUES (?,?,?,?,?)",
                (ts, kind, body, prev, self._hash(body, prev)),
            )
            self.db.commit()
            return int(cur.lastrowid)

    # -- ledger reads ----------------------------------------------------------
    def ledger_rows(self, kind: str | None = None, limit: int = 100, since_id: int = 0) -> list[dict]:
        q = "SELECT id, ts, kind, payload FROM ledger WHERE id>?"
        args: list = [since_id]
        if kind:
            q += " AND kind=?"
            args.append(kind)
        q += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        out = []
        for rid, ts, k, payload in self.db.execute(q, args).fetchall():
            d = json.loads(payload)
            d["id"] = rid
            out.append(d)
        return out

    def last_ledger(self, kind: str) -> dict | None:
        rows = self.ledger_rows(kind, limit=1)
        return rows[0] if rows else None

    def count(self, table: str = "memories") -> int:
        return int(self.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

    # -- derived views -------------------------------------------------------
    def _recall_events(self, memory_id: int) -> list[float]:
        rows = self.db.execute(
            "SELECT ts FROM recalls WHERE memory_id=? ORDER BY ts", (memory_id,)
        ).fetchall()
        return [r[0] for r in rows]

    def _verdict(self, memory_id: int, verb: str) -> bool:
        """A human verdict is itself a memory row referencing the id ('keep #3')."""
        rows = self.db.execute(
            "SELECT 1 FROM memories WHERE provenance='verdict' AND content LIKE ?",
            (f"{verb} #{memory_id}",),
        ).fetchall()
        return bool(rows)

    def state_of(self, memory_id: int) -> str:
        """Truth status. Stored at insert; changed ONLY by a later verdict event
        ('confirm #id' / 'invalidate #id'). Recall frequency has no path here."""
        if self._verdict(memory_id, "invalidate"):
            return "invalidated"
        if self._verdict(memory_id, "confirm"):
            return "confirmed"
        row = self.db.execute("SELECT state FROM memories WHERE id=?", (memory_id,)).fetchone()
        return row[0] if row else "unknown"

    def recall_count(self, memory_id: int) -> int:
        row = self.db.execute(
            "SELECT COUNT(*) FROM recalls WHERE memory_id=?", (memory_id,)).fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    def tier(self, memory_id: int) -> str:
        if self._verdict(memory_id, "keep"):
            return "permanent"
        n = self.recall_count(memory_id)
        if n >= PROMOTE_PERMANENT:
            return "permanent"
        if n >= PROMOTE_LONG_TERM:
            return "long_term"
        if n >= PROMOTE_WORKING:
            return "working"
        return "short_term"

    def strength(self, memory_id: int) -> float:
        """Retrieval weight.

        never used  -> fades on the clock (0.5 ** (age / 1 day)), approaches 0
        used        -> protected: floor rises with each use (capped), evidence
                       half-life doubles with each use, importance multiplies
        kept/confirmed verdict -> permanent / x1.5
        """
        if self._verdict(memory_id, "keep"):
            return float("inf")
        row = self.db.execute("SELECT ts FROM memories WHERE id=?", (memory_id,)).fetchone()
        if not row:
            return 0.0
        now = _now()
        events = self._recall_events(memory_id)
        importance = IMPORTANCE_CONFIRMED if self.state_of(memory_id) == "confirmed" else 1.0
        if not events:
            age_days = max(0.0, (now - row[0]) / 86400.0)
            return importance * 0.5 ** (age_days / UNUSED_HALF_LIFE_DAYS)
        n = len(events)
        floor = min(FLOOR_CAP, FLOOR_PER_RECALL * n)
        hl = BASE_RECALL_HALF_LIFE_DAYS * (2 ** min(n, 20))
        ev = sum(0.5 ** (max(0.0, (now - t) / 86400.0) / hl) for t in events)
        return importance * (1.0 + floor + ev)

    # -- retrieval ------------------------------------------------------------
    # stopwords carry no retrieval signal ("the" alone must never pull a memory)
    STOPWORDS = {"the", "a", "an", "what", "does", "do", "mean", "means",
                 "is", "of", "?", "word", ".", ":",
                 # v3 (sentences): little words that pulled unrelated memories ("at" matched "at generation 152")
                 "at", "in", "on", "to", "and", "it", "i", "you", "he", "she", "they", "we", "are", "was", ","}

    @classmethod
    def _overlap(cls, query: str, content: str) -> float:
        # "" must be discarded: "?" and "." strip to "" and would match everything
        # (run 1 bug: one self memory got recalled on every query and went permanent)
        q = {w.strip(".,?:!") for w in query.lower().split()} - cls.STOPWORDS - {""}
        c = {w.strip(".,?:!") for w in content.lower().split()} - cls.STOPWORDS - {""}
        if not q or not c:
            return 0.0
        return len(q & c) / len(q | c)

    @staticmethod
    def top_k_for(n: int) -> int:
        """Recall width grows with the store: 4 at first, sqrt(n) later."""
        return max(4, int(math.sqrt(max(0, n))))

    def search(self, query: str, limit: int | None = None, min_score: float = 0.05,
               embed_fn=None, record: bool = True) -> list[dict]:
        """embed_fn(text) -> vector (list[float]) lets the core's own embeddings
        join the score (0.6 lexical + 0.4 cosine). Without it, lexical only."""
        rows = self.db.execute(
            "SELECT id, ts, content, provenance, state FROM memories WHERE provenance!='verdict'"
        ).fetchall()
        k = limit if limit is not None else self.top_k_for(len(rows))
        qvec = embed_fn(query) if embed_fn else None
        scored = []
        for mid, ts, content, prov, state in rows:
            rel = self._overlap(query, content)
            if rel <= 0:
                continue
            if qvec is not None:
                # the core's own embeddings RE-RANK lexical hits; they never
                # create a hit on their own (a tiny core's mean-pooled vectors
                # are all alike, so cosine alone pulled unrelated memories)
                rel *= 0.7 + 0.3 * max(0.0, _cosine(qvec, embed_fn(content)))
            st = self.strength(mid)
            score = rel * st
            if score >= min_score:
                scored.append(
                    {"id": mid, "content": content, "provenance": prov,
                     "state": self.state_of(mid), "tier": self.tier(mid),
                     "rel": round(rel, 3), "strength": round(st, 3) if st != float("inf") else "inf",
                     "score": score}
                )
        scored.sort(key=lambda r: r["score"], reverse=True)
        hits = scored[:k]
        if record:
            for hit in hits:
                self.record_recall(hit["id"], query)  # recall is itself recorded
        return hits

    def report(self, limit: int | None = None) -> list[dict]:
        rows = self.db.execute(
            "SELECT id, ts, content, provenance, state FROM memories ORDER BY id DESC"
            + (f" LIMIT {int(limit)}" if limit else "")
        ).fetchall()
        out = []
        for mid, ts, content, prov, state in rows:
            st = self.strength(mid)
            out.append(
                {"id": mid, "age_h": round((_now() - ts) / 3600, 2), "content": content,
                 "provenance": prov, "state": self.state_of(mid), "recalls": self.recall_count(mid),
                 "tier": self.tier(mid), "strength": round(st, 3) if st != float("inf") else "inf"}
            )
        return out

    def facts(self) -> list[dict]:
        """Everything taught or found (not verdicts / exam rows), with derived state."""
        rows = self.db.execute(
            "SELECT id, ts, content, provenance FROM memories "
            "WHERE provenance IN ('user','self','retrieval') ORDER BY id"
        ).fetchall()
        return [{"id": mid, "ts": ts, "content": c, "provenance": p, "state": self.state_of(mid),
                 "recalls": self.recall_count(mid)} for mid, ts, c, p in rows]

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


def _cosine(a, b) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a)) or 1e-9
    db = math.sqrt(sum(y * y for y in b)) or 1e-9
    return num / (da * db)


if __name__ == "__main__":
    import sys

    store = MemoryStore(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB)
    print(f"chain ok: {store.verify_chain()}")
    for row in store.report():
        print(f"  #{row['id']:<3} [{row['tier']:<10}] recalls={row['recalls']:<3} "
              f"strength={row['strength']:<7} ({row['provenance']}/{row['state']}) "
              f"{row['content']}")
