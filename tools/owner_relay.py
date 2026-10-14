"""owner_relay.py — the OWNER's door to sapling. Runs as its own process (Windows
scheduled task "Sapling Relay"), outside the AI's heartbeat process, on purpose.

Jaron 10/14 (2:59 PM): the control room + chat must be reachable from his PHONE,
behind the owner login, never an open URL; control-room clicks may change the
rule file as an owner action (logged; the AI still has no write path); after a
self-stop the owner decides parole with a "Restart (parole)" button.

How it is reached
- Local:  http://127.0.0.1:7338/  (this process serves the same page as the
  heartbeat's own http://127.0.0.1:7337/ and proxies chat/state to it).
- Phone:  https://heartbeatobservatory.com/admin/sapling/  (the site's own
  admin login: Supabase session + is_admin()). That page never talks to the MSI
  directly. It reads `sapling_state` and inserts rows into `sapling_commands`;
  this relay (outbound only, service role) pushes state every few seconds and
  polls the commands. No port on the MSI is opened to the internet.

Owner actions this relay performs (each one: results/owner_actions.jsonl + a
ledger row of kind owner_action in the AI's own hash-chained record):
  rule        {id, enabled}   -> tools/owner_rules.py flips one switch in membrane.yaml
  restart     {}              -> `python -m sapling.heartbeat start` (parole), logged next to the stop report
  stop        {}              -> `python -m sapling.heartbeat stop` (owner stop -> stop report)
  restore     {gen}           -> sapling.archive.restore (download gen_N.pt back to the MSI)
  archive_now {}              -> sapling.archive.archive_step

Usage: python tools/owner_relay.py [--no-cloud]
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sapling import membrane                                    # noqa: E402
from sapling.archive import archive_step, creds, list_local, manifest_local, restore, _req   # noqa: E402
from sapling.control_room import OWNER_LOG, PAGE_FILE            # noqa: E402
from sapling.heartbeat import PID_FILE, STOP_DIR, _pid_alive     # noqa: E402
from sapling.memory import MemoryStore                           # noqa: E402
from sapling.train import RESULTS_DIR                            # noqa: E402
import owner_rules                                               # noqa: E402

SET = membrane.settings()
HB = f"http://127.0.0.1:{SET['control_room_port']}"
PORT = int(SET.get("relay_port", 7338))
LOG = os.path.join(RESULTS_DIR, "relay.log")
STARTED = time.time()


def log(msg: str):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def hb_get(path: str, timeout: float = 8):
    with urllib.request.urlopen(HB + path, timeout=timeout) as r:
        return json.loads(r.read())


def hb_post(path: str, body: dict, timeout: float = 60):
    req = urllib.request.Request(HB + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def heartbeat_alive() -> bool:
    try:
        return os.path.exists(PID_FILE) and _pid_alive(int(open(PID_FILE).read().strip() or 0))
    except Exception:
        return False


def owner_actions_tail(n: int = 20) -> list[dict]:
    if not os.path.exists(OWNER_LOG):
        return []
    try:
        lines = open(OWNER_LOG, encoding="utf-8").read().splitlines()
        return [json.loads(x) for x in lines[-n:]][::-1]
    except Exception:
        return []


def down_state() -> dict:
    """What the page shows while the heartbeat process is not running: the
    record it left (last ticks, stop reports, rules, archive) + the parole button."""
    cfg = membrane.load()
    stops = []
    if os.path.isdir(STOP_DIR):
        for name in sorted(os.listdir(STOP_DIR), reverse=True)[:10]:
            try:
                r = json.load(open(os.path.join(STOP_DIR, name), encoding="utf-8"))
                stops.append({k: r.get(k) for k in ("ts", "who", "reason", "tick", "kind")})
            except Exception:
                pass
    ticks, cons, memories, last_arch, last_vocab = [], [], 0, None, None
    birth_ts = last_input_ts = None
    try:
        store = MemoryStore()
        ticks = store.ledger_rows("tick", limit=30)
        cons = [{k: r.get(k) for k in ("generation", "facts_self", "facts_user", "facts_total", "self_new")}
                for r in store.ledger_rows("consolidation", limit=12)]
        memories = store.count()
        last_arch = store.last_ledger("archive")
        r1 = store.db.execute("SELECT ts FROM ledger WHERE kind='birth' ORDER BY id LIMIT 1").fetchone()
        r2 = store.db.execute("SELECT ts FROM ledger WHERE kind='input' ORDER BY id DESC LIMIT 1").fetchone()
        birth_ts = float(r1[0]) if r1 else None
        last_input_ts = float(r2[0]) if r2 else None
        last_vocab = (store.last_ledger("consolidation") or {}).get("vocab_size")
        store.db.close()
    except Exception:
        pass
    local = list_local()
    rows = manifest_local()
    # the record the heartbeat left: last tick has the budget; birth / last input come from the ledger
    clock, generation = {}, None
    try:
        if ticks:
            clock = dict(ticks[0].get("clock") or {})
        if birth_ts is not None:
            clock["born_iso"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(birth_ts))
            clock["age_s"] = round(time.time() - birth_ts, 1)
        clock["since_last_input_s"] = None if last_input_ts is None else round(time.time() - last_input_ts, 1)
        clock["uptime_s"] = 0
        if cons:
            generation = cons[0].get("generation")
    except Exception:
        pass
    return {"running": False, "down": True, "tick": ticks[0]["tick"] if ticks else None,
            "generation": generation, "vocab_size": last_vocab,
            "last_action": ticks[0]["action"] if ticks else None, "clock": clock, "memories": memories,
            "rules": cfg["rules"], "settings": cfg["settings"], "goals": [], "exams": [], "ticks": ticks,
            "stops": stops, "last_stop": stops[0] if stops else None, "memory": [], "consolidations": cons,
            "archive": {"rows": rows, "local_gens": [c["gen"] for c in local],
                        "local_mb": round(sum(c["size"] for c in local) / 1e6, 1),
                        "cloud_mb": round(sum(int(r.get("size_bytes") or 0) for r in rows) / 1e6, 1),
                        "last_run": time.strftime("%m-%d %H:%M", time.localtime(float(last_arch["ts"]))) if last_arch else None},
            "owner_actions": owner_actions_tail()}


def full_state() -> dict:
    try:
        st = hb_get("/api/state")
    except Exception as e:
        st = down_state()
        st["hb_error"] = str(e)
    st["relay"] = f"up {int(time.time() - STARTED)} s, pid {os.getpid()}"
    return st


# -- owner actions ---------------------------------------------------------------------
def do_owner(kind: str, payload: dict, by: str) -> dict:
    payload = payload or {}
    store = None
    try:
        store = MemoryStore()
        if kind == "rule":
            rid, en = str(payload.get("id", "")), bool(payload.get("enabled"))
            row = owner_rules.set_enabled(rid, en, by, str(payload.get("why", "control room switch")), store=store)
            return {"ok": True, "row": row}
        if kind == "restart":
            if heartbeat_alive():
                return {"ok": True, "note": "already running"}
            last_stop = (down_state().get("last_stop") or {})
            p = subprocess.Popen([sys.executable, "-m", "sapling.heartbeat", "start"], cwd=ROOT,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            out, _ = p.communicate(timeout=60)
            row = owner_rules.log_owner_action({"action": "parole_start", "by": by, "after_stop": last_stop,
                                                "start_output": out.strip()[:300]}, store=store)
            return {"ok": True, "row": row, "out": out.strip()}
        if kind == "stop":
            if not heartbeat_alive():
                return {"ok": True, "note": "not running"}
            subprocess.run([sys.executable, "-m", "sapling.heartbeat", "stop"], cwd=ROOT, timeout=30)
            row = owner_rules.log_owner_action({"action": "owner_stop", "by": by}, store=store)
            return {"ok": True, "row": row}
        if kind == "restore":
            gen = int(payload.get("gen"))
            r = restore(gen, store=store, who=by)
            row = owner_rules.log_owner_action({"action": "restore", "gen": gen, "by": by, "dest": r["dest"]}, store=store)
            return {"ok": True, "row": row, "restore": r}
        if kind == "archive_now":
            r = archive_step(store)
            row = owner_rules.log_owner_action({"action": "archive_now", "by": by, "moved": r.get("moved"),
                                                "errors": r.get("errors")}, store=store)
            return {"ok": not r.get("errors"), "row": row, "archive": r, "error": "; ".join(r.get("errors") or []) or None}
        return {"ok": False, "error": f"unknown owner action '{kind}'"}
    except Exception as e:
        log(f"owner {kind} failed: {type(e).__name__}: {e}")
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    finally:
        if store is not None:
            try:
                store.db.close()
            except Exception:
                pass


def do_chat(text: str) -> dict:
    if not text.strip():
        return {"out": ""}
    try:
        return hb_post("/api/chat", {"text": text})
    except Exception as e:
        err = f"{type(e).__name__}"
    # heartbeat is down: answer from a plain session (same memory + weights, no ticking), if that works
    try:
        from sapling.session import Session
        sess = Session()
        try:
            out = sess.handle(text)
        finally:
            try:
                sess.store.db.close()
            except Exception:
                pass
        return {"out": out, "via": "relay session (the heartbeat is down, so this answer did not come from a ticking sapling)"}
    except Exception as e2:
        return {"out": "", "down": True,
                "error": f"heartbeat is down ({err}) and the relay could not answer either ({type(e2).__name__}: {e2}). Press Restart (parole)."}


# -- local server (127.0.0.1:7338) -----------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _cors(self):
        origin = self.headers.get("origin") or ""
        if origin in (f"http://127.0.0.1:{SET['control_room_port']}", f"http://localhost:{SET['control_room_port']}"):
            self.send_header("access-control-allow-origin", origin)
            self.send_header("access-control-allow-headers", "content-type")

    def _json(self, obj, code=200):
        body = json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("access-control-allow-methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/", "/index.html"):
            body = open(PAGE_FILE, encoding="utf-8").read().encode()
            self.send_response(200)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif p == "/api/state":
            self._json(full_state())
        elif p == "/health":
            self._json({"relay": "ok", "heartbeat_alive": heartbeat_alive()})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("content-length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}") if n else {}
        except Exception:
            body = {}
        p = self.path.split("?")[0]
        if p == "/api/chat":
            self._json(do_chat(str(body.get("text", ""))))
        elif p.startswith("/owner/"):
            kind = p[len("/owner/"):]
            r = do_owner(kind, body, by="owner@local")
            log(f"owner/{kind} by owner@local -> {json.dumps(r, default=str)[:200]}")
            self._json(r)
        else:
            self._json({"error": "not found"}, 404)


# -- cloud loop (outbound only) -------------------------------------------------------
def cloud_loop(c: dict):
    log("cloud loop: on (state push + command poll)")
    last_push = 0.0
    retry_at: dict = {}
    while True:
        try:
            now = time.time()
            if now - last_push >= 5:
                st = full_state()
                st["heartbeat_alive"] = heartbeat_alive()
                code, out = _req(c, "POST", "/rest/v1/sapling_state", json.dumps(
                    {"id": 1, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                     "heartbeat_alive": st["heartbeat_alive"], "state": st}, default=str).encode(),
                    {"content-type": "application/json", "prefer": "resolution=merge-duplicates,return=minimal"})
                if code not in (200, 201, 204):
                    log(f"state push failed: {code} {out[:120]!r}")
                last_push = now
            code, out = _req(c, "GET", "/rest/v1/sapling_commands?status=eq.pending&order=id.asc&limit=5")
            if code == 200:
                for cmd in json.loads(out):
                    by = str(cmd.get("created_by") or "owner@site")
                    kind, payload = cmd["kind"], cmd.get("payload") or {}
                    if kind == "chat":
                        if time.time() < retry_at.get(cmd["id"], 0):
                            continue
                        result = do_chat(str(payload.get("text", "")))
                        if result.get("down"):
                            # heartbeat down and no session answer: leave it pending; the page says so; retried in 60 s
                            retry_at[cmd["id"]] = time.time() + 60
                            log(f"cloud cmd #{cmd['id']} chat held (heartbeat down)")
                            continue
                    else:
                        result = do_owner(kind, payload, by)
                    log(f"cloud cmd #{cmd['id']} {kind} by {by} -> {json.dumps(result, default=str)[:160]}")
                    status = "error" if (isinstance(result, dict) and result.get("error") and not result.get("ok")) else "done"
                    _req(c, "PATCH", f"/rest/v1/sapling_commands?id=eq.{cmd['id']}",
                         json.dumps({"status": status, "result": result, "done_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
                                    default=str).encode(),
                         {"content-type": "application/json", "prefer": "return=minimal"})
                    last_push = 0.0        # push fresh state right after an action
            else:
                log(f"command poll failed: {code} {out[:120]!r}")
        except Exception as e:
            log(f"cloud loop error: {type(e).__name__}: {e}")
        time.sleep(3)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True, name="relay-http").start()
    log(f"relay up on http://127.0.0.1:{PORT}/ (heartbeat at {HB}, alive={heartbeat_alive()})")
    c = None if "--no-cloud" in argv else creds()
    if c is None:
        log("cloud loop: OFF (no credentials or --no-cloud); local page only")
        while True:
            time.sleep(60)
    cloud_loop(c)


if __name__ == "__main__":
    main()
