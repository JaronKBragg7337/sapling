"""The control room — a plain-words page on localhost, served by the heartbeat.

Jaron's control-rooms principle: every system ships with a page a non-dev can
run it from. This one shows: every rule with its origin and switch state,
live heartbeat status, the budget meter, the tick log, stop reports, goals,
exam history, memory size, vocab size - and a chat box to talk to it.

The rules on this page come straight from membrane.yaml on every request.
The switches are READ-ONLY here on purpose: only the owner's edit of the
file changes a rule, and this process (the AI's own process) has no write
path to that file. "Open rule file" opens it in the owner's editor.

Bound to 127.0.0.1 only. Not exposed publicly (a tunnel can read it later).
"""

from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import membrane
from .exams import REPORT_CARD, WRITING_DIR
from .archive import list_local, manifest_local
from .heartbeat import STOP_DIR

OWNER_LOG = os.path.join(os.path.dirname(STOP_DIR), "owner_actions.jsonl")

PAGE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "control_room.html")


def page() -> str:
    """Read on every request so an edit shows without a restart. The same file is
    copied to heartbeatobservatory.com/admin/sapling/ (phone path, owner login)."""
    with open(PAGE_FILE, encoding="utf-8") as f:
        return f.read()


def state_of(hb) -> dict:
    cfg = membrane.load()
    exams = []
    if os.path.exists(REPORT_CARD):
        try:
            card = json.load(open(REPORT_CARD, encoding="utf-8"))
            hist = card.get("history", []) + [{k: card.get(k) for k in ("ts", "trigger", "generation")} | {"exams": card.get("exams", [])}]
            exams = list(reversed(hist))[:12]
        except Exception:
            exams = []
    stops = []
    if os.path.isdir(STOP_DIR):
        for name in sorted(os.listdir(STOP_DIR), reverse=True)[:10]:
            try:
                r = json.load(open(os.path.join(STOP_DIR, name), encoding="utf-8"))
                stops.append({k: r.get(k) for k in ("ts", "who", "reason", "tick", "kind")})
            except Exception:
                pass
    ticks = hb.store.ledger_rows("tick", limit=30)
    local = list_local()
    arch_rows = manifest_local()
    last_arch = hb.store.last_ledger("archive")
    owner_actions = []
    if os.path.exists(OWNER_LOG):
        try:
            lines = open(OWNER_LOG, encoding="utf-8").read().splitlines()
            owner_actions = [json.loads(x) for x in lines[-20:]][::-1]
        except Exception:
            owner_actions = []
    writing = []
    if os.path.isdir(WRITING_DIR):
        for name in sorted(os.listdir(WRITING_DIR), reverse=True)[:3]:
            try:
                writing.append({"file": name, "text": open(os.path.join(WRITING_DIR, name), encoding="utf-8").read()[:3000]})
            except Exception:
                pass
    growth = hb.store.last_ledger("growth")
    from .reader import documents_summary
    try:
        docs = [{k: d.get(k) for k in ("id", "title", "source", "passages", "recalls")} | {"ts": time.strftime("%m-%d %H:%M", time.localtime(float(d["ts"])))}
                for d in documents_summary(hb.store)][-20:]
    except Exception:
        docs = []
    from .reading import summary as reading_summary
    try:
        reading = reading_summary()
    except Exception:
        reading = None
    return {
        "documents": docs,
        "reading": reading,
        "core": hb.brain.core_info(),
        "growth": {k: growth.get(k) for k in ("event", "ts", "seconds")} | {
            "from": {k: (growth.get("from") or {}).get(k) for k in ("generation", "params", "vocab_size")},
            "to": {k: (growth.get("to") or {}).get(k) for k in ("generation", "params", "vocab_size")}} if growth else None,
        "writing": writing,
        "consolidations": [{k: r.get(k) for k in ("generation", "facts_self", "facts_user", "facts_total", "self_new")}
                           for r in hb.store.ledger_rows("consolidation", limit=12)],
        "archive": {"rows": arch_rows, "local_gens": [c["gen"] for c in local],
                    "local_mb": round(sum(c["size"] for c in local) / 1e6, 1),
                    "cloud_mb": round(sum(int(r.get("size_bytes") or 0) for r in arch_rows) / 1e6, 1),
                    "last_run": time.strftime("%m-%d %H:%M", time.localtime(float(last_arch["ts"]))) if last_arch else None},
        "owner_actions": owner_actions, "last_stop": stops[0] if stops else None,
        "running": hb.running, "tick": hb.tick_no,
        "last_action": ticks[0]["action"] if ticks else None,
        "generation": hb.brain.generation, "memories": hb.store.count(),
        "vocab_size": hb.brain.vocab_size(), "clock": hb.clock.snapshot(),
        "rules": cfg["rules"], "settings": cfg["settings"],
        "goals": hb.goals.update(), "exams": exams, "ticks": ticks, "stops": stops,
        "memory": hb.store.report(limit=30),
    }


def make_handler(hb):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj, default=str).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.split("?")[0] in ("/", "/index.html"):
                body = page().encode()
                self.send_response(200)
                self.send_header("content-type", "text/html; charset=utf-8")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path.startswith("/api/state"):
                self._json(state_of(hb))
            elif self.path.startswith("/api/ticks"):
                self._json(hb.store.ledger_rows("tick", limit=200))
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            n = int(self.headers.get("content-length") or 0)
            raw = self.rfile.read(n) if n else b""
            if self.path == "/api/chat":
                try:
                    text = json.loads(raw or b"{}").get("text", "").strip()
                except Exception:
                    text = ""
                if not text:
                    return self._json({"out": ""})
                try:
                    out = hb.chat(text)
                except SystemExit:
                    out = "(/quit does nothing here; the heartbeat decides when it stops)"
                except Exception as e:
                    out = f"error: {type(e).__name__}: {e}"
                self._json({"out": out})
            elif self.path == "/api/open_rules":
                try:
                    os.startfile(membrane.MEMBRANE_PATH)  # the OWNER's editor, not this process, edits it
                    self._json({"ok": True})
                except Exception as e:
                    self._json({"ok": False, "error": str(e)})
            else:
                self._json({"error": "not found"}, 404)

    return Handler


def serve(hb, port: int = 7337):
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(hb))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True, name="control-room").start()
    return server
