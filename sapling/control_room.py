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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import membrane
from .exams import REPORT_CARD
from .heartbeat import STOP_DIR

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sapling control room</title>
<style>
:root{--bg:#f6f1e7;--card:#fffdf8;--ink:#2b2520;--mut:#7a6f64;--line:#e3d9c8;--ok:#2e7d4f;--bad:#b23a3a;--acc:#b8732a}
@media (prefers-color-scheme:dark){:root{--bg:#1d1a16;--card:#2a2520;--ink:#efe7da;--mut:#a89c8c;--line:#3d362e;--acc:#e0a35a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;padding:12px 16px 40px}
h1{font-size:22px;margin:4px 0 12px}h2{font-size:17px;margin:0 0 8px}
.grid{display:grid;grid-template-columns:1fr;gap:12px}@media(min-width:900px){.grid{grid-template-columns:1fr 1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;min-width:0;overflow:hidden;overflow-wrap:anywhere}
.tbl{overflow-x:auto}
.kv{display:grid;grid-template-columns:auto 1fr;gap:4px 12px;font-size:15px}.kv b{color:var(--mut);font-weight:500}
.meter{height:14px;background:var(--line);border-radius:7px;overflow:hidden;margin:6px 0}.meter i{display:block;height:100%;background:var(--acc)}
.rule{border-top:1px solid var(--line);padding:8px 0}.rule:first-of-type{border-top:0}
.pill{display:inline-block;font-size:12px;padding:1px 8px;border-radius:9px;border:1px solid var(--line);color:var(--mut);margin-right:6px}
.on{color:var(--ok);border-color:var(--ok)}.off{color:var(--bad);border-color:var(--bad)}
pre{white-space:pre-wrap;word-break:break-word;font-size:13px;margin:0;max-height:320px;overflow:auto;background:var(--bg);padding:8px;border-radius:8px}
.chat{display:flex;gap:8px}.chat input{flex:1;font:inherit;padding:12px;border:1px solid var(--line);border-radius:10px;background:var(--bg);color:var(--ink)}
button{font:inherit;padding:12px 16px;border-radius:10px;border:0;background:var(--acc);color:#fff;min-height:44px}
#log div{padding:6px 0;border-top:1px solid var(--line)}#log .me{color:var(--mut)}
table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;padding:4px 6px;border-top:1px solid var(--line)}td:first-child,td:nth-child(2){white-space:nowrap}
.PASS{color:var(--ok)}.FAIL{color:var(--bad)}.INFO{color:var(--mut)}
small{color:var(--mut)}
</style></head><body>
<h1>Sapling control room <small id="alive"></small></h1>
<div class="grid">
<div class="card"><h2>Talk to it</h2>
<div id="log" style="max-height:260px;overflow:auto;margin-bottom:8px"></div>
<form class="chat" onsubmit="return send()"><input id="msg" placeholder="ask, or /teach the word glorp means happy" autocomplete="off"><button>Send</button></form>
<small>/teach &lt;fact&gt; &middot; /confirm &lt;id&gt; &middot; /invalidate &lt;id&gt; &middot; /keep &lt;id&gt; &middot; /report</small></div>
<div class="card"><h2>Heartbeat</h2><div class="kv" id="hb"></div>
<h2 style="margin-top:12px">Energy today</h2><div class="meter"><i id="bar" style="width:0"></i></div><div class="kv" id="budget"></div></div>
<div class="card"><h2>Rules <small>(from membrane.yaml; only the owner's file edit changes them)</small></h2><div id="rules"></div>
<button onclick="fetch('/api/open_rules',{method:'POST'})" style="margin-top:8px">Open rule file</button></div>
<div class="card"><h2>Goals it set itself</h2><div id="goals"></div></div>
<div class="card"><h2>Exams <small>(outside checks; E5 = facts in weights, info only)</small></h2><div id="exams"></div></div>
<div class="card"><h2>Tick log <small>(newest first)</small></h2><pre id="ticks"></pre></div>
<div class="card"><h2>Stop reports</h2><pre id="stops"></pre></div>
<div class="card"><h2>Memory <small id="memsize"></small></h2><pre id="mem"></pre></div>
</div>
<script>
const $=id=>document.getElementById(id);const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function fmt(s){if(s==null)return'never';s=Math.round(s);if(s<90)return s+' s';if(s<5400)return Math.round(s/60)+' min';if(s<172800)return (s/3600).toFixed(1)+' h';return (s/86400).toFixed(1)+' days'}
async function refresh(){try{const r=await fetch('/api/state');const d=await r.json();
$('alive').textContent=d.running?'beating':'stopped';
const c=d.clock;$('hb').innerHTML=`<b>tick</b><span>${d.tick} (last: ${esc(d.last_action||'-')})</span><b>born</b><span>${esc(c.born_iso)}</span><b>age</b><span>${fmt(c.age_s)}</span><b>this run</b><span>${fmt(c.uptime_s)}</span><b>since last input</b><span>${fmt(c.since_last_input_s)}</span><b>generation</b><span>${d.generation} (consolidations)</span><b>memories</b><span>${d.memories}</span><b>vocab</b><span>${d.vocab_size} word parts</span>`;
$('bar').style.width=(100*c.budget_left_frac).toFixed(1)+'%';
$('budget').innerHTML=`<b>left</b><span>${fmt(c.budget_left_s)} of ${fmt(c.budget_total_s)} (${(100*c.budget_left_frac).toFixed(1)}%)</span><b>spent</b><span>${c.budget_spent_s.toFixed(1)} compute-s</span><b>kWh today</b><span>${c.kwh_today.toExponential(2)} (est.)</span><b>kWh total</b><span>${c.kwh_total.toExponential(2)} (est.)</span><b>GPU now</b><span>${c.gpu_watts_now==null?'n/a':c.gpu_watts_now+' W'}</span><b>day resets in</b><span>${fmt(c.day_ends_in_s)}</span>`;
$('rules').innerHTML=d.rules.map(r=>`<div class="rule"><span class="pill ${r.enabled?'on':'off'}">${r.enabled?'ON':'OFF'}</span><span class="pill">${esc(r.kind)}</span><span class="pill">origin: ${esc(r.origin)}</span><br>${esc(r.text)}<br><small>${esc(r.decided||'')}</small></div>`).join('')+`<div class="rule"><small>settings: ${esc(JSON.stringify(d.settings))}</small></div>`;
$('goals').innerHTML=d.goals.length?'<div class="tbl"><table><tr><th>part</th><th>status</th><th>recent</th><th>target</th></tr>'+d.goals.map(g=>`<tr><td>${esc(g.part)}</td><td class="${g.status=='open'?'FAIL':'PASS'}">${g.status}</td><td>${g.misses_recent} misses / ${g.tries_recent}</td><td>${esc(g.target)}</td></tr>`).join('')+'</table></div>':'<small>none yet (it needs a few practice ticks)</small>';
$('exams').innerHTML='<div class="tbl"><table><tr><th>when</th><th>trigger</th>'+['E1','E2','E3','E4','E5'].map(e=>'<th>'+e+'</th>').join('')+'</tr>'+d.exams.map(x=>`<tr><td>${esc((x.ts||'').slice(5,16))}</td><td>${esc(x.trigger||'')}</td>`+['E1_composition','E2_affix_recall','E3_recall_echo','E4_deferral','E5_facts_in_weights'].map(n=>{const e=(x.exams||[]).find(q=>q.exam==n);return e?`<td class="${e.verdict}">${e.score}</td>`:'<td>-</td>'}).join('')+'</tr>').join('')+'</table></div>';
$('ticks').textContent=d.ticks.map(t=>`#${t.tick} ${t.action.padEnd(11)} ${JSON.stringify(t.result).slice(0,120)}  drives=${JSON.stringify(t.drives)}`).join('\n');
$('stops').textContent=d.stops.length?d.stops.map(s=>`${s.ts} by ${s.who} at tick ${s.tick}: ${s.reason}`).join('\n'):'none';
$('memsize').textContent=d.memories+' rows';$('mem').textContent=d.memory.map(m=>`#${m.id} [${m.tier}] recalls=${m.recalls} strength=${m.strength} (${m.provenance}/${m.state}) ${m.content}`).join('\n');
}catch(e){$('alive').textContent='no connection'}}
async function send(){const m=$('msg').value.trim();if(!m)return false;$('msg').value='';$('log').insertAdjacentHTML('afterbegin',`<div class="me">&gt; ${esc(m)}</div>`);
const r=await fetch('/api/chat',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({text:m})});const d=await r.json();
$('log').insertAdjacentHTML('afterbegin',`<div><pre>${esc(d.out)}</pre></div>`);refresh();return false}
refresh();setInterval(refresh,4000);
</script></body></html>"""


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
                stops.append({k: r.get(k) for k in ("ts", "who", "reason", "tick")})
            except Exception:
                pass
    ticks = hb.store.ledger_rows("tick", limit=40)
    return {
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
                body = PAGE.encode()
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
