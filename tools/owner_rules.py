"""owner_rules.py — the OWNER's hand on membrane.yaml. Lives outside the `sapling`
package on purpose: the AI's own process still has no write path to its rules
(the anti-tamper test asserts `sapling/` never imports this file).

Jaron 10/14 (2:59 PM): "Control room clicks can change the rule file (owner
action, logged; AI still has no write path)." A click on the owner's page
(local 127.0.0.1:7338, or the signed-in site page relayed from his phone)
calls this script; it flips ONE `enabled:` switch, keeps every comment, and
writes two logs: results/owner_actions.jsonl (plain file) and a ledger row of
kind `owner_action` in the AI's own hash-chained record, so the change sits
next to the ticks it affected.

Usage:
    python tools/owner_rules.py set <rule_id> on|off --by <who> [--why <text>]
    python tools/owner_rules.py show
"""

from __future__ import annotations

import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMBRANE = os.path.join(ROOT, "membrane.yaml")
LOG = os.path.join(ROOT, "results", "owner_actions.jsonl")
sys.path.insert(0, ROOT)


def set_enabled(rule_id: str, enabled: bool, by: str, why: str = "", path: str = MEMBRANE,
                log: str = LOG, store=None) -> dict:
    text = open(path, encoding="utf-8").read()
    # find the rule block that starts with "- id: <rule_id>" and its first "enabled:" line
    m = re.search(rf"^(\s*)- id:\s*{re.escape(rule_id)}\s*$", text, re.M)
    if not m:
        raise KeyError(f"no rule with id '{rule_id}'")
    indent = m.group(1)
    block_end = re.search(rf"^{indent}- id:|^\S", text[m.end():], re.M)
    end = m.end() + (block_end.start() if block_end else len(text) - m.end())
    block = text[m.end():end]
    em = re.search(r"^(\s*enabled:\s*)(true|false)(.*)$", block, re.M)
    if not em:
        raise KeyError(f"rule '{rule_id}' has no enabled: line")
    before = em.group(2) == "true"
    new_block = block[:em.start(2)] + ("true" if enabled else "false") + block[em.end(2):]
    out = text[:m.end()] + new_block + text[end:]
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(out)
    os.replace(tmp, path)
    row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "action": "rule_set", "rule": rule_id,
           "before": before, "after": enabled, "by": by, "why": why, "via": "tools/owner_rules.py"}
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    try:
        if store is None:
            from sapling.memory import MemoryStore
            store = MemoryStore()
        store.meter("owner_action", row)
    except Exception as e:   # the plain-file log already has it
        row["ledger_error"] = str(e)
    return row


def log_owner_action(row: dict, log: str = LOG, store=None) -> dict:
    """Other owner actions (parole restart, stop, restore) use the same two logs."""
    row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **row, "via": row.get("via", "tools/owner_relay.py")}
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    try:
        if store is None:
            from sapling.memory import MemoryStore
            store = MemoryStore()
        store.meter("owner_action", row)
    except Exception as e:
        row["ledger_error"] = str(e)
    return row


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] == "show":
        from sapling import membrane
        for r in membrane.rules():
            print(f"{'ON ' if r.get('enabled', True) else 'OFF'} {r['id']:<24} origin={r.get('origin')}  {r['text'][:70]}")
        return
    if argv[0] == "set":
        rule, val = argv[1], argv[2].lower() in ("on", "true", "1", "yes")
        by = argv[argv.index("--by") + 1] if "--by" in argv else "owner"
        why = argv[argv.index("--why") + 1] if "--why" in argv else ""
        print(json.dumps(set_enabled(rule, val, by, why)))
        return
    print(__doc__)


if __name__ == "__main__":
    main()
