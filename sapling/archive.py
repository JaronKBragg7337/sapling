"""Archive — checkpoints leave the MSI, never history.

Jaron 10/14 (2:59 PM, disk): "never delete history forever. Keep the last 10
checkpoints + 1 per day on the MSI; move every other checkpoint OFF the MSI
to cloud storage where it can still be listed, viewed and restored."

Where: Supabase Storage, private bucket `sapling-archive` (project
ygjpnvrwhkrowkrskftk, PRO plan). A manifest row per checkpoint lives in the
`sapling_archive` table (gen, checkpoint time, exams at that generation,
size, why) so the control room can list it; a copy of the manifest is also
written next to the files as `manifest.json` and cached locally in
`results/archive_manifest.json`.

This is a housekeeping step the heartbeat runs on a schedule (not a drive it
chooses - it is chores, like breathing). Every run is a ledger row of kind
`archive`. A restore downloads `gen_N.pt` back into results/checkpoints/.

Credentials: SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY from ~/.secrets/keys.env
(names only ever appear in logs). DESIGN-CHOICES.md D4 notes that this key is
wider than the job needs; a storage-only key is the follow-up.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime

from .train import CHECKPOINT_DIR, RESULTS_DIR

KEEP_LAST = 10          # Jaron: last 10 checkpoints stay on the MSI
BUCKET = "sapling-archive"
MANIFEST_LOCAL = os.path.join(RESULTS_DIR, "archive_manifest.json")
KEYS_ENV = os.path.join(os.path.expanduser("~"), ".secrets", "keys.env")
BIG_FILE_BYTES = 50 * 1024 * 1024   # report (never touch) other files above this


# -- credentials -----------------------------------------------------------------------
def creds() -> dict | None:
    env = {}
    if os.path.exists(KEYS_ENV):
        for line in open(KEYS_ENV, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    url = os.environ.get("SUPABASE_URL") or env.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or env.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        return None
    return {"url": url.rstrip("/"), "key": key}


def _req(c: dict, method: str, path: str, body: bytes | None = None, headers: dict | None = None,
         timeout: int = 120) -> tuple[int, bytes]:
    h = {"apikey": c["key"], "authorization": f"Bearer {c['key']}", **(headers or {})}
    r = urllib.request.Request(c["url"] + path, data=body, method=method, headers=h)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


# -- the policy ------------------------------------------------------------------------
def list_local() -> list[dict]:
    out = []
    if not os.path.isdir(CHECKPOINT_DIR):
        return out
    for name in sorted(os.listdir(CHECKPOINT_DIR)):
        m = re.fullmatch(r"gen_(\d+)\.pt", name)
        if not m:
            continue
        p = os.path.join(CHECKPOINT_DIR, name)
        st = os.stat(p)
        out.append({"gen": int(m.group(1)), "path": p, "size": st.st_size, "mtime": st.st_mtime,
                    "day": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d")})
    return sorted(out, key=lambda x: x["gen"])


def plan(local: list[dict] | None = None, keep_last: int = KEEP_LAST) -> dict:
    """Which checkpoints stay (last N + the last one of every day) and which move."""
    local = list_local() if local is None else local
    keep = {c["gen"] for c in local[-keep_last:]}
    last_of_day: dict[str, int] = {}
    for c in local:
        last_of_day[c["day"]] = c["gen"]          # sorted by gen -> last wins
    keep |= set(last_of_day.values())
    move = [c for c in local if c["gen"] not in keep]
    why = {}
    for c in move:
        why[c["gen"]] = f"older than the last {keep_last} and not the last checkpoint of {c['day']}"
    return {"keep": sorted(keep), "move": move, "why": why}


def exams_for(store, gen: int) -> list | None:
    for row in store.ledger_rows("consolidation", limit=5000):
        if int(row.get("generation", -1)) == gen:
            return row.get("exams")
    return None


# -- cloud side ------------------------------------------------------------------------
def upload(c: dict, local_path: str, gen: int) -> str:
    key = f"checkpoints/gen_{gen:04d}.pt"
    with open(local_path, "rb") as f:
        data = f.read()
    code, body = _req(c, "POST", f"/storage/v1/object/{BUCKET}/{key}", data,
                      {"content-type": "application/octet-stream", "x-upsert": "true"}, timeout=600)
    if code not in (200, 201):
        raise RuntimeError(f"upload gen {gen} failed: {code} {body[:200]!r}")
    # verify it is really there before anything local goes
    code, body = _req(c, "GET", f"/storage/v1/object/info/{BUCKET}/{key}")
    if code != 200:
        raise RuntimeError(f"upload gen {gen} not visible after upload: {code}")
    size = int((json.loads(body) or {}).get("size") or (json.loads(body).get("metadata") or {}).get("size") or 0)
    if size and size != len(data):
        raise RuntimeError(f"upload gen {gen} size mismatch {size} != {len(data)}")
    return key


def download(c: dict, gen: int, dest: str | None = None) -> str:
    key = f"checkpoints/gen_{gen:04d}.pt"
    code, body = _req(c, "GET", f"/storage/v1/object/{BUCKET}/{key}", timeout=600)
    if code != 200:
        raise RuntimeError(f"download gen {gen} failed: {code} {body[:200]!r}")
    dest = dest or os.path.join(CHECKPOINT_DIR, f"gen_{gen:04d}.pt")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(body)
    return dest


def manifest_upsert(c: dict, row: dict) -> None:
    body = json.dumps(row, default=str).encode()
    code, out = _req(c, "POST", "/rest/v1/sapling_archive", body,
                     {"content-type": "application/json", "prefer": "resolution=merge-duplicates,return=minimal"})
    if code not in (200, 201, 204):
        raise RuntimeError(f"manifest row gen {row.get('gen')} failed: {code} {out[:200]!r}")


def manifest_cloud(c: dict) -> list[dict]:
    code, body = _req(c, "GET", "/rest/v1/sapling_archive?select=*&order=gen.asc")
    if code != 200:
        raise RuntimeError(f"manifest read failed: {code} {body[:200]!r}")
    return json.loads(body)


def manifest_local() -> list[dict]:
    if os.path.exists(MANIFEST_LOCAL):
        try:
            return json.load(open(MANIFEST_LOCAL, encoding="utf-8"))
        except Exception:
            return []
    return []


def _write_manifest_copies(c: dict | None, rows: list[dict]) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(MANIFEST_LOCAL, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1, default=str)
    if c:
        _req(c, "POST", f"/storage/v1/object/{BUCKET}/manifest.json", json.dumps(rows, default=str).encode(),
             {"content-type": "application/json", "x-upsert": "true"})


def big_files() -> list[dict]:
    """Other growing files (ledger db, logs). Reported, never touched (Jaron: only if big)."""
    out = []
    for name in sorted(os.listdir(RESULTS_DIR)) if os.path.isdir(RESULTS_DIR) else []:
        p = os.path.join(RESULTS_DIR, name)
        if os.path.isfile(p):
            s = os.path.getsize(p)
            out.append({"file": name, "mb": round(s / 1e6, 2), "big": s >= BIG_FILE_BYTES})
    return out


# -- the step ----------------------------------------------------------------------------
def archive_step(store, keep_last: int = KEEP_LAST, dry: bool = False, verbose: bool = False) -> dict:
    """Move every checkpoint the policy says to move. Local file is removed ONLY after
    the cloud copy is verified. Logged as one `archive` ledger row."""
    t0 = time.time()
    c = creds()
    local = list_local()
    p = plan(local, keep_last)
    row = {"keep": p["keep"], "planned": [m["gen"] for m in p["move"]], "moved": [], "errors": [],
           "dry": dry, "cloud": bool(c), "local_mb_before": round(sum(x["size"] for x in local) / 1e6, 1)}
    if not c:
        row["errors"].append("no SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY in ~/.secrets/keys.env (names only)")
    elif not dry:
        existing = {}
        try:
            existing = {r["gen"]: r for r in manifest_cloud(c)}
        except Exception as e:
            row["errors"].append(f"manifest read: {e}")
        for m in p["move"]:
            try:
                key = upload(c, m["path"], m["gen"])
                manifest_upsert(c, {"gen": m["gen"], "path": key, "size_bytes": m["size"],
                                    "checkpoint_time": datetime.fromtimestamp(m["mtime"]).isoformat(),
                                    "exams": exams_for(store, m["gen"]) if store else None,
                                    "why": p["why"][m["gen"]], "local_present": False,
                                    "archived_at": datetime.now().astimezone().isoformat()})
                os.remove(m["path"])            # the cloud copy was verified above
                row["moved"].append(m["gen"])
                if verbose:
                    print(f"[archive] gen {m['gen']} -> {BUCKET}/{key} ({m['size']/1e6:.1f} MB)")
            except Exception as e:
                row["errors"].append(f"gen {m['gen']}: {e}")
        try:
            rows = manifest_cloud(c)
            # mark which archived gens are also present locally (after a restore)
            local_gens = {x["gen"] for x in list_local()}
            for r in rows:
                want = r["gen"] in local_gens
                if bool(r.get("local_present")) != want:
                    manifest_upsert(c, {"gen": r["gen"], "path": r["path"], "local_present": want})
                    r["local_present"] = want
            _write_manifest_copies(c, rows)
            row["archived_total"] = len(rows)
            row["archived_mb"] = round(sum(int(r.get("size_bytes") or 0) for r in rows) / 1e6, 1)
        except Exception as e:
            row["errors"].append(f"manifest write: {e}")
    row["local_mb_after"] = round(sum(x["size"] for x in list_local()) / 1e6, 1)
    row["other_files"] = [f for f in big_files() if f["big"]]
    row["seconds"] = round(time.time() - t0, 1)
    if store is not None:
        store.meter("archive", row)
    return row


def restore(gen: int, store=None, who: str = "owner") -> dict:
    c = creds()
    if not c:
        raise RuntimeError("no Supabase credentials (names: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)")
    dest = download(c, gen)
    try:
        manifest_upsert(c, {"gen": gen, "path": f"checkpoints/gen_{gen:04d}.pt", "local_present": True})
        _write_manifest_copies(c, manifest_cloud(c))
    except Exception:
        pass
    row = {"gen": gen, "dest": dest, "size": os.path.getsize(dest), "who": who}
    if store is not None:
        store.meter("archive_restore", row)
    return row


def main(argv=None):
    import sys
    from .memory import MemoryStore
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "plan"
    if cmd == "plan":
        p = plan()
        print(json.dumps({"keep": p["keep"], "move": [m["gen"] for m in p["move"]]}, indent=1))
        print("other files:", json.dumps(big_files()))
    elif cmd == "run":
        print(json.dumps(archive_step(MemoryStore(), verbose=True), indent=1, default=str))
    elif cmd == "restore":
        print(json.dumps(restore(int(argv[1]), MemoryStore()), indent=1))
    elif cmd == "manifest":
        c = creds()
        print(json.dumps(manifest_cloud(c) if c else manifest_local(), indent=1))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
