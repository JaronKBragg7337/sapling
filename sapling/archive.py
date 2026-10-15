"""Archive — checkpoints leave the MSI, never history.

Jaron 10/14 (2:59 PM, disk): "never delete history forever. Keep the last 10
checkpoints + 1 per day on the MSI; move every other checkpoint OFF the MSI
to cloud storage where it can still be listed, viewed and restored."

Where (Jaron 10/15 6:06 PM): **GitHub Releases on JaronKBragg7337/sapling**, one release per day
(`ckpt-YYYY-MM-DD`, rolling to `-b`, `-c` ... at 1,000 assets), uploaded with the gh CLI that is already
signed in on the MSI. "Nothing deleted forever" = a VERIFIED copy in a release (asset size read back with
`gh release view --json assets` must match) before the local file goes. Before 10/15 the target was Supabase
Storage (`sapling-archive`); its 50 MB per-object limit rejected every checkpoint from v3.1 on (17.6M params
= 71 MB), which is why 100+ generations had piled up on the MSI. Gens already in Supabase stay there and still
restore. The manifest (one row per checkpoint: gen, where, size, exams, why) still lives in the Supabase table
`sapling_archive` (when the key is there) and always in `results/archive_manifest.json`.

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
GH_REPO = "JaronKBragg7337/sapling"
GH_MAX_ASSETS = 1000    # GitHub's per-release asset limit
GH_MAX_BYTES = 2 * 1024 ** 3
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


# -- GitHub Releases (the archive from 10/15) ------------------------------------------------
def _gh(args: list[str], timeout: int = 1800) -> tuple[int, str]:
    import subprocess
    p = subprocess.run(["gh"] + args + ["-R", GH_REPO], capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def gh_assets(tag: str) -> dict | None:
    """name -> size of a release's assets, or None if the release does not exist."""
    code, out = _gh(["release", "view", tag, "--json", "assets"], timeout=120)
    if code != 0:
        return None
    try:
        return {a["name"]: int(a["size"]) for a in json.loads(out[out.index("{"):]).get("assets", [])}
    except Exception:
        return None


def gh_release_for(day: str, create: bool = True) -> tuple[str, dict]:
    """The day's release with room left (ckpt-DAY, then ckpt-DAY-b, -c, ...); created if missing."""
    for suffix in [""] + [f"-{chr(c)}" for c in range(ord("b"), ord("z") + 1)]:
        tag = f"ckpt-{day}{suffix}"
        assets = gh_assets(tag)
        if assets is None:
            if not create:
                return tag, {}
            code, out = _gh(["release", "create", tag, "--title", f"Sapling checkpoints {day}{suffix}", "--latest=false",
                             "--notes", f"Sapling's generation checkpoints from {day} (archive: nothing deleted forever - "
                                        f"each file here was verified by size before the MSI's copy was removed). "
                                        f"Restore one from the control room or `python -m sapling.archive restore <gen>`."],
                            timeout=120)
            if code != 0:
                raise RuntimeError(f"release create {tag} failed: {out[-300:]}")
            return tag, {}
        if len(assets) < GH_MAX_ASSETS:
            return tag, assets
    raise RuntimeError(f"no room in any release for {day}")


def gh_upload(local_path: str, day: str) -> str:
    """Upload one checkpoint to the day's release and verify its size there. -> 'gh:<tag>/<name>'."""
    name, size = os.path.basename(local_path), os.path.getsize(local_path)
    if size >= GH_MAX_BYTES:
        raise RuntimeError(f"{name} is {size} bytes - over GitHub's 2 GB asset limit")
    tag, assets = gh_release_for(day)
    if assets.get(name) != size:
        code, out = _gh(["release", "upload", tag, local_path, "--clobber"])
        if code != 0:
            raise RuntimeError(f"upload {name} -> {tag} failed: {out[-300:]}")
        got = (gh_assets(tag) or {}).get(name)
        if got != size:
            raise RuntimeError(f"upload {name} -> {tag} not verified (release says {got}, local {size})")
    return f"gh:{tag}/{name}"


def gh_download(path: str, dest_dir: str = CHECKPOINT_DIR) -> str:
    tag, name = path[3:].split("/", 1)
    os.makedirs(dest_dir, exist_ok=True)
    code, out = _gh(["release", "download", tag, "-p", name, "-D", dest_dir, "--clobber"])
    if code != 0:
        raise RuntimeError(f"download {path} failed: {out[-300:]}")
    return os.path.join(dest_dir, name)


def _manifest_merge(rows: list[dict]) -> list[dict]:
    """Local manifest = the source of truth for where each gen lives (cloud table mirrors it when reachable)."""
    by = {r["gen"]: r for r in manifest_local()}
    for r in rows:
        by[r["gen"]] = {**by.get(r["gen"], {}), **r}
    out = [by[g] for g in sorted(by)]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(MANIFEST_LOCAL, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=str)
    return out


# -- cloud side (Supabase: manifest table; the pre-10/15 storage bucket) ----------------------
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
    """Move every checkpoint the policy says to move to the day's GitHub release. The local file is removed ONLY
    after the release reports the same size. The manifest row goes to results/archive_manifest.json and, when the
    Supabase key is there, to the sapling_archive table. Logged as one `archive` ledger row."""
    t0 = time.time()
    c = creds()
    local = list_local()
    p = plan(local, keep_last)
    row = {"keep": p["keep"], "planned": [m["gen"] for m in p["move"]], "moved": [], "errors": [],
           "dry": dry, "target": f"github releases {GH_REPO}", "cloud_manifest": bool(c),
           "local_mb_before": round(sum(x["size"] for x in local) / 1e6, 1), "releases": []}
    if not dry:
        for m in p["move"]:
            try:
                where = gh_upload(m["path"], m["day"])
                mrow = {"gen": m["gen"], "path": where, "size_bytes": m["size"],
                        "checkpoint_time": datetime.fromtimestamp(m["mtime"]).isoformat(),
                        "exams": exams_for(store, m["gen"]) if store else None,
                        "why": p["why"][m["gen"]], "local_present": False,
                        "archived_at": datetime.now().astimezone().isoformat()}
                _manifest_merge([mrow])
                if c:
                    try:
                        manifest_upsert(c, mrow)
                    except Exception as e:
                        row["errors"].append(f"gen {m['gen']} cloud manifest (file is safe in the release): {e}")
                os.remove(m["path"])            # the release copy was verified above
                row["moved"].append(m["gen"])
                tag = where[3:].split("/")[0]
                if tag not in row["releases"]:
                    row["releases"].append(tag)
                if verbose:
                    print(f"[archive] gen {m['gen']} -> {where} ({m['size'] / 1e6:.1f} MB)", flush=True)
            except Exception as e:
                row["errors"].append(f"gen {m['gen']}: {e}")
        local_gens = {x["gen"] for x in list_local()}
        rows = _manifest_merge([{"gen": r["gen"], "local_present": r["gen"] in local_gens} for r in manifest_local()])
        if c:
            try:
                _write_manifest_copies(c, rows)
            except Exception as e:
                row["errors"].append(f"cloud manifest copy: {e}")
        row["archived_total"] = len(rows)
        row["archived_mb"] = round(sum(int(r.get("size_bytes") or 0) for r in rows) / 1e6, 1)
    row["local_mb_after"] = round(sum(x["size"] for x in list_local()) / 1e6, 1)
    row["other_files"] = [f for f in big_files() if f["big"]]
    row["seconds"] = round(time.time() - t0, 1)
    if store is not None:
        store.meter("archive", row)
    return row


def restore(gen: int, store=None, who: str = "owner") -> dict:
    """Bring an archived gen back into results/checkpoints/ - from its GitHub release (10/15 on) or Supabase (before)."""
    c = creds()
    rows = {r["gen"]: r for r in manifest_local()}
    if gen not in rows and c:
        try:
            rows.update({r["gen"]: r for r in manifest_cloud(c)})
        except Exception:
            pass
    path = (rows.get(gen) or {}).get("path") or f"checkpoints/gen_{gen:04d}.pt"
    if path.startswith("gh:"):
        dest = gh_download(path)
    else:
        if not c:
            raise RuntimeError("no Supabase credentials (names: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)")
        dest = download(c, gen)
    _manifest_merge([{"gen": gen, "path": path, "local_present": True}])
    if c:
        try:
            manifest_upsert(c, {"gen": gen, "path": path, "local_present": True})
        except Exception:
            pass
    row = {"gen": gen, "dest": dest, "size": os.path.getsize(dest), "who": who, "from": path}
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
