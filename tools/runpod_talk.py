"""Run one bounded Sapling v5 conversation lesson on a rented H100 and leave the staged core local.

  python tools/runpod_talk.py --dry-run
  python tools/runpod_talk.py
  python tools/runpod_talk.py --reap
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import runpod_run as R

HARD_CAP = 10.0
WORKING_CAP = 9.50
NAME_PREFIX = "sapling-talk-"
REAPER_TASK = "Sapling Talk Cloud Reaper"
SOURCES = ("stories", "adult", "web")
REPLAY_WINDOWS = 384
SEED = 7337


def cloud_log(line: str) -> None:
    os.makedirs(os.path.dirname(R.CLOUD_LOG), exist_ok=True)
    with open(R.CLOUD_LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %I:%M %p").replace(" 0", " ") + " | Kimi | " + line + "\n")
        f.flush()
        os.fsync(f.fileno())


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_bundle(run: R.Run, stamp: str, windows: int = REPLAY_WINDOWS, core: str | None = None,
                 lessons: str | None = None) -> tuple[str, dict]:
    import numpy as np
    from sapling.reading import rehearsal_sentences

    bundle = os.path.join(run.dir, "bundle")
    os.makedirs(bundle, exist_ok=True)
    core_src = core or os.path.join(ROOT, "results", "core_checkpoint.pt")
    lessons_src = lessons or os.path.join(ROOT, "data", "talk", "lessons.jsonl")
    shard_dir = os.path.join(ROOT, "data", "reading", "shards")
    for path in (core_src, lessons_src, os.path.join(shard_dir, "index.json")):
        if not os.path.isfile(path):
            raise RuntimeError(f"required live input is missing: {path}")
    shutil.copyfile(core_src, os.path.join(bundle, "core.pt"))
    shutil.copyfile(lessons_src, os.path.join(bundle, "lessons.jsonl"))

    school = rehearsal_sentences(SEED, None)
    if not school:
        raise RuntimeError("school rehearsal is empty")
    with open(os.path.join(bundle, "school.json"), "w", encoding="utf-8") as f:
        json.dump(school, f, ensure_ascii=False)

    old_index = json.load(open(os.path.join(shard_dir, "index.json"), encoding="utf-8"))
    index = {k: old_index.get(k) for k in ("vocab_size", "vocab_sha")}
    index.update({"ts": stamp, "shard_tokens": windows * 513, "replay_windows_per_source": windows,
                  "seed": SEED, "sources": {}})
    rng = random.Random(SEED)
    for source in SOURCES:
        shards = old_index.get("sources", {}).get(source, {}).get("shards") or []
        if not shards:
            raise RuntimeError(f"reading source has no shards: {source}")
        arrays = {}
        rows = []
        weights = [int(s.get("tokens") or 1) for s in shards]
        for shard in rng.choices(shards, weights=weights, k=windows):
            name = shard["file"]
            if name not in arrays:
                arrays[name] = np.load(os.path.join(shard_dir, name), mmap_mode="r")
            arr = arrays[name]
            if len(arr) < 513:
                raise RuntimeError(f"reading shard is too short: {name}")
            offset = rng.randrange(0, len(arr) - 512)
            rows.append(np.asarray(arr[offset:offset + 513]))
        name = f"replay_{source}.npy"
        replay = np.stack(rows).astype(np.uint16, copy=False).reshape(-1)
        np.save(os.path.join(bundle, name), replay)
        tokens = int(replay.size)
        index["sources"][source] = {"shards": [{"file": name, "tokens": tokens}], "tokens": tokens,
                                    "docs": 0, "windows": windows}
    with open(os.path.join(bundle, "reading_index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=1)

    names = ["core.pt", "lessons.jsonl", "school.json", "reading_index.json"] + [f"replay_{s}.npy" for s in SOURCES]
    files = {name: os.path.getsize(os.path.join(bundle, name)) for name in names}
    hashes = {name: _sha256(os.path.join(bundle, name)) for name in names}
    info = {"built": stamp, "core_file": "core.pt", "files": files, "sha256": hashes,
            "school_sentences": len(school), "replay_windows_per_source": windows,
            "note": "Sapling v5 conversation inputs: live core, current lessons, reading replay and school rehearsal."}
    with open(os.path.join(bundle, "bundle.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    run.log(step="bundle built", core_bytes=files["core.pt"], school_sentences=len(school),
            replay_windows_per_source=windows, mb=round(sum(files.values()) / 1e6, 1))
    return bundle, info


def _assets(tag: str) -> dict[str, int]:
    code, out = R.gh(["release", "view", tag, "--json", "assets"])
    if code:
        raise RuntimeError(f"release view {tag}: {out[-400:]}")
    return {a["name"]: int(a["size"]) for a in json.loads(out[out.index("{"):])["assets"]}


def upload_bundle(run: R.Run, stamp: str, bundle: str, info: dict) -> tuple[str, str]:
    tag = "talk-" + stamp
    code, out = R.gh(["release", "create", tag, "--title", f"Sapling v5 conversation inputs {stamp}",
                      "--latest=false", "--notes",
                      "Inputs for one bounded Sapling v5 conversation run. FineWeb-Edu (ODC-By 1.0), Simple English "
                      "Wikipedia (CC BY-SA 3.0), Project Gutenberg (public domain), TinyStories (CDLA-Sharing-1.0), "
                      "oasst1 (Apache-2.0), dolly-15k (CC BY-SA 3.0), and owner-written lessons."])
    if code:
        raise RuntimeError(f"release create {tag}: {out[-400:]}")
    paths = [os.path.join(bundle, name) for name in info["files"]]
    paths += [os.path.join(bundle, "bundle.json"), os.path.join(ROOT, "tools", "cloud_talk.py")]
    expected = {}
    for path in paths:
        name = os.path.basename(path)
        expected[name] = os.path.getsize(path)
        t0 = time.time()
        code, out = R.gh(["release", "upload", tag, path, "--clobber"])
        if code:
            raise RuntimeError(f"upload {name}: {out[-400:]}")
        run.log(step="uploaded", file=name, mb=round(expected[name] / 1e6, 1),
                seconds=round(time.time() - t0, 1))
    got = _assets(tag)
    for name, size in expected.items():
        if got.get(name) != size:
            raise RuntimeError(f"release asset not verified: {name} ({got.get(name)} != {size})")
    run.log(step="upload verified", release=tag, files=len(expected))
    return tag, f"https://github.com/{R.REPO}/releases/download/{tag}"


def start_cmd(base: str, minutes: float, max_pod_minutes: float, max_steps: int, epochs: float, micro: int) -> list[str]:
    script = f"""set -x
mkdir -p /workspace/out
echo '{{"state":"booting"}}' > /workspace/out/status.json
( sleep {int(max_pod_minutes * 60)}; echo "watchdog $(date)" >> /workspace/out/watchdog.txt;
  runpodctl remove pod "$RUNPOD_POD_ID" || runpodctl pod delete "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID" ) &
cd /workspace/out && nohup python -m http.server 8000 > /dev/null 2>&1 &
cd /workspace && git clone --depth 1 https://github.com/{R.REPO} && cd sapling && pip install -q numpy pyyaml
python -c "import urllib.request; urllib.request.urlretrieve('{base}/cloud_talk.py', 'tools/cloud_talk.py')"
python tools/cloud_talk.py --base {shlex.quote(base)} --out /workspace/out --minutes {minutes:g} --epochs {epochs:g} --max-steps {max_steps:d} --micro {micro:d} > /workspace/out/run.txt 2>&1
sleep infinity
"""
    return ["bash", "-c", script]


def create_pod(run: R.Run, stamp: str, base: str, minutes: float, max_pod_minutes: float,
               max_steps: int, epochs: float, micro: int, secure_price: float) -> dict:
    last = None
    for cloud in ("COMMUNITY", "SECURE"):
        command = start_cmd(base, minutes, max_pod_minutes, max_steps, epochs, micro)
        body = R.pod_create_body(NAME_PREFIX + stamp, cloud, 40, command)
        code, response = R._http("POST", f"{R.REST}/pods", body, timeout=120)
        if code in (200, 201) and isinstance(response, dict) and response.get("id"):
            pod = {"id": response["id"], "cloud": cloud,
                   "cost_per_hr": float(response.get("cost") or secure_price)}
            run.log(step="pod created", cloud=cloud, pod=pod["id"], cost_per_hr=pod["cost_per_hr"])
            return pod
        last = f"{cloud}: {code} {str(response)[:300]}"
        run.log(step="pod not created", cloud=cloud, why=last)
    raise RuntimeError(f"no H100 pod: {last}")


def set_reaper(minutes: float) -> None:
    when = time.localtime(time.time() + minutes * 60)
    cmd = f'"{sys.executable}" "{os.path.join(ROOT, "tools", "runpod_talk.py")}" --reap'
    p = subprocess.run(["schtasks", "/Create", "/F", "/TN", REAPER_TASK, "/SC", "ONCE", "/ST", time.strftime("%H:%M", when),
                        "/SD", time.strftime("%m/%d/%Y", when), "/TR", cmd], capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(f"could not set Windows reaper: {(p.stdout or p.stderr)[-400:]}")


def clear_reaper() -> None:
    subprocess.run(["schtasks", "/Delete", "/F", "/TN", REAPER_TASK], capture_output=True, text=True)


def terminate(run: R.Run, pod: dict, started: float, purpose: str) -> tuple[float, list[dict]]:
    ok = False
    error = None
    try:
        ok = R.delete_pod(pod["id"])
    except Exception as e:
        error = str(e)[:300]
    hours = (time.time() - started) / 3600
    cost = hours * pod["cost_per_hr"]
    try:
        left = R.list_pods()
    except Exception as e:
        left = []
        error = (error or "") + f"; final pod listing failed: {str(e)[:200]}"
    names = [p.get("name") for p in left]
    cloud_log(f"TERMINATE | {pod['id']} | H100 SXM | {purpose} | deleted={ok}, "
              f"{hours * 60:.1f} min x ${pod['cost_per_hr']:.2f}/hr = ${cost:.2f}; pods left: {names or 'none'}"
              + (f"; error={error}" if error else ""))
    run.log(step="pod terminated", deleted=ok, minutes=round(hours * 60, 1), cost=round(cost, 4),
            pods_left=names, error=error)
    if ok and not any(str(p.get("name", "")).startswith(NAME_PREFIX) for p in left):
        clear_reaper()
    if not ok:
        raise RuntimeError(f"pod {pod['id']} did not terminate: {error or 'delete not verified'}")
    return cost, left


def reap() -> tuple[list[str], list[dict]]:
    gone = []
    for pod in R.list_pods():
        if str(pod.get("name", "")).startswith(NAME_PREFIX):
            ok = R.delete_pod(pod["id"])
            cloud_log(f"TERMINATE | {pod['id']} | H100 SXM | conversation reaper | deleted={ok}")
            if not ok:
                raise RuntimeError(f"reaper could not verify deletion of {pod['id']}")
            gone.append(pod["id"])
    clear_reaper()
    return gone, R.list_pods()


def _save_summary(run: R.Run, summary: dict) -> None:
    summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(os.path.join(run.dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=str)


def main(argv=None) -> dict | None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reap", action="store_true")
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--minutes", type=float, default=25, help="hard training wall-clock limit on the pod")
    ap.add_argument("--max-pod-minutes", type=float, default=45, help="in-pod self-deletion deadline")
    ap.add_argument("--micro", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--core", default=None)
    ap.add_argument("--lessons", default=None)
    a = ap.parse_args(argv)
    if a.reap:
        gone, left = reap()
        print(json.dumps({"reaped": gone, "pods_left": [p.get("name") for p in left]}))
        return None
    if min(a.max_steps, a.micro) <= 0 or min(a.minutes, a.max_pod_minutes, a.epochs) <= 0:
        ap.error("steps, micro, epochs and minute limits must be positive")
    if a.max_pod_minutes <= a.minutes:
        ap.error("--max-pod-minutes must exceed --minutes to leave boot/download/fetch time")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run = R.Run(stamp, a.dry_run)
    run.log(step="start", dry=a.dry_run, plan="v5 conversation", minutes=a.minutes,
            max_pod_minutes=a.max_pod_minutes, max_steps=a.max_steps, epochs=a.epochs, micro=a.micro,
            hard_cap=HARD_CAP, working_cap=WORKING_CAP, core=a.core, lessons=a.lessons)
    bundle, info = build_bundle(run, stamp, core=a.core, lessons=a.lessons)
    if a.dry_run:
        cmd = start_cmd("<release-url>", a.minutes, a.max_pod_minutes, a.max_steps, a.epochs, a.micro)
        run.log(step="dry run ends here - no upload, price query, pod, local GPU training or swap", pod_start_cmd=cmd[2])
        summary = {"stamp": stamp, "dry_run": True, "release": None, "max_steps": a.max_steps,
                   "minutes": a.minutes, "max_pod_minutes": a.max_pod_minutes, "micro": a.micro,
                   "working_cap": WORKING_CAP, "pods_left": [], "run_dir": os.path.relpath(run.dir, ROOT)}
        _save_summary(run, summary)
        print(json.dumps(summary), flush=True)
        return summary

    summary = {"stamp": stamp, "dry_run": False, "release": None, "cost": 0.0, "pods_left": None,
               "run_dir": os.path.relpath(run.dir, ROOT)}
    pod = None
    started = None
    status = {}
    try:
        quote = R.price_quote()
        secure_price = float(quote.get("securePrice") or 0)
        if secure_price <= 0:
            raise RuntimeError(f"no usable secure H100 price; refusing to create a pod: {quote}")
        worst_case = secure_price * a.max_pod_minutes / 60
        run.log(step="price and cap", quote=quote, secure_worst_case=round(worst_case, 3), working_cap=WORKING_CAP)
        summary.update(secure_price=secure_price, secure_worst_case=worst_case)
        if worst_case > WORKING_CAP:
            raise RuntimeError(f"secure-price worst case ${worst_case:.2f} exceeds ${WORKING_CAP:.2f} working cap")
        existing = R.list_pods()
        if existing:
            summary["pods_left"] = [p.get("name") for p in existing]
            raise RuntimeError(f"a pod already exists; refusing to add cost: {[p.get('name') for p in existing]}")
        tag, base = upload_bundle(run, stamp, bundle, info)
        summary["release"] = tag
        existing = R.list_pods()
        if existing:
            summary["pods_left"] = [p.get("name") for p in existing]
            raise RuntimeError("a pod appeared before creation; refusing to add cost")
        try:
            pod = create_pod(run, stamp, base, a.minutes, a.max_pod_minutes, a.max_steps, a.epochs, a.micro, secure_price)
            started = time.time()
            cloud_log(f"START | {pod['id']} | H100 SXM ({pod['cloud'].lower()}) | v5 conversation {stamp}; "
                      f"training {a.minutes:g} min / {a.max_steps} steps; watchdog {a.max_pod_minutes:g} min; "
                      f"Windows reaper | ${pod['cost_per_hr']:.2f}/hr")
            try:
                set_reaper(a.max_pod_minutes + 10)
                run.log(step="Windows reaper set", task=REAPER_TASK)
            except Exception as e:
                run.log(step="Windows reaper failed", error=str(e)[:400])
                cloud_log(f"REAPER | {pod['id']} | H100 SXM | v5 conversation {stamp} | setup failed: {str(e)[:200]}")
                raise
            status = R.watch(run, pod, started + a.max_pod_minutes * 60 - 120)
            R.fetch(run, pod, names=("core_staged.pt", "log.jsonl", "status.json", "run.txt"))
        finally:
            if pod:
                summary["cost"], left = terminate(run, pod, started, f"v5 conversation {stamp}")
                summary["pods_left"] = [p.get("name") for p in left]
        left = R.list_pods()
        summary["pods_left"] = [p.get("name") for p in left]
        if left:
            raise RuntimeError(f"final pod list is not empty: {summary['pods_left']}")
        staged = os.path.join(run.dir, "core_staged.pt")
        summary.update(status=status, core_ready=bool(status.get("state") == "done" and os.path.getsize(staged) > 1_000_000)
                       if os.path.exists(staged) else False,
                       staged=os.path.relpath(staged, ROOT) if os.path.exists(staged) else None)
        if not summary["core_ready"]:
            raise RuntimeError(f"conversation run did not return a complete staged core: {status}")
        run.log(step="finished", **summary)
        print(json.dumps(summary, default=str), flush=True)
        return summary
    finally:
        _save_summary(run, summary)


if __name__ == "__main__":
    main()
