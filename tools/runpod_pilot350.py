"""Run the fresh ~350M, 800M-token Sapling pilot on one bounded H100.

  python tools/runpod_pilot350.py --dry-run
  python tools/runpod_pilot350.py
  python tools/runpod_pilot350.py --reap

The staged checkpoint is downloaded for local evaluation and is never installed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import runpod_run as R
import cloud_pilot350 as C

HARD_CAP = 9.0
WORKING_CAP = 8.75
TARGET_TOKENS = C.TARGET_TOKENS
NAME_PREFIX = "sapling-pilot350-"
REAPER_TASK = "Sapling Pilot350 Cloud Reaper"
BUNDLE = R.BUNDLE
FETCHES = ("core_staged.pt", "log.jsonl", "status.json", "benchmark.json", "run.txt")


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


def validate_bundle(bundle: str = BUNDLE) -> dict:
    meta = os.path.join(bundle, "bundle.json")
    if not os.path.isfile(meta):
        raise RuntimeError(f"clean bundle is missing: {meta}")
    with open(meta, encoding="utf-8") as f:
        info = json.load(f)
    if str(info.get("status", "")).upper().startswith("INCOMPLETE"):
        raise RuntimeError("clean bundle is incomplete; refusing the 350M pilot")
    core = info.get("core_file")
    if not core:
        raise RuntimeError("clean bundle has no current core/tokenizer")
    required = [core, *C.REQUIRED]
    missing = [name for name in required if name not in info.get("files", {})]
    if "reading.npy" not in info.get("files", {}) or not os.path.isfile(os.path.join(bundle, "reading.npy")):
        raise RuntimeError("clean bundle is missing required reading.npy; refusing the 350M pilot")
    if missing:
        raise RuntimeError(f"clean bundle is missing required assets: {missing}")
    if set(info["files"]) != set(required):
        raise RuntimeError("clean bundle contains unexpected assets; pilot accepts only current core/tokenizer, reading, lesson, talk, school and test")
    if info.get("urls"):
        raise RuntimeError("clean bundle must not reuse assets from another release")
    for name in required:
        if os.path.basename(name) != name:
            raise RuntimeError(f"unsafe clean bundle asset name: {name}")
        path = os.path.join(bundle, name)
        if not os.path.isfile(path):
            raise RuntimeError(f"clean bundle asset is missing: {name}")
        size = os.path.getsize(path)
        if size != int(info["files"][name]):
            raise RuntimeError(f"clean bundle asset size does not match manifest: {name}")
        expected_hash = info.get("sha256", {}).get(name)
        if expected_hash and _sha256(path) != expected_hash:
            raise RuntimeError(f"clean bundle asset hash does not match manifest: {name}")
    if os.path.getsize(os.path.join(bundle, "reading.npy")) <= (C.CONFIG["block_size"] + 1) * 2:
        raise RuntimeError("reading.npy is too short for the pilot")
    return info


def _assets(tag: str) -> dict[str, int]:
    code, out = R.gh(["release", "view", tag, "--json", "assets"])
    if code:
        raise RuntimeError(f"release view {tag}: {out[-400:]}")
    return {a["name"]: int(a["size"]) for a in json.loads(out[out.index("{"):])["assets"]}


def upload_bundle(run: R.Run, stamp: str, bundle: str, info: dict) -> tuple[str, str]:
    tag = "pilot350-" + stamp
    code, out = R.gh(["release", "create", tag, "--title", f"Sapling fresh 350M pilot inputs {stamp}",
                      "--latest=false", "--notes",
                      "Clean inputs for one bounded fresh 350M, 800M-token pilot. No old-release or quarantined assets. "
                      "The current checkpoint supplies only the tokenizer; training starts from fresh weights."])
    if code:
        raise RuntimeError(f"release create {tag}: {out[-400:]}")
    paths = [os.path.join(bundle, name) for name in info["files"]]
    paths += [os.path.join(bundle, "bundle.json"), os.path.join(ROOT, "tools", "cloud_pilot350.py")]
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


def start_cmd(base: str, minutes: float, max_pod_minutes: float, micro: int, compile_model: bool = True) -> list[str]:
    compile_arg = "" if compile_model else " --no-compile"
    script = f"""set -x
mkdir -p /workspace/out
echo '{{"state":"booting"}}' > /workspace/out/status.json
( sleep {int(max_pod_minutes * 60)}; echo "watchdog $(date)" >> /workspace/out/watchdog.txt;
  runpodctl remove pod "$RUNPOD_POD_ID" || runpodctl pod delete "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID" ) &
cd /workspace/out && nohup python -m http.server 8000 > /dev/null 2>&1 &
cd /workspace && git clone --depth 1 https://github.com/{R.REPO} && cd sapling && pip install -q numpy pyyaml
python -c "import urllib.request; urllib.request.urlretrieve('{base}/cloud_pilot350.py', 'tools/cloud_pilot350.py')"
python tools/cloud_pilot350.py --base {shlex.quote(base)} --out /workspace/out --target-tokens {TARGET_TOKENS} --minutes {minutes:g} --micro {micro:d}{compile_arg} > /workspace/out/run.txt 2>&1
sleep infinity
"""
    return ["bash", "-c", script]


def create_pod(run: R.Run, stamp: str, base: str, minutes: float, max_pod_minutes: float,
               micro: int, secure_price: float, compile_model: bool = True) -> dict:
    last = None
    for cloud in ("COMMUNITY", "SECURE"):
        command = start_cmd(base, minutes, max_pod_minutes, micro, compile_model)
        body = R.pod_create_body(NAME_PREFIX + stamp, cloud, 80, command)
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
    cmd = f'"{sys.executable}" "{os.path.join(ROOT, "tools", "runpod_pilot350.py")}" --reap'
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
            cloud_log(f"TERMINATE | {pod['id']} | H100 SXM | 350M pilot reaper | deleted={ok}")
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
    ap.add_argument("--minutes", type=float, default=105, help="hard training wall-clock limit on the pod")
    ap.add_argument("--max-pod-minutes", type=float, default=125, help="in-pod self-deletion deadline including overhead")
    ap.add_argument("--micro", type=int, default=16)
    ap.add_argument("--no-compile", action="store_true")
    a = ap.parse_args(argv)
    if a.reap:
        gone, left = reap()
        print(json.dumps({"reaped": gone, "pods_left": [p.get("name") for p in left]}))
        return None
    if min(a.micro, a.minutes, a.max_pod_minutes) <= 0 or 64 % a.micro:
        ap.error("micro and minute limits must be positive, and micro must divide batch 64")
    if a.max_pod_minutes <= a.minutes:
        ap.error("--max-pod-minutes must exceed --minutes to leave boot/download/fetch time")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run = R.Run(stamp, a.dry_run)
    run.log(step="start", dry=a.dry_run, plan="fresh 350M 800M-token pilot", fresh=True,
            target_tokens=TARGET_TOKENS, config=C.CONFIG, minutes=a.minutes,
            max_pod_minutes=a.max_pod_minutes, micro=a.micro, compiled=not a.no_compile,
            hard_cap=HARD_CAP, working_cap=WORKING_CAP, install_or_swap=False)
    info = validate_bundle(BUNDLE)
    if a.dry_run:
        cmd = start_cmd("<release-url>", a.minutes, a.max_pod_minutes, a.micro, not a.no_compile)
        worst_case = 3.99 * a.max_pod_minutes / 60
        if worst_case > HARD_CAP:
            raise RuntimeError(f"default-price worst case ${worst_case:.2f} exceeds ${HARD_CAP:.2f} hard cap")
        run.log(step="dry run ends here - no upload, cloud call, pod, GPU training, install or swap", pod_start_cmd=cmd[2])
        summary = {"stamp": stamp, "dry_run": True, "release": None, "pod_created": False,
                   "target_tokens": TARGET_TOKENS, "config": C.CONFIG, "minutes": a.minutes,
                   "max_pod_minutes": a.max_pod_minutes, "micro": a.micro, "compiled": not a.no_compile,
                   "default_price_worst_case": round(worst_case, 3), "hard_cap": HARD_CAP,
                   "pods_left": None, "run_dir": os.path.relpath(run.dir, ROOT)}
        _save_summary(run, summary)
        print(json.dumps(summary), flush=True)
        return summary

    summary = {"stamp": stamp, "dry_run": False, "release": None, "cost": 0.0, "pods_left": None,
               "target_tokens": TARGET_TOKENS, "config": C.CONFIG, "run_dir": os.path.relpath(run.dir, ROOT)}
    pod = None
    started = None
    status = {}
    try:
        quote = R.price_quote()
        secure_price = float(quote.get("securePrice") or 0)
        if secure_price <= 0:
            raise RuntimeError(f"no usable secure H100 price; refusing to create a pod: {quote}")
        worst_case = secure_price * a.max_pod_minutes / 60
        run.log(step="price and cap", quote=quote, secure_worst_case=round(worst_case, 3), hard_cap=HARD_CAP)
        summary.update(secure_price=secure_price, secure_worst_case=worst_case)
        if worst_case > WORKING_CAP:
            raise RuntimeError(f"secure-price worst case ${worst_case:.2f} exceeds ${WORKING_CAP:.2f} working cap")
        existing = R.list_pods()
        if existing:
            summary["pods_left"] = [p.get("name") for p in existing]
            raise RuntimeError(f"a pod already exists; refusing to add cost: {[p.get('name') for p in existing]}")
        tag, base = upload_bundle(run, stamp, BUNDLE, info)
        summary["release"] = tag
        if R.list_pods():
            raise RuntimeError("a pod appeared before creation; refusing to add cost")
        try:
            pod = create_pod(run, stamp, base, a.minutes, a.max_pod_minutes, a.micro, secure_price, not a.no_compile)
            started = time.time()
            actual_worst_case = pod["cost_per_hr"] * a.max_pod_minutes / 60
            if actual_worst_case > HARD_CAP:
                raise RuntimeError(f"created pod's ${actual_worst_case:.2f} worst case exceeds ${HARD_CAP:.2f} hard cap; terminating it")
            cloud_log(f"START | {pod['id']} | H100 SXM ({pod['cloud'].lower()}) | fresh 350M pilot {stamp}; "
                      f"target {TARGET_TOKENS:,} tokens; training {a.minutes:g} min; watchdog {a.max_pod_minutes:g} min; "
                      f"Windows reaper; staged only | ${pod['cost_per_hr']:.2f}/hr")
            set_reaper(a.max_pod_minutes + 10)
            try:
                status = R.watch(run, pod, started + a.max_pod_minutes * 60 - 120)
            finally:
                R.fetch(run, pod, names=FETCHES)
        finally:
            if pod:
                summary["cost"], left = terminate(run, pod, started, f"fresh 350M pilot {stamp}")
                summary["pods_left"] = [p.get("name") for p in left]
        left = R.list_pods()
        summary["pods_left"] = [p.get("name") for p in left]
        cloud_log(f"FINISH | - | H100 SXM | fresh 350M pilot {stamp} | total ${summary['cost']:.2f}; "
                  f"pods left: {summary['pods_left'] or 'none'}")
        if left:
            raise RuntimeError(f"final pod list is not empty: {summary['pods_left']}")
        staged = os.path.join(run.dir, "core_staged.pt")
        benchmark = os.path.join(run.dir, "benchmark.json")
        summary.update(status=status,
                       core_ready=bool(status.get("state") == "done" and (status.get("run") or {}).get("target_reached")
                                       and os.path.exists(staged) and os.path.getsize(staged) > 1_000_000
                                       and os.path.exists(benchmark)),
                       staged=os.path.relpath(staged, ROOT) if os.path.exists(staged) else None,
                       benchmark=os.path.relpath(benchmark, ROOT) if os.path.exists(benchmark) else None,
                       installed=False)
        if not summary["core_ready"]:
            raise RuntimeError(f"350M pilot did not return a complete 800M-token staged core: {status}")
        run.log(step="finished", **summary)
        print(json.dumps(summary, default=str), flush=True)
        return summary
    finally:
        _save_summary(run, summary)


if __name__ == "__main__":
    main()
