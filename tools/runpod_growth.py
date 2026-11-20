"""Run the 10/18 growth plan: measure a fresh d768/12L core, then train 20 tokens/parameter.

  python tools/runpod_growth.py
  python tools/runpod_growth.py --reap
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import runpod_run as R

OLD_RELEASE = "cloud-20261016-2304"
HARD_CAP = 15.0
WORKING_CAP = 14.70
RESERVE_MINUTES = 15


def cloud_log(line: str):
    os.makedirs(os.path.dirname(R.CLOUD_LOG), exist_ok=True)
    with open(R.CLOUD_LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %I:%M %p").replace(" 0", " ") + " | Kimi | " + line + "\n")


def assets(tag: str) -> dict[str, int]:
    code, out = R.gh(["release", "view", tag, "--json", "assets"])
    if code:
        raise RuntimeError(f"release view {tag}: {out[-400:]}")
    return {a["name"]: int(a["size"]) for a in json.loads(out[out.index("{"):])["assets"]}


def upload(run: R.Run, tag: str, path: str) -> None:
    name = os.path.basename(path)
    t0 = time.time()
    code, out = R.gh(["release", "upload", tag, path, "--clobber"])
    if code:
        raise RuntimeError(f"upload {name}: {out[-400:]}")
    run.log(step="uploaded", file=name, mb=round(os.path.getsize(path) / 1e6, 1),
            seconds=round(time.time() - t0, 1))


def start_cmd(base: str, mode: str, max_minutes: float, train_minutes: float, bench_seconds: float,
              bench_micro: int = 32, use_compile: bool = False) -> list[str]:
    args = (f"bench --bench-seconds {bench_seconds:g} --micro {bench_micro}" if mode.startswith("bench") else
            f"train --minutes {train_minutes:g} --micro {64 if use_compile else 32}")
    if use_compile:
        args += " --compile"
    script = f"""set -x
mkdir -p /workspace/out
echo '{{"state":"booting"}}' > /workspace/out/status.json
( sleep {int(max_minutes * 60)}; echo "watchdog $(date)" >> /workspace/out/watchdog.txt;
  runpodctl remove pod $RUNPOD_POD_ID || runpodctl pod delete $RUNPOD_POD_ID || runpodctl stop pod $RUNPOD_POD_ID ) &
cd /workspace/out && nohup python -m http.server 8000 > /dev/null 2>&1 &
cd /workspace && git clone --depth 1 https://github.com/{R.REPO} && cd sapling && pip install -q numpy pyyaml
python -c "import urllib.request; urllib.request.urlretrieve('{base}/cloud_growth.py', 'tools/cloud_growth.py')"
python tools/cloud_growth.py {args} --base {base} --out /workspace/out > /workspace/out/run.txt 2>&1
sleep infinity
"""
    return ["bash", "-c", script]


def create_pod(run: R.Run, stamp: str, mode: str, base: str, max_minutes: float,
               train_minutes: float, bench_seconds: float, bench_micro: int = 32, use_compile: bool = False) -> dict:
    last = None
    for cloud in ("COMMUNITY", "SECURE"):
        command = start_cmd(base, mode, max_minutes, train_minutes, bench_seconds, bench_micro, use_compile)
        body = R.pod_create_body(f"sapling-growth-{mode}-{stamp}", cloud, 50, command)
        code, response = R._http("POST", f"{R.REST}/pods", body, timeout=120)
        if code in (200, 201) and isinstance(response, dict) and response.get("id"):
            pod = {"id": response["id"], "cloud": cloud,
                   "cost_per_hr": float(response.get("cost") or 0)}
            run.log(step="pod created", mode=mode, cloud=cloud, pod=pod["id"], cost_per_hr=pod["cost_per_hr"])
            return pod
        last = f"{cloud}: {code} {str(response)[:300]}"
        run.log(step="pod not created", mode=mode, cloud=cloud, why=last)
    raise RuntimeError(f"no H100 pod: {last}")


def set_reaper(minutes: float):
    when = time.localtime(time.time() + minutes * 60)
    cmd = f'"{sys.executable}" "{os.path.join(ROOT, "tools", "runpod_growth.py")}" --reap'
    p = subprocess.run(["schtasks", "/Create", "/F", "/TN", R.REAPER_TASK, "/SC", "ONCE", "/ST", time.strftime("%H:%M", when),
                        "/SD", time.strftime("%m/%d/%Y", when), "/TR", cmd], capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(f"could not set reaper: {(p.stdout or p.stderr)[-400:]}")


def terminate(run: R.Run, pod: dict, started: float, purpose: str) -> float:
    ok = R.delete_pod(pod["id"])
    hours = (time.time() - started) / 3600
    cost = hours * pod["cost_per_hr"]
    left = [p.get("name") for p in R.list_pods() if str(p.get("name", "")).startswith(R.NAME_PREFIX)]
    cloud_log(f"TERMINATE | {pod['id']} | H100 SXM | {purpose} | deleted={ok}, "
              f"{hours * 60:.1f} min x ${pod['cost_per_hr']:.2f}/hr = ${cost:.2f}; sapling pods left: {left or 'none'}")
    run.log(step="pod terminated", deleted=ok, minutes=round(hours * 60, 1), cost=round(cost, 3), left=left)
    if ok and not left:
        R.clear_reaper()
    if not ok:
        raise RuntimeError(f"pod {pod['id']} did not terminate")
    return cost


def deadline_tonight() -> dt.datetime:
    now = dt.datetime.now()
    deadline = now.replace(hour=2, minute=30, second=0, microsecond=0)
    if now.hour >= 3:
        deadline += dt.timedelta(days=1)
    return deadline


def reap() -> list[str]:
    gone = []
    for p in R.list_pods():
        if str(p.get("name", "")).startswith(R.NAME_PREFIX):
            ok = R.delete_pod(p["id"])
            cloud_log(f"TERMINATE | {p['id']} | H100 SXM | growth reaper | deleted={ok}")
            gone.append(p["id"])
    R.clear_reaper()
    return gone


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reap", action="store_true")
    ap.add_argument("--bench-seconds", type=float, default=45)
    ap.add_argument("--archive", default="results/checkpoints/pre_growth_20261028-223553.pt")
    a = ap.parse_args(argv)
    if a.reap:
        print(json.dumps({"reaped": reap(), "pods_left": [p.get("name") for p in R.list_pods()]}))
        return

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run = R.Run(stamp, False)
    tag = "growth-" + stamp
    base = f"https://github.com/{R.REPO}/releases/download/{tag}"
    quote = R.price_quote()
    secure = float(quote.get("securePrice") or 3.99)
    run.log(step="start", plan="10/18 growth", stamp=stamp, quote=quote, hard_cap=HARD_CAP,
            deadline=deadline_tonight().isoformat(), archive=a.archive)
    if [p for p in R.list_pods() if p.get("status") == "RUNNING"]:
        raise RuntimeError("a pod is already running; refusing to add cost")
    old_assets = assets(OLD_RELEASE)
    old_base = f"https://github.com/{R.REPO}/releases/download/{OLD_RELEASE}"
    bundle_dir = os.path.join(ROOT, "data", "cloud", "bundle")
    old_bundle = json.load(open(os.path.join(bundle_dir, "bundle.json"), encoding="utf-8"))
    core_src = os.path.join(ROOT, a.archive)
    import torch
    old_ck = torch.load(core_src, map_location="cpu", weights_only=False)
    core_name = f"core_gen{int(old_ck.get('generation') or 0):04d}.pt"
    core_copy = os.path.join(run.dir, core_name)
    shutil.copy(core_src, core_copy)
    index_src = os.path.join(ROOT, "data", "reading", "shards", "index.json")
    index = json.load(open(index_src, encoding="utf-8"))
    index_copy = os.path.join(run.dir, "reading_index.json")
    shutil.copy(index_src, index_copy)
    paths = {core_name: core_copy, "reading_index.json": index_copy,
             os.path.join(ROOT, "tools", "cloud_growth.py"): os.path.join(ROOT, "tools", "cloud_growth.py")}
    files = {core_name: os.path.getsize(core_copy), "reading_index.json": os.path.getsize(index_copy)}
    urls = {}
    for name in ("lesson.npz", "talk.npz", "school.npy", "test.jsonl"):
        expected = int(old_bundle["files"][name])
        if old_assets.get(name) != expected or os.path.getsize(os.path.join(bundle_dir, name)) != expected:
            raise RuntimeError(f"reused lesson asset not verified: {name}")
        files[name] = expected
        urls[name] = f"{old_base}/{name}"
    new_uploads = []
    for source in index["sources"].values():
        for shard in source["shards"]:
            name = shard["file"]
            path = os.path.join(ROOT, "data", "reading", "shards", name)
            files[name] = os.path.getsize(path)
            paths[name] = path
            new_uploads.append(path)
    public = {"built": stamp, "core_file": core_name, "files": files, "urls": urls,
              "note": "Fresh d768/12L core; licensed v4 reading plus rule-based long-reading and licensed conversation/school replay."}
    public_path = os.path.join(run.dir, "bundle.json")
    json.dump(public, open(public_path, "w", encoding="utf-8"), indent=1)
    code, out = R.gh(["release", "create", tag, "--title", f"Sapling growth inputs {stamp}", "--latest=false",
                      "--notes", "Inputs for the 10/18 growth run. FineWeb-Edu (ODC-By 1.0), Simple English Wikipedia "
                                 "(CC BY-SA 3.0), Project Gutenberg (public domain), TinyStories (CDLA-Sharing-1.0), "
                                 "oasst1 (Apache-2.0), dolly-15k (CC BY-SA 3.0). Long-reading questions are rule-based code templates."])
    if code:
        raise RuntimeError(f"release create: {out[-400:]}")
    code_path = os.path.join(ROOT, "tools", "cloud_growth.py")
    for path in (code_path, core_copy, public_path):
        upload(run, tag, path)
    run.log(step="initial upload verified", release=tag, assets=assets(tag))
    costs = []
    growth_ready = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = None
        pod = None
        t_pod = None
        try:
            pod = create_pod(run, stamp, "bench", base, 12, 0, a.bench_seconds)
            t_pod = time.time()
            set_reaper(22)
            cloud_log(f"START | {pod['id']} | H100 SXM ({pod['cloud'].lower()}) | growth speed test {stamp}; "
                      f"watchdog 12 min; reaper task | ${pod['cost_per_hr']:.2f}/hr")
            future = pool.submit(lambda: [upload(run, tag, p) for p in [index_copy] + new_uploads])
            st = R.watch(run, pod, t_pod + 10 * 60)
            if st.get("state") in ("done", "failed"):
                R.fetch(run, pod, names=("benchmark.json", "status.json", "run.txt"))
        finally:
            if pod:
                costs.append(terminate(run, pod, t_pod, f"growth speed test {stamp}"))
        bench_path = os.path.join(run.dir, "benchmark.json")
        if st.get("state") != "done" or not os.path.exists(bench_path):
            raise RuntimeError(f"speed test failed: {st}")
        bench = json.load(open(bench_path, encoding="utf-8"))
        run.log(step="speed measured", **bench)
        future.result()
        got = assets(tag)
        for name in [core_name, "reading_index.json", "cloud_growth.py", "bundle.json"] + [os.path.basename(p) for p in new_uploads]:
            if got.get(name) != (os.path.getsize(public_path) if name == "bundle.json" else files.get(name, os.path.getsize(code_path))):
                raise RuntimeError(f"growth asset not verified: {name}")
        run.log(step="growth upload verified", files=len(got), mb=round(sum(got.values()) / 1e6, 1))
        rate = float(bench["tok_per_s"])
        target = int(bench["target_tokens"])
        train_h = target / rate / 3600
        estimated_growth = secure * (train_h + 20 / 60)
        estimated_total = costs[0] + estimated_growth
        now = dt.datetime.now()
        deadline = deadline_tonight()
        max_by_cost = (WORKING_CAP - costs[0]) / secure * 60
        max_by_deadline = (deadline - now).total_seconds() / 60 - RESERVE_MINUTES
        max_minutes = min(max_by_cost, max_by_deadline)
        estimated_finish = now + dt.timedelta(minutes=train_h * 60 + 20)
        plan = {"measured_tok_per_s": rate, "params": bench["params"], "target_tokens": target,
                "training_hours": round(train_h, 3), "growth_cost_worst_case": round(estimated_growth, 2),
                "total_with_speed_test_worst_case": round(estimated_total, 2), "estimated_finish": estimated_finish.isoformat(),
                "hard_finish": deadline.isoformat(), "max_pod_minutes": round(max_minutes, 1)}
        run.log(step="growth dollar plan", **plan)
        cloud_log(f"PLAN | - | H100 SXM | growth {stamp} measured {rate:,} tok/s; target {target:,} tokens | "
                  f"growth ${estimated_growth:.2f} worst case, total ${estimated_total:.2f} including speed test; hard finish {deadline:%Y-%m-%d %I:%M %p}")
        if estimated_total >= WORKING_CAP or estimated_finish > deadline - dt.timedelta(minutes=10):
            raise RuntimeError(f"measured plan does not fit cost/time limits: {plan}")
        train_minutes = max_minutes - 20
        if train_minutes <= train_h * 60:
            raise RuntimeError(f"insufficient safe pod window: {plan}")
        pod = None
        t_pod = None
        try:
            pod = create_pod(run, stamp, "train", base, max_minutes, train_minutes, a.bench_seconds)
            t_pod = time.time()
            set_reaper(max_minutes + 10)
            cloud_log(f"START | {pod['id']} | H100 SXM ({pod['cloud'].lower()}) | growth training {stamp}; "
                      f"target {target:,} tokens; watchdog {max_minutes:.1f} min; reaper task | ${pod['cost_per_hr']:.2f}/hr")
            st = R.watch(run, pod, t_pod + max_minutes * 60 - 120)
            if st.get("state") in ("done", "failed"):
                R.fetch(run, pod, names=("core_staged.pt", "log.jsonl", "status.json", "run.txt"))
        finally:
            if pod:
                costs.append(terminate(run, pod, t_pod, f"growth training {stamp}"))
        growth_ready = bool(st.get("state") == "done" and (st.get("run") or {}).get("target_reached") and
                            os.path.exists(os.path.join(run.dir, "core_staged.pt")))
        run.log(step="growth result", ready=growth_ready, status=st, total_cost=round(sum(costs), 2))
    left = [p.get("name") for p in R.list_pods()]
    if any(str(n).startswith(R.NAME_PREFIX) for n in left):
        reap()
        left = [p.get("name") for p in R.list_pods()]
    summary = {"stamp": stamp, "release": tag, "speed_test": bench, "growth_ready": growth_ready,
               "cost": round(sum(costs), 2), "pods_left": left, "run_dir": os.path.relpath(run.dir, ROOT)}
    json.dump(summary, open(os.path.join(run.dir, "summary.json"), "w"), indent=1)
    cloud_log(f"FINISH | - | - | growth {stamp}; core ready={growth_ready} | total ${sum(costs):.2f}; pods left: {left or 'none'}")
    run.log(step="finished", **summary)
    if not growth_ready:
        raise RuntimeError(f"growth did not produce a complete core: {st}")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        try:
            reap()
            left = [p.get("name") for p in R.list_pods()]
            cloud_log(f"FINISH | - | - | growth run stopped on error | pods left: {left or 'none'}")
        finally:
            raise
