"""Run the long-reading lesson on ONE rented H100 (RunPod), bring the core back, always terminate the pod.

  python tools/runpod_run.py --dry-run          # everything except creating the pod (bundle, upload, price quote, plan)
  python tools/runpod_run.py                    # the real run (~$4-5, ~75 min door to door)
  python tools/runpod_run.py --reap             # terminate every pod named sapling-* (the safety task calls this)

What happens (each step is a line in results/cloud/<stamp>/runner.jsonl):
 1. bundle: `python -m sapling.longread_lesson build` -> data/cloud/bundle/ (lesson examples, held-out test, talk +
    reading + school replay, the live core). Reused if built from the same live core within 24 h.
 2. upload: the bundle -> a GitHub release `cloud-<stamp>` on the public repo (gh CLI, signed in on the MSI), every
    asset size verified. The pod downloads it from there (public URL - nothing secret leaves the MSI).
 3. price: the H100's community and secure prices are read from RunPod (RUNPOD_API_KEY from ~/.secrets/keys.env;
    the value is never printed or logged).
 4. pod: COMMUNITY cloud first, SECURE if none is free; on-demand. Its start command clones the repo, starts a
    watchdog that deletes the pod after --max-pod-minutes even if the MSI is gone (runpodctl, the pod's own key),
    serves /workspace/out on port 8000, and runs tools/cloud_lesson.py.
    A Windows one-shot task "Sapling Cloud Reaper" is also set for max-pod-minutes + 10: it runs `--reap`.
 5. watch: status.json through RunPod's proxy every 60 s.
 6. fetch: core_staged.pt + log.jsonl + status.json -> results/cloud/<stamp>/.
 7. ALWAYS (finally): DELETE the pod, check it is gone, log TERMINATE with the cost to AI-Shared/state/cloud-gpu-log.md,
    remove the reaper task.
 8. --swap (default on): measure the new core on the MSI, stop the heartbeat (gpu_yield, waits), grow.swap, restart,
    chat samples. --no-swap leaves it staged.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

REPO = "JaronKBragg7337/sapling"
KEYS_ENV = os.path.join(os.path.expanduser("~"), ".secrets", "keys.env")
CLOUD_LOG = r"C:\Users\lilli\AI-Shared\state\cloud-gpu-log.md"
BUNDLE = os.path.join(ROOT, "data", "cloud", "bundle")
GPU = "NVIDIA H100 80GB HBM3"           # H100 SXM
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
REST = "https://rest.runpod.io/v1"
GQL = "https://api.runpod.io/graphql"
REAPER_TASK = "Sapling Cloud Reaper"
NAME_PREFIX = "sapling-"


# -- secrets (names only ever printed) --------------------------------------------------------
def _key() -> str:
    for line in open(KEYS_ENV, encoding="utf-8"):
        line = line.strip()
        if line.startswith("RUNPOD_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("RUNPOD_API_KEY is not in ~/.secrets/keys.env")


def _ntfy_topic() -> str | None:
    try:
        for line in open(KEYS_ENV, encoding="utf-8"):
            if line.startswith("NTFY_TOPIC="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return None


def _http(method: str, url: str, body: dict | None = None, timeout: int = 60, auth: bool = True) -> tuple[int, dict | str]:
    h = {"content-type": "application/json", "user-agent": "sapling-runpod-runner/1.0"}   # Cloudflare 1010 blocks urllib's default agent
    if auth:
        h["authorization"] = f"Bearer {_key()}"
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode() or "{}"
            code = r.status
    except urllib.error.HTTPError as e:
        raw, code = e.read().decode(errors="replace"), e.code
    try:
        return code, json.loads(raw)
    except ValueError:
        return code, raw


def price_quote(gpu: str = GPU) -> dict:
    q = {"query": 'query { gpuTypes(input: {id: "%s"}) { id displayName memoryInGb securePrice communityPrice '
                  'secureCloud communityCloud lowestPrice(input: {gpuCount: 1}) { minimumBidPrice uninterruptablePrice } } }' % gpu}
    code, r = _http("POST", GQL, q)
    if code != 200 or not isinstance(r, dict) or "data" not in r:
        return {"error": f"price query {code}: {str(r)[:200]}"}
    g = (r["data"]["gpuTypes"] or [{}])[0]
    return {k: g.get(k) for k in ("id", "displayName", "memoryInGb", "securePrice", "communityPrice", "secureCloud",
                                  "communityCloud")} | {"lowest": g.get("lowestPrice")}


def list_pods() -> list[dict]:
    code, r = _http("GET", f"{REST}/pods")
    if code != 200:
        raise RuntimeError(f"list pods {code}: {str(r)[:200]}")
    return r if isinstance(r, list) else r.get("pods", [])


def delete_pod(pod_id: str) -> bool:
    code, r = _http("DELETE", f"{REST}/pods/{pod_id}")
    if code not in (200, 202, 204, 404):
        return False
    for _ in range(12):
        c, g = _http("GET", f"{REST}/pods/{pod_id}")
        if c == 404 or (isinstance(g, dict) and g.get("desiredStatus") == "TERMINATED"):
            return True
        time.sleep(5)
    return pod_id not in {p.get("id") for p in list_pods()}


def cloud_log(line: str):
    os.makedirs(os.path.dirname(CLOUD_LOG), exist_ok=True)
    with open(CLOUD_LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %I:%M %p").replace(" 0", " ") + " | Claude | " + line + "\n")


# -- the run log ------------------------------------------------------------------------------------
class Run:
    def __init__(self, stamp: str, dry: bool):
        self.stamp, self.dry = stamp, dry
        self.dir = os.path.join(ROOT, "results", "cloud", stamp + ("_dry" if dry else ""))
        os.makedirs(self.dir, exist_ok=True)

    def log(self, **row):
        row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **row}
        with open(os.path.join(self.dir, "runner.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        print(json.dumps(row, default=str), flush=True)


# -- steps --------------------------------------------------------------------------------------------
def ensure_bundle(run: Run, examples: int, read_tokens: float, rebuild: bool) -> dict:
    from sapling.train import CHECKPOINT
    info_p = os.path.join(BUNDLE, "bundle.json")
    if not rebuild and os.path.exists(info_p):
        info = json.load(open(info_p, encoding="utf-8"))
        fresh = time.time() - os.path.getmtime(os.path.join(BUNDLE, "lesson.npz")) < 48 * 3600
        if fresh and info.get("lesson", {}).get("train", 0) >= examples * 0.9:
            import shutil
            import torch
            gen = int(torch.load(CHECKPOINT, map_location="cpu", weights_only=False).get("generation") or 0)
            if gen != info.get("core_generation"):
                # the lesson is still good; only the core moved on (the heartbeat consolidated) -> swap in the live core
                new = f"core_gen{gen:04d}.pt"
                for n in list(info["files"]):
                    if n.startswith("core") and n.endswith(".pt"):
                        info["files"].pop(n)
                        if os.path.exists(os.path.join(BUNDLE, n)):
                            os.remove(os.path.join(BUNDLE, n))
                shutil.copy(CHECKPOINT, os.path.join(BUNDLE, new))
                info.update({"core_generation": gen, "core_file": new, "core_updated": time.strftime("%Y-%m-%dT%H:%M:%S")})
                info["files"][new] = os.path.getsize(os.path.join(BUNDLE, new))
                json.dump(info, open(info_p, "w", encoding="utf-8"), indent=1)
            run.log(step="bundle reused", built=info["built"], core=info.get("core_file"), files=info["files"])
            return info
    t0 = time.time()
    from sapling.longread_lesson import build_bundle
    info = build_bundle(n_examples=examples, read_tokens=read_tokens, out=BUNDLE, verbose=True)
    run.log(step="bundle built", seconds=round(time.time() - t0), files=info["files"], lesson={k: info["lesson"][k] for k in ("train", "test", "train_tokens", "kinds", "distractor")})
    return info


def gh(args: list[str], timeout: int = 3600) -> tuple[int, str]:
    p = subprocess.run(["gh"] + args + ["-R", REPO], capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def upload_bundle(run: Run, info: dict) -> str:
    # one release per BUILT bundle, so a dry run's upload is reused by the real run (same bundle, same tag)
    tag = "cloud-" + info["built"].replace("-", "").replace(":", "").replace("T", "-")[:13]
    code, out = gh(["release", "view", tag, "--json", "assets"])
    if code != 0:
        code, out = gh(["release", "create", tag, "--title", f"Sapling cloud lesson bundle {run.stamp}", "--latest=false",
                        "--notes", "Inputs for one rented-GPU lesson run (tools/runpod_run.py). Text: Simple English "
                                   "Wikipedia (CC BY-SA 3.0, Wikipedia contributors), Project Gutenberg (public domain), "
                                   "FineWeb-Edu (ODC-By 1.0, Hugging Face / Common Crawl), oasst1 (Apache-2.0), dolly-15k "
                                   "(CC BY-SA 3.0); tokenized. Questions are rule-based templates (code), not generated text."])
        if code != 0:
            raise RuntimeError(f"release create: {out[-300:]}")
        have = {}
    else:
        have = {a["name"]: int(a["size"]) for a in json.loads(out[out.index("{"):])["assets"]}
    t0, sent = time.time(), 0
    for name in list(info["files"]) + ["bundle.json"]:
        p = os.path.join(BUNDLE, name)
        size = os.path.getsize(p)
        if have.get(name) == size and name != "bundle.json":     # bundle.json always goes (it names the core)
            continue
        t1 = time.time()
        code, out = gh(["release", "upload", tag, p, "--clobber"])
        if code != 0:
            raise RuntimeError(f"upload {name}: {out[-300:]}")
        sent += size
        run.log(step="uploaded", file=name, mb=round(size / 1e6, 1), seconds=round(time.time() - t1, 1),
                mb_per_s=round(size / 1e6 / max(0.1, time.time() - t1), 2))
    code, out = gh(["release", "view", tag, "--json", "assets"])
    got = {a["name"]: int(a["size"]) for a in json.loads(out[out.index("{"):])["assets"]}
    for name in list(info["files"]) + ["bundle.json"]:
        if got.get(name) != os.path.getsize(os.path.join(BUNDLE, name)):
            raise RuntimeError(f"release asset {name} not verified ({got.get(name)})")
    run.log(step="upload verified", release=tag, mb_sent=round(sent / 1e6, 1), seconds=round(time.time() - t0, 1))
    return f"https://github.com/{REPO}/releases/download/{tag}"


def start_cmd(base: str, minutes: float, max_pod_minutes: float, micro: int) -> list[str]:
    script = f"""set -x
mkdir -p /workspace/out
echo '{{"state":"booting"}}' > /workspace/out/status.json
( sleep {int(max_pod_minutes * 60)}; echo "watchdog $(date)" >> /workspace/out/watchdog.txt;
  runpodctl remove pod $RUNPOD_POD_ID || runpodctl pod delete $RUNPOD_POD_ID || runpodctl stop pod $RUNPOD_POD_ID ) &
cd /workspace/out && nohup python -m http.server 8000 > /dev/null 2>&1 &
cd /workspace && git clone --depth 1 https://github.com/{REPO} && cd sapling && pip install -q numpy pyyaml
python tools/cloud_lesson.py --base {base} --out /workspace/out --minutes {minutes} --micro {micro} > /workspace/out/run.txt 2>&1
sleep infinity
"""
    return ["bash", "-c", script]


def create_pod(run: Run, base: str, minutes: float, max_pod_minutes: float, micro: int) -> dict:
    last = None
    for cloud in ("COMMUNITY", "SECURE"):
        body = {"name": f"{NAME_PREFIX}{run.stamp}", "imageName": IMAGE, "gpuTypeIds": [GPU], "gpuCount": 1,
                "cloudType": cloud, "interruptible": False, "containerDiskInGb": 40, "volumeInGb": 0,
                "ports": ["8000/http"], "env": {}, "dockerStartCmd": start_cmd(base, minutes, max_pod_minutes, micro)}
        code, r = _http("POST", f"{REST}/pods", body, timeout=120)
        if code in (200, 201) and isinstance(r, dict) and r.get("id"):
            run.log(step="pod created", cloud=cloud, pod=r["id"], cost_per_hr=r.get("costPerHr"),
                    adjusted=r.get("adjustedCostPerHr"))
            return {"id": r["id"], "cloud": cloud, "cost_per_hr": float(r.get("adjustedCostPerHr") or r.get("costPerHr") or 0)}
        last = f"{cloud}: {code} {str(r)[:300]}"
        run.log(step="pod not created", cloud=cloud, why=last)
    raise RuntimeError(f"no H100 pod: {last}")


def set_reaper(minutes: float):
    when = time.localtime(time.time() + minutes * 60)
    cmd = f'"{sys.executable}" "{os.path.join(ROOT, "tools", "runpod_run.py")}" --reap'
    subprocess.run(["schtasks", "/Create", "/F", "/TN", REAPER_TASK, "/SC", "ONCE", "/ST", time.strftime("%H:%M", when),
                    "/SD", time.strftime("%m/%d/%Y", when), "/TR", cmd], capture_output=True, text=True)


def clear_reaper():
    subprocess.run(["schtasks", "/Delete", "/F", "/TN", REAPER_TASK], capture_output=True, text=True)


UA = {"user-agent": "sapling-runpod-runner/1.0"}


def _download(url: str, dest: str, timeout: int = 600):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest + ".part", "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    os.replace(dest + ".part", dest)


def watch(run: Run, pod: dict, deadline: float) -> dict:
    url = f"https://{pod['id']}-8000.proxy.runpod.net"
    last_state, last = None, {}
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(urllib.request.Request(f"{url}/status.json", headers=UA), timeout=30) as r:
                last = json.loads(r.read().decode())
        except Exception as e:
            last = {"state": "unreachable", "error": str(e)[:120]}
        if last.get("state") != last_state or int(time.time()) % 600 < 60:
            run.log(step="pod status", state=last.get("state"), last=last.get("last"), error=last.get("error"))
            last_state = last.get("state")
        if last.get("state") in ("done", "failed"):
            return last
        time.sleep(60)
    return {"state": "timeout", **last}


def fetch(run: Run, pod: dict, names=("core_staged.pt", "log.jsonl", "status.json", "run.txt")) -> dict:
    url = f"https://{pod['id']}-8000.proxy.runpod.net"
    got = {}
    for n in names:
        dest = os.path.join(run.dir, n)
        try:
            t0 = time.time()
            _download(f"{url}/{n}", dest)
            got[n] = {"bytes": os.path.getsize(dest), "seconds": round(time.time() - t0, 1)}
        except Exception as e:
            got[n] = {"error": str(e)[:200]}
    run.log(step="fetched", files=got)
    return got


def notify(text: str):
    t = _ntfy_topic()
    if t:
        try:
            urllib.request.urlopen(urllib.request.Request(f"https://ntfy.sh/{t}", data=text.encode(), method="POST"), timeout=20)
        except Exception:
            pass


def swap_in(run: Run, staged: str):
    py = sys.executable
    def sh(args, env=None):
        p = subprocess.run([py] + args, cwd=ROOT, capture_output=True, text=True, env={**os.environ, **(env or {})})
        if p.returncode:
            raise RuntimeError(p.stderr[-800:])
        return p.stdout
    who = f"Claude (teacher) - cloud lesson {run.stamp}"
    sh(["tools/measure_core.py", "--core", staged, "--label", f"after long-reading lesson (cloud {run.stamp})"])
    sh(["tools/chat_samples.py", "--core", staged, "--label", f"after long-reading lesson (cloud {run.stamp})"])
    out = sh(["tools/gpu_yield.py", "stop", who]).strip()
    run.log(step="heartbeat stop", result=out)
    try:
        gen = sh(["-c", f"from sapling.grow import swap; r = swap(staged={staged!r}); print(r['event']['to']['generation'])"]).strip().splitlines()[-1]
        run.log(step="swapped", generation=gen)
    finally:
        run.log(step="heartbeat start", result=sh(["tools/gpu_yield.py", "start", who]).strip())


def reap() -> list[str]:
    gone = []
    for p in list_pods():
        if str(p.get("name", "")).startswith(NAME_PREFIX):
            ok = delete_pod(p["id"])
            cloud_log(f"TERMINATE | {p['id']} | {p.get('gpu', {}).get('displayName', GPU) if isinstance(p.get('gpu'), dict) else GPU} | "
                      f"reaper (--reap) | deleted={ok}")
            gone.append(p["id"])
    clear_reaper()
    return gone


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reap", action="store_true")
    ap.add_argument("--minutes", type=float, default=55, help="training minutes on the pod")
    ap.add_argument("--max-pod-minutes", type=float, default=95, help="the pod deletes itself after this, whatever happens")
    ap.add_argument("--examples", type=int, default=400_000)
    ap.add_argument("--read-tokens", type=float, default=150e6)
    ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--no-swap", action="store_true")
    ap.add_argument("--skip-upload", action="store_true", help="dry run only: do not upload the bundle")
    a = ap.parse_args(argv)
    if a.reap:
        print(json.dumps({"reaped": reap()}))
        return
    stamp = time.strftime("%Y%m%d-%H%M")
    run = Run(stamp, a.dry_run)
    run.log(step="start", dry=a.dry_run, minutes=a.minutes, max_pod_minutes=a.max_pod_minutes, gpu=GPU, image=IMAGE)
    quote = price_quote()
    run.log(step="price", quote=quote)
    running = [p for p in list_pods() if p.get("desiredStatus") == "RUNNING"]
    run.log(step="pods running before", n=len(running), names=[p.get("name") for p in running])
    info = ensure_bundle(run, a.examples, a.read_tokens, a.rebuild)
    base = None if (a.dry_run and a.skip_upload) else upload_bundle(run, info)
    cmd = start_cmd(base or "<release-url>", a.minutes, a.max_pod_minutes, a.micro)
    est_h = (a.minutes + 15) / 60                 # boot + download ~10 min, training, fetch ~5 min
    cp = quote.get("communityPrice") or quote.get("securePrice") or 3.99
    run.log(step="plan", release=base, pod_start_cmd=cmd[2], expected_pod_hours=round(est_h, 2),
            expected_cost=f"${cp * est_h:.2f} community / ${(quote.get('securePrice') or 3.99) * est_h:.2f} secure",
            hard_cap_cost=f"${(quote.get('securePrice') or 3.99) * a.max_pod_minutes / 60:.2f} (watchdog at {a.max_pod_minutes} min)")
    if a.dry_run:
        run.log(step="dry run ends here - no pod created")
        return
    pod, t_pod, st = None, None, {}
    try:
        pod = create_pod(run, base, a.minutes, a.max_pod_minutes, a.micro)
        t_pod = time.time()
        set_reaper(a.max_pod_minutes + 10)
        cloud_log(f"START | {pod['id']} | H100 SXM ({pod['cloud'].lower()}) | long-reading lesson {stamp} "
                  f"(tools/runpod_run.py; watchdog {a.max_pod_minutes:g} min; reaper task) | ${pod['cost_per_hr']:.2f}/hr")
        notify(f"Sapling cloud lesson started: H100 {pod['cloud'].lower()} ${pod['cost_per_hr']:.2f}/hr, back in ~{a.minutes + 15:.0f} min")
        st = watch(run, pod, t_pod + a.max_pod_minutes * 60 - 120)
        if st.get("state") in ("done", "failed"):
            fetch(run, pod)
    finally:
        if pod:
            ok = delete_pod(pod["id"])
            hours = (time.time() - t_pod) / 3600
            left = [p.get("name") for p in list_pods() if str(p.get("name", "")).startswith(NAME_PREFIX)]
            cloud_log(f"TERMINATE | {pod['id']} | H100 SXM | long-reading lesson {stamp} | deleted={ok}, "
                      f"{hours * 60:.0f} min x ${pod['cost_per_hr']:.2f}/hr = ${hours * pod['cost_per_hr']:.2f}; sapling pods left: {left or 'none'}")
            run.log(step="pod terminated", deleted=ok, minutes=round(hours * 60, 1), cost=round(hours * pod["cost_per_hr"], 2), left=left)
            if ok and not left:
                clear_reaper()
    staged = os.path.join(run.dir, "core_staged.pt")
    if os.path.exists(staged) and os.path.getsize(staged) > 1e6:
        notify(f"Sapling cloud lesson back: {json.dumps((st.get('run') or {}).get('heldout_after', {}))[:120]}")
        if not a.no_swap:
            swap_in(run, staged)
    else:
        notify("Sapling cloud lesson: no core came back - see results/cloud/" + stamp)
    run.log(step="finished")


if __name__ == "__main__":
    main()
