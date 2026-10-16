"""Runs ON the rented GPU pod (started by tools/runpod_run.py; never on the MSI).

Downloads the lesson bundle from the public GitHub release the MSI uploaded, checks every file's size against
bundle.json, trains (sapling.longread_lesson.train_bundle), and writes everything into --out, which the pod serves on
port 8000 for the MSI to fetch: status.json (always current), log.jsonl, core_staged.pt, run.txt.

    python tools/cloud_lesson.py --base https://github.com/<repo>/releases/download/<tag> --out /workspace/out --minutes 55
"""
import argparse
import json
import os
import sys
import time
import traceback
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", default="/workspace/out")
    ap.add_argument("--bundle", default="/workspace/bundle")
    ap.add_argument("--minutes", type=float, default=55)
    ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--smoke", action="store_true", help="tiny batch, 3 steps (a CPU test of the whole path)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.bundle, exist_ok=True)
    st_path = os.path.join(a.out, "status.json")
    state = {"state": "downloading", "since": time.time()}

    def status(row=None, **kw):
        if row:
            state["last"] = row
        state.update(kw)
        state["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        tmp = st_path + ".tmp"
        json.dump(state, open(tmp, "w"), indent=1)
        os.replace(tmp, st_path)

    try:
        status()
        t0 = time.time()
        meta = os.path.join(a.bundle, "bundle.json")
        urllib.request.urlretrieve(f"{a.base}/bundle.json", meta)
        info = json.load(open(meta))
        for name, size in info["files"].items():
            dest = os.path.join(a.bundle, name)
            for attempt in range(5):
                try:
                    urllib.request.urlretrieve(f"{a.base}/{name}", dest)
                    if os.path.getsize(dest) == size:
                        break
                except Exception as e:
                    status(error=f"download {name} try {attempt + 1}: {e}")
                time.sleep(10)
            if os.path.getsize(dest) != size:
                raise RuntimeError(f"{name}: got {os.path.getsize(dest)} bytes, expected {size}")
        status(state="training", download_s=round(time.time() - t0, 1))
        import torch
        status(gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
        from sapling.longread_lesson import train_bundle
        kw = dict(batch=8, micro=8, shares={"lesson": 3, "reading": 2, "talk": 2, "school": 1}, max_steps=3, device="cpu") if a.smoke else {}
        run = train_bundle(a.bundle, a.out, minutes=a.minutes, status=lambda r: status(r), **({"micro": a.micro} | kw))
        status(state="done", run={k: run[k] for k in ("steps", "tokens", "minutes", "loss_last", "heldout_before", "heldout_after")},
               files={n: os.path.getsize(os.path.join(a.out, n)) for n in os.listdir(a.out)})
    except Exception as e:
        status(state="failed", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-3000:])
        raise


if __name__ == "__main__":
    main()
