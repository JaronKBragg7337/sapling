"""Download a conversation bundle and run Sapling's existing talk.train on a rented GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import sys
import time
import traceback
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class TrainingTimeLimit(TimeoutError):
    pass


def _download(base: str, name: str, dest: str, expected: int, sha256: str | None) -> None:
    if os.path.basename(name) != name:
        raise ValueError(f"unsafe bundle asset name: {name}")
    error = None
    for attempt in range(5):
        try:
            req = urllib.request.Request(f"{base}/{name}", headers={"user-agent": "sapling-cloud-talk/1.0"})
            with urllib.request.urlopen(req, timeout=180) as response, open(dest + ".part", "wb") as f:
                shutil.copyfileobj(response, f, length=1 << 20)
            if os.path.getsize(dest + ".part") != expected:
                raise RuntimeError(f"size {os.path.getsize(dest + '.part')} != {expected}")
            if sha256:
                h = hashlib.sha256()
                with open(dest + ".part", "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
                if h.hexdigest() != sha256:
                    raise RuntimeError("SHA-256 mismatch")
            os.replace(dest + ".part", dest)
            return
        except Exception as e:
            error = e
            try:
                os.remove(dest + ".part")
            except OSError:
                pass
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"download failed for {name}: {error}")


def _install_reading_replay(bundle: str, info: dict) -> None:
    shard_dir = os.path.join(ROOT, "data", "reading", "shards")
    os.makedirs(shard_dir, exist_ok=True)
    index = json.load(open(os.path.join(bundle, "reading_index.json"), encoding="utf-8"))
    for source in index.get("sources", {}).values():
        for shard in source.get("shards", []):
            name = shard["file"]
            if name not in info["files"]:
                raise RuntimeError(f"reading shard is not in the verified bundle: {name}")
            dest = os.path.join(shard_dir, name)
            shutil.copyfile(os.path.join(bundle, name), dest + ".tmp")
            os.replace(dest + ".tmp", dest)
    dest = os.path.join(shard_dir, "index.json")
    shutil.copyfile(os.path.join(bundle, "reading_index.json"), dest + ".tmp")
    os.replace(dest + ".tmp", dest)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", default="/workspace/out")
    ap.add_argument("--bundle", default="/workspace/talk_bundle")
    ap.add_argument("--minutes", type=float, default=25)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--micro", type=int, default=16)
    a = ap.parse_args(argv)
    if a.minutes <= 0 or a.epochs <= 0 or a.max_steps <= 0 or a.micro <= 0:
        ap.error("--minutes, --epochs, --max-steps and --micro must be positive")
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.bundle, exist_ok=True)
    status_path = os.path.join(a.out, "status.json")
    state = {"state": "downloading", "since": time.time()}

    def status(**row):
        state.update(row)
        state["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        tmp = status_path + ".tmp"
        json.dump(state, open(tmp, "w"), indent=1, default=str)
        os.replace(tmp, status_path)

    try:
        status()
        meta = os.path.join(a.bundle, "bundle.json")
        req = urllib.request.Request(f"{a.base}/bundle.json", headers={"user-agent": "sapling-cloud-talk/1.0"})
        with urllib.request.urlopen(req, timeout=120) as response, open(meta + ".part", "wb") as f:
            shutil.copyfileobj(response, f)
        os.replace(meta + ".part", meta)
        info = json.load(open(meta, encoding="utf-8"))
        core_name = info["core_file"]
        for name, expected in info["files"].items():
            status(last=f"downloading {name}")
            _download(a.base, name, os.path.join(a.bundle, name), int(expected), (info.get("sha256") or {}).get(name))
        status(last="downloads verified", download_s=round(time.time() - state["since"], 1))

        _install_reading_replay(a.bundle, info)
        school = json.load(open(os.path.join(a.bundle, "school.json"), encoding="utf-8"))
        if not isinstance(school, list) or not school or not all(isinstance(s, str) for s in school):
            raise RuntimeError("school.json must be a non-empty list of sentences")
        import sapling.reading as reading
        import sapling.talk as talk
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on the rented GPU")
        reading.rehearsal_sentences = lambda seed, store=None, verbose=True: school
        core = os.path.join(a.bundle, core_name)
        lessons = os.path.join(a.bundle, "lessons.jsonl")
        staged = os.path.join(a.out, "core_staged.pt")
        status(state="training", gpu=torch.cuda.get_device_name(0), epochs=a.epochs,
               max_steps=a.max_steps, minutes=a.minutes, micro=a.micro)

        def time_limit(signum, frame):
            raise TrainingTimeLimit(f"talk.train exceeded {a.minutes:g} minutes")

        old_handler = signal.signal(signal.SIGALRM, time_limit)
        signal.alarm(max(1, int(a.minutes * 60)))
        try:
            result = talk.train(core=core, staged=staged, epochs=a.epochs, max_steps=a.max_steps,
                                micro=a.micro, lessons=lessons, out_dir=a.out, wait_for_gpu=False,
                                label="v5 cloud conversation lessons")
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        if not os.path.exists(staged) or os.path.getsize(staged) < 1_000_000:
            raise RuntimeError("talk.train did not save a usable core_staged.pt")
        status(state="done", run=result["run"], staged=staged, staged_bytes=os.path.getsize(staged))
        return result["run"]
    except Exception as e:
        status(state="failed", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-4000:])
        raise


if __name__ == "__main__":
    main()
