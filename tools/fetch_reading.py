"""Fetch the reading-stage text (v3.2 "much more reading", Jaron 10/14 9:40 PM) into data/reading/raw/.

Three sources, each license-checked LIVE at fetch time and written to data/reading/manifest.json:
- TinyStories, the full GPT-4 train split (roneneldan/TinyStories on Hugging Face, CDLA-Sharing-1.0)
- Simple English Wikipedia (wikimedia/wikipedia on Hugging Face, config 20231101.simple; CC BY-SA 3.0 + GFDL;
  attribution: Wikipedia contributors - kept in README)
- Project Gutenberg books (public domain in the US; each file's own header must carry the "anyone anywhere ...
  almost no restrictions" line or the book is NOT kept; the Gutenberg header/footer/licence text is stripped
  before training, nothing of theirs is trained on)

Everything lands under data/ (gitignored). Usage:
    python tools/fetch_reading.py              # all three (skips files already present)
    python tools/fetch_reading.py gutenberg    # one source
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "reading", "raw")
MANIFEST = os.path.join(ROOT, "data", "reading", "manifest.json")
UA = {"User-Agent": "sapling-reading-stage/1.0 (public-domain text fetch; contact via heartbeatobservatory.com)"}

TINYSTORIES_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
SIMPLEWIKI_URL = "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.simple/train-00000-of-00001.parquet"

# Public-domain books (US) beyond the five v3.1 school books and the held-out Thoreau essay (pg1022 stays out of
# every lesson - it is E7L). Mostly prose: novels, essays, history, philosophy, letters. A wrong id just fetches a
# different public-domain book; the header check below decides whether it is kept, and the title is read from it.
GUTENBERG_IDS = [
    1342, 11, 84, 1661, 2701, 98, 1400, 345, 174, 76, 74, 1184, 120, 16, 35, 36, 43, 158, 161, 1260, 768, 2591,
    1727, 2500, 5200, 244, 2852, 1232, 7370, 3300, 3600, 1497, 2300, 4300, 2554, 28054, 600, 1399, 135, 2413,
    2814, 4217, 33, 2148, 2147, 215, 910, 12, 113, 236, 1998, 19942, 996, 1404, 147, 3207, 2641, 141, 105, 121,
    203, 23, 1023, 730, 766, 41, 2097, 3296, 5827, 4705, 55, 1257, 103, 164, 18857, 521, 829, 24, 834, 2160,
    1251, 2680, 10, 8800, 1322, 4363, 30254, 1080, 1952, 2500, 25344, 2554, 46, 1155, 1250, 140, 1065, 2246,
    3177, 4280, 1946, 7849, 2680, 3090, 2199, 2554, 6593, 20417, 1526, 1524, 1513, 2267, 23700, 29870, 15399,
    4085, 2554, 2591, 6400, 1399, 7142, 5740, 408, 17135, 2446, 3268, 2708, 2852, 2097, 1155, 1998, 1342,
]
# v4 (10/15, Claude as teacher): more adult text + conversation lessons. Licences checked live on the HF API 10/15.
FINEWEB_EDU_URL = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/{name}"
FINEWEB_EDU_FILES = ["000_00000.parquet"]            # one 2.15 GB file of the 10BT sample (~0.7B GPT-2 tokens)
OASST1_URL = "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/2023-04-12_oasst_ready.trees.jsonl.gz"
DOLLY_URL = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
GUTENBERG_OK = "almost no restrictions whatsoever"
GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"


def _manifest() -> dict:
    try:
        return json.load(open(MANIFEST, encoding="utf-8"))
    except Exception:
        return {"sources": {}, "gutenberg": {}}


def _save(m: dict):
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    json.dump(m, open(MANIFEST, "w", encoding="utf-8"), indent=2)


def _get(url: str, dest: str, resume: bool = True, tries: int = 8) -> bool:
    for i in range(tries):
        try:
            have = os.path.getsize(dest) if resume and os.path.exists(dest) else 0
            req = urllib.request.Request(url, headers={**UA, **({"Range": f"bytes={have}-"} if have else {})})
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:
                    have = 0
                with open(dest, "ab" if have else "wb") as f:
                    while True:
                        b = r.read(1 << 20)
                        if not b:
                            break
                        f.write(b)
            return True
        except Exception as e:                    # a reset mid-file resumes where it stopped
            print(f"  retry {i + 1}: {type(e).__name__}: {str(e)[:80]}", flush=True)
            time.sleep(5 + 5 * i)
    return False


def fetch_tinystories(m: dict):
    dest = os.path.join(RAW, "TinyStoriesV2-GPT4-train.txt")
    if not os.path.exists(dest) or os.path.getsize(dest) < 2_000_000_000:
        print("[fetch] TinyStories train split (~2.2 GB) ...", flush=True)
        _get(TINYSTORIES_URL, dest)
    m["sources"]["tinystories"] = {"file": os.path.basename(dest), "bytes": os.path.getsize(dest), "url": TINYSTORIES_URL,
                                   "license": "CDLA-Sharing-1.0 (dataset card roneneldan/TinyStories, checked live 2026-10-14)",
                                   "attribution": "TinyStories (Ronen Eldan & Yuanzhi Li, 2023)", "fetched": time.strftime("%Y-%m-%dT%H:%M:%S")}


def fetch_simplewiki(m: dict):
    dest = os.path.join(RAW, "simplewiki.parquet")
    if not os.path.exists(dest):
        print("[fetch] Simple English Wikipedia (20231101.simple parquet, ~150 MB) ...", flush=True)
        _get(SIMPLEWIKI_URL, dest)
    m["sources"]["simplewiki"] = {"file": os.path.basename(dest), "bytes": os.path.getsize(dest), "url": SIMPLEWIKI_URL,
                                  "license": "CC BY-SA 3.0 + GFDL (dataset card wikimedia/wikipedia, checked live 2026-10-14)",
                                  "attribution": "Wikipedia contributors, Simple English Wikipedia, dump 2023-11-01 (wikimedia/wikipedia on Hugging Face)",
                                  "fetched": time.strftime("%Y-%m-%dT%H:%M:%S")}


def fetch_gutenberg(m: dict, pause: float = 1.5):
    gdir = os.path.join(RAW, "gutenberg")
    os.makedirs(gdir, exist_ok=True)
    seen = set()
    for gid in GUTENBERG_IDS:
        if gid in seen or str(gid) in m["gutenberg"]:
            continue
        seen.add(gid)
        dest = os.path.join(gdir, f"pg{gid}.txt")
        if not os.path.exists(dest):
            ok = _get(GUTENBERG_URL.format(id=gid), dest, resume=False, tries=3)
            time.sleep(pause)
            if not ok:
                m["gutenberg"][str(gid)] = {"kept": False, "why": "fetch failed"}
                continue
        raw = open(dest, encoding="utf-8", errors="replace").read()
        head = " ".join(raw[:4000].split())          # the licence line wraps mid-phrase in the file
        title = next((ln.split(" of ", 1)[1].strip().rstrip(".,") for ln in raw.splitlines()[:5]
                      if ln.lower().startswith("the project gutenberg ebook of")), f"pg{gid}")
        kept = GUTENBERG_OK in head and "*** START OF" in raw and "copyrighted project gutenberg" not in head.lower()
        m["gutenberg"][str(gid)] = {"kept": kept, "title": title, "bytes": len(raw),
                                    "why": "header: public domain, no restrictions" if kept else "header lacks the no-restrictions line (not used)"}
        print(f"[fetch] pg{gid} {'KEEP' if kept else 'skip'} {title[:60]} ({len(raw) // 1000} KB)", flush=True)
        if not kept:
            os.remove(dest)
    m["sources"]["gutenberg"] = {"dir": "gutenberg", "books": sum(1 for v in m["gutenberg"].values() if v.get("kept")),
                                 "license": "public domain in the US; each kept file's own header carries the no-restrictions line (checked at fetch)",
                                 "note": "Project Gutenberg header/footer/licence text is stripped before training; the trademark text is not trained on",
                                 "fetched": time.strftime("%Y-%m-%dT%H:%M:%S")}


def fetch_fineweb_edu(m: dict):
    files = []
    for name in FINEWEB_EDU_FILES:
        dest = os.path.join(RAW, f"fineweb_edu_{name}")
        if not os.path.exists(dest) or os.path.getsize(dest) < 2_000_000_000:
            print(f"[fetch] FineWeb-Edu sample/10BT/{name} (~2.15 GB) ...", flush=True)
            _get(FINEWEB_EDU_URL.format(name=name), dest)
        files.append({"file": os.path.basename(dest), "bytes": os.path.getsize(dest)})
    m["sources"]["fineweb_edu"] = {"files": files, "url": FINEWEB_EDU_URL.format(name="..."),
                                   "license": "ODC-By 1.0 (dataset card HuggingFaceFW/fineweb-edu, checked live 2026-10-15); "
                                              "also subject to the Common Crawl Terms of Use",
                                   "attribution": "FineWeb-Edu (Hugging Face, Lozhkov et al. 2024), built from Common Crawl",
                                   "fetched": time.strftime("%Y-%m-%dT%H:%M:%S")}


def fetch_dialogue(m: dict):
    for key, url, fn, lic, attr in [
        ("oasst1", OASST1_URL, "oasst1_ready.trees.jsonl.gz", "Apache-2.0 (dataset card OpenAssistant/oasst1, checked live 2026-10-15)",
         "OpenAssistant Conversations (oasst1), LAION / OpenAssistant contributors"),
        ("dolly15k", DOLLY_URL, "databricks-dolly-15k.jsonl", "CC BY-SA 3.0 (dataset card databricks/databricks-dolly-15k, checked live 2026-10-15)",
         "databricks-dolly-15k, Databricks employees (CC BY-SA 3.0)")]:
        dest = os.path.join(RAW, fn)
        if not os.path.exists(dest):
            print(f"[fetch] {key} ...", flush=True)
            _get(url, dest, resume=False)
        m["sources"][key] = {"file": fn, "bytes": os.path.getsize(dest), "url": url, "license": lic, "attribution": attr,
                             "fetched": time.strftime("%Y-%m-%dT%H:%M:%S")}


def main(argv=None):
    what = (argv or sys.argv[1:]) or ["tinystories", "simplewiki", "gutenberg", "fineweb_edu", "dialogue"]
    os.makedirs(RAW, exist_ok=True)
    m = _manifest()
    for w in what:
        {"tinystories": fetch_tinystories, "simplewiki": fetch_simplewiki, "gutenberg": fetch_gutenberg,
         "fineweb_edu": fetch_fineweb_edu, "dialogue": fetch_dialogue}[w](m)
        _save(m)
    print(json.dumps(m["sources"], indent=1))


if __name__ == "__main__":
    main()
