"""Ten fixed chat prompts, answered through the real chat path (session.Session), saved to results/chat_samples/.

Claude as teacher, 10/15: "show the change" - the same ten prompts before and after each stage, so anyone can read
what changed in how Sapling talks. Runs on a COPY of memory.db (the chat path writes metering rows; a sample run
must not add rows to the real chain), with whatever core file is given (default: the live one).

    python tools/chat_samples.py --label "v3.2 gen 287 (before)" [--core results/checkpoints/gen_0287.pt]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sapling.memory import MemoryStore  # noqa: E402
from sapling.session import Brain, Session  # noqa: E402
from sapling.train import CHECKPOINT, RESULTS_DIR  # noqa: E402

OUT_DIR = os.path.join(RESULTS_DIR, "chat_samples")
PROMPTS = [
    "hello! who are you?",
    "what is your name?",
    "what is the sun?",
    "why is the sky blue?",
    "tell me a short story about a dog.",
    "what does respect mean?",
    "how do plants make their food?",
    "can you help me write a short note to thank a friend?",
    "what is a roof made of?",
    "what would you like to learn next?",
]
# one short conversation, so the page can show whether it keeps the thread (v4: the chat box keeps history)
CONVERSATION = ["hi, i am jaron.", "what is my name?", "what do you know about rain?", "why does it fall down?"]


def run(core: str, label: str, memory: str | None = None) -> dict:
    if os.environ.get("SAPLING_DEVICE", "cuda") != "cpu":
        from sapling.reading import wait_until_gpu_free
        wait_until_gpu_free(who="chat_samples")
    tmp = tempfile.mkdtemp(prefix="sapling_samples_")
    src = memory or os.path.join(RESULTS_DIR, "memory.db")
    dst = os.path.join(tmp, "memory.db")
    if os.path.exists(src):
        # the WAL holds the newest rows; a sqlite backup copies a consistent snapshot
        import sqlite3
        with sqlite3.connect(src) as a, sqlite3.connect(dst) as b:
            a.backup(b)
    store = MemoryStore(dst)
    out = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "label": label, "core": os.path.relpath(core, ROOT), "single": [], "conversation": []}
    s = Session(store=store, brain=Brain(store=store, path=core))
    out["params"] = s.brain.model.n_params()
    out["generation"] = s.brain.generation
    for p in PROMPTS:
        t0 = time.time()
        one = Session(store=store, brain=s.brain)         # each fixed prompt is its own first turn
        out["single"].append({"prompt": p, "answer": one.ask(p), "seconds": round(time.time() - t0, 2)})
    s2 = Session(store=store, brain=s.brain)
    for p in CONVERSATION:
        out["conversation"].append({"prompt": p, "answer": s2.ask(p)})
    store.close() if hasattr(store, "close") else None
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    fn = os.path.join(OUT_DIR, f"{out['ts'].replace(':', '')}_{''.join(c if c.isalnum() else '_' for c in label)[:40]}.json")
    json.dump(out, open(fn, "w", encoding="utf-8"), indent=1)
    out["file"] = os.path.relpath(fn, ROOT)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", default=CHECKPOINT)
    ap.add_argument("--label", required=True)
    a = ap.parse_args(argv)
    r = run(a.core, a.label)
    for row in r["single"] + [{"prompt": "(conv) " + c["prompt"], "answer": c["answer"]} for c in r["conversation"]]:
        print(f"> {row['prompt']}\n  {row['answer']}")
    print("->", r["file"])


if __name__ == "__main__":
    main()
