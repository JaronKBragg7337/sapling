"""Deterministic held-out multi-turn evaluation with no retrieval or gold answers in history."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sapling.talk import encode_turns  # noqa: E402
from sapling.text import normalize  # noqa: E402
from sapling.train import load_checkpoint  # noqa: E402

HELDOUT = os.path.join(ROOT, "data", "talk", "conversation_heldout.jsonl")
OUT_DIR = os.path.join(ROOT, "results", "conversation")


def hit(answer: str, expected: str, kind: str = "") -> bool:
    expected = normalize(expected)
    if not expected:
        return False
    answer = normalize(answer)
    if kind == "name":
        prefixes = ("your name is ", "you are ", "you're ")
        return answer == expected or any(answer.startswith(p + expected) and
                                        answer[len(p + expected):].strip() in ("", ".", "!") for p in prefixes)
    return f" {expected} " in f" {answer} "


@torch.no_grad()
def evaluate(core: str, label: str, heldout: str = HELDOUT, max_new: int = 48) -> dict:
    model, tok, ck = load_checkpoint(path=core)
    device = next(model.parameters()).device
    model.eval()
    rows = [json.loads(line) for line in open(heldout, encoding="utf-8") if line.strip()]
    chats = []
    for row in rows:
        history, generated = [], []
        for turn, (prompt, _) in enumerate(row["turns"]):
            turns = history[-5:] + [[prompt, ""]]
            while True:
                ids, _ = encode_turns(tok, turns, answer_last=False)
                if len(ids) <= model.block_size - max_new or len(turns) == 1:
                    break
                turns = turns[1:]
            x = torch.tensor([ids], device=device)
            out = model.generate(x, max_new=max_new, eos_id=tok.stoi["<eos>"], temperature=0.0)
            answer = tok.decode(out[0, len(ids):].tolist()).replace("<eos>", "").strip()
            generated.append({"prompt": prompt, "answer": answer, "gold_not_used": True})
            history.append([prompt, answer])
        ok = hit(generated[-1]["answer"], row["expected"], row["kind"])
        chats.append({"id": row["id"], "kind": row["kind"], "expected": row["expected"], "pass": ok,
                      "turns": generated})
    by = {}
    for chat in chats:
        score = by.setdefault(chat["kind"], [0, 0])
        score[0] += int(chat["pass"])
        score[1] += 1
    names = [c for c in chats if c["kind"] == "name"]
    name_ok = sum(c["pass"] for c in names)
    sample_kinds = ["name", "object", "followup"]
    samples = [next(c for c in chats if c["kind"] == k) for k in sample_kinds]
    result = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "label": label, "core": os.path.relpath(core, ROOT),
        "generation": ck.get("generation"), "params": model.n_params(), "method": "greedy core-only; generated history; final answer scored",
        "score": f"{sum(c['pass'] for c in chats)}/{len(chats)}",
        "by_kind": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by.items())},
        "name_thread": {"score": f"{name_ok}/{len(names)}", "pass_min": 10, "pass": name_ok >= 10},
        "acorn_candidate": name_ok >= 10, "samples": samples, "chats": chats,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    fn = os.path.join(OUT_DIR, f"{result['ts'].replace(':', '')}_{''.join(c if c.isalnum() else '_' for c in label)[:50]}.json")
    json.dump(result, open(fn, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    result["file"] = os.path.relpath(fn, ROOT)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--heldout", default=HELDOUT)
    ap.add_argument("--max-new", type=int, default=48)
    a = ap.parse_args(argv)
    r = evaluate(a.core, a.label, a.heldout, a.max_new)
    print(json.dumps({k: v for k, v in r.items() if k != "chats"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
