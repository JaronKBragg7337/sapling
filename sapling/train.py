"""Train the Sapling core on the staged morphology curriculum.

School-style: lessons in order, loss reported per lesson, checkpoint saved,
and every run appends to results/train_log.jsonl. Deterministic under --seed.

Usage (RTX 4060 or CPU — device is auto-detected):
    python -m sapling.train --steps-per-lesson 300 --seed 7337
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch

from .curriculum import LESSONS, build_corpus, build_vocab, corpus_hash
from .model import SaplingCore

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
CHECKPOINT = os.path.join(RESULTS_DIR, "core_checkpoint.pt")
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
TRAIN_LOG = os.path.join(RESULTS_DIR, "train_log.jsonl")


class Tokenizer:
    """Whitespace/morpheme tokenizer. The vocabulary IS the curriculum."""

    def __init__(self, stoi: dict, itos: list[str]):
        self.stoi = stoi
        self.itos = itos

    def encode(self, text: str, bos: bool = True, eos: bool = True) -> list[int]:
        ids = [self.stoi.get(t, self.stoi["<unk>"]) for t in text.split()]
        if bos:
            ids = [self.stoi["<bos>"]] + ids
        if eos:
            ids = ids + [self.stoi["<eos>"]]
        return ids

    def unknown_tokens(self, text: str) -> list[str]:
        seen, out = set(), []
        for t in text.split():
            if t not in self.stoi and t not in seen:
                seen.add(t)
                out.append(t)
        return out

    def add_tokens(self, tokens: list[str]) -> int:
        """Grow the vocabulary in place; returns how many were new."""
        n = 0
        for t in tokens:
            if t not in self.stoi:
                self.stoi[t] = len(self.itos)
                self.itos.append(t)
                n += 1
        return n

    def decode(self, ids: list[int]) -> str:
        toks = []
        for i in ids:
            t = self.itos[i]
            if t in ("<bos>", "<pad>"):
                continue
            if t == "<eos>":
                break
            toks.append(t)
        return " ".join(toks)


def sentences_to_tensor(sentences: list[str], tok: Tokenizer, block_size: int) -> torch.Tensor:
    ids: list[int] = []
    for s in sentences:
        ids.extend(tok.encode(s))
    ids = ids[: (len(ids) // block_size) * block_size]
    return torch.tensor(ids, dtype=torch.long)


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, rng: random.Random):
    hi = len(data) - block_size - 1
    starts = [rng.randrange(0, hi) for _ in range(batch_size)]
    x = torch.stack([data[s: s + block_size] for s in starts])
    y = torch.stack([data[s + 1: s + block_size + 1] for s in starts])
    return x, y


def train(
    steps_per_lesson: int = 500,
    batch_size: int = 32,
    lr: float = 3e-3,
    block_size: int = 64,
    d_model: int = 128,
    n_head: int = 4,
    n_layer: int = 4,
    seed: int = 7337,
    device: str | None = None,
    verbose: bool = True,
) -> dict:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = random.Random(seed)
    torch.manual_seed(seed)

    corpus = build_corpus(seed)
    vocab = build_vocab(corpus)
    tok = Tokenizer(vocab["stoi"], vocab["itos"])

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SaplingCore(len(vocab["itos"]), d_model, n_head, n_layer, block_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    if verbose:
        print(f"[train] device={device} params={model.n_params():,} "
              f"vocab={len(vocab['itos'])} corpus_sha={corpus_hash(corpus)}")

    run_record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": seed, "device": device, "params": model.n_params(),
        "corpus_sha": corpus_hash(corpus), "lessons": [],
    }

    # Staged schooling: each lesson adds its data to the training mix
    # (no catastrophic-forgetting tricks yet — that is a next step).
    mix: list[str] = []
    for lesson in LESSONS:
        mix += corpus["lessons"][lesson]
        rng.shuffle(mix)
        data = sentences_to_tensor(mix, tok, block_size).to(device)
        t0 = time.time()
        losses = []
        for step in range(steps_per_lesson):
            x, y = get_batch(data, block_size, batch_size, rng)
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        row = {
            "lesson": lesson,
            "examples": len(mix),
            "tokens": int(data.numel()),
            "loss_first": round(losses[0], 4),
            "loss_last": round(sum(losses[-20:]) / 20, 4),
            "seconds": round(time.time() - t0, 1),
        }
        run_record["lessons"].append(row)
        if verbose:
            print(f"[train] lesson={lesson:<12} examples={row['examples']:<5} "
                  f"loss {row['loss_first']:.3f} -> {row['loss_last']:.3f} "
                  f"({row['seconds']}s)")

    save_checkpoint(model, tok, {"seed": seed, "corpus_sha": corpus_hash(corpus),
                                 "generation": 0, "note": "school (from scratch)"})
    with open(TRAIN_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(run_record) + "\n")
    if verbose:
        print(f"[train] checkpoint -> {CHECKPOINT}")
    return {"model": model, "tok": tok, "run_record": run_record}


def save_checkpoint(model, tok: Tokenizer, meta: dict, path: str = CHECKPOINT) -> str:
    """Write the live model + its (possibly grown) vocabulary. The model config
    is taken from the model itself so a grown vocabulary reloads correctly."""
    payload = {
        "model_state": model.state_dict(),
        "config": model.config(),
        "params": model.n_params(),
        "vocab": {"stoi": dict(tok.stoi), "itos": list(tok.itos)},
        **meta,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)
    return path


def load_checkpoint(device: str | None = None, path: str = CHECKPOINT):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = SaplingCore(**cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    tok = Tokenizer(ckpt["vocab"]["stoi"], ckpt["vocab"]["itos"])
    return model, tok, ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps-per-lesson", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=7337)
    args = ap.parse_args()
    train(steps_per_lesson=args.steps_per_lesson, batch_size=args.batch_size,
          lr=args.lr, seed=args.seed)


if __name__ == "__main__":
    main()
