"""Growth - the brain swap. Core v2 (~0.82M, word parts) -> core v3 (~12M, sentences).

Jaron (10/14 7:02 PM): "school level 2: real sentences ... language first, then
everything grows." And: carry forward what it already knows - it is the same
individual, growing. So:

- the new core goes to school on the OLD curriculum (level 1, same seed, same
  held-out split, so E1 stays cold) + level 2 (curriculum2.py) + its own
  RECORD (every fact it was taught and everything it said, from memory.db -
  the same mix consolidate.py trains on);
- every token the old core knew is in the new vocabulary (nothing it met
  becomes unknown);
- memory.db and its hash chain are not touched by school; the swap adds one
  ledger row: kind "growth", "growth: core v2 -> v3", with params / vocab /
  exams before and after; the generation counter keeps counting (the new
  school checkpoint = old generation + 1);
- the last v2 checkpoint is kept beside the archive as results/checkpoints/
  core_v2_gen_NNNN.pt (outside the gen_N archive policy: a small keepsake).

Two steps, so the heartbeat keeps beating while the new core is in school
(the ledger is one hash chain; two processes appending at once could break it):

    python -m sapling.grow school      # GPU: ~15-25 min; writes results/core_v3_staged.pt, no ledger writes
    python -m sapling.heartbeat stop   # then, when ready to swap brains:
    python -m sapling.grow swap        # before-exams on v2, install v3, "growth" ledger row, after-exams
    python -m sapling.heartbeat start

    python -m sapling.grow --smoke     # CPU, under a minute, temp dir: proves the whole pipeline
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import tempfile
import time

import torch

from .consolidate import record_sentences
from .curriculum import FUNCTION_WORDS, LESSONS, SLOTS, build_corpus, corpus_hash
from .curriculum2 import LESSONS2, build_corpus2
from .exams import run_exams
from .memory import MemoryStore
from .model import V3_CONFIG, SaplingCore, autocast_ctx
from .train import CHECKPOINT, CHECKPOINT_DIR, RESULTS_DIR, TRAIN_LOG, Tokenizer, get_batch, load_checkpoint, \
    save_checkpoint, sentences_to_tensor

STAGED = os.path.join(RESULTS_DIR, "core_v3_staged.pt")
ORDER = LESSONS + LESSONS2 + ["record", "all"]
# steps per stage on the GPU (the CPU smoke test uses 3 each)
STEPS = {"roots": 300, "affixes": 300, "composition": 600, "sentences": 300, "recall": 600,
         "simple": 500, "grammar": 1500, "paragraphs": 600, "comprehension": 1500,
         "stories": 4000, "record": 400, "all": 1500}
BATCH = 48
LR = 5e-4


def _tokens(sents: list[str]) -> int:
    return sum(len(s.split()) for s in sents)


def _sample_to_tokens(sents: list[str], target: int, rng) -> list[str]:
    if not sents or target <= 0:
        return []
    out, tot = [], 0
    while tot < target:
        s = rng.choice(sents)
        out.append(s)
        tot += len(s.split())
    return out


def stage_mix(groups: dict[str, list[str]], order: list[str], i: int, rng) -> list[str]:
    """School mix for stage i: the current lesson in full + the earlier lessons
    sampled to half of its tokens between them (so the first lessons are never
    drowned by the stories). The final 'all' pass: stories 35%, the rest equal."""
    name = order[i]
    if name == "all":
        names = [n for n in order[:i] if groups.get(n)]
        total = max(200_000, _tokens(groups.get("stories", [])) // 4)
        mix = []
        rest = [n for n in names if n != "stories"]
        for n in names:
            share = 0.35 if n == "stories" else 0.65 / max(1, len(rest))
            mix += _sample_to_tokens(groups[n], int(total * share), rng)
        rng.shuffle(mix)
        return mix
    cur = list(groups.get(name, []))
    earlier = [n for n in order[:i] if groups.get(n)]
    mix = list(cur)
    if earlier and cur:
        per = int(0.5 * _tokens(cur) / len(earlier))
        for n in earlier:
            mix += _sample_to_tokens(groups[n], max(per, 1), rng)
    rng.shuffle(mix)
    return mix


def build_groups(seed: int, store, stories: bool = True, max_stories: int = 12000, sizes: dict | None = None):
    c1 = build_corpus(seed)
    c2 = build_corpus2(seed, stories=stories, max_stories=max_stories, sizes=sizes)
    groups = {n: c1["lessons"][n] for n in LESSONS}
    groups.update({n: c2["lessons"][n] for n in LESSONS2})
    groups["record"] = record_sentences(store) if store is not None else []
    return groups, c1, c2


def build_vocab_v3(groups: dict, old_itos: list[str] | None) -> Tokenizer:
    toks = set(FUNCTION_WORDS) | set(SLOTS) | {"<pad>", "<bos>", "<eos>", "<unk>"}
    for sents in groups.values():
        for s in sents:
            toks.update(s.split())
    if old_itos:
        toks.update(old_itos)            # everything the old core knew stays known
    itos = sorted(toks)
    return Tokenizer({t: i for i, t in enumerate(itos)}, itos)


def _old_info(path: str, device: str) -> dict | None:
    if not os.path.exists(path):
        return None
    om, ot, oc = load_checkpoint(device=device, path=path)
    info = {"generation": int(oc.get("generation", 0)), "params": om.n_params(), "vocab_size": len(ot.itos),
            "config": om.config(), "core_version": int(oc.get("core_version", 2)), "path": path,
            "corpus_sha": oc.get("corpus_sha"), "itos": list(ot.itos)}
    del om
    return info


# -- step 1: school (no ledger writes; the heartbeat may keep running) ------------------------
def school(store: MemoryStore | None = None, seed: int = 7337, device: str | None = None,
           config: dict | None = None, steps: dict | None = None, batch: int = BATCH, lr: float = LR,
           stories: bool = True, max_stories: int = 12000, sizes: dict | None = None,
           old_path: str = CHECKPOINT, staged: str = STAGED, verbose: bool = True) -> dict:
    t_all = time.time()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = {**V3_CONFIG, **(config or {})}
    steps = {**STEPS, **(steps or {})}
    store = store or MemoryStore()
    rng = random.Random(seed)
    torch.manual_seed(seed)

    old = _old_info(old_path, device)
    if verbose and old:
        print(f"[grow] old core: v{old['core_version']} gen {old['generation']} params={old['params']:,} vocab={old['vocab_size']}")
    groups, c1, c2 = build_groups(seed, store, stories=stories, max_stories=max_stories, sizes=sizes)
    tok = build_vocab_v3(groups, old["itos"] if old else None)
    model = SaplingCore(len(tok.itos), **config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    if verbose:
        print(f"[grow] new core: params={model.n_params():,} vocab={len(tok.itos)} config={config} device={device}")
        print(f"[grow] stories: {json.dumps(c2['story_stats'])[:300]}")
        print(f"[grow] record: {len(groups['record'])} sentences from memory.db")
    run = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": "growth school", "seed": seed, "device": device,
           "params": model.n_params(), "vocab": len(tok.itos), "config": config, "corpus_sha": corpus_hash(c1),
           "story_stats": c2["story_stats"], "record_sentences": len(groups["record"]), "stages": []}
    order = [n for n in ORDER if groups.get(n) or n == "all"]
    model.train()
    ctx = autocast_ctx(device)
    for i, name in enumerate(order):
        mix = stage_mix(groups, order, i, rng)
        if not mix:
            continue
        data = sentences_to_tensor(mix, tok, model.block_size).to(device)
        if len(data) < model.block_size + 2:
            continue
        n_steps = int(steps.get(name, 300))
        t0, losses = time.time(), []
        for step in range(n_steps):
            x, y = get_batch(data, model.block_size, batch, rng)
            with ctx:
                _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        row = {"stage": name, "examples": len(mix), "tokens": int(data.numel()), "steps": n_steps,
               "loss_first": round(losses[0], 4), "loss_last": round(sum(losses[-20:]) / min(20, len(losses)), 4),
               "seconds": round(time.time() - t0, 1)}
        run["stages"].append(row)
        if verbose:
            print(f"[grow] stage={name:<14} examples={row['examples']:<6} tokens={row['tokens']:<8} "
                  f"loss {row['loss_first']:.3f} -> {row['loss_last']:.3f} ({row['seconds']}s)", flush=True)
    model.eval()
    run["seconds"] = round(time.time() - t_all, 1)
    gen = (old["generation"] + 1) if old else 0
    meta = {"seed": seed, "corpus_sha": corpus_hash(c1), "generation": gen, "core_version": 3, "staged": True,
            "note": f"growth: core v2 -> v3 (school level 2); grown from gen {old['generation'] if old else '-'}",
            "grown_from": {k: old[k] for k in ("generation", "params", "vocab_size", "config", "core_version")} if old else None,
            "story_stats": c2["story_stats"], "school": run}
    save_checkpoint(model, tok, meta, path=staged)
    if staged == STAGED:
        os.makedirs(os.path.dirname(TRAIN_LOG), exist_ok=True)
        with open(TRAIN_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(run, default=str) + "\n")
    if verbose:
        print(f"[grow] school done in {run['seconds']}s -> {staged} (staged; run `swap` to install)")
    return {"model": model, "tok": tok, "meta": meta, "run": run, "staged": staged}


# -- step 2: the swap (heartbeat stopped; one ledger row) --------------------------------------
def swap(store: MemoryStore | None = None, seed: int = 7337, device: str | None = None, staged: str = STAGED,
         checkpoint: str = CHECKPOINT, checkpoint_dir: str = CHECKPOINT_DIR, exams: bool = True,
         verbose: bool = True) -> dict:
    t0 = time.time()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    store = store or MemoryStore()
    new_model, new_tok, new_ck = load_checkpoint(device=device, path=staged)
    user_facts = [f["content"] for f in store.facts() if f["provenance"] == "user"]

    old = None
    if os.path.exists(checkpoint):
        om, ot, oc = load_checkpoint(device=device, path=checkpoint)
        old = {"generation": int(oc.get("generation", 0)), "params": om.n_params(), "vocab_size": len(ot.itos),
               "config": om.config(), "core_version": int(oc.get("core_version", 2))}
        if exams:
            card = run_exams(seed=seed, verbose=verbose, model=om, tok=ot, store=store,
                             meta={"generation": old["generation"], "corpus_sha": oc.get("corpus_sha")},
                             consolidated_facts=user_facts,
                             trigger=f"before growth (core v{old['core_version']} gen {old['generation']})")
            old["exams"] = [{k: e[k] for k in ("exam", "score", "accuracy", "verdict")} for e in card["exams"]]
        os.makedirs(checkpoint_dir, exist_ok=True)
        keep = os.path.join(checkpoint_dir, f"core_v{old['core_version']}_gen_{old['generation']:04d}.pt")
        shutil.copy(checkpoint, keep)
        old["kept_as"] = os.path.relpath(keep, os.path.dirname(checkpoint_dir))
        del om

    # the generation keeps counting from where the old core left off (it may have consolidated since school)
    gen = (old["generation"] + 1) if old else int(new_ck.get("generation", 0))
    meta = {k: new_ck.get(k) for k in ("seed", "corpus_sha", "core_version", "note", "grown_from", "story_stats", "school")}
    meta["generation"] = gen
    meta["staged"] = False
    os.makedirs(checkpoint_dir, exist_ok=True)
    kept = save_checkpoint(new_model, new_tok, meta, path=os.path.join(checkpoint_dir, f"gen_{gen:04d}.pt"))
    save_checkpoint(new_model, new_tok, meta, path=checkpoint)

    event = {"event": "growth: core v2 -> v3",
             "from": old,
             "to": {"core_version": 3, "generation": gen, "params": new_model.n_params(), "vocab_size": len(new_tok.itos),
                    "config": new_model.config(), "checkpoint": os.path.relpath(kept, os.path.dirname(checkpoint_dir))},
             "school": (new_ck.get("school") or {}).get("stages"), "school_seconds": (new_ck.get("school") or {}).get("seconds"),
             "story_stats": new_ck.get("story_stats"), "record_sentences": (new_ck.get("school") or {}).get("record_sentences"),
             "device": device, "seed": seed}
    if exams:
        card = run_exams(seed=seed, verbose=verbose, model=new_model, tok=new_tok, meta=meta, store=store,
                         consolidated_facts=user_facts, trigger=f"growth: core v2 -> v3 (gen {gen})")
        event["to"]["exams"] = [{k: e[k] for k in ("exam", "score", "accuracy", "verdict")} for e in card["exams"]]
    event["seconds"] = round(time.time() - t0, 1)
    # one row for the whole swap: before and after on the same line of the chain
    store.meter("growth", event)
    store.remember(f"growth: core v2 -> v3 at generation {gen} ({new_model.n_params():,} parameters, vocab {len(new_tok.itos)})",
                   provenance="self", state="confirmed")
    if verbose:
        print(f"[grow] swapped: gen {gen}, {new_model.n_params():,} params, vocab {len(new_tok.itos)} -> {checkpoint}")
    return {"model": new_model, "tok": new_tok, "meta": meta, "event": event}


def grow(store: MemoryStore | None = None, **kw) -> dict:
    """school + swap in one go (used by the smoke test and when the heartbeat is already stopped)."""
    store = store or MemoryStore()
    swap_kw = {k: kw.pop(k) for k in ("checkpoint", "checkpoint_dir", "exams") if k in kw}
    s = school(store=store, **kw)
    out = swap(store=store, seed=kw.get("seed", 7337), device=kw.get("device"), staged=s["staged"],
               verbose=kw.get("verbose", True), **swap_kw)
    out["run"] = s["run"]
    return out


def smoke(verbose: bool = True) -> dict:
    """CPU, under a minute: tiny core, tiny lessons, temp dir, temp memory. Proves every
    stage, the vocab carry-over, the ledger row and E1-E8 run end to end."""
    tmp = tempfile.mkdtemp(prefix="sapling_smoke_")
    store = MemoryStore(os.path.join(tmp, "m.db"))
    store.remember("the word hello means a greeting", provenance="user")
    store.remember("the word glorp means happy", provenance="user")
    from . import exams as E
    old_card, old_wd = E.REPORT_CARD, E.WRITING_DIR
    E.REPORT_CARD = os.path.join(tmp, "card.json")
    E.WRITING_DIR = os.path.join(tmp, "writing")
    ck = os.path.join(tmp, "core.pt")
    if os.path.exists(CHECKPOINT):
        shutil.copy(CHECKPOINT, ck)          # a copy of the live core plays "old"
    try:
        out = grow(store=store, device="cpu", config=dict(d_model=64, n_head=4, n_layer=2, block_size=64),
                   steps={k: 3 for k in STEPS}, batch=8, max_stories=150,
                   sizes={"simple": 60, "grammar": 80, "paragraphs": 40, "comprehension": 80},
                   old_path=ck, staged=os.path.join(tmp, "staged.pt"),
                   checkpoint=ck, checkpoint_dir=os.path.join(tmp, "ck"), verbose=verbose)
    finally:
        E.REPORT_CARD, E.WRITING_DIR = old_card, old_wd
    out["tmp"] = tmp
    out["store"] = store
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="both", choices=["school", "swap", "both"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=7337)
    ap.add_argument("--no-stories", action="store_true")
    ap.add_argument("--max-stories", type=int, default=12000)
    args = ap.parse_args(argv)
    if args.smoke:
        out = smoke()
        print("smoke ok:", out["tmp"], json.dumps(out["event"]["to"], default=str)[:300])
        return
    if args.cmd == "school":
        school(seed=args.seed, stories=not args.no_stories, max_stories=args.max_stories)
    elif args.cmd == "swap":
        swap(seed=args.seed)
    else:
        grow(seed=args.seed, stories=not args.no_stories, max_stories=args.max_stories)


if __name__ == "__main__":
    main()
