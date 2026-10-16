"""Growth - the brain grows. v2 (~0.82M, word parts) -> v3 (~12M, sentences) -> v3.1 (reading).

Jaron (10/14 7:02 PM): "school level 2: real sentences ... language first, then
everything grows." And: carry forward what it already knows - it is the same
individual, growing. So:

- v2 -> v3 (base "fresh"): a new core goes to school on the OLD curriculum
  (level 1, same seed, same held-out split, so E1 stays cold) + level 2
  (curriculum2.py) + its own RECORD (every fact it was taught and everything it
  said, from memory.db - the same mix consolidate.py trains on);
- v3 -> v3.1 (base "live", 10/14 night, DESIGN-CHOICES D17): the SAME weights
  keep going. The vocabulary gets new rows (whole words for adult text + word
  pieces + the forced splits of the school's composed words, pieces.py), the
  context gets new position rows (256 -> 512, model.grow_context), and school
  is level 3 (curriculum3.py: copy, bridge, prose, docs, longread) with every
  earlier lesson rehearsed in each stage;
- every token the old core knew is in the new vocabulary (nothing it met
  becomes unknown);
- memory.db and its hash chain are not touched by school; the swap adds one
  ledger row: kind "growth", "growth: core vX -> vY", with params / vocab /
  exams before and after; the generation counter keeps counting (the new
  school checkpoint = old generation + 1);
- the last old checkpoint is kept beside the archive as results/checkpoints/
  core_vX_gen_NNNN.pt (outside the gen_N archive policy: a small keepsake).

Two steps, so the heartbeat keeps beating while the new core is in school
(the ledger is one hash chain; two processes appending at once could break it):

    python -m sapling.grow school      # GPU: writes results/core_staged.pt, no ledger writes
    python -m sapling.heartbeat stop   # then, when ready to swap brains:
    python -m sapling.grow swap        # before-exams on the old core, install, "growth" ledger row, after-exams
    python -m sapling.heartbeat start

    python -m sapling.grow --smoke     # CPU, ~1-2 min, temp dir: both growths end to end on tiny cores
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import tempfile
import time
from collections import Counter

import torch

from .consolidate import record_sentences
from .curriculum import FUNCTION_WORDS, LESSONS, ROOTS, SLOTS, SUFFIXES, build_corpus, corpus_hash
from .curriculum2 import LESSONS2, build_corpus2
from .curriculum3 import DOCS_DIR, LESSONS3, build_corpus3, composed_words, load_docs
from .exams import run_exams
from .memory import MemoryStore
from .model import V3_CONFIG, V31_CONFIG, SaplingCore, autocast_ctx
from .pieces import MARK, base_pieces, learn_pieces
from .train import CHECKPOINT, CHECKPOINT_DIR, RESULTS_DIR, TRAIN_LOG, Tokenizer, get_batch, load_checkpoint, \
    save_checkpoint, sentences_to_tensor

STAGED = os.path.join(RESULTS_DIR, "core_staged.pt")
ORDER = LESSONS + LESSONS2 + LESSONS3 + ["record", "all"]
# steps per stage on the GPU for a FRESH core (the CPU smoke test uses 3 each)
STEPS = {"roots": 300, "affixes": 300, "composition": 600, "sentences": 300, "recall": 600,
         "simple": 500, "grammar": 1500, "paragraphs": 600, "comprehension": 1500,
         "stories": 4000, "copy": 1000, "bridge": 400, "prose": 2500, "docs": 2000, "longread": 1200,
         "record": 400, "all": 1500}
# steps per stage when growing the LIVE core in place (it already knows levels 1-2; those are rehearsed inside
# every stage, so only the new lessons, the record and one mixed pass run as stages)
STEPS_LIVE = {"copy": 800, "bridge": 300, "prose": 1200, "docs": 1500, "longread": 1000, "record": 300, "all": 1500}
BATCH = 32            # v3.1: 32 at 512 tokens (48 left no room on the 8 GB card beside the running heartbeat)
LR = 5e-4
LR_LIVE = 2e-4             # gentler on weights that already know things (D14 lesson)
STORY_WORDS = 6000         # whole-word tokens from the stories (was 3,000 in v3)
DOC_WORD_MIN = 4           # a document word this common becomes a whole token; rarer ones are spelled from pieces
N_PIECES = 1500


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


def stage_mix(groups: dict[str, list[str]], order: list[str], i: int, rng, earlier: list[str] | None = None) -> list[str]:
    """School mix for stage i: the current lesson in full + the earlier lessons
    sampled to half of its tokens between them (so the first lessons are never
    drowned by the stories). The final 'all' pass: stories + prose 35%, the rest equal.
    `earlier` overrides which lessons count as earlier (in-place growth: every old lesson)."""
    name = order[i]
    if name == "all":
        names = [n for n in (earlier if earlier is not None else order[:i]) if groups.get(n)]
        big = [n for n in names if n in ("stories", "prose")]
        total = max(200_000, sum(_tokens(groups[n]) for n in big) // 4)
        mix = []
        rest = [n for n in names if n not in big]
        for n in names:
            share = 0.35 / max(1, len(big)) if n in big else 0.65 / max(1, len(rest))
            mix += _sample_to_tokens(groups[n], int(total * share), rng)
        rng.shuffle(mix)
        return mix
    cur = list(groups.get(name, []))
    earlier = [n for n in (earlier if earlier is not None else order[:i]) if groups.get(n) and n != name]
    mix = list(cur)
    if earlier and cur:
        per = int(0.5 * _tokens(cur) / len(earlier))
        for n in earlier:
            mix += _sample_to_tokens(groups[n], max(per, 1), rng)
    rng.shuffle(mix)
    return mix


def build_groups(seed: int, store, stories: bool = True, max_stories: int = 20000, sizes: dict | None = None,
                 level3: bool = True, docs_dir: str = DOCS_DIR):
    c1 = build_corpus(seed)
    c2 = build_corpus2(seed, stories=stories, max_stories=max_stories, sizes=sizes,
                       vocab_cap=STORY_WORDS if level3 else 3000)   # v3.1: 6,000 story words are whole tokens (rarer: pieces)
    groups = {n: c1["lessons"][n] for n in LESSONS}
    groups.update({n: c2["lessons"][n] for n in LESSONS2})
    c3 = None
    if level3:
        story_words = Counter(t for s in c2["lessons"]["stories"] for t in s.split() if t.isalpha())
        docs = load_docs(docs_dir=docs_dir)
        doc_words = Counter()
        for d in docs.values():
            doc_words.update(d["words"])
        wide = [w for w, _ in (story_words + doc_words).most_common(4000) if w.isalpha() and len(w) > 2]
        c3 = build_corpus3(seed, docs=bool(docs), words=wide, story_sents=c2["lessons"]["stories"][:2000],
                           sizes=sizes, docs_dir=docs_dir)
        groups.update({n: c3["lessons"][n] for n in LESSONS3})
    groups["record"] = record_sentences(store) if store is not None else []
    return groups, c1, c2, c3


def build_vocab_v3(groups: dict, old_itos: list[str] | None) -> Tokenizer:
    """v3: whole words from every lesson + everything the old core knew, sorted (a fresh core)."""
    toks = set(FUNCTION_WORDS) | set(SLOTS) | {"<pad>", "<bos>", "<eos>", "<unk>"}
    for sents in groups.values():
        for s in sents:
            toks.update(s.split())
    if old_itos:
        toks.update(old_itos)            # everything the old core knew stays known
    itos = sorted(toks)
    return Tokenizer({t: i for i, t in enumerate(itos)}, itos)


def build_vocab_v31(groups: dict, old_itos: list[str] | None, story_words: int = STORY_WORDS,
                    doc_word_min: int = DOC_WORD_MIN, n_pieces: int = N_PIECES, docs_dir: str = DOCS_DIR) -> dict:
    """v3.1 (D16): the old vocabulary in its old order (ids unchanged, the live
    core keeps its rows) + whole words (lessons; the most common story words;
    document words seen >= doc_word_min times) + the school's roots/suffixes as
    continuation pieces + every letter + BPE pieces learned from the rarer
    words - and the forced splits for every composed school word."""
    splits = composed_words()
    itos = list(old_itos) if old_itos else sorted(set(FUNCTION_WORDS) | set(SLOTS) | {"<pad>", "<bos>", "<eos>", "<unk>"})
    have = set(itos)
    add: set[str] = set()
    for name, sents in groups.items():
        if name in ("stories", "prose", "docs", "longread"):
            continue
        for s in sents:
            add.update(s.split())
    sw = Counter(t for s in groups.get("stories", []) for t in s.split())
    add.update(w for w, _ in sw.most_common(story_words))
    dw = Counter()
    for d in load_docs(docs_dir=docs_dir).values():
        dw.update(d["words"])
    for s in groups.get("prose", []):
        dw.update(s.split())
    add.update(w for w, c in dw.items() if c >= doc_word_min)
    add -= set(splits)                              # composed words are spelled from their parts, always
    add -= have
    rare = Counter({w: c for w, c in (sw + dw).items() if w not in add and w not in have and w not in splits})
    pieces = base_pieces() + [MARK + r for r in ROOTS] + [MARK + s for s in SUFFIXES] + learn_pieces(rare, n_merges=n_pieces)
    for t in sorted(add) + [p for p in pieces if p not in have]:
        if t not in have:
            itos.append(t); have.add(t)
    tok = Tokenizer({t: i for i, t in enumerate(itos)}, itos, splits=splits)
    return {"tok": tok, "stats": {"old": len(old_itos or []), "whole_words_added": len(add), "pieces": len([p for p in itos if p.startswith(MARK)]),
                                  "story_words": story_words, "doc_word_min": doc_word_min, "doc_distinct": len(dw), "splits": len(splits)}}


def _old_info(path: str, device: str) -> dict | None:
    if not os.path.exists(path):
        return None
    om, ot, oc = load_checkpoint(device=device, path=path)
    info = {"generation": int(oc.get("generation", 0)), "params": om.n_params(), "vocab_size": len(ot.itos),
            "config": om.config(), "core_version": float(oc.get("core_version", 2)), "path": path,
            "corpus_sha": oc.get("corpus_sha"), "itos": list(ot.itos), "model": om, "tok": ot, "ckpt": oc}
    return info


# -- step 1: school (no ledger writes; the heartbeat may keep running) ------------------------
def school(store: MemoryStore | None = None, seed: int = 7337, device: str | None = None,
           config: dict | None = None, steps: dict | None = None, batch: int = BATCH, lr: float | None = None,
           stories: bool = True, max_stories: int = 20000, sizes: dict | None = None, base: str = "live",
           old_path: str = CHECKPOINT, staged: str = STAGED, verbose: bool = True, to_version: float = 3.1,
           n_pieces: int = N_PIECES, docs_dir: str = DOCS_DIR, label: str = "") -> dict:
    """base="fresh": a new core (v3 recipe + level 3). base="live": grow the old core in place (v3.1)."""
    t_all = time.time()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    store = store or MemoryStore()
    rng = random.Random(seed)
    torch.manual_seed(seed)

    old = _old_info(old_path, device)
    if base == "live" and not old:
        base = "fresh"
    if verbose and old:
        print(f"[grow] old core: v{old['core_version']} gen {old['generation']} params={old['params']:,} vocab={old['vocab_size']} ctx={old['config']['block_size']}")
    groups, c1, c2, c3 = build_groups(seed, store, stories=stories, max_stories=max_stories, sizes=sizes, docs_dir=docs_dir)
    if base == "live":
        config = {**old["config"], "block_size": V31_CONFIG["block_size"], **(config or {})}   # 256 -> 512
        config.pop("vocab_size", None)
        steps = {**STEPS_LIVE, **(steps or {})}
        lr = LR_LIVE if lr is None else lr
        vb = build_vocab_v31(groups, old["itos"], n_pieces=n_pieces, docs_dir=docs_dir)
        tok = vb["tok"]
        model = old["model"]
        model.grow_vocab(len(tok.itos))
        model.grow_context(int(config["block_size"]))
        order = [n for n in LESSONS3 + ["record", "all"] if groups.get(n) or n == "all"]
        earlier_all = [n for n in LESSONS + LESSONS2 + LESSONS3 + ["record"] if groups.get(n)]
    else:
        config = {**V31_CONFIG, **(config or {})}
        steps = {**STEPS, **(steps or {})}
        lr = LR if lr is None else lr
        vb = build_vocab_v31(groups, old["itos"] if old else None, n_pieces=n_pieces, docs_dir=docs_dir)
        tok = vb["tok"]
        model = SaplingCore(len(tok.itos), **config).to(device)
        order = [n for n in ORDER if groups.get(n) or n == "all"]
        earlier_all = None
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    if verbose:
        print(f"[grow] {'growing live core' if base == 'live' else 'new core'}: params={model.n_params():,} vocab={len(tok.itos)} "
              f"pieces={vb['stats']['pieces']} config={model.config()} device={device}")
        print(f"[grow] stories: {json.dumps(c2['story_stats'])[:300]}")
        if c3:
            print(f"[grow] docs: {json.dumps(c3['docs_stats'].get('docs'))[:400]}")
        print(f"[grow] record: {len(groups['record'])} sentences from memory.db")
    run = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": f"growth school ({base})", "seed": seed, "device": device,
           "params": model.n_params(), "vocab": len(tok.itos), "vocab_stats": vb["stats"], "config": model.config(),
           "corpus_sha": corpus_hash(c1), "story_stats": c2["story_stats"], "docs_stats": c3["docs_stats"] if c3 else None,
           "record_sentences": len(groups["record"]), "stages": []}
    model.train()
    ctx = autocast_ctx(device)
    for i, name in enumerate(order):
        earlier = None
        if earlier_all is not None:
            earlier = [n for n in earlier_all if n in LESSONS + LESSONS2 or n in order[:i]]
        mix = stage_mix(groups, order, i, rng, earlier=earlier)
        if not mix:
            continue
        data = sentences_to_tensor(mix, tok, model.block_size).to(device)
        if len(data) < model.block_size + 2:
            continue
        n_steps = int(steps.get(name, 300))
        if n_steps <= 0:
            continue
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
    from_v = old["core_version"] if old else None
    meta = {"seed": seed, "corpus_sha": corpus_hash(c1), "generation": gen, "core_version": to_version, "staged": True, "label": label,
            "note": f"growth: core v{from_v} -> v{to_version} ({'in place' if base == 'live' else 'fresh'}{(' - ' + label) if label else ''}); grown from gen {old['generation'] if old else '-'}",
            "grown_from": {k: old[k] for k in ("generation", "params", "vocab_size", "config", "core_version")} if old else None,
            "story_stats": c2["story_stats"], "docs_stats": c3["docs_stats"] if c3 else None, "school": run}
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
    if str(device).startswith("cuda"):
        from .reading import wait_until_gpu_free
        wait_until_gpu_free(who="grow.swap")
    store = store or MemoryStore()
    new_model, new_tok, new_ck = load_checkpoint(device=device, path=staged)
    to_v = float(new_ck.get("core_version", 3))
    user_facts = [f["content"] for f in store.facts() if f["provenance"] == "user"]

    old = None
    if os.path.exists(checkpoint):
        om, ot, oc = load_checkpoint(device=device, path=checkpoint)
        old = {"generation": int(oc.get("generation", 0)), "params": om.n_params(), "vocab_size": len(ot.itos),
               "config": om.config(), "core_version": float(oc.get("core_version", 2))}
        if exams:
            card = run_exams(seed=seed, verbose=verbose, model=om, tok=ot, store=store,
                             meta={"generation": old["generation"], "corpus_sha": oc.get("corpus_sha")},
                             consolidated_facts=user_facts,
                             trigger=f"before growth (core v{old['core_version']:g} gen {old['generation']})")
            old["exams"] = [{k: e[k] for k in ("exam", "score", "accuracy", "verdict")} for e in card["exams"]]
        os.makedirs(checkpoint_dir, exist_ok=True)
        keep = os.path.join(checkpoint_dir, f"core_v{old['core_version']:g}_gen_{old['generation']:04d}.pt")
        shutil.copy(checkpoint, keep)
        old["kept_as"] = os.path.relpath(keep, os.path.dirname(checkpoint_dir))
        del om
    from_v = f"{old['core_version']:g}" if old else "-"
    event_name = f"growth: core v{from_v} -> v{to_v:g}" + (f" ({new_ck['label']})" if new_ck.get("label") else "")

    # the generation keeps counting from where the old core left off (it may have consolidated since school)
    gen = (old["generation"] + 1) if old else int(new_ck.get("generation", 0))
    meta = {k: new_ck.get(k) for k in ("seed", "corpus_sha", "core_version", "note", "grown_from", "story_stats", "docs_stats", "school", "label", "reading", "talk")}
    meta["generation"] = gen
    meta["staged"] = False
    os.makedirs(checkpoint_dir, exist_ok=True)
    kept = save_checkpoint(new_model, new_tok, meta, path=os.path.join(checkpoint_dir, f"gen_{gen:04d}.pt"), half=True)
    save_checkpoint(new_model, new_tok, meta, path=checkpoint)

    event = {"event": event_name,
             "from": old,
             "to": {"core_version": to_v, "generation": gen, "params": new_model.n_params(), "vocab_size": len(new_tok.itos),
                    "pieces": new_tok.has_pieces(), "config": new_model.config(),
                    "checkpoint": os.path.relpath(kept, os.path.dirname(checkpoint_dir))},
             "school": (new_ck.get("school") or {}).get("stages"), "school_seconds": (new_ck.get("school") or {}).get("seconds"),
             "vocab_stats": (new_ck.get("school") or {}).get("vocab_stats"),
             "story_stats": new_ck.get("story_stats"), "docs_stats": new_ck.get("docs_stats"),
             "record_sentences": (new_ck.get("school") or {}).get("record_sentences"),
             # v3.2: a reading stage (reading.py) instead of / beside school - tokens read, probes, pauses
             "reading": {k: v for k, v in (new_ck.get("reading") or {}).items() if k != "probes"} | (
                 {"probes": (new_ck.get("reading") or {}).get("probes")} if new_ck.get("reading") else {}),
             "device": device, "seed": seed}
    if exams:
        card = run_exams(seed=seed, verbose=verbose, model=new_model, tok=new_tok, meta=meta, store=store,
                         consolidated_facts=user_facts, trigger=f"{event_name} (gen {gen})")
        event["to"]["exams"] = [{k: e[k] for k in ("exam", "score", "accuracy", "verdict")} for e in card["exams"]]
    event["seconds"] = round(time.time() - t0, 1)
    # one row for the whole swap: before and after on the same line of the chain
    store.meter("growth", event)
    store.remember(f"{event_name} at generation {gen} ({new_model.n_params():,} parameters, vocab {len(new_tok.itos)}, "
                   f"context {new_model.block_size})", provenance="self", state="confirmed")
    if verbose:
        print(f"[grow] swapped: {event_name}, gen {gen}, {new_model.n_params():,} params, vocab {len(new_tok.itos)} -> {checkpoint}")
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
    """CPU, a minute or two: tiny cores, tiny lessons, temp dir, temp memory. Proves BOTH growths end to
    end - a fresh tiny v3-shaped core (base 'fresh', ctx 64), then the same core grown in place to v3.1
    (base 'live': new vocab rows + pieces + context 64 -> 128) - two `growth` ledger rows, one chain."""
    tmp = tempfile.mkdtemp(prefix="sapling_smoke_")
    store = MemoryStore(os.path.join(tmp, "m.db"))
    store.remember("the word hello means a greeting", provenance="user")
    store.remember("the word glorp means happy", provenance="user")
    from . import exams as E
    old_card, old_wd = E.REPORT_CARD, E.WRITING_DIR
    E.REPORT_CARD = os.path.join(tmp, "card.json")
    E.WRITING_DIR = os.path.join(tmp, "writing")
    ck = os.path.join(tmp, "core.pt")
    sizes = {"simple": 60, "grammar": 80, "paragraphs": 40, "comprehension": 80, "copy": 60, "prose": 30, "docs": 30, "longread": 20}
    try:
        first = grow(store=store, device="cpu", base="fresh", config=dict(d_model=64, n_head=4, n_layer=2, block_size=64),
                     steps={k: 3 for k in STEPS}, batch=8, max_stories=150, sizes=sizes, n_pieces=40, to_version=3,
                     old_path=os.path.join(tmp, "none.pt"), staged=os.path.join(tmp, "staged1.pt"),
                     checkpoint=ck, checkpoint_dir=os.path.join(tmp, "ck"), verbose=verbose)
        out = grow(store=store, device="cpu", base="live", config=dict(block_size=128),
                   steps={k: 3 for k in STEPS}, batch=8, max_stories=150, sizes=sizes, n_pieces=40, to_version=3.1,
                   old_path=ck, staged=os.path.join(tmp, "staged2.pt"),
                   checkpoint=ck, checkpoint_dir=os.path.join(tmp, "ck"), verbose=verbose)
    finally:
        E.REPORT_CARD, E.WRITING_DIR = old_card, old_wd
    out["tmp"] = tmp
    out["store"] = store
    out["first"] = first["event"]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="both", choices=["school", "swap", "both"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=7337)
    ap.add_argument("--base", default="live", choices=["live", "fresh"])
    ap.add_argument("--no-stories", action="store_true")
    ap.add_argument("--max-stories", type=int, default=20000)
    args = ap.parse_args(argv)
    if args.smoke:
        out = smoke()
        print("smoke ok:", out["tmp"], json.dumps(out["event"]["to"], default=str)[:300])
        return
    if args.cmd == "school":
        school(seed=args.seed, stories=not args.no_stories, max_stories=args.max_stories, base=args.base)
    elif args.cmd == "swap":
        swap(seed=args.seed)
    else:
        grow(seed=args.seed, stories=not args.no_stories, max_stories=args.max_stories, base=args.base)


if __name__ == "__main__":
    main()
