"""The long-reading lesson (10/16, Claude as teacher; Jaron: step up on a rented GPU).

Target: E7L (reading a long, never-seen passage and answering with a span copied out of it: 0/40 so far) and the
look-back skill under it (E10, the induction probe: 109/2016 at 1.08B tokens of plain reading, rising).

Every example is a passage of 200-450 tokens of LICENSE-CLEAN HUMAN TEXT already on disk (Simple English Wikipedia,
Project Gutenberg books, FineWeb-Edu pages), then a question whose answer is a span copied from EARLIER in that
passage, in the chat-box format (talk.encode_turns):

    <bos> passage : <text> q : <question> a : <span copied from the text> <eos>

Loss is on the answer only. The questions are made BY RULE from the text's own structure - code templates, no model
writes anything (Jaron 10/15: no outputs we generate from other AIs become training data):
    after    "what comes after <3 words> ?"            -> the next 1-6 words
    word     "what word comes after / before <2 words> ?" -> one word
    which    "which sentence has the word <w> ?"        -> that sentence
    first    "copy the first / second / last sentence ." -> that sentence
    year     "what does the passage say about <year> ?" -> the sentence with that year
    number   "what number goes with <noun> ?"           -> "<number> <noun>"
    name     "what name comes right before <3 words> ?" -> the capitalised name (found in the raw text)
    define   "what is / was <X> ?"                      -> the "<X> is a ..." sentence
The fact sits anywhere in the passage (position drawn uniformly), and in ~30% of examples a second, unrelated passage
(a distractor) is added before or after, so the look-back distance runs from a few tokens to ~450.

Held out: 1% of documents (by a hash of their id) give the test set only - those documents never give a training
example. E7L's essay (pg1022, Thoreau's "Walking") is never read here, and any passage sharing an 8-word run with it is
dropped, so no E7L text can leak in.

    python -m sapling.longread_lesson build [--examples 400000]     # CPU, MSI -> data/cloud/bundle/
    (training runs on the rented GPU: tools/cloud_lesson.py, started by tools/runpod_run.py)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time

import numpy as np

from .text import normalize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "reading", "raw")
BUNDLE = os.path.join(ROOT, "data", "cloud", "bundle")
P_MIN, P_MAX = 200, 450              # passage length in core tokens
DISTRACT = 0.30
HOLDOUT_PCT = 1
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
NUM_NOUN = re.compile(r"\b(\d{1,6}) ([a-z]{3,}s)\b")      # "12 years", "500 soldiers": plural nouns only
NAME = re.compile(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]+(?: [A-Z][a-z]+){1,2})\b")
DEFINE = re.compile(r"^((?:[A-Z][\w'-]*)(?: [\w'-]+){0,4}) (is|was|are|were) (?:a|an|the) ")
STOP = {"the", "a", "an", "and", "but", "or", "in", "on", "at", "of", "for", "to", "by", "with", "from", "as", "if", "when",
        "then", "this", "that", "these", "those", "he", "she", "it", "they", "we", "i", "you", "his", "her", "its", "their",
        "our", "my", "your", "so", "yet", "after", "before", "while", "during", "since", "until", "mr", "mrs", "dr", "st"}
NOTE = ("Long-reading lesson: passages of Simple English Wikipedia (CC BY-SA 3.0 + GFDL, Wikipedia contributors), Project "
        "Gutenberg public-domain books, and FineWeb-Edu (ODC-By 1.0, Hugging Face, from Common Crawl); questions made by "
        "rule-based code templates from the text itself (no generated text). pg1022 (E7L) never used.")


# ---------------------------------------------------------------------------
# text sources (raw, so capitals are still there for names / definitions)
# ---------------------------------------------------------------------------

def iter_raw_docs(raw_dir: str = RAW, limit_per_source: int | None = None):
    """(source, doc_id, raw_text), the three sources interleaved."""
    import pyarrow.parquet as pq
    from .curriculum3 import DOCS_HOLDOUT, strip_gutenberg

    def wiki():
        p = os.path.join(raw_dir, "simplewiki.parquet")
        if not os.path.exists(p):
            return
        pf = pq.ParquetFile(p)
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=["id", "title", "text"])
            for i, ti, tx in zip(t.column("id").to_pylist(), t.column("title").to_pylist(), t.column("text").to_pylist()):
                if tx and len(tx) > 1200:
                    yield "wiki", f"wiki:{i}", tx

    def web():
        for name in sorted(n for n in os.listdir(raw_dir) if n.startswith("fineweb_edu_") and n.endswith(".parquet")):
            pf = pq.ParquetFile(os.path.join(raw_dir, name))
            for rg in range(pf.num_row_groups):
                t = pf.read_row_group(rg, columns=["id", "text"])
                for i, tx in zip(t.column("id").to_pylist(), t.column("text").to_pylist()):
                    if tx and len(tx) > 1200:
                        yield "web", f"web:{i}", tx

    def books():
        gdir = os.path.join(raw_dir, "gutenberg")
        if not os.path.isdir(gdir):
            return
        for name in sorted(os.listdir(gdir)):
            if not name.endswith(".txt") or name.startswith(DOCS_HOLDOUT + "."):
                continue
            raw = open(os.path.join(gdir, name), encoding="utf-8", errors="replace").read()
            body = strip_gutenberg(raw) if "Project Gutenberg" in raw[:2000] else raw
            paras = [p.strip() for p in re.split(r"\n\s*\n", body) if len(p.strip()) > 40]
            buf, k = [], 0
            for p in paras:
                buf.append(" ".join(p.split()))
                if sum(len(b) for b in buf) > 4000:
                    yield "book", f"book:{name}:{k}", "\n\n".join(buf)
                    buf, k = [], k + 1

    gens = {"web": web(), "wiki": wiki(), "book": books()}
    weights = {"web": 6, "wiki": 3, "book": 1}
    counts = {k: 0 for k in gens}
    rng = random.Random(11)
    while gens:
        name = rng.choices(list(gens), [weights[k] for k in gens])[0]
        try:
            item = next(gens[name])
        except StopIteration:
            del gens[name]
            continue
        counts[name] += 1
        if limit_per_source and counts[name] > limit_per_source:
            del gens[name]
            continue
        yield item


def held_out(doc_id: str) -> bool:
    return int(hashlib.sha256(doc_id.encode()).hexdigest()[:8], 16) % 100 < HOLDOUT_PCT


def e7l_ngrams(n: int = 8) -> set:
    """8-word runs of E7L's essay (so no passage that quotes it is ever used)."""
    from .curriculum3 import DOCS_DIR, DOCS_HOLDOUT
    out = set()
    for d in (DOCS_DIR, os.path.join(RAW, "gutenberg")):
        p = os.path.join(d, DOCS_HOLDOUT + ".txt")
        if os.path.exists(p):
            w = normalize(open(p, encoding="utf-8", errors="replace").read()).split()
            out |= {" ".join(w[i:i + n]) for i in range(0, max(0, len(w) - n))}
    return out


# ---------------------------------------------------------------------------
# passages and questions
# ---------------------------------------------------------------------------

def passages(raw: str, ntok, rng: random.Random) -> list[list[str]]:
    """Raw text -> passages as lists of RAW sentences, each passage P_MIN..P_MAX core tokens (normalised)."""
    sents = [s.strip() for s in SENT_SPLIT.split(" ".join(raw.replace("\n", " \n ").split())) if s.strip()]
    out, cur, n = [], [], 0
    target = rng.randint(P_MIN + 20, P_MAX)
    for s in sents:
        k = ntok(normalize(s))
        if k == 0 or k > 120:
            continue
        if n + k > P_MAX:
            if n >= P_MIN:
                out.append(cur)
            cur, n, target = [], 0, rng.randint(P_MIN + 20, P_MAX)
        cur.append(s)
        n += k
        if n >= target:
            out.append(cur)
            cur, n, target = [], 0, rng.randint(P_MIN + 20, P_MAX)
    return out


def _unique(norm_passage: str, span: str) -> bool:
    return f" {norm_passage} ".count(f" {span} ") == 1


def question(rng: random.Random, raw_sents: list[str], kinds: list[str] | None = None) -> tuple[str, str, str, float] | None:
    """-> (question, answer, kind, where) or None. `where` = position of the fact in the passage, 0..1."""
    norm = [normalize(s) for s in raw_sents]
    P = " ".join(norm)
    toks = P.split()
    if len(toks) < 30:
        return None
    kind = rng.choice(kinds or ["after", "after", "word", "word", "which", "first", "year", "number", "name", "define"])
    if kind == "after":
        i = rng.randrange(0, len(toks) - 8)
        span = " ".join(toks[i:i + 3])
        if not _unique(P, span) or any(t in ".,;:!?" for t in toks[i:i + 3]):
            return None
        ans = []
        for t in toks[i + 3:i + 3 + rng.randint(1, 6)]:
            if t in ".!?":
                break
            ans.append(t)
        if not ans or ans[0] in ".,;:!?":
            return None
        return f"what comes after {span} ?", " ".join(ans) + " .", kind, i / len(toks)
    if kind == "word":
        before = rng.random() < 0.5
        i = rng.randrange(1, len(toks) - 3)
        span = " ".join(toks[i:i + 2])
        w = toks[i - 1] if before else toks[i + 2]
        if not _unique(P, span) or w in ".,;:!?" or any(t in ".,;:!?" for t in toks[i:i + 2]):
            return None
        return f"what word comes {'before' if before else 'after'} {span} ?", f"{w} .", kind, i / len(toks)
    if kind == "which":
        j = rng.randrange(len(norm))
        content = [t for t in norm[j].split() if len(t) > 3 and t.isalpha()]
        if not content or len(norm[j].split()) > 45:
            return None
        w = rng.choice(content)
        if P.split().count(w) != 1:
            return None
        return f"which sentence has the word {w} ?", norm[j], kind, j / len(norm)
    if kind == "first":
        which = rng.choice(["first", "second", "last"])
        j = {"first": 0, "second": 1, "last": len(norm) - 1}[which]
        if len(norm[j].split()) > 45:
            return None
        return f"copy the {which} sentence .", norm[j], kind, j / len(norm)
    if kind == "year":
        cands = [(j, m.group(1)) for j, s in enumerate(raw_sents) for m in YEAR.finditer(s)]
        rng.shuffle(cands)
        for j, y in cands:
            if P.split().count(y) == 1 and len(norm[j].split()) <= 45:
                return f"what does the passage say about {y} ?", norm[j], kind, j / len(norm)
        return None
    if kind == "number":
        cands = [(j, m.group(1), m.group(2)) for j, s in enumerate(norm) for m in NUM_NOUN.finditer(s)]
        rng.shuffle(cands)
        for j, n, noun in cands:
            if P.split().count(noun) == 1 and _unique(P, f"{n} {noun}"):
                return f"what number goes with {noun} ?", f"{n} {noun} .", kind, j / len(norm)
        return None
    if kind == "name":
        cands = []
        for j, s in enumerate(raw_sents):
            for m in NAME.finditer(s):
                if m.start() == 0:
                    continue
                words = m.group(1).split()
                while words and words[0].lower() in STOP:
                    words = words[1:]
                if len(words) < 2:
                    continue
                name = normalize(" ".join(words))
                after = normalize(s[m.end():]).split()[:3]
                if len(after) == 3 and not any(t in ".,;:!?" for t in after):
                    cands.append((j, name, " ".join(after)))
        rng.shuffle(cands)
        for j, name, after in cands:
            if _unique(P, after) and _unique(P, f"{name} {after}"):
                return f"what name comes right before {after} ?", f"{name} .", kind, j / len(norm)
        return None
    if kind == "define":
        for j in rng.sample(range(len(raw_sents)), len(raw_sents)):
            m = DEFINE.match(raw_sents[j])
            if m and len(norm[j].split()) <= 45:
                x, verb = normalize(m.group(1)), m.group(2)
                if f" {P} ".count(f" {x} {verb} ") == 1:
                    return f"what {verb} {x} ?", norm[j], kind, j / len(norm)
        return None
    return None


# ---------------------------------------------------------------------------
# build the bundle (MSI, CPU)
# ---------------------------------------------------------------------------

def build_examples(tok, n_examples: int = 400_000, n_test: int = 2000, seed: int = 7337, verbose: bool = True,
                   limit_per_source: int | None = None) -> dict:
    """-> {"train": [(ids, tg)], "test": [{passage, q, a, kind, where}], "stats": ...}"""
    from .reading import FastEncoder
    from .talk import encode_turns
    rng = random.Random(seed + 101)
    enc = FastEncoder(tok)
    cache = {}

    def ntok(norm_text):
        return len(enc.ids(norm_text)) - 2

    banned = e7l_ngrams()
    train, test = [], []
    stats = {"kinds": {}, "sources": {}, "distractor": 0, "dropped_e7l_overlap": 0, "docs": 0, "where": [0] * 10}
    pool: list[list[str]] = []          # recent passages for distractors (training docs only)
    t0 = time.time()
    for src, did, raw in iter_raw_docs(limit_per_source=limit_per_source):
        if len(train) >= n_examples and len(test) >= n_test:
            break
        is_test = held_out(did)
        if (is_test and len(test) >= n_test) or (not is_test and len(train) >= n_examples):
            continue
        stats["docs"] += 1
        for ps in passages(raw, ntok, rng):
            normp = " ".join(normalize(s) for s in ps)
            w = normp.split()
            if banned and any(" ".join(w[i:i + 8]) in banned for i in range(0, len(w) - 8, 4)):
                stats["dropped_e7l_overlap"] += 1
                continue
            for _ in range(rng.choice([1, 1, 2])):
                qa = question(rng, ps)
                if not qa:
                    continue
                q, a, kind, where = qa
                body = normp
                n_p = ntok(normp)
                if not is_test and pool and kind != "first" and n_p <= 380 and rng.random() < DISTRACT / 0.5:
                    # a piece of another document (whole sentences), so the fact must be found past or before other text
                    room, d = 470 - n_p, []
                    for s_ in rng.choice(pool):
                        k = ntok(s_)
                        if k > room:
                            break
                        d.append(s_); room -= k
                    d = " ".join(d)
                    key = q.split(" ")[-2] if q.endswith(" ?") else q
                    if len(d.split()) >= 30 and f" {key} " not in f" {d} " and a.rstrip(" .") not in d:
                        body = f"{d} {normp}" if rng.random() < 0.5 else f"{normp} {d}"
                        stats["distractor"] += 1
                if is_test:
                    if len(test) < n_test:
                        test.append({"passage": body, "q": q, "a": a, "kind": kind, "where": round(where, 3), "doc": did})
                else:
                    ids, tg = encode_turns(tok, [[q, a]], passage=body)
                    if len(ids) > 513:
                        continue
                    train.append((ids, tg))
                    stats["kinds"][kind] = stats["kinds"].get(kind, 0) + 1
                    stats["sources"][src] = stats["sources"].get(src, 0) + 1
                    stats["where"][min(9, int(where * 10))] += 1
            if not is_test:
                pool.append([normalize(s_) for s_ in ps])
                if len(pool) > 500:
                    pool.pop(rng.randrange(len(pool)))
        if verbose and stats["docs"] % 5000 == 0:
            print(f"[longread] {stats['docs']:,} docs -> {len(train):,} train / {len(test):,} test ({time.time() - t0:.0f}s)", flush=True)
    stats["train"], stats["test"], stats["seconds"] = len(train), len(test), round(time.time() - t0, 1)
    stats["train_tokens"] = int(sum(len(x[0]) for x in train))
    return {"train": train, "test": test, "stats": stats}


def pack(examples: list[tuple[list[int], list[int]]]) -> dict:
    """Ragged examples -> flat uint16 ids + a 0/1 'learn here' mask + offsets (small to upload)."""
    offs = np.zeros(len(examples) + 1, dtype=np.int64)
    for i, (ids, _) in enumerate(examples):
        offs[i + 1] = offs[i] + len(ids)
    ids = np.zeros(int(offs[-1]), dtype=np.uint16)
    mask = np.zeros(int(offs[-1]), dtype=np.uint8)
    for i, (a, t) in enumerate(examples):
        ids[offs[i]:offs[i + 1]] = a
        mask[offs[i]:offs[i + 1]] = [x != -1 for x in t]
    return {"ids": ids, "mask": mask, "offsets": offs}


def build_bundle(n_examples: int = 400_000, read_tokens: float = 150e6, out: str = BUNDLE, core: str | None = None,
                 seed: int = 7337, verbose: bool = True, limit_per_source: int | None = None, n_test: int = 2000) -> dict:
    """Everything the rented GPU needs, as a few files: lesson.npz (train), test.jsonl, talk.npz (conversation
    replay), reading.npy (reading replay windows), school.npy (rehearsal stream), core.pt (the live core)."""
    import shutil
    from .reading import SHARD_DIR, rehearsal_sentences
    from .talk import LESSONS, load_examples
    from .train import CHECKPOINT, load_checkpoint, sentences_to_tensor
    t0 = time.time()
    core = core or CHECKPOINT
    os.makedirs(out, exist_ok=True)
    model, tok, ck = load_checkpoint(device="cpu", path=core)
    block = model.block_size
    ex = build_examples(tok, n_examples=n_examples, n_test=n_test, seed=seed, verbose=verbose, limit_per_source=limit_per_source)
    np.savez(os.path.join(out, "lesson.npz"), **pack(ex["train"]))
    with open(os.path.join(out, "test.jsonl"), "w", encoding="utf-8") as f:
        for t in ex["test"]:
            f.write(json.dumps(t) + "\n")
    talk = load_examples(tok, block, LESSONS) if os.path.exists(LESSONS) else []
    np.savez(os.path.join(out, "talk.npz"), **pack(talk))
    # reading replay: random 513-token windows from every shard, by the v4 shares (stories 11 / adult 4 / web 14)
    idx = json.load(open(os.path.join(SHARD_DIR, "index.json"), encoding="utf-8"))
    rng = np.random.default_rng(seed)
    n_win = int(read_tokens // (block + 1))
    share = {"stories": 11, "adult": 4, "web": 14}
    wins = []
    for name, sh in share.items():
        files = idx["sources"].get(name, {}).get("shards", [])
        if not files:
            continue
        k = int(n_win * sh / sum(share.values()))
        per = max(1, k // len(files))
        for f in files:
            a = np.load(os.path.join(SHARD_DIR, f["file"]), mmap_mode="r")
            offs = rng.integers(0, len(a) - block - 2, size=per)
            wins.append(np.stack([np.asarray(a[o:o + block + 1]) for o in offs]))
    reading = np.concatenate(wins).astype(np.uint16)
    rng.shuffle(reading)
    np.save(os.path.join(out, "reading.npy"), reading)
    school = sentences_to_tensor(rehearsal_sentences(seed, None, verbose=verbose), tok, block).numpy().astype(np.uint16)
    np.save(os.path.join(out, "school.npy"), school)
    core_file = f"core_gen{int(ck.get('generation') or 0):04d}.pt"     # a new live core = a new asset name
    for n in os.listdir(out):
        if n.startswith("core") and n.endswith(".pt"):
            os.remove(os.path.join(out, n))
    shutil.copy(core, os.path.join(out, core_file))
    info = {"built": time.strftime("%Y-%m-%dT%H:%M:%S"), "core": os.path.relpath(core, ROOT), "core_generation": ck.get("generation"),
            "core_file": core_file,
            "params": model.n_params(), "lesson": ex["stats"], "talk_examples": len(talk), "reading_windows": int(len(reading)),
            "school_tokens": int(len(school)), "note": NOTE, "seconds": round(time.time() - t0, 1),
            "files": {n: os.path.getsize(os.path.join(out, n)) for n in os.listdir(out) if not n.endswith(".json")}}
    json.dump(info, open(os.path.join(out, "bundle.json"), "w", encoding="utf-8"), indent=1)
    if verbose:
        print(json.dumps({k: v for k, v in info.items() if k != "lesson"} | {"lesson": {k: ex["stats"][k] for k in ("train", "test", "train_tokens", "kinds", "sources", "distractor", "dropped_e7l_overlap", "where")}}, indent=1))
    return info


# ---------------------------------------------------------------------------
# training on the bundle (runs on the rented GPU; also runs on the MSI for a smoke test)
# ---------------------------------------------------------------------------

class Ragged:
    def __init__(self, path: str):
        z = np.load(path)
        self.ids, self.mask, self.offs = z["ids"], z["mask"], z["offsets"]

    def __len__(self):
        return len(self.offs) - 1

    def batch(self, idx, block: int, pad: int):
        import torch
        X = torch.full((len(idx), block + 1), pad, dtype=torch.long)
        Y = torch.full((len(idx), block + 1), -1, dtype=torch.long)
        for r, i in enumerate(idx):
            a, b = int(self.offs[i]), int(self.offs[i + 1])
            n = min(b - a, block + 1)
            ids = torch.from_numpy(self.ids[a:a + n].astype(np.int64))
            m = torch.from_numpy(self.mask[a:a + n].astype(bool))
            X[r, :n] = ids
            Y[r, :n] = torch.where(m, ids, torch.full_like(ids, -1))
        return X[:, :-1], Y[:, 1:]


def heldout_accuracy(model, tok, test: list[dict], n: int = 300) -> dict:
    """Greedy answers on held-out documents (never trained on); exact match, by kind and by distance."""
    import torch
    from .talk import encode_turns
    dev = next(model.parameters()).device
    eos = tok.stoi["<eos>"]
    model.eval()
    ok, by = 0, {}
    for t in test[:n]:
        ids, _ = encode_turns(tok, [[t["q"], ""]], passage=t["passage"], answer_last=False)
        ids = ids[-(model.block_size - 48):]
        with torch.no_grad():
            out = model.generate(torch.tensor([ids], device=dev), max_new=48, eos_id=eos)
        got = tok.decode(out[0, len(ids):].tolist()).replace("<eos>", "").strip()
        hit = got == t["a"].strip()
        ok += hit
        k = by.setdefault(t["kind"], [0, 0]); k[0] += hit; k[1] += 1
    model.train()
    n = min(n, len(test))
    return {"score": f"{ok}/{n}", "accuracy": round(ok / max(1, n), 3), "by_kind": {k: f"{a}/{b}" for k, a_b in by.items() for a, b in [a_b]}}


def train_bundle(bundle: str, out: str, minutes: float = 60.0, batch: int = 64, micro: int = 32, lr: float = 2e-4,
                 min_lr: float = 2e-5, warmup: int = 300, shares: dict | None = None, seed: int = 7337,
                 device: str | None = None, max_steps: int | None = None, eval_every_min: float = 10.0,
                 status=None) -> dict:
    """Fine-tune the bundle's core on the lesson + replay for `minutes` of wall clock (the LR schedule is laid out over
    the steps that time allows, measured after warm-up). Writes out/core_staged.pt, out/log.jsonl, out/status.json."""
    import torch
    from .model import autocast_ctx
    from .train import load_checkpoint, save_checkpoint
    t_start = time.time()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    shares = shares or {"lesson": 28, "reading": 16, "talk": 12, "school": 8}       # of 64
    assert sum(shares.values()) == batch
    os.makedirs(out, exist_ok=True)
    logp = os.path.join(out, "log.jsonl")

    def log(row):
        row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **row}
        with open(logp, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if status:
            status(row)

    torch.manual_seed(seed)
    rng = random.Random(seed)
    core_file = json.load(open(os.path.join(bundle, "bundle.json"), encoding="utf-8")).get("core_file", "core.pt")
    model, tok, ck = load_checkpoint(device=device, path=os.path.join(bundle, core_file))
    block, pad = model.block_size, tok.stoi["<pad>"]
    lesson = Ragged(os.path.join(bundle, "lesson.npz"))
    talk = Ragged(os.path.join(bundle, "talk.npz"))
    reading = np.load(os.path.join(bundle, "reading.npy"), mmap_mode="r")
    school = torch.from_numpy(np.load(os.path.join(bundle, "school.npy")).astype(np.int64))
    test = [json.loads(l) for l in open(os.path.join(bundle, "test.jsonl"), encoding="utf-8")]
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.95), fused=device.startswith("cuda"))
    ctx = autocast_ctx(device)
    before = heldout_accuracy(model, tok, test)
    log({"kind": "start", "device": device, "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu",
         "lesson_examples": len(lesson), "talk_examples": len(talk), "reading_windows": int(len(reading)),
         "batch": batch, "micro": micro, "shares": shares, "minutes": minutes, "heldout_before": before})
    budget_s = minutes * 60
    total = max_steps or 10 ** 9
    step, tokens, losses = 0, 0, []
    t_train, next_eval = time.time(), time.time() + eval_every_min * 60
    evals = [{"step": 0, **before}]
    while step < total:
        if step == 50 and not max_steps:                 # lay out the cosine over what the time budget allows
            rate = 50 / (time.time() - t_train)
            total = int(rate * (budget_s - (time.time() - t_start) - 300)) + 50
            log({"kind": "plan", "steps_per_s": round(rate, 3), "total_steps": total,
                 "tokens_planned": int(total * batch * block)})
        xs, ys = [], []
        x, y = lesson.batch([rng.randrange(len(lesson)) for _ in range(shares["lesson"])], block, pad); xs.append(x); ys.append(y)
        if shares["talk"] and len(talk):
            x, y = talk.batch([rng.randrange(len(talk)) for _ in range(shares["talk"])], block, pad); xs.append(x); ys.append(y)
        w = torch.from_numpy(np.stack([reading[rng.randrange(len(reading))] for _ in range(shares["reading"])]).astype(np.int64))
        xs.append(w[:, :-1]); ys.append(w[:, 1:])
        starts = [rng.randrange(0, len(school) - block - 1) for _ in range(shares["school"])]
        s = torch.stack([school[a:a + block + 1] for a in starts]); xs.append(s[:, :-1]); ys.append(s[:, 1:])
        X, Y = torch.cat(xs).to(device, non_blocking=True), torch.cat(ys).to(device, non_blocking=True)
        p = min(1.0, max(0, step - warmup) / max(1, total - warmup))
        cur = lr * (step + 1) / warmup if step < warmup else min_lr + 0.5 * (lr - min_lr) * (1 + np.cos(np.pi * p))
        for g in opt.param_groups:
            g["lr"] = cur
        try:
            opt.zero_grad(set_to_none=True)
            perm = torch.randperm(len(X), device=X.device)
            tot = 0.0
            for i in range(0, len(X), micro):
                j = perm[i:i + micro]
                with ctx:
                    _, l = model(X[j], Y[j])
                (l * len(j) / len(X)).backward()
                tot += float(l.detach()) * len(j) / len(X)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            micro = max(4, micro // 2)
            log({"kind": "oom", "step": step, "micro_now": micro})
            continue
        losses.append(tot)
        step += 1
        tokens += len(X) * block
        if step % 200 == 0:
            el = time.time() - t_train
            log({"kind": "step", "step": step, "of": total, "loss": round(sum(losses[-200:]) / 200, 4), "lr": round(cur, 7),
                 "tok_per_s": round(tokens / el), "minutes": round((time.time() - t_start) / 60, 1)})
        if time.time() >= next_eval:
            ev = heldout_accuracy(model, tok, test)
            evals.append({"step": step, **ev})
            log({"kind": "heldout", "step": step, **ev})
            next_eval = time.time() + eval_every_min * 60
        if time.time() - t_start > budget_s - 240:     # leave time to save and hand back
            log({"kind": "time budget reached", "step": step})
            break
    after = heldout_accuracy(model, tok, test, n=len(test) if len(test) <= 1000 else 1000)
    run = {"kind": "long-reading lesson (cloud)", "steps": step, "tokens": tokens, "batch": batch, "micro": micro,
           "shares": shares, "lr": lr, "minutes": round((time.time() - t_start) / 60, 1),
           "heldout_before": before, "heldout_after": after, "evals": evals,
           "loss_last": round(sum(losses[-200:]) / max(1, len(losses[-200:])), 4), "note": NOTE}
    meta = {k: ck.get(k) for k in ("seed", "corpus_sha", "generation", "core_version", "grown_from", "story_stats",
                                    "docs_stats", "school", "reading", "talk")}
    meta.update({"staged": True, "label": (ck.get("label") or "v4") + " + long-reading lesson", "longread": run,
                 "note": f"growth: core v{float(ck.get('core_version') or 4):g} + long-reading lesson (cloud)"})
    model.eval()
    save_checkpoint(model, tok, meta, path=os.path.join(out, "core_staged.pt"))
    log({"kind": "done", **{k: run[k] for k in ("steps", "tokens", "minutes", "loss_last")}, "heldout_after": after})
    return run


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "train"])
    ap.add_argument("--examples", type=int, default=400_000)
    ap.add_argument("--read-tokens", type=float, default=150e6)
    ap.add_argument("--bundle", default=BUNDLE)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "longread"))
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build_bundle(n_examples=a.examples, read_tokens=a.read_tokens, out=a.bundle)
    else:
        train_bundle(a.bundle, a.out, minutes=a.minutes, micro=a.micro, max_steps=a.max_steps)


if __name__ == "__main__":
    main()
