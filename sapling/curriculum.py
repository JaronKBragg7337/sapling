"""Sapling curriculum — school-style, morphology-first, deterministic.

Design notes (lineage):
- "School hours" and exams come from fly-school: a lesson is taught, then measured
  cold, and the FAIL stays on the record.
- The corpus is generated, not scraped: every sentence is causally valid and
  labelled by construction (fed-not-filled: "train on your own simulations,
  where the label is free and checkable").
- Morphemes are real English word parts (prefixes, roots, suffixes) with
  simplified school glosses. The glosses are a closed teaching language,
  not a claim about etymology.
- Lessons are staged: roots -> affixes -> composition -> sentences -> recall
  patterns with slots. The recall lesson is what lets the trained core use
  the external memory layer (Live-Reference Principle, in architecture).

Everything here is CHOSEN (ours, swappable), deterministic under --seed.
"""

from __future__ import annotations

import hashlib
import random

# ---------------------------------------------------------------------------
# The word parts (real morphemes, school glosses)
# ---------------------------------------------------------------------------

ROOTS = {
    "spect": "look", "port": "carry", "dict": "say", "graph": "write",
    "aud": "hear", "vis": "see", "struct": "build", "ject": "throw",
    "mit": "send", "tract": "pull", "rupt": "break", "scrib": "write",
    "fer": "bring", "form": "shape", "pend": "hang", "pos": "place",
    "ten": "hold", "cred": "believe", "log": "speech", "phon": "sound",
    "bio": "life", "geo": "earth", "chron": "time", "luc": "light",
}

PREFIXES = {
    "re": "again", "un": "not", "pre": "before", "dis": "away",
    "trans": "across", "sub": "under", "super": "above", "inter": "between",
    "mis": "wrong", "ex": "out", "pro": "forward", "co": "together",
}

SUFFIXES = {
    "er": "one who", "ing": "ongoing", "ed": "in the past",
    "tion": "the act of", "able": "can be", "less": "without",
    "ful": "full of", "ly": "in that manner", "ness": "the state of",
    "ive": "tending to", "ology": "the study of", "ist": "one who practices",
}

# Special tokens for the memory-recall interface. v2: 8 slots = up to 4
# recalled facts per prompt (Jaron 10/14: "4 recall slots -> increase and let
# it grow"). The memory store's top-k grows with its size; the core's context
# window (block_size) is what bounds how many facts ride in one prompt.
SLOTS = [f"<m{i}>" for i in range(1, 9)]
MAX_RECALLS = len(SLOTS) // 2

FUNCTION_WORDS = (
    "i you she he we they the a an is was will can do does did today now "
    "good small new old box cart load light sound story book house ship "
    "q a recall user confirmed means word root prefix suffix what does of "
    "+ = : . ? [ ]"
).split()


def compose(word_parts: list[str]) -> str:
    """The school language spells composed words with visible parts
    ("co graph", teaching-orthography style). This is a design decision, not
    a shortcut: a held-out word must be *representable* for an exam of it to
    mean anything. Run 1 (atomic word tokens) failed E1 0/86 precisely
    because unseen words had untrained embeddings — the FAIL is on the
    record in results/report_card.json history."""
    return " ".join(word_parts)


def prefix_gloss(root: str, prefix: str) -> str:
    """Rule: composed word gloss = root meaning, then prefix meaning."""
    return f"{ROOTS[root]} {PREFIXES[prefix]}"


def suffix_gloss(root: str, suffix: str) -> str:
    """Rule: suffix phrase first, then the root meaning."""
    return f"{SUFFIXES[suffix]} {ROOTS[root]}"


def all_combos() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    pre = [(p, r) for p in PREFIXES for r in ROOTS]
    suf = [(r, s) for r in ROOTS for s in SUFFIXES]
    return pre, suf


def split_combos(seed: int = 7337, holdout_frac: float = 0.15):
    """Deterministic train/exam split. Exam combos are never taught."""
    rng = random.Random(seed)
    pre, suf = all_combos()
    rng.shuffle(pre)
    rng.shuffle(suf)
    n_pre = max(1, int(len(pre) * holdout_frac))
    n_suf = max(1, int(len(suf) * holdout_frac))
    return {
        "pre_train": pre[n_pre:], "pre_exam": pre[:n_pre],
        "suf_train": suf[n_suf:], "suf_exam": suf[:n_suf],
    }


# ---------------------------------------------------------------------------
# Lesson generators. Each returns a list of whitespace-tokenised sentences.
# ---------------------------------------------------------------------------

def lesson_roots(rng: random.Random) -> list[str]:
    out = []
    phrasings = [
        "the root {r} means {g} .",
        "root {r} : {g} .",
        "{r} is a root . it means {g} .",
        "q : what does the root {r} mean ? a : the root {r} means {g} .",
    ]
    for r, g in ROOTS.items():
        for p in phrasings:
            out.append(p.format(r=r, g=g))
    rng.shuffle(out)
    return out


def lesson_affixes(rng: random.Random) -> list[str]:
    out = []
    for p, g in PREFIXES.items():
        out += [
            f"the prefix {p} means {g} .",
            f"prefix {p} : {g} .",
            f"q : what does the prefix {p} mean ? a : {g} .",
        ]
    for s, g in SUFFIXES.items():
        out += [
            f"the suffix {s} means {g} .",
            f"suffix {s} : {g} .",
            f"q : what does the suffix {s} mean ? a : {g} .",
        ]
    rng.shuffle(out)
    return out


def lesson_composition(rng: random.Random, split: dict) -> list[str]:
    out = []
    for p, r in split["pre_train"]:
        w = compose([p, r])
        g = prefix_gloss(r, p)
        out += [
            f"{p} + {r} = {w} .",
            f"{w} means {g} .",
            f"q : what does {w} mean ? a : {w} means {g} .",
            f"q : what does {p} + {r} make ? a : {p} + {r} = {w} .",
        ]
    for r, s in split["suf_train"]:
        w = compose([r, s])
        g = suffix_gloss(r, s)
        out += [
            f"{r} + {s} = {w} .",
            f"{w} means {g} .",
            f"q : what does {w} mean ? a : {w} means {g} .",
            f"q : what does {r} + {s} make ? a : {r} + {s} = {w} .",
        ]
    rng.shuffle(out)
    return out


def lesson_sentences(rng: random.Random, split: dict) -> list[str]:
    out = []
    subjects = ["i", "you", "she", "he", "we", "they"]
    objects = ["the box", "the cart", "the load", "the story", "the book",
               "the house", "the ship", "a light", "a sound"]
    for p, r in split["pre_train"]:
        w = compose([p, r])
        s = rng.choice(subjects)
        o = rng.choice(objects)
        out.append(f"{s} will {w} {o} .")
        out.append(f"q : what will {s} do ? a : {s} will {w} {o} .")
    for r, s_ in split["suf_train"]:
        w = compose([r, s_])
        sub = rng.choice(subjects)
        if s_ in ("er", "ist"):
            out.append(f"{sub} is a good {w} .")
        else:
            out.append(f"the {w} was good .")
    rng.shuffle(out)
    return out


def lesson_recall(rng: random.Random) -> list[str]:
    """The memory-interface lesson.

    Teaches the core to (a) echo a recalled fact through slot tokens,
    (b) pick the right slot among 1..4 recalls in any order, and (c) defer
    when no recall exists. This is the trained half of the Live-Reference
    architecture: the core never memorises the fact, it learns the
    *pattern* of answering from recall.
    """
    out = []
    filler_roots = list(ROOTS.keys())
    filler_glosses = list(set(ROOTS.values()))
    pairs = [(SLOTS[i], SLOTS[i + 1]) for i in range(0, len(SLOTS), 2)]
    for _ in range(320):
        n = rng.choice([1, 1, 2, 2, 3, 4])
        chosen = pairs[:n]
        order = list(chosen)
        rng.shuffle(order)
        asked = rng.choice(chosen)
        ctx = " ".join(f"recall [ user ] : the word {w} means {g} ." for w, g in order)
        out.append(f"{ctx} q : what does {asked[0]} mean ? a : the word {asked[0]} means {asked[1]} .")
        # no recall -> defer (judgment: cite or defer)
        out.append(f"q : what does {asked[0]} mean ? a : i do not have that yet .")
    # a few grounded variants with real morphemes in slot position
    for _ in range(80):
        r = rng.choice(filler_roots)
        g = rng.choice(filler_glosses)
        out.append(
            f"recall [ confirmed ] : the root {r} means {g} . "
            f"q : what does the root {r} mean ? a : the root {r} means {g} ."
        )
    rng.shuffle(out)
    return out


LESSONS = ["roots", "affixes", "composition", "sentences", "recall"]


def build_corpus(seed: int = 7337) -> dict:
    """Returns {lesson_name: [sentences]} plus the held-out exam split."""
    rng = random.Random(seed)
    split = split_combos(seed)
    corpus = {
        "roots": lesson_roots(rng),
        "affixes": lesson_affixes(rng),
        "composition": lesson_composition(rng, split),
        "sentences": lesson_sentences(rng, split),
        "recall": lesson_recall(rng),
    }
    return {"lessons": corpus, "split": split}


def build_vocab(corpus: dict) -> dict:
    toks = set(FUNCTION_WORDS) | set(SLOTS) | {"<pad>", "<bos>", "<eos>", "<unk>"}
    for sents in corpus["lessons"].values():
        for s in sents:
            toks.update(s.split())
    vocab = sorted(toks)
    stoi = {t: i for i, t in enumerate(vocab)}
    return {"stoi": stoi, "itos": vocab}


def corpus_hash(corpus: dict) -> str:
    h = hashlib.sha256()
    for name in LESSONS:
        for s in corpus["lessons"][name]:
            h.update(s.encode() + b"\n")
    return h.hexdigest()[:16]
