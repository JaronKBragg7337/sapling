"""Sapling curriculum, level 2 - real sentences, staged like school.

Jaron (10/14 7:02 PM): "school level 2: real sentences" - grow it from word
parts to reading and writing real sentences; "language first, then everything
grows". The stages, after the level-1 lessons (roots -> affixes -> composition
-> sentences -> recall):

    simple         subject / verb / object sentences, and "what does X do ?"
    grammar        tense (today / yesterday / tomorrow / now), plurals,
                   agreement, questions, negation - each as a "fill" task
    paragraphs     2-4 sentences about one or two characters
    comprehension  read a 2-4 sentence passage, answer a question about it
    stories        real children's stories (TinyStories, CDLA-Sharing-1.0;
                   see STORIES_NOTE) - the only text here a person did not
                   generate by rule
    record         what it has already been taught and said (memory.db), so
                   the grown core is the same individual (train.py adds it)

Everything generated here is labelled by construction (fed-not-filled), is
deterministic under the seed, and keeps a cold held-out split:
- E6 (grammar): held-out (verb, object) pairs never appear in ANY generated
  level-2 lesson. (A real story may still contain "ate the apple" - that is
  exposure to language, not the exam item; noted, not hidden.)
- E7 (reading): held-out (name, noun, colour) and (name, place) facts never
  appear in any generated passage.
License note: no document of Jaron's is used here (his step 2, after he checks
this one).
"""

from __future__ import annotations

import os
import random
from collections import Counter

from .text import normalize

# ---------------------------------------------------------------------------
# Word banks (small on purpose - a first reader's words)
# ---------------------------------------------------------------------------

NAMES = ["tom", "anna", "sam", "lily", "ben", "mia", "max", "zoe", "leo", "emma", "jack", "nina"]

# noun -> plural
NOUNS = {
    "dog": "dogs", "cat": "cats", "ball": "balls", "box": "boxes", "bird": "birds",
    "cup": "cups", "hat": "hats", "book": "books", "fish": "fish", "child": "children",
    "apple": "apples", "car": "cars", "tree": "trees", "house": "houses", "bus": "buses",
    "mouse": "mice", "foot": "feet", "man": "men", "toy": "toys", "cake": "cakes",
    "frog": "frogs", "duck": "ducks", "bag": "bags", "kite": "kites", "boat": "boats",
}

# verb -> (3rd person singular, past, -ing); transitive ones take an object
VERBS = {
    "run": ("runs", "ran", "running"), "jump": ("jumps", "jumped", "jumping"),
    "eat": ("eats", "ate", "eating"), "see": ("sees", "saw", "seeing"),
    "play": ("plays", "played", "playing"), "walk": ("walks", "walked", "walking"),
    "read": ("reads", "read", "reading"), "make": ("makes", "made", "making"),
    "find": ("finds", "found", "finding"), "like": ("likes", "liked", "liking"),
    "want": ("wants", "wanted", "wanting"), "hold": ("holds", "held", "holding"),
    "carry": ("carries", "carried", "carrying"), "build": ("builds", "built", "building"),
    "sing": ("sings", "sang", "singing"), "sit": ("sits", "sat", "sitting"),
    "sleep": ("sleeps", "slept", "sleeping"), "look": ("looks", "looked", "looking"),
    "help": ("helps", "helped", "helping"), "push": ("pushes", "pushed", "pushing"),
    "wash": ("washes", "washed", "washing"), "open": ("opens", "opened", "opening"),
    "throw": ("throws", "threw", "throwing"), "draw": ("draws", "drew", "drawing"),
    "swim": ("swims", "swam", "swimming"), "go": ("goes", "went", "going"),
}
TRANSITIVE = ["eat", "see", "make", "find", "like", "want", "hold", "carry", "build",
              "look at", "help", "push", "wash", "open", "throw", "draw", "read", "play with"]
INTRANSITIVE = ["run", "jump", "walk", "sing", "sit", "sleep", "swim", "go", "play"]

COLOURS = ["red", "blue", "green", "yellow", "brown", "white", "black", "pink"]
ADJECTIVES = ["big", "small", "old", "new", "happy", "sad", "warm", "cold", "soft", "fast"]
PLACES = ["the park", "the house", "the school", "the shop", "the garden", "the river",
          "the beach", "the farm"]
OBJECTS = [f"the {n}" for n in NOUNS] + [f"a {n}" for n in ("ball", "cake", "book", "kite", "hat")]


def _forms(verb: str) -> tuple[str, str, str, str]:
    """base, 3sg, past, ing - also for two-word verbs ('look at')."""
    head, _, rest = verb.partition(" ")
    v3, vp, vi = VERBS[head]
    tail = (" " + rest) if rest else ""
    return verb, v3 + tail, vp + tail, vi + tail


# ---------------------------------------------------------------------------
# Held-out splits (cold exams)
# ---------------------------------------------------------------------------

def split_level2(seed: int = 7337, holdout_frac: float = 0.15) -> dict:
    rng = random.Random(seed + 2)
    pairs = [(v, o) for v in TRANSITIVE for o in OBJECTS]
    rng.shuffle(pairs)
    n = max(1, int(len(pairs) * holdout_frac))
    facts = [(nm, n_, c) for nm in NAMES for n_ in NOUNS for c in COLOURS]
    rng.shuffle(facts)
    nf = max(1, int(len(facts) * holdout_frac))
    places = [(nm, p) for nm in NAMES for p in PLACES]
    rng.shuffle(places)
    npl = max(1, int(len(places) * holdout_frac))
    return {
        "vo_train": pairs[n:], "vo_exam": pairs[:n],
        "fact_train": facts[nf:], "fact_exam": facts[:nf],
        "place_train": places[npl:], "place_exam": places[:npl],
    }


# ---------------------------------------------------------------------------
# Lesson generators
# ---------------------------------------------------------------------------

def _subject(rng: random.Random) -> tuple[str, bool]:
    """(subject, is_plural)"""
    k = rng.random()
    if k < 0.5:
        return rng.choice(NAMES), False
    if k < 0.8:
        return f"the {rng.choice(list(NOUNS))}", False
    n = rng.choice(list(NOUNS))
    return f"the {NOUNS[n]}", True


def lesson_simple(rng: random.Random, split: dict, n: int = 900) -> list[str]:
    out = []
    for _ in range(n):
        s, plural = _subject(rng)
        if rng.random() < 0.65:
            v, o = rng.choice(split["vo_train"])
            base, v3, vp, vi = _forms(v)
            pred = f"{base if plural else v3} {o}"
            out.append(f"{s} {pred} .")
            if rng.random() < 0.5:
                out.append(f"q : what does {s} do ? a : {s} {pred} .")
        else:
            v = rng.choice(INTRANSITIVE)
            base, v3, vp, vi = _forms(v)
            where = f" at {rng.choice(PLACES)}" if rng.random() < 0.6 else ""
            out.append(f"{s} {base if plural else v3}{where} .")
            if rng.random() < 0.4:
                out.append(f"q : where does {s} {base} ? a : {s} {base if plural else v3} at {rng.choice(PLACES)} ."
                           if not where else f"q : where does {s} {base} ? a : {s} {base if plural else v3}{where} .")
    rng.shuffle(out)
    return out


def grammar_item(kind: str, rng: random.Random, pairs: list[tuple[str, str]]) -> tuple[str, str]:
    """One (question, answer) for a grammar task. The same function makes exam
    items from the held-out pairs, so training and exam share a format and
    nothing else."""
    s, plural = _subject(rng)
    v, o = rng.choice(pairs)
    base, v3, vp, vi = _forms(v)
    pres = f"{base if plural else v3} {o}"
    if kind == "past":
        return f"q : fill : yesterday {s} ( {base} ) {o} . a :", f"yesterday {s} {vp} {o} ."
    if kind == "future":
        return f"q : fill : tomorrow {s} ( {base} ) {o} . a :", f"tomorrow {s} will {base} {o} ."
    if kind == "now":
        be = "are" if plural else "is"
        return f"q : fill : now {s} ( {base} ) {o} . a :", f"now {s} {be} {vi} {o} ."
    if kind == "present":
        return f"q : fill : today {s} ( {base} ) {o} . a :", f"today {s} {pres} ."
    if kind == "plural":
        n = rng.choice(list(NOUNS))
        k = rng.choice(["two", "three", "many", "four"])
        return f"q : fill : {s} {v3 if not plural else base} {k} ( {n} ) . a :", f"{s} {v3 if not plural else base} {k} {NOUNS[n]} ."
    if kind == "question":
        do = "do" if plural else "does"
        return f"q : make a question : {s} {pres} . a :", f"{do} {s} {base} {o} ?"
    if kind == "yesno":
        do = "do" if plural else "does"
        return f"q : {do} {s} {base} {o} ? a :", f"yes , {s} {pres} ."
    if kind == "negation":
        do = "do" if plural else "does"
        return f"q : say no : {s} {pres} . a :", f"{s} {do} not {base} {o} ."
    raise ValueError(kind)


GRAMMAR_KINDS = ["past", "future", "now", "present", "plural", "question", "yesno", "negation"]


def lesson_grammar(rng: random.Random, split: dict, n: int = 2400) -> list[str]:
    out = []
    for _ in range(n):
        kind = rng.choice(GRAMMAR_KINDS)
        q, a = grammar_item(kind, rng, split["vo_train"])
        out.append(f"{q} {a}")
    # a few plain statements of the same rules, no task frame
    for _ in range(400):
        s, plural = _subject(rng)
        v, o = rng.choice(split["vo_train"])
        base, v3, vp, vi = _forms(v)
        pres = f"{base if plural else v3} {o}"
        be = "are" if plural else "is"
        out.append(f"today {s} {pres} . yesterday {s} {vp} {o} . tomorrow {s} will {base} {o} . now {s} {be} {vi} {o} .")
        n_ = rng.choice(list(NOUNS))
        out.append(f"one {n_} . two {NOUNS[n_]} .")
    rng.shuffle(out)
    return out


def passage(rng: random.Random, split: dict, n_chars: int = 1) -> tuple[list[str], list[tuple[str, str]]]:
    """A 2-4 sentence passage about 1-2 characters + the (question, answer)
    pairs it supports. Facts come from the given split lists only."""
    sents, qa = [], []
    names = rng.sample(NAMES, n_chars)
    for nm in names:
        kinds = rng.sample(["has", "place", "likes", "adj"], rng.choice([2, 2, 3]) if n_chars == 1 else 2)
        for k in kinds:
            if k == "has":
                cands = [f for f in split["fact_train"] if f[0] == nm]
                if not cands:
                    continue
                _, noun, col = rng.choice(cands)
                sents.append(f"{nm} has a {col} {noun} .")
                qa.append((f"what color is {nm} 's {noun} ?", f"{col} ."))
                qa.append((f"who has the {col} {noun} ?", f"{nm} ."))
            elif k == "place":
                cands = [p for p in split["place_train"] if p[0] == nm]
                if not cands:
                    continue
                _, pl = rng.choice(cands)
                sents.append(f"{nm} is at {pl} .")
                qa.append((f"where is {nm} ?", f"at {pl} ."))
            elif k == "likes":
                v = rng.choice(INTRANSITIVE)
                sents.append(f"{nm} likes to {v} .")
                qa.append((f"what does {nm} like to do ?", f"{v} ."))
            elif k == "adj":
                adj = rng.choice(ADJECTIVES)
                sents.append(f"{nm} is {adj} .")
                qa.append((f"how is {nm} ?", f"{adj} ."))
    return sents, qa


def lesson_paragraphs(rng: random.Random, split: dict, n: int = 1200) -> list[str]:
    out = []
    for _ in range(n):
        sents, _ = passage(rng, split, n_chars=rng.choice([1, 1, 2]))
        if len(sents) >= 2:
            out.append(" ".join(sents))
    rng.shuffle(out)
    return out


def comprehension_item(rng: random.Random, split: dict) -> tuple[str, str]:
    while True:
        sents, qa = passage(rng, split, n_chars=rng.choice([1, 2, 2]))
        if len(sents) >= 2 and qa:
            q, a = rng.choice(qa)
            return f"passage : {' '.join(sents)} q : {q} a :", a


def lesson_comprehension(rng: random.Random, split: dict, n: int = 2400) -> list[str]:
    out = []
    for _ in range(n):
        q, a = comprehension_item(rng, split)
        out.append(f"{q} {a}")
    rng.shuffle(out)
    return out


# ---------------------------------------------------------------------------
# Stories (real text)
# ---------------------------------------------------------------------------

STORIES_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "tinystories", "TinyStoriesV2-GPT4-valid.txt")
STORIES_NOTE = ("TinyStories (Eldan & Li, 2023), dataset roneneldan/TinyStories on Hugging Face, "
                "license CDLA-Sharing-1.0 (checked live 2026-10-14); the GPT-4 validation split only "
                "(27,630 stories, 22.5 MB). Attribution kept here and in README.")
STORY_VOCAB_CAP = 3000      # whole words: the most common N in the file; stories with rarer words are left out
STORY_MAX_TOKENS = 300      # longer stories are left out (the context window is 256)


def load_stories(path: str = STORIES_PATH, max_stories: int = 12000, vocab_cap: int = STORY_VOCAB_CAP,
                 max_tokens: int = STORY_MAX_TOKENS, seed: int = 7337) -> dict:
    """Read, normalise, keep only stories inside the capped vocabulary. Returns
    {"stories": [...], "vocab": [...], "stats": {...}}. Empty if the file is absent."""
    if not os.path.exists(path):
        return {"stories": [], "vocab": [], "stats": {"present": False, "path": path}}
    raw = open(path, encoding="utf-8", errors="replace").read()
    chunks = [c.strip() for c in raw.split("<|endoftext|>") if c.strip()]
    norm = [normalize(c) for c in chunks]
    counts = Counter(t for s in norm for t in s.split())
    keep_words = {w for w, _ in counts.most_common(vocab_cap)}
    kept = [s for s in norm if 20 <= len(s.split()) <= max_tokens and all(t in keep_words for t in s.split())]
    rng = random.Random(seed + 3)
    rng.shuffle(kept)
    kept = kept[:max_stories]
    vocab = sorted({t for s in kept for t in s.split()})
    return {"stories": kept, "vocab": vocab, "stats": {
        "present": True, "path": path, "stories_in_file": len(chunks), "distinct_words_in_file": len(counts),
        "vocab_cap": vocab_cap, "kept": len(kept), "tokens": sum(len(s.split()) for s in kept),
        "note": STORIES_NOTE}}


# ---------------------------------------------------------------------------
# The level
# ---------------------------------------------------------------------------

LESSONS2 = ["simple", "grammar", "paragraphs", "comprehension", "stories"]


def build_corpus2(seed: int = 7337, stories: bool = True, max_stories: int = 12000,
                  sizes: dict | None = None, vocab_cap: int = STORY_VOCAB_CAP) -> dict:
    """{lessons: {name: [sentences]}, split, story_stats}. `sizes` lets the CPU
    smoke test shrink every lesson."""
    sizes = sizes or {}
    rng = random.Random(seed + 1)
    split = split_level2(seed)
    lessons = {
        "simple": lesson_simple(rng, split, sizes.get("simple", 900)),
        "grammar": lesson_grammar(rng, split, sizes.get("grammar", 2400)),
        "paragraphs": lesson_paragraphs(rng, split, sizes.get("paragraphs", 1200)),
        "comprehension": lesson_comprehension(rng, split, sizes.get("comprehension", 2400)),
    }
    st = load_stories(max_stories=max_stories, seed=seed, vocab_cap=vocab_cap) if stories else {"stories": [], "stats": {"present": False}}
    lessons["stories"] = st["stories"]
    return {"lessons": lessons, "split": split, "story_stats": st["stats"]}


def exam_items_grammar(seed: int = 7337, n: int = 48) -> list[dict]:
    """E6: held-out (verb, object) pairs, every task kind, seeded."""
    split = split_level2(seed)
    rng = random.Random(seed + 61)
    out = []
    for i in range(n):
        kind = GRAMMAR_KINDS[i % len(GRAMMAR_KINDS)]
        q, a = grammar_item(kind, rng, split["vo_exam"])
        out.append({"kind": kind, "q": q, "a": a})
    return out


def exam_items_reading(seed: int = 7337, n: int = 40) -> list[dict]:
    """E7: passages built ONLY from held-out facts/places, never seen in a lesson."""
    split = split_level2(seed)
    cold = {"fact_train": split["fact_exam"], "place_train": split["place_exam"]}
    rng = random.Random(seed + 71)
    out = []
    while len(out) < n:
        q, a = comprehension_item(rng, cold)
        out.append({"q": q, "a": a})
    return out
