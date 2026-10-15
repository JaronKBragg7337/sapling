"""Sapling curriculum, level 3 - getting ready to read real documents.

v3.1 (10/14 night). Jaron's step 2 is "read my documents"; NEXT.md section 0
listed what must come first. These lessons are that list, each one labelled
by construction (fed-not-filled) and deterministic under the seed:

    copy       the answer must reuse the words of the prompt - "copy :", "say
               again :", and every grammar task with an object it has never
               seen (so it cannot answer from habit; E6's misses were exactly
               that: right tense, wrong object, swapped for a familiar one)
    bridge     the two spellings of one word are one word: "re port" (the
               morpheme school) and "report" (a document). Train combos only,
               so E1 stays cold; the whole-word form of a held-out combo is a
               cold check of its own (E1b, info)
    prose      real adult nonfiction, public domain (Project Gutenberg - see
               DOCS_NOTE), in passages of up to ~150 words
    docs       reading tasks on those passages in the E7 frame, the answer
               always a copy out of the passage ("which sentence says X ?",
               "what comes after X Y ?", "copy the first sentence")
    longread   the same, but the prompt holds TWO passages and only one has
               the answer: retrieval with a distractor, inside the context

Held out: one whole essay (DOCS_HOLDOUT, Thoreau's "Walking") never enters a
lesson; its tasks are E7L (info). No document of Jaron's is used (his step 2,
after he checks this one).
"""

from __future__ import annotations

import os
import random
import re
from collections import Counter

from .curriculum import PREFIXES, ROOTS, SUFFIXES, prefix_gloss, split_combos, suffix_gloss
from .curriculum2 import ADJECTIVES, GRAMMAR_KINDS, NOUNS, OBJECTS, _forms, _subject, split_level2
from .text import normalize, sentences

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DOCS_DIR = os.path.join(DATA_DIR, "gutenberg")
DOCS_NOTE = ("Project Gutenberg plain-text editions (public domain in the US; the Gutenberg header/footer and "
             "licence text are stripped, nothing of theirs is trained on): pg2944 Emerson, Essays First Series; "
             "pg20203 Franklin, Autobiography; pg2680 Marcus Aurelius, Meditations (Long); pg1228 Darwin, On the "
             "Origin of Species; pg205 Thoreau, Walden. Held out entirely: pg1022 Thoreau, Walking (E7L + the demo).")
DOCS_TRAIN = ["pg2944", "pg20203", "pg2680", "pg1228", "pg205"]
DOCS_HOLDOUT = "pg1022"
PASSAGE_MAX = 150        # whitespace tokens per passage (two fit in 512 with a question)
PASSAGE_MIN = 30


# ---------------------------------------------------------------------------
# Documents (real text)
# ---------------------------------------------------------------------------

def strip_gutenberg(raw: str) -> str:
    """Keep only the book: between '*** START OF ...' and '*** END OF ...'."""
    lo = raw.find("*** START OF")
    if lo >= 0:
        lo = raw.find("\n", lo) + 1
    hi = raw.find("*** END OF")
    return raw[max(lo, 0): hi if hi > 0 else None]


def chunk_passages(norm_text: str, max_tokens: int = PASSAGE_MAX, min_tokens: int = PASSAGE_MIN) -> list[str]:
    """Sentence-bounded passages of at most max_tokens; a sentence longer than
    that is cut at max_tokens. Short tails are dropped."""
    out, cur, n = [], [], 0
    for s in sentences(norm_text):
        toks = s.split()
        while len(toks) > max_tokens:
            if cur:
                out.append(" ".join(cur)); cur, n = [], 0
            out.append(" ".join(toks[:max_tokens])); toks = toks[max_tokens:]
        if n + len(toks) > max_tokens and cur:
            out.append(" ".join(cur)); cur, n = [], 0
        cur += toks; n += len(toks)
    if cur and n >= min_tokens:
        out.append(" ".join(cur))
    return [p for p in out if len(p.split()) >= min_tokens]


_PUNCT = frozenset({".", ",", ";", ":"})


def doc_text(body: str) -> str:
    """A document body -> one normalised, spaced line (blank lines become sentence ends;
    ". . ." / ", ." runs from blank lines collapse to one ".")."""
    toks = normalize(body.replace("\n\n", " . ").replace("\n", " ")).split()
    # In each run of punctuation tokens, drop everything before the run's last "." (what the old
    # regex did). Token by token because that regex backtracked exponentially on long runs like
    # ", , , , ," in Wikipedia tables and stalled the reading prepare for hours (10/14 night).
    out, i, n = [], 0, len(toks)
    while i < n:
        if toks[i] in _PUNCT:
            j = i
            while j < n and toks[j] in _PUNCT:
                j += 1
            run = toks[i:j]
            last = max((k for k, t in enumerate(run) if t == "."), default=-1)
            out.extend(run[last:] if last >= 0 else run)
            i = j
        else:
            out.append(toks[i])
            i += 1
    return " ".join(out)


def gutenberg_title(raw: str, default: str) -> str:
    for line in raw.splitlines()[:5]:
        if line.lower().startswith("the project gutenberg ebook of"):
            return line.split(" of ", 1)[1].strip().rstrip(".,")
    return default


def load_document(path: str) -> dict:
    """{title, passages, words} for one text file (Gutenberg or any plain text)."""
    raw = open(path, encoding="utf-8", errors="replace").read()
    body = strip_gutenberg(raw) if "Project Gutenberg" in raw[:2000] else raw
    title = gutenberg_title(raw, os.path.splitext(os.path.basename(path))[0])
    norm = doc_text(body)
    passages = chunk_passages(norm)
    return {"title": title, "path": path, "passages": passages,
            "words": Counter(t for p in passages for t in p.split())}


def load_docs(names: list[str] | None = None, docs_dir: str = DOCS_DIR) -> dict:
    """The training documents (DOCS_TRAIN) that are present on disk."""
    names = DOCS_TRAIN if names is None else names
    docs = {}
    for n in names:
        p = os.path.join(docs_dir, n + ".txt")
        if os.path.exists(p):
            docs[n] = load_document(p)
    return docs


# ---------------------------------------------------------------------------
# Lesson generators
# ---------------------------------------------------------------------------

def composed_words() -> dict[str, list[str]]:
    """Every word the level-1 school composes, whole -> parts: the forced splits."""
    out = {}
    for p in PREFIXES:
        for r in ROOTS:
            out[p + r] = [p, "##" + r]
    for r in ROOTS:
        for s in SUFFIXES:
            out[r + s] = [r, "##" + s]
    return out


def _novel_object(rng: random.Random, words: list[str]) -> str:
    """An object no grammar lesson used: adjective + noun, or a word from the
    wider vocabulary. Never a bare OBJECTS entry (those are E6's material)."""
    k = rng.random()
    if k < 0.5:
        return f"the {rng.choice(ADJECTIVES)} {rng.choice(list(NOUNS))}"
    if k < 0.75 and words:
        return f"the {rng.choice(words)}"
    if words:
        return f"the {rng.choice(ADJECTIVES)} {rng.choice(words)}"
    return f"a {rng.choice(ADJECTIVES)} {rng.choice(list(NOUNS))}"


def copy_item(kind: str, rng: random.Random, words: list[str], story_sents: list[str]) -> tuple[str, str]:
    """One (question, answer) where the answer is a copy of prompt words."""
    if kind == "copy":
        n = rng.randint(3, 9)
        toks = [rng.choice(words) if words else rng.choice(list(NOUNS)) for _ in range(n)]
        s = " ".join(toks)
        return f"q : copy : {s} . a :", f"{s} ."
    if kind == "again" and story_sents:
        s = rng.choice(story_sents)
        return f"q : say again : {s} a :", s
    if kind == "last":
        n = rng.randint(3, 8)
        toks = [rng.choice(words) if words else rng.choice(list(NOUNS)) for _ in range(n)]
        return f"q : last word : {' '.join(toks)} . a :", f"{toks[-1]} ."
    # a grammar task with a never-seen object: the only way to be right is to copy it
    s, plural = _subject(rng)
    v = rng.choice(["eat", "see", "make", "find", "like", "want", "hold", "carry", "build", "push", "wash",
                    "open", "throw", "draw", "read", "help", "look at", "play with"])
    o = _novel_object(rng, words)
    base, v3, vp, vi = _forms(v)
    pres = f"{base if plural else v3} {o}"
    do = "do" if plural else "does"
    be = "are" if plural else "is"
    g = kind if kind in GRAMMAR_KINDS else rng.choice(GRAMMAR_KINDS)
    if g == "past":
        return f"q : fill : yesterday {s} ( {base} ) {o} . a :", f"yesterday {s} {vp} {o} ."
    if g == "future":
        return f"q : fill : tomorrow {s} ( {base} ) {o} . a :", f"tomorrow {s} will {base} {o} ."
    if g == "now":
        return f"q : fill : now {s} ( {base} ) {o} . a :", f"now {s} {be} {vi} {o} ."
    if g == "present":
        return f"q : fill : today {s} ( {base} ) {o} . a :", f"today {s} {pres} ."
    if g == "plural":
        n_ = rng.choice(list(NOUNS)); k = rng.choice(["two", "three", "many", "four"])
        return f"q : fill : {s} {v3 if not plural else base} {k} ( {n_} ) . a :", f"{s} {v3 if not plural else base} {k} {NOUNS[n_]} ."
    if g == "question":
        return f"q : make a question : {s} {pres} . a :", f"{do} {s} {base} {o} ?"
    if g == "yesno":
        return f"q : {do} {s} {base} {o} ? a :", f"yes , {s} {pres} ."
    return f"q : say no : {s} {pres} . a :", f"{s} {do} not {base} {o} ."


COPY_KINDS = ["copy", "again", "last"] + GRAMMAR_KINDS


def lesson_copy(rng: random.Random, words: list[str], story_sents: list[str], n: int = 2400) -> list[str]:
    out = []
    for _ in range(n):
        q, a = copy_item(rng.choice(COPY_KINDS), rng, words, story_sents)
        out.append(f"{q} {a}")
    rng.shuffle(out)
    return out


def bridge_sentences(p_or_r: str, r_or_s: str, kind: str) -> list[str]:
    """The lines that tie one composed word to its parts, both spellings."""
    if kind == "pre":
        whole, parts, gloss = p_or_r + r_or_s, f"{p_or_r} {r_or_s}", prefix_gloss(r_or_s, p_or_r)
        plus = f"{p_or_r} + {r_or_s}"
    else:
        whole, parts, gloss = p_or_r + r_or_s, f"{p_or_r} {r_or_s}", suffix_gloss(p_or_r, r_or_s)
        plus = f"{p_or_r} + {r_or_s}"
    return [
        f"{parts} is also written {whole} .",
        f"{whole} is {plus} .",
        f"the word {whole} means {gloss} .",
        f"q : what does {whole} mean ? a : the word {whole} means {gloss} .",
        f"q : what are the parts of {whole} ? a : {plus} .",
        f"q : write {parts} as one word . a : {whole} .",
        f"{whole} and {parts} are the same word .",
    ]


def lesson_bridge(rng: random.Random, split: dict) -> list[str]:
    out = []
    for p, r in split["pre_train"]:
        out += bridge_sentences(p, r, "pre")
    for r, s in split["suf_train"]:
        out += bridge_sentences(r, s, "suf")
    rng.shuffle(out)
    return out


def doc_item(rng: random.Random, passage: str, distractor: str | None = None) -> tuple[str, str] | None:
    """One reading task on a passage, answer = a copy out of it. With a
    distractor, the prompt holds two passages (order random) and the task is
    about `passage` only."""
    sents = [s for s in sentences(passage) if 3 <= len(s.split()) <= 40]
    if not sents:
        return None
    toks = passage.split()
    kind = rng.choice(["which", "which", "after", "first", "before"])
    if kind == "which":
        s = rng.choice(sents)
        content = [t for t in s.split() if len(t) > 3 and t.isalpha()]
        if not content:
            return None
        w = rng.choice(content)
        if sum(1 for x in sents if w in x.split()) != 1:
            return None
        if distractor and w in distractor.split():
            return None
        q, a = f"which sentence has the word {w} ?", s
    elif kind == "after":
        i = rng.randrange(0, len(toks) - 3)
        a2 = " ".join(toks[i:i + 2])
        if passage.count(" " + a2 + " ") > 1 or (distractor and a2 in distractor) or toks[i + 2] in ".,;:!?":
            return None
        q, a = f"what word comes after {a2} ?", f"{toks[i + 2]} ."
    elif kind == "before":
        i = rng.randrange(1, len(toks) - 2)
        a2 = " ".join(toks[i:i + 2])
        if passage.count(" " + a2 + " ") > 1 or (distractor and a2 in distractor) or toks[i - 1] in ".,;:!?":
            return None
        q, a = f"what word comes before {a2} ?", f"{toks[i - 1]} ."
    else:
        q, a = "copy the first sentence .", sentences(passage)[0]
        if distractor:
            q = f"copy the first sentence of the passage with the word {next((t for t in sentences(passage)[0].split() if len(t) > 4 and t.isalpha() and t not in distractor.split()), '')} ."
            if q.endswith("word  ."):
                return None
    body = passage if not distractor else (f"{passage} {distractor}" if rng.random() < 0.5 else f"{distractor} {passage}")
    return f"passage : {body} q : {q} a :", a


def _split_passages(docs: dict, which: str) -> list[str]:
    """Every document's passages, halved: even-numbered ones are PROSE (read as text), odd-numbered
    ones are only ever seen inside a reading TASK. Observed 10/14 20:48 (first v3.1 swap): with tasks
    built on the same passages it had read as prose, it learned the passages, not the copying - docs
    loss 0.17, E7L on a cold essay 0/40. Disjoint halves leave copying as the only way to be right."""
    out = []
    for d in docs.values():
        out += [p for i, p in enumerate(d["passages"]) if (i % 2 == 0) == (which == "prose")]
    return out


def lesson_prose(rng: random.Random, docs: dict, n: int = 6000) -> list[str]:
    allp = _split_passages(docs, "prose")
    rng.shuffle(allp)
    return allp[:n]


def lesson_docs(rng: random.Random, docs: dict, n: int = 3000) -> list[str]:
    allp = _split_passages(docs, "task")
    out = []
    while allp and len(out) < n:
        it = doc_item(rng, rng.choice(allp))
        if it:
            out.append(f"{it[0]} {it[1]}")
    rng.shuffle(out)
    return out


def lesson_longread(rng: random.Random, docs: dict, n: int = 2000) -> list[str]:
    allp = _split_passages(docs, "task")
    out = []
    while len(allp) >= 2 and len(out) < n:
        p, d = rng.sample(allp, 2)
        if len(p.split()) + len(d.split()) > 2 * PASSAGE_MAX:
            continue
        it = doc_item(rng, p, distractor=d)
        if it:
            out.append(f"{it[0]} {it[1]}")
    rng.shuffle(out)
    return out


# ---------------------------------------------------------------------------
# The level
# ---------------------------------------------------------------------------

LESSONS3 = ["copy", "bridge", "prose", "docs", "longread"]


def build_corpus3(seed: int = 7337, docs: bool = True, words: list[str] | None = None,
                  story_sents: list[str] | None = None, sizes: dict | None = None,
                  docs_dir: str = DOCS_DIR) -> dict:
    """{lessons, docs_stats}. `words` = the wider vocabulary the copy lesson
    draws from (whole words from stories + documents); `story_sents` = real
    sentences for "say again". `sizes` shrinks lessons for the CPU smoke test."""
    sizes = sizes or {}
    rng = random.Random(seed + 31)
    split1 = split_combos(seed)
    D = load_docs(docs_dir=docs_dir) if docs else {}
    words = list(words or [])
    lessons = {
        "copy": lesson_copy(rng, words, list(story_sents or []), sizes.get("copy", 2400)),
        "bridge": lesson_bridge(rng, split1),
        "prose": lesson_prose(rng, D, sizes.get("prose", 6000)),
        "docs": lesson_docs(rng, D, sizes.get("docs", 3000)),
        "longread": lesson_longread(rng, D, sizes.get("longread", 2000)),
    }
    stats = {"present": bool(D), "docs": {k: {"title": v["title"], "passages": len(v["passages"]),
                                               "tokens": sum(len(p.split()) for p in v["passages"])} for k, v in D.items()},
             "note": DOCS_NOTE, "holdout": DOCS_HOLDOUT}
    return {"lessons": lessons, "docs_stats": stats, "split1": split1}


# ---------------------------------------------------------------------------
# Exam items (cold)
# ---------------------------------------------------------------------------

def exam_items_longread(seed: int = 7337, n: int = 40, docs_dir: str = DOCS_DIR, text: str | None = None) -> list[dict]:
    """E7L (info): reading tasks on the held-out essay, half with a distractor
    passage. Empty if the essay is not on disk (and no text given)."""
    if text is None:
        p = os.path.join(docs_dir, DOCS_HOLDOUT + ".txt")
        if not os.path.exists(p):
            return []
        d = load_document(p)
    else:
        d = {"passages": chunk_passages(normalize(text))}
    ps = d["passages"]
    if len(ps) < 2:
        return []
    rng = random.Random(seed + 73)
    out = []
    tries = 0
    while len(out) < n and tries < n * 50:
        tries += 1
        if len(out) % 2 == 0:
            it = doc_item(rng, rng.choice(ps))
        else:
            p, dd = rng.sample(ps, 2)
            it = doc_item(rng, p, distractor=dd) if len(p.split()) + len(dd.split()) <= 2 * PASSAGE_MAX else None
        if it:
            out.append({"q": it[0], "a": it[1], "kind": "two passages" if len(out) % 2 else "one passage"})
    return out


def exam_items_bridge(seed: int = 7337) -> list[dict]:
    """E1b (info): the held-out E1 combos asked in their whole-word spelling -
    never taught in either spelling."""
    split = split_combos(seed)
    out = []
    for p, r in split["pre_exam"]:
        out.append({"q": f"q : what does {p + r} mean ? a :", "a": f"the word {p + r} means {prefix_gloss(r, p)} .", "whole": p + r})
    for r, s in split["suf_exam"]:
        out.append({"q": f"q : what does {r + s} mean ? a :", "a": f"the word {r + s} means {suffix_gloss(r, s)} .", "whole": r + s})
    return out
