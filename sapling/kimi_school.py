"""Kimi-authored school generators for the Sapling provenance rebuild.

The lesson goals, rough sizes, seed, and cold-split boundaries match the old
school. The teaching sentences and task wording are newly written here. This
module has no imports from the old curriculum and never writes outside the
rebuild directory by itself.
"""
from __future__ import annotations

import hashlib
import os
import random
import re
from collections import Counter
from pathlib import Path

REPAIR = os.environ.get("SAPLING_KIMI_REPAIR", "1") == "1"
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[.,!?;:]|<[a-z0-9|]+>|\+|=|\[|\]")
PUNCTUATION = {".", ",", ";", ":"}


def normalize(text: str) -> str:
    text = text.lower().replace("’", "'").replace("‘", "'")
    return " ".join(TOKEN_RE.findall(text))


def sentence_list(text: str) -> list[str]:
    rows, current = [], []
    for token in text.split():
        current.append(token)
        if token in (".", "!", "?"):
            rows.append(" ".join(current))
            current = []
    if current:
        rows.append(" ".join(current))
    return rows


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
SLOTS = [f"<m{i}>" for i in range(1, 9)]
MAX_RECALLS = 4
FUNCTION_WORDS = (
    "i you she he we they the a an is was will can do does did today now "
    "good small new old box cart load light sound story book house ship "
    "q a recall user confirmed means word root prefix suffix what does of "
    "+ = : . ? [ ]"
).split()
LESSONS = ["roots", "affixes", "composition", "sentences", "recall"]


def compose(parts: list[str]) -> str:
    return " ".join(parts)


def prefix_gloss(root: str, prefix: str) -> str:
    return f"{ROOTS[root]} {PREFIXES[prefix]}"


def suffix_gloss(root: str, suffix: str) -> str:
    return f"{SUFFIXES[suffix]} {ROOTS[root]}"


def split_combos(seed: int = 7337, holdout_frac: float = 0.15) -> dict:
    rng = random.Random(seed)
    pre = [(p, r) for p in PREFIXES for r in ROOTS]
    suf = [(r, s) for r in ROOTS for s in SUFFIXES]
    rng.shuffle(pre)
    rng.shuffle(suf)
    n_pre = max(1, int(len(pre) * holdout_frac))
    n_suf = max(1, int(len(suf) * holdout_frac))
    return {
        "pre_train": pre[n_pre:], "pre_exam": pre[:n_pre],
        "suf_train": suf[n_suf:], "suf_exam": suf[:n_suf],
    }


def lesson_roots(rng: random.Random) -> list[str]:
    out = []
    for root, gloss in ROOTS.items():
        out.extend([
            f"learn the root {root} . it means {gloss} .",
            f"{root} is a root . its meaning is {gloss} .",
            f"the meaning of root {root} is {gloss} .",
            f"q : tell me the meaning of root {root} . a : {root} means {gloss} .",
        ])
    rng.shuffle(out)
    return out


def lesson_affixes(rng: random.Random) -> list[str]:
    out = []
    for prefix, gloss in PREFIXES.items():
        rows = [
            f"learn the prefix {prefix} . it adds the meaning {gloss} .",
            f"{prefix} is a prefix meaning {gloss} .",
            f"q : tell me what prefix {prefix} means . a : it means {gloss} .",
        ]
        if REPAIR:
            rows += [
                f"q : what meaning does the prefix {prefix} carry ? a : {gloss} .",
                f"q : state the prefix {prefix} meaning briefly . a : {gloss} .",
            ]
        out.extend(rows)
    for suffix, gloss in SUFFIXES.items():
        rows = [
            f"learn the suffix {suffix} . it adds the meaning {gloss} .",
            f"{suffix} is a suffix meaning {gloss} .",
            f"q : tell me what suffix {suffix} means . a : it means {gloss} .",
        ]
        if REPAIR:
            rows += [
                f"q : give the short meaning of the suffix {suffix} . a : {gloss} .",
                f"q : state the suffix {suffix} meaning briefly . a : {gloss} .",
            ]
        out.extend(rows)
    rng.shuffle(out)
    return out


def lesson_composition(rng: random.Random, split: dict) -> list[str]:
    out = []
    for first, second, kind in [(p, r, "pre") for p, r in split["pre_train"]] + [(r, s, "suf") for r, s in split["suf_train"]]:
        word = compose([first, second])
        gloss = prefix_gloss(second, first) if kind == "pre" else suffix_gloss(first, second)
        rows = [
            f"join {first} and {second} to get {word} .",
            f"the parts {first} + {second} give {word} .",
            f"the meaning of {word} is {gloss} .",
            f"q : explain {word} . a : it means {gloss} .",
        ]
        if REPAIR:
            rows += [
                f"q : give the meaning of {word} . a : {word} means {gloss} .",
                f"q : describe the word {word} . a : the combined word {word} means {gloss} .",
            ]
        out.extend(rows)
    rng.shuffle(out)
    return out


def lesson_sentences(rng: random.Random, split: dict) -> list[str]:
    out = []
    subjects = ["i", "you", "she", "he", "we", "they"]
    objects = ["the box", "the cart", "the load", "the story", "the book",
               "the house", "the ship", "a light", "a sound"]
    for prefix, root in split["pre_train"]:
        word = compose([prefix, root])
        subject = rng.choice(subjects)
        obj = rng.choice(objects)
        out.append(f"{subject} will {word} {obj} .")
        out.append(f"q : tell me what {subject} will do . a : {subject} will {word} {obj} .")
    for root, suffix in split["suf_train"]:
        word = compose([root, suffix])
        subject = rng.choice(subjects)
        if suffix in ("er", "ist"):
            out.append(f"{subject} is a good {word} .")
        else:
            out.append(f"the {word} was good .")
    rng.shuffle(out)
    return out


def lesson_recall(rng: random.Random) -> list[str]:
    out = []
    pairs = [(SLOTS[i], SLOTS[i + 1]) for i in range(0, len(SLOTS), 2)]
    for _ in range(320):
        count = rng.choice([1, 1, 2, 2, 3, 4])
        chosen = pairs[:count]
        order = list(chosen)
        rng.shuffle(order)
        asked = rng.choice(chosen)
        context = " ".join(f"recall [ user ] : the word {word} means {gloss} ." for word, gloss in order)
        out.append(f"{context} q : use the recall to explain {asked[0]} . a : the word {asked[0]} means {asked[1]} .")
        out.append(f"q : without a recall , what does {asked[0]} mean ? a : i do not have that yet .")
    for _ in range(80):
        root = rng.choice(list(ROOTS))
        gloss = rng.choice(sorted(set(ROOTS.values())))
        out.append(
            f"recall [ confirmed ] : the root {root} means {gloss} . "
            f"q : from the confirmed recall , explain root {root} . a : the root {root} means {gloss} ."
        )
    rng.shuffle(out)
    return out


def build_corpus(seed: int = 7337) -> dict:
    rng = random.Random(seed)
    split = split_combos(seed)
    lessons = {
        "roots": lesson_roots(rng),
        "affixes": lesson_affixes(rng),
        "composition": lesson_composition(rng, split),
        "sentences": lesson_sentences(rng, split),
        "recall": lesson_recall(rng),
    }
    return {"lessons": lessons, "split": split}


def build_vocab(corpus: dict) -> dict:
    tokens = set(FUNCTION_WORDS) | set(SLOTS) | {"<pad>", "<bos>", "<eos>", "<unk>"}
    for rows in corpus["lessons"].values():
        for row in rows:
            tokens.update(row.split())
    vocab = sorted(tokens)
    return {"stoi": {token: i for i, token in enumerate(vocab)}, "itos": vocab}


def corpus_hash(corpus: dict) -> str:
    digest = hashlib.sha256()
    for name in LESSONS:
        for row in corpus["lessons"][name]:
            digest.update(row.encode("utf-8") + b"\n")
    return digest.hexdigest()[:16]


NAMES = ["tom", "anna", "sam", "lily", "ben", "mia", "max", "zoe", "leo", "emma", "jack", "nina"]
NOUNS = {
    "dog": "dogs", "cat": "cats", "ball": "balls", "box": "boxes", "bird": "birds",
    "cup": "cups", "hat": "hats", "book": "books", "fish": "fish", "child": "children",
    "apple": "apples", "car": "cars", "tree": "trees", "house": "houses", "bus": "buses",
    "mouse": "mice", "foot": "feet", "man": "men", "toy": "toys", "cake": "cakes",
    "frog": "frogs", "duck": "ducks", "bag": "bags", "kite": "kites", "boat": "boats",
}
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
OBJECTS = [f"the {noun}" for noun in NOUNS] + [f"a {noun}" for noun in ("ball", "cake", "book", "kite", "hat")]
GRAMMAR_KINDS = ["past", "future", "now", "present", "plural", "question", "yesno", "negation"]
LESSONS2 = ["simple", "grammar", "paragraphs", "comprehension", "stories"]


def verb_forms(verb: str) -> tuple[str, str, str, str]:
    head, _, rest = verb.partition(" ")
    third, past, ongoing = VERBS[head]
    tail = f" {rest}" if rest else ""
    return verb, third + tail, past + tail, ongoing + tail


def split_level2(seed: int = 7337, holdout_frac: float = 0.15) -> dict:
    rng = random.Random(seed + 2)
    pairs = [(verb, obj) for verb in TRANSITIVE for obj in OBJECTS]
    rng.shuffle(pairs)
    n_pairs = max(1, int(len(pairs) * holdout_frac))
    facts = [(name, noun, colour) for name in NAMES for noun in NOUNS for colour in COLOURS]
    rng.shuffle(facts)
    n_facts = max(1, int(len(facts) * holdout_frac))
    places = [(name, place) for name in NAMES for place in PLACES]
    rng.shuffle(places)
    n_places = max(1, int(len(places) * holdout_frac))
    return {
        "vo_train": pairs[n_pairs:], "vo_exam": pairs[:n_pairs],
        "fact_train": facts[n_facts:], "fact_exam": facts[:n_facts],
        "place_train": places[n_places:], "place_exam": places[:n_places],
    }


def make_subject(rng: random.Random) -> tuple[str, bool]:
    pick = rng.random()
    if pick < 0.5:
        return rng.choice(NAMES), False
    if pick < 0.8:
        return f"the {rng.choice(list(NOUNS))}", False
    noun = rng.choice(list(NOUNS))
    return f"the {NOUNS[noun]}", True


def lesson_simple(rng: random.Random, split: dict, n: int = 900) -> list[str]:
    out = []
    for _ in range(n):
        subject, plural = make_subject(rng)
        if rng.random() < 0.65:
            verb, obj = rng.choice(split["vo_train"])
            base, third, _, _ = verb_forms(verb)
            predicate = f"{base if plural else third} {obj}"
            out.append(f"{subject} {predicate} .")
            if rng.random() < 0.5:
                out.append(f"q : describe what {subject} does . a : {subject} {predicate} .")
        else:
            verb = rng.choice(INTRANSITIVE)
            base, third, _, _ = verb_forms(verb)
            place = rng.choice(PLACES)
            has_place = rng.random() < 0.6
            ending = f" at {place}" if has_place else ""
            out.append(f"{subject} {base if plural else third}{ending} .")
            if rng.random() < 0.4:
                answer_place = place if has_place else rng.choice(PLACES)
                out.append(
                    f"q : tell me where {subject} {base} . a : {subject} "
                    f"{base if plural else third} at {answer_place} ."
                )
    rng.shuffle(out)
    return out


def grammar_item(kind: str, rng: random.Random, pairs: list[tuple[str, str]]) -> tuple[str, str]:
    subject, plural = make_subject(rng)
    verb, obj = rng.choice(pairs)
    base, third, past, ongoing = verb_forms(verb)
    present = f"{base if plural else third} {obj}"
    if kind == "past":
        return f"q : use the past : yesterday {subject} ( {base} ) {obj} . a :", f"yesterday {subject} {past} {obj} ."
    if kind == "future":
        return f"q : use the future : tomorrow {subject} ( {base} ) {obj} . a :", f"tomorrow {subject} will {base} {obj} ."
    if kind == "now":
        be = "are" if plural else "is"
        return f"q : use the present continuing form : now {subject} ( {base} ) {obj} . a :", f"now {subject} {be} {ongoing} {obj} ."
    if kind == "present":
        return f"q : use the present : today {subject} ( {base} ) {obj} . a :", f"today {subject} {present} ."
    if kind == "plural":
        noun = rng.choice(list(NOUNS))
        count = rng.choice(["two", "three", "many", "four"])
        return (
            f"q : write the plural : {subject} {third if not plural else base} {count} ( {noun} ) . a :",
            f"{subject} {third if not plural else base} {count} {NOUNS[noun]} .",
        )
    if kind == "question":
        do = "do" if plural else "does"
        return f"q : turn this into a question : {subject} {present} . a :", f"{do} {subject} {base} {obj} ?"
    if kind == "yesno":
        do = "do" if plural else "does"
        return f"q : answer yes : {do} {subject} {base} {obj} ? a :", f"yes , {subject} {present} ."
    if kind == "negation":
        do = "do" if plural else "does"
        return f"q : make this negative : {subject} {present} . a :", f"{subject} {do} not {base} {obj} ."
    raise ValueError(kind)


def lesson_grammar(rng: random.Random, split: dict, n: int = 2400) -> list[str]:
    out = []
    for _ in range(n):
        question, answer = grammar_item(rng.choice(GRAMMAR_KINDS), rng, split["vo_train"])
        out.append(f"{question} {answer}")
    for _ in range(400):
        subject, plural = make_subject(rng)
        verb, obj = rng.choice(split["vo_train"])
        base, third, past, ongoing = verb_forms(verb)
        be = "are" if plural else "is"
        out.append(
            f"today {subject} {base if plural else third} {obj} . "
            f"yesterday {subject} {past} {obj} . tomorrow {subject} will {base} {obj} . "
            f"now {subject} {be} {ongoing} {obj} ."
        )
        noun = rng.choice(list(NOUNS))
        out.append(f"here is one {noun} . here are two {NOUNS[noun]} .")
    rng.shuffle(out)
    return out


def passage(rng: random.Random, split: dict, n_chars: int = 1) -> tuple[list[str], list[tuple[str, str]]]:
    rows, questions = [], []
    for name in rng.sample(NAMES, n_chars):
        kinds = rng.sample(["has", "place", "likes", "adj"], rng.choice([2, 2, 3]) if n_chars == 1 else 2)
        for kind in kinds:
            if kind == "has":
                candidates = [fact for fact in split["fact_train"] if fact[0] == name]
                if not candidates:
                    continue
                _, noun, colour = rng.choice(candidates)
                rows.append(f"a {colour} {noun} belongs to {name} .")
                questions.append((f"tell me the colour of {name} 's {noun} .", f"{colour} ."))
                questions.append((f"which person has a {colour} {noun} ?", f"{name} ."))
            elif kind == "place":
                candidates = [place for place in split["place_train"] if place[0] == name]
                if not candidates:
                    continue
                _, place = rng.choice(candidates)
                rows.append(f"{name} is at {place} .")
                questions.append((f"at what place is {name} ?", f"at {place} ."))
            elif kind == "likes":
                verb = rng.choice(INTRANSITIVE)
                rows.append(f"{name} likes to {verb} .")
                questions.append((f"which activity does {name} enjoy ?", f"{verb} ."))
            else:
                adjective = rng.choice(ADJECTIVES)
                rows.append(f"{name} is {adjective} .")
                questions.append((f"describe {name} .", f"{adjective} ."))
    return rows, questions


def lesson_paragraphs(rng: random.Random, split: dict, n: int = 1200) -> list[str]:
    out = []
    while len(out) < n:
        rows, _ = passage(rng, split, n_chars=rng.choice([1, 1, 2]))
        if len(rows) >= 2:
            out.append(" ".join(rows))
    rng.shuffle(out)
    return out


def comprehension_item(rng: random.Random, split: dict) -> tuple[str, str]:
    while True:
        rows, questions = passage(rng, split, n_chars=rng.choice([1, 2, 2]))
        if len(rows) >= 2 and questions:
            question, answer = rng.choice(questions)
            return f"passage : {' '.join(rows)} q : {question} a :", answer


def lesson_comprehension(rng: random.Random, split: dict, n: int = 2400) -> list[str]:
    out = [f"{question} {answer}" for question, answer in
           (comprehension_item(rng, split) for _ in range(n))]
    rng.shuffle(out)
    return out


def build_corpus2(seed: int = 7337, story_rows: list[str] | None = None,
                  sizes: dict | None = None) -> dict:
    sizes = sizes or {}
    rng = random.Random(seed + 1)
    split = split_level2(seed)
    lessons = {
        "simple": lesson_simple(rng, split, sizes.get("simple", 900)),
        "grammar": lesson_grammar(rng, split, sizes.get("grammar", 2400)),
        "paragraphs": lesson_paragraphs(rng, split, sizes.get("paragraphs", 1200)),
        "comprehension": lesson_comprehension(rng, split, sizes.get("comprehension", 2400)),
        "stories": list(story_rows or []),
    }
    return {"lessons": lessons, "split": split}


DOCS_TRAIN = ["pg2944", "pg20203", "pg2680", "pg1228", "pg205"]
DOCS_HOLDOUT = "pg1022"
PASSAGE_MAX = 150
PASSAGE_MIN = 30
LESSONS3 = ["copy", "bridge", "prose", "docs", "longread"]
COPY_KINDS = ["copy", "again", "last"] + GRAMMAR_KINDS


def strip_gutenberg(raw: str) -> str:
    start = raw.find("*** START OF")
    if start >= 0:
        start = raw.find("\n", start) + 1
    end = raw.find("*** END OF")
    return raw[max(start, 0):end if end > 0 else None]


def document_text(body: str) -> str:
    tokens = normalize(body.replace("\n\n", " . ").replace("\n", " ")).split()
    out, index = [], 0
    while index < len(tokens):
        if tokens[index] not in PUNCTUATION:
            out.append(tokens[index])
            index += 1
            continue
        end = index
        while end < len(tokens) and tokens[end] in PUNCTUATION:
            end += 1
        run = tokens[index:end]
        last_stop = max((i for i, token in enumerate(run) if token == "."), default=-1)
        out.extend(run[last_stop:] if last_stop >= 0 else run)
        index = end
    return " ".join(out)


def chunk_passages(text: str, max_tokens: int = PASSAGE_MAX, min_tokens: int = PASSAGE_MIN) -> list[str]:
    passages, current = [], []
    for sentence in sentence_list(text):
        tokens = sentence.split()
        while len(tokens) > max_tokens:
            if current:
                passages.append(" ".join(current))
                current = []
            passages.append(" ".join(tokens[:max_tokens]))
            tokens = tokens[max_tokens:]
        if current and len(current) + len(tokens) > max_tokens:
            passages.append(" ".join(current))
            current = []
        current.extend(tokens)
    if len(current) >= min_tokens:
        passages.append(" ".join(current))
    return [row for row in passages if len(row.split()) >= min_tokens]


def load_document(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8", errors="replace")
    body = strip_gutenberg(raw) if "Project Gutenberg" in raw[:2000] else raw
    text = document_text(body)
    passages = chunk_passages(text)
    return {
        "path": str(path), "passages": passages,
        "words": Counter(token for row in passages for token in row.split()),
    }


def load_docs(docs_dir: Path) -> dict:
    docs = {}
    for name in DOCS_TRAIN:
        path = docs_dir / f"{name}.txt"
        if path.exists():
            docs[name] = load_document(path)
    return docs


def novel_object(rng: random.Random, words: list[str]) -> str:
    pick = rng.random()
    if pick < 0.5 or not words:
        article = "the" if pick < 0.75 else "a"
        return f"{article} {rng.choice(ADJECTIVES)} {rng.choice(list(NOUNS))}"
    if pick < 0.75:
        return f"the {rng.choice(words)}"
    return f"the {rng.choice(ADJECTIVES)} {rng.choice(words)}"


def copy_item(kind: str, rng: random.Random, words: list[str], story_rows: list[str]) -> tuple[str, str]:
    if kind == "copy":
        tokens = [rng.choice(words) if words else rng.choice(list(NOUNS))
                  for _ in range(rng.randint(3, 9))]
        text = " ".join(tokens)
        return f"q : repeat these words exactly : {text} . a :", f"{text} ."
    if kind == "again" and story_rows:
        text = rng.choice(story_rows)
        return f"q : repeat this passage : {text} a :", text
    if kind == "last":
        tokens = [rng.choice(words) if words else rng.choice(list(NOUNS))
                  for _ in range(rng.randint(3, 8))]
        return f"q : give only the final word : {' '.join(tokens)} . a :", f"{tokens[-1]} ."
    subject, plural = make_subject(rng)
    verb = rng.choice(TRANSITIVE)
    obj = novel_object(rng, words)
    base, third, past, ongoing = verb_forms(verb)
    present = f"{base if plural else third} {obj}"
    do = "do" if plural else "does"
    be = "are" if plural else "is"
    grammar = kind if kind in GRAMMAR_KINDS else rng.choice(GRAMMAR_KINDS)
    if grammar == "past":
        return f"q : use the past : yesterday {subject} ( {base} ) {obj} . a :", f"yesterday {subject} {past} {obj} ."
    if grammar == "future":
        return f"q : use the future : tomorrow {subject} ( {base} ) {obj} . a :", f"tomorrow {subject} will {base} {obj} ."
    if grammar == "now":
        return f"q : use the present continuing form : now {subject} ( {base} ) {obj} . a :", f"now {subject} {be} {ongoing} {obj} ."
    if grammar == "present":
        return f"q : use the present : today {subject} ( {base} ) {obj} . a :", f"today {subject} {present} ."
    if grammar == "plural":
        noun = rng.choice(list(NOUNS))
        count = rng.choice(["two", "three", "many", "four"])
        return (
            f"q : write the plural : {subject} {third if not plural else base} {count} ( {noun} ) . a :",
            f"{subject} {third if not plural else base} {count} {NOUNS[noun]} .",
        )
    if grammar == "question":
        return f"q : turn this into a question : {subject} {present} . a :", f"{do} {subject} {base} {obj} ?"
    if grammar == "yesno":
        return f"q : answer yes : {do} {subject} {base} {obj} ? a :", f"yes , {subject} {present} ."
    return f"q : make this negative : {subject} {present} . a :", f"{subject} {do} not {base} {obj} ."


def lesson_copy(rng: random.Random, words: list[str], story_rows: list[str], n: int = 2400) -> list[str]:
    out = []
    for _ in range(n):
        question, answer = copy_item(rng.choice(COPY_KINDS), rng, words, story_rows)
        out.append(f"{question} {answer}")
    rng.shuffle(out)
    return out


def bridge_rows(first: str, second: str, kind: str) -> list[str]:
    if kind == "pre":
        whole, parts, gloss = first + second, f"{first} {second}", prefix_gloss(second, first)
        joined = f"{first} + {second}"
    else:
        whole, parts, gloss = first + second, f"{first} {second}", suffix_gloss(first, second)
        joined = f"{first} + {second}"
    rows = [
        f"write {parts} without the space to get {whole} .",
        f"{whole} contains the parts {joined} .",
        f"the whole word {whole} means {gloss} .",
        f"q : explain the whole word {whole} . a : it means {gloss} .",
        f"q : split {whole} into its two parts . a : {joined} .",
        f"q : join the parts in {parts} . a : {whole} .",
        f"{whole} and {parts} have the same meaning .",
    ]
    if REPAIR:
        rows.append(f"q : give the meaning of the whole word {whole} . a : the word {whole} means {gloss} .")
    return rows


def lesson_bridge(rng: random.Random, split: dict) -> list[str]:
    out = []
    for prefix, root in split["pre_train"]:
        out.extend(bridge_rows(prefix, root, "pre"))
    for root, suffix in split["suf_train"]:
        out.extend(bridge_rows(root, suffix, "suf"))
    rng.shuffle(out)
    return out


def passage_halves(docs: dict, which: str) -> list[str]:
    out = []
    for doc in docs.values():
        out.extend(row for i, row in enumerate(doc["passages"]) if (i % 2 == 0) == (which == "prose"))
    return out


def document_item(rng: random.Random, passage_text: str, distractor: str | None = None) -> tuple[str, str] | None:
    rows = [row for row in sentence_list(passage_text) if 3 <= len(row.split()) <= 40]
    if not rows:
        return None
    tokens = passage_text.split()
    kind = rng.choice(["which", "which", "after", "first", "before"])
    if kind == "which":
        sentence = rng.choice(rows)
        content = [token for token in sentence.split() if len(token) > 3 and token.isalpha()]
        if not content:
            return None
        word = rng.choice(content)
        if sum(word in row.split() for row in rows) != 1:
            return None
        if distractor and word in distractor.split():
            return None
        question, answer = f"find the sentence containing {word} .", sentence
    elif kind == "after":
        index = rng.randrange(0, len(tokens) - 3)
        pair = " ".join(tokens[index:index + 2])
        if passage_text.count(f" {pair} ") > 1 or (distractor and pair in distractor) or tokens[index + 2] in ".,;:!?":
            return None
        question, answer = f"name the word directly after {pair} .", f"{tokens[index + 2]} ."
    elif kind == "before":
        index = rng.randrange(1, len(tokens) - 2)
        pair = " ".join(tokens[index:index + 2])
        if passage_text.count(f" {pair} ") > 1 or (distractor and pair in distractor) or tokens[index - 1] in ".,;:!?":
            return None
        question, answer = f"name the word directly before {pair} .", f"{tokens[index - 1]} ."
    else:
        question, answer = "repeat the opening sentence .", sentence_list(passage_text)[0]
        if distractor:
            marker = next((token for token in sentence_list(passage_text)[0].split()
                           if len(token) > 4 and token.isalpha() and token not in distractor.split()), "")
            if not marker:
                return None
            question = f"repeat the opening sentence from the text containing {marker} ."
    body = passage_text
    if distractor:
        body = f"{passage_text} {distractor}" if rng.random() < 0.5 else f"{distractor} {passage_text}"
    return f"passage : {body} q : {question} a :", answer


def lesson_prose(rng: random.Random, docs: dict, n: int = 6000) -> list[str]:
    rows = passage_halves(docs, "prose")
    rng.shuffle(rows)
    return rows[:n]


def lesson_docs(rng: random.Random, docs: dict, n: int = 3000) -> list[str]:
    rows = passage_halves(docs, "task")
    out = []
    while rows and len(out) < n:
        item = document_item(rng, rng.choice(rows))
        if item:
            out.append(f"{item[0]} {item[1]}")
    rng.shuffle(out)
    return out


def lesson_longread(rng: random.Random, docs: dict, n: int = 2000) -> list[str]:
    rows = passage_halves(docs, "task")
    out = []
    while len(rows) >= 2 and len(out) < n:
        passage_text, distractor = rng.sample(rows, 2)
        item = document_item(rng, passage_text, distractor)
        if item:
            out.append(f"{item[0]} {item[1]}")
    rng.shuffle(out)
    return out


def build_corpus3(seed: int = 7337, docs_dir: Path | None = None,
                  words: list[str] | None = None, story_rows: list[str] | None = None,
                  sizes: dict | None = None) -> dict:
    sizes = sizes or {}
    rng = random.Random(seed + 31)
    docs = load_docs(docs_dir) if docs_dir else {}
    split = split_combos(seed)
    lessons = {
        "copy": lesson_copy(rng, list(words or []), list(story_rows or []), sizes.get("copy", 2400)),
        "bridge": lesson_bridge(rng, split),
        "prose": lesson_prose(rng, docs, sizes.get("prose", 6000)),
        "docs": lesson_docs(rng, docs, sizes.get("docs", 3000)),
        "longread": lesson_longread(rng, docs, sizes.get("longread", 2000)),
    }
    stats = {
        "documents": len(docs), "holdout": DOCS_HOLDOUT,
        "passages": {name: len(doc["passages"]) for name, doc in docs.items()},
    }
    return {"lessons": lessons, "docs_stats": stats, "split1": split}
