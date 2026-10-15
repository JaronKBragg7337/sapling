"""Word pieces - how an unknown word is spelled from parts it already knows.

v3.1 (10/14 night, Jaron: "prepare it to read my documents"; DESIGN-CHOICES D16).
The tokenizer stays whitespace + a fixed list (every old checkpoint and exam
unchanged). What changes: a word that is NOT in the list no longer becomes
<unk>. It is spelled from pieces, WordPiece-style - the first piece plain, the
rest marked with "##" ("archaeology" -> "arch ##ae ##ology"), and decode()
joins them back. The pieces are learned here (plain BPE merges over the words
that are too rare to be whole tokens), plus every single letter/digit, so any
word made of those characters is always spellable.

Forced splits (the bridge to the morpheme school): every word the level-1
school composes from parts - "report" = re + port, "spector" = spect + er -
is spelled as its parts, "re ##port", never as a whole token or a greedy
guess. So a word the school taught in parts and the same word met whole in a
document share their tokens ("##port" is one step from "port"); the bridge
lesson (curriculum3.py) teaches that step. Held-out E1 combos get the same
treatment, so the bridge can be examined cold.
"""

from __future__ import annotations

from collections import Counter, defaultdict

MARK = "##"
CHARS = list("abcdefghijklmnopqrstuvwxyz0123456789'")


def base_pieces() -> list[str]:
    """Every character, plain and as a continuation piece: the floor under spelling."""
    return CHARS + [MARK + c for c in CHARS]


def _symbols(word: str) -> tuple[str, ...]:
    return tuple([word[0]] + [MARK + c for c in word[1:]])


def learn_pieces(word_counts: dict[str, int], n_merges: int = 1500, min_count: int = 2) -> list[str]:
    """Plain BPE over the given words (weighted by count). Returns the learned
    pieces (merged symbols) in the order they were made. Deterministic."""
    words = {w: _symbols(w) for w, c in word_counts.items()
             if c >= min_count and w and all(ch in CHARS for ch in w) and len(w) > 1}
    if not words:
        return []
    pair_counts: Counter = Counter()
    where: dict[tuple, set] = defaultdict(set)
    for w, syms in words.items():
        c = word_counts[w]
        for a, b in zip(syms, syms[1:]):
            pair_counts[(a, b)] += c
            where[(a, b)].add(w)
    learned: list[str] = []
    for _ in range(n_merges):
        if not pair_counts:
            break
        (a, b), cnt = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))
        if cnt < min_count:
            break
        merged = a + b[len(MARK):]
        learned.append(merged)
        for w in list(where[(a, b)]):
            syms = words[w]
            c = word_counts[w]
            for x, y in zip(syms, syms[1:]):          # take the old pairs out
                pair_counts[(x, y)] -= c
                if pair_counts[(x, y)] <= 0:
                    del pair_counts[(x, y)]
                where[(x, y)].discard(w)
            out, i = [], 0
            while i < len(syms):
                if i + 1 < len(syms) and syms[i] == a and syms[i + 1] == b:
                    out.append(merged); i += 2
                else:
                    out.append(syms[i]); i += 1
            syms = tuple(out)
            words[w] = syms
            for x, y in zip(syms, syms[1:]):          # and the new pairs in
                pair_counts[(x, y)] += c
                where[(x, y)].add(w)
    return learned


_LONGEST: dict[int, tuple[int, int]] = {}


def _longest_key(stoi: dict) -> int:
    k = id(stoi)
    v = _LONGEST.get(k)
    if v is None or v[1] != len(stoi):
        v = _LONGEST[k] = (max((len(t) for t in stoi), default=1), len(stoi))
    return v[0]


def split_word(word: str, stoi: dict, splits: dict | None = None) -> list[str] | None:
    """Spell one unknown word from pieces: greedy longest match from the left,
    first piece plain, the rest "##"-marked. A forced split wins. None if the
    word cannot be spelled (a character that has no piece)."""
    if splits and word in splits:
        return list(splits[word])
    out, i, n = [], 0, len(word)
    # No piece is longer than the longest key, so longer candidates can't match: same split, and
    # a 100k-character "word" (a table or URL with no spaces) no longer costs n^2 lookups.
    longest = _longest_key(stoi)
    while i < n:
        found = None
        for j in range(min(n, i + longest + len(MARK)), i, -1):
            cand = word[i:j] if i == 0 else MARK + word[i:j]
            if cand in stoi:
                found = cand; i = j
                break
        if found is None:
            return None
        out.append(found)
    return out


def join_pieces(tokens: list[str]) -> list[str]:
    """'arch ##ae ##ology' -> 'archaeology' (decode side)."""
    out: list[str] = []
    for t in tokens:
        if t.startswith(MARK) and out:
            out[-1] = out[-1] + t[len(MARK):]
        else:
            out.append(t[len(MARK):] if t.startswith(MARK) else t)
    return out
