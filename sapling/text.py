"""Text normalisation for level 2 (real sentences).

The core's tokenizer splits on whitespace and the vocabulary is a list of
strings, so everything that reaches it must already be "spaced": lower case,
punctuation as its own token, one space between tokens. Level-1 school text
was written that way by hand ("re + spect = re spect ."); level-2 text (stories,
chat typed on a phone) goes through normalize() first.

Design choice (Claude, 10/14 v3, DESIGN-CHOICES D11): whole words become
tokens. The morpheme-spelled school words ("re spect") stay as they are, so a
word the school taught in parts and the same word met whole in a story are
two different tokens. That is on purpose for now - the school language is a
closed teaching language - and a bridge lesson is the obvious next step.
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[.,!?;:]|<[a-z0-9|]+>|\+|=|\[|\]")

# punctuation the tokenizer keeps as tokens
PUNCT = {".", ",", "!", "?", ";", ":"}


def normalize(text: str) -> str:
    """'Tom said, "I like it!"' -> 'tom said , i like it !'  (quotes dropped,
    numbers kept, apostrophes kept inside a word: don't -> don't)."""
    text = text.lower().replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    toks = _TOKEN.findall(text)
    return " ".join(toks)


def sentences(text: str) -> list[str]:
    """Split normalised text into sentences on . ! ? (each keeps its mark)."""
    out, cur = [], []
    for t in text.split():
        cur.append(t)
        if t in (".", "!", "?"):
            out.append(" ".join(cur))
            cur = []
    if cur:
        out.append(" ".join(cur))
    return out
