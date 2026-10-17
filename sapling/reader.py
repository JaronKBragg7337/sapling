"""Reading a long document in chunks and pulling the right passages into the prompt.

v3.1 (10/14 night). The E7 frame - "passage : ... q : ... a :" - scaled up:

    read_document(store, path | text)  -> the document becomes one `documents`
        row (title, source, sha256, chained) and one memory per passage
        (provenance "retrieval", doc=<id>, passage=<n>); nothing is trained
        here. Reading the same text twice adds nothing (same sha256).
    retrieve(store, question)          -> the passages that cover the most of
        the question's content words (a coverage score, not Jaccard: a long
        passage must not lose to a short fact), re-ranked by the core's own
        embeddings when given; every pick is a recorded recall, so a passage
        that keeps answering is protected the way any used memory is.
    ask(brain, store, question)        -> up to `context budget` tokens of the
        best passages go into the prompt in front of the question; the answer
        comes back with [doc:<id> "<title>" passage <n> memory:<id>] citations.

What it reads goes into memory with provenance, and the heartbeat's
consolidation trains on retrieval memories like any other (no gate; Jaron
10/14). Jaron's own documents are his step 2 - this module is tested on
public-domain text (curriculum3.DOCS_NOTE) until he says go.
"""

from __future__ import annotations

import hashlib
import os

from .curriculum3 import chunk_passages, load_document
from .memory import MemoryStore, _cosine
from .text import normalize

STOP = MemoryStore.STOPWORDS | {"which", "who", "where", "when", "how", "why", "that", "this", "with", "for",
                                "from", "by", "or", "but", "not", "his", "her", "their", "its", "did", "has",
                                "have", "had", "be", "been", "were", "as", "sentence", "say", "says", "about"}


def _content_words(text: str) -> set[str]:
    return {w.strip(".,?:!;") for w in text.lower().split()} - STOP - {""}


def read_document(store: MemoryStore, path: str | None = None, text: str | None = None,
                  title: str | None = None, source: str | None = None, brain=None) -> dict:
    """Chunk a document into passages and remember each with its provenance."""
    if path:
        d = load_document(path)
        passages, title = d["passages"], title or d["title"]
        source = source or os.path.abspath(path)
        raw = open(path, encoding="utf-8", errors="replace").read()
    else:
        raw = text or ""
        passages = chunk_passages(normalize(raw.replace("\n\n", " . ").replace("\n", " ")), min_tokens=3)
        title = title or (normalize(raw)[:40] + ("..." if len(raw) > 40 else ""))
        source = source or "chat"
    sha = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()
    dup = store.document_by_sha(sha)
    if dup:
        return {"doc": dup["id"], "title": dup["title"], "passages": dup["passages"], "new": False}
    doc_id = store.add_document(title, source, sha, len(raw), len(passages))
    ids = []
    for i, p in enumerate(passages):
        ids.append(store.remember(p, provenance="retrieval", state="captured", doc=doc_id, passage=i))
    store.meter("read_document", {"doc": doc_id, "title": title, "source": source, "sha256": sha[:16],
                                  "chars": len(raw), "passages": len(passages), "memory_ids": [ids[0], ids[-1]] if ids else []})
    return {"doc": doc_id, "title": title, "passages": len(passages), "new": True, "memory_ids": ids}


def retrieve(store: MemoryStore, question: str, k: int = 3, doc: int | None = None,
             embed_fn=None, record: bool = True) -> list[dict]:
    """Best passages for a question: coverage of the question's content words
    (x the memory's strength), embeddings re-rank, recalls recorded."""
    q = _content_words(question)
    if not q:
        return []
    qvec = embed_fn(question) if embed_fn else None
    scored = []
    q_long = {w for w in q if len(w) > 3}
    for p in store.passages(doc):
        c = _content_words(p["content"])
        cov = len(q & c) / len(q)
        if cov <= 0:
            continue
        rel = cov
        if qvec is not None:
            rel *= 0.7 + 0.3 * max(0.0, _cosine(qvec, embed_fn(p["content"])))
        st = store.strength(p["id"])
        st = 3.0 if st == float("inf") else min(st, 3.0)        # strength breaks ties; it never outranks coverage
        scored.append({**p, "cov": round(cov, 3), "nmatch_long": len(q_long & c), "rel": round(rel, 3),
                       "score": rel * (1 + 0.1 * st)})
    scored.sort(key=lambda r: (-r["score"], r["doc"], r["passage"]))
    hits = scored[:k]
    if record:
        for h in hits:
            store.record_recall(h["id"], question)
    return hits


def build_prompt(question: str, hits: list[dict], block_size: int, tok, max_new: int = 48) -> tuple[str, list[dict]]:
    """Pack the best passages (in document order) in front of the question,
    inside the context: block_size - question - room for the answer."""
    q = normalize(question)
    if not q.endswith(("?", ".", "!")):
        q += " ?"
    budget = block_size - len(tok.tokens(f"passage : q : {q} a :")) - max_new - 2
    used, n = [], 0
    for h in hits:
        t = len(tok.tokens(h["content"]))
        if n + t > budget:
            continue
        used.append(h); n += t
    used.sort(key=lambda h: (h["doc"], h["passage"]))
    body = " ".join(h["content"] for h in used)
    return f"passage : {body} q : {q} a :", used


def ask(brain, store: MemoryStore, question: str, k: int = 3, doc: int | None = None, max_new: int = 48) -> dict:
    """Retrieve, read, answer, cite."""
    hits = retrieve(store, question, k=k, doc=doc, embed_fn=brain.embed if brain.ok else None)
    if not hits:
        return {"answer": "i do not have that yet .", "used": [], "prompt": None}
    prompt, used = build_prompt(question, hits, brain.model.block_size, brain.tok, max_new=max_new)
    out = brain.generate(prompt, max_new=max_new).strip()
    titles = {d["id"]: d["title"] for d in store.documents()}
    cites = " ".join(f'[doc:{h["doc"]} "{titles.get(h["doc"], "?")[:30]}" passage {h["passage"]} memory:{h["id"]}]' for h in used)
    return {"answer": out, "cites": cites, "used": used, "prompt": prompt}


def documents_summary(store: MemoryStore) -> list[dict]:
    out = []
    for d in store.documents():
        ps = store.passages(d["id"])
        out.append({**d, "recalls": sum(store.recall_count(p["id"]) for p in ps)})
    return out
