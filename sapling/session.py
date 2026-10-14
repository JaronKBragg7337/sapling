"""Sapling session — one turn: observe -> meter -> recall -> answer -> record.

Answer layering (fed-not-filled: cheapest first, expensive path rare):
    1. memory hits with parseable facts -> core answers through recall slots
       (up to 4 facts in one prompt; the store's top-k grows with its size)
    2. memory hit, unparseable          -> deterministic cite with provenance
    3. no memory                        -> core free generation (what it has
       become, including facts consolidated into its weights)
    4. core defers                      -> "i do not have that yet" is kept

v2 (Jaron 10/14): the core is no longer forbidden from holding facts - taught
facts live in memory first and can be consolidated into the weights by the
heartbeat (consolidate.py). The vocabulary grows as it meets new words.

Usage:
    python -m sapling.session            # interactive (no heartbeat needed)
    python -m sapling.session --demo     # scripted two-turn demo, fresh memory
"""

from __future__ import annotations

import re
import threading
import time

import torch

from .curriculum import MAX_RECALLS, SLOTS
from .memory import MemoryStore
from .text import normalize, sentences
from .train import load_checkpoint

WORD_MEANS = re.compile(r"^the word (\S+) means (.+?)\s*\.?$", re.I)
SLOT_PAIRS = [(SLOTS[i], SLOTS[i + 1]) for i in range(0, len(SLOTS), 2)]


class Brain:
    """The live core + tokenizer, with a lock (the heartbeat and the chat box
    share it) and vocabulary growth."""

    def __init__(self, store: MemoryStore | None = None, path: str | None = None):
        self.store = store
        self.lock = threading.RLock()
        try:
            self.model, self.tok, self.meta = load_checkpoint(path=path) if path else load_checkpoint()
            self.ok = True
        except FileNotFoundError:
            self.model = self.tok = self.meta = None
            self.ok = False
        self.generation = int((self.meta or {}).get("generation", 0))

    @property
    def device(self):
        return next(self.model.parameters()).device

    def vocab_size(self) -> int:
        return len(self.tok.itos) if self.ok else 0

    def grow_vocab_for(self, text: str, why: str = "met") -> list[str]:
        """Add every word part it has not seen before (Jaron 10/14: vocab grows)."""
        if not self.ok:
            return []
        new = self.tok.unknown_tokens(text)
        if not new:
            return []
        with self.lock:
            self.tok.add_tokens(new)
            self.model.grow_vocab(len(self.tok.itos))
        if self.store is not None:
            self.store.meter("vocab_grow", {"added": new, "why": why, "vocab_size": len(self.tok.itos)})
        return new

    @torch.no_grad()
    def generate(self, prompt: str, max_new: int = 24, temperature: float = 0.0, seed: int | None = None) -> str:
        with self.lock:
            was_training = self.model.training
            self.model.eval()
            ids = torch.tensor([self.tok.encode(prompt, eos=False)], device=self.device)
            out = self.model.generate(ids, max_new=max_new, eos_id=self.tok.stoi["<eos>"],
                                      temperature=temperature, seed=seed)
            if was_training:
                self.model.train()
            return self.tok.decode(out[0, ids.shape[1]:].tolist())

    def core_info(self) -> dict:
        if not self.ok:
            return {"version": None, "params": 0, "config": None}
        return {"version": int((self.meta or {}).get("core_version", 2)), "params": self.model.n_params(),
                "config": self.model.config(), "grown_from": (self.meta or {}).get("grown_from")}

    @torch.no_grad()
    def embed(self, text: str) -> list[float]:
        """Mean-pooled token embeddings from its own core: its sense of
        similarity, which grows as its vocabulary and weights do."""
        ids = [self.tok.stoi[t] for t in text.lower().split() if t in self.tok.stoi
               and t not in ("<unk>", ".", "?", ":")]
        if not ids:
            return [0.0] * self.model.tok_emb.weight.shape[1]
        with self.lock:
            v = self.model.tok_emb.weight[torch.tensor(ids, device=self.device)].mean(dim=0)
        return v.float().cpu().tolist()


class Session:
    def __init__(self, memory_path: str | None = None, brain: Brain | None = None,
                 store: MemoryStore | None = None, clock=None):
        self.store = store or (MemoryStore(memory_path) if memory_path else MemoryStore())
        self.brain = brain or Brain(store=self.store)
        self.clock = clock
        self.turns = 0

    @property
    def has_core(self) -> bool:
        return self.brain.ok

    # -- the one-loop turn -----------------------------------------------------
    def ask(self, query: str) -> str:
        t0 = time.time()
        query = normalize(query)            # v3: phone typing ("mean?") -> school spacing ("mean ?")
        if self.has_core:
            self.brain.grow_vocab_for(query, why="chat")
        embed_fn = self.brain.embed if self.has_core else None
        hits = self.store.search(query, embed_fn=embed_fn)
        slots: dict[str, str] = {}

        parsed = []
        for h in hits:
            if h["state"] == "invalidated":
                continue
            m = WORD_MEANS.match(h["content"].strip())
            if m:
                parsed.append((h, m.group(1).lower(), m.group(2).strip()))
        # the fact the question is about goes first; at most MAX_RECALLS ride along
        qwords = {w.strip(".,?:!") for w in query.lower().split()}
        parsed.sort(key=lambda t: t[1] not in qwords)
        parsed = parsed[:MAX_RECALLS]

        if parsed and self.has_core:
            recall_lines = []
            for (h, word, gloss), (sw, sg) in zip(parsed, SLOT_PAIRS):
                slots[sw], slots[sg] = word, gloss
                recall_lines.append(f"recall [ {h['provenance']} ] : the word {sw} means {sg} .")
            asked = SLOT_PAIRS[0][0]
            prompt = " ".join(recall_lines) + f" q : what does {asked} mean ? a :"
            raw = self.brain.generate(prompt)
            for tok, real in slots.items():
                raw = raw.replace(tok, real)
            cites = " ".join(f"[memory:{h['id']} {h['provenance']}/{h['state']}]" for h, _, _ in parsed)
            answer = f"{raw}  {cites}"
        elif hits and not self.has_core:
            h = hits[0]
            answer = (f"i recall : {h['content']} "
                      f"[memory:{h['id']} {h['provenance']}/{h['state']}, tier={h['tier']}]")
        elif self.has_core:
            # v3: it answers in sentences where it can (level 2 taught "q : ... ? a : <sentence>").
            # Two or more sentences ending in a question go in the reading frame it was taught
            # ("passage : ... q : ... ? a :"); an unparseable memory hit rides along as a citation.
            q = query if query.endswith(("?", ".", "!")) else query + " ?"
            parts = sentences(q)
            if len(parts) >= 2 and parts[-1].endswith("?"):
                prompt = f"passage : {' '.join(parts[:-1])} q : {parts[-1]} a :"
                answer = self.brain.generate(prompt, max_new=24)
            else:
                answer = self.brain.generate(f"q : {q} a :", max_new=48)
            if hits:
                h = hits[0]
                answer += f"  [memory:{h['id']} {h['provenance']}/{h['state']}: {h['content'][:80]}]"
        else:
            answer = "i do not have that yet . (no trained core found — run train.py)"

        self.turns += 1
        self.store.meter("consumption", {
            "turn": self.turns, "tokens_in": len(query.split()),
            "tokens_out": len(answer.split()), "recalls": len(hits),
            "ms": round((time.time() - t0) * 1000, 1),
        })
        return answer

    # -- commands ---------------------------------------------------------------
    def write(self, prompt: str) -> str:
        """Free writing (seeded sampling, reproducible from the seed = turn count)."""
        if not self.has_core:
            return "i do not have that yet . (no trained core found)"
        prompt = normalize(prompt) or "once upon a time ,"
        self.brain.grow_vocab_for(prompt, why="chat")
        out = self.brain.generate(prompt, max_new=80, temperature=0.8, seed=self.turns + 1)
        self.turns += 1
        self.store.meter("consumption", {"turn": self.turns, "tokens_in": len(prompt.split()),
                                         "tokens_out": len(out.split()), "recalls": 0, "kind": "write"})
        return f"{prompt} {out}".strip()

    def teach(self, fact: str) -> str:
        fact = normalize(fact).rstrip(".").strip()
        new = self.brain.grow_vocab_for(fact, why="teach") if self.has_core else []
        mid = self.store.remember(fact, provenance="user", state="captured")
        extra = f"; new word parts: {' '.join(new)}" if new else ""
        return f"remembered as memory:{mid} (user/captured{extra})"

    def keep(self, memory_id: int) -> str:
        mid = self.store.remember(f"keep #{memory_id}", provenance="verdict", state="confirmed")
        return f"verdict recorded as memory:{mid}; memory:{memory_id} -> permanent"

    def confirm(self, memory_id: int) -> str:
        mid = self.store.remember(f"confirm #{memory_id}", provenance="verdict", state="confirmed")
        return f"verdict recorded as memory:{mid}; memory:{memory_id} -> confirmed (resolved)"

    def invalidate(self, memory_id: int) -> str:
        mid = self.store.remember(f"invalidate #{memory_id}", provenance="verdict", state="confirmed")
        return f"verdict recorded as memory:{mid}; memory:{memory_id} -> invalidated"

    def report(self) -> str:
        lines = ["memory report (chain ok: %s, %d memories, vocab %d)" % (
            self.store.verify_chain(), self.store.count(), self.brain.vocab_size())]
        for r in self.store.report(limit=40):
            lines.append(
                f"  #{r['id']:<3} [{r['tier']:<10}] recalls={r['recalls']:<3} "
                f"strength={r['strength']:<8} ({r['provenance']}/{r['state']}) "
                f"{r['content']}"
            )
        return "\n".join(lines)

    def handle(self, line: str) -> str:
        line = line.strip()
        if self.clock is not None:
            self.clock.note_input()
        if line.startswith("/teach "):
            return self.teach(line[len("/teach "):])
        if line.startswith("/write"):
            return self.write(line[len("/write"):])
        if line.startswith("/keep "):
            return self.keep(int(line[len("/keep "):].strip()))
        if line.startswith("/confirm "):
            return self.confirm(int(line[len("/confirm "):].strip()))
        if line.startswith("/invalidate "):
            return self.invalidate(int(line[len("/invalidate "):].strip()))
        if line == "/report":
            return self.report()
        if line in ("/quit", "/exit"):
            raise SystemExit
        return self.ask(line)


def demo():
    """Scripted proof: taught once here, recalled in a LATER process (demo2)."""
    import tempfile, os
    path = os.path.join(tempfile.gettempdir(), "sapling_demo_memory.db")
    if os.path.exists(path):
        os.remove(path)
    s = Session(memory_path=path)
    print("== session 1 (teach) ==")
    for fact in ("the word glorp means happy", "the word brindle means a small brave boat"):
        print(f"> /teach {fact}")
        print(s.teach(fact))
    print("> what does glorp mean ?")
    print(s.ask("what does glorp mean ?"))
    print(f"demo memory file: {path}  (kept for --demo2)")


def demo2():
    import tempfile, os
    path = os.path.join(tempfile.gettempdir(), "sapling_demo_memory.db")
    s = Session(memory_path=path)
    print("== session 2 (new process, same memory) ==")
    for q in ("what does glorp mean ?", "what does brindle mean ?", "what does the root spect mean ?"):
        print(f"> {q}")
        print(s.ask(q))
    print("> /report")
    print(s.report())


def main():
    import sys
    if "--demo2" in sys.argv:
        demo2(); return
    if "--demo" in sys.argv:
        demo(); return
    s = Session()
    print("sapling session. commands: /teach <fact>  /keep <id>  /confirm <id>  /invalidate <id>  /report  /quit")
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            break
        try:
            print(s.handle(line))
        except SystemExit:
            break


if __name__ == "__main__":
    main()
