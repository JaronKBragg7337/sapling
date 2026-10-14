"""Sapling session — one loop turn: observe -> meter -> recall -> answer -> record.

Answer layering (fed-not-filled: cheapest first, expensive path rare):
    1. memory strong hit with a parseable fact  -> core answers through recall slots
    2. memory hit, unparseable                   -> deterministic cite with provenance
    3. no memory                                 -> core free generation
    4. core defers                               -> "i do not have that yet" is kept

The core never memorises taught facts — they live in the memory layer, which is
why session 2 (a new process) can recall what session 1 was taught.

Usage:
    python -m sapling.session            # interactive
    python -m sapling.session --demo     # scripted two-turn demo, fresh memory
"""

from __future__ import annotations

import re
import time

import torch

from .memory import MemoryStore
from .train import load_checkpoint

WORD_MEANS = re.compile(r"^the word (\w+) means (.+)$", re.I)


def load_core():
    try:
        return (*load_checkpoint(), True)
    except FileNotFoundError:
        return None, None, None, False


class Session:
    def __init__(self, memory_path: str | None = None):
        self.store = MemoryStore(memory_path) if memory_path else MemoryStore()
        self.model, self.tok, _ckpt, self.has_core = load_core()
        self.turns = 0

    # -- core generation -----------------------------------------------------
    def _generate(self, prompt: str, max_new: int = 24) -> str:
        ids = torch.tensor([self.tok.encode(prompt, eos=False)],
                           device=next(self.model.parameters()).device)
        out = self.model.generate(ids, max_new=max_new, eos_id=self.tok.stoi["<eos>"])
        return self.tok.decode(out[0, ids.shape[1]:].tolist())

    # -- the one-loop turn -----------------------------------------------------
    def ask(self, query: str) -> str:
        t0 = time.time()
        hits = self.store.search(query, limit=2)
        slots: dict[str, str] = {}
        answer = ""

        parsed = []
        for h in hits:
            m = WORD_MEANS.match(h["content"].strip())
            if m:
                parsed.append((h, m.group(1).lower(), m.group(2).strip()))

        if parsed:
            # map recalled words onto slot tokens, strongest hit first
            slot_pairs = [("<m1>", "<m2>"), ("<m3>", "<m4>")]
            recall_lines = []
            for (h, word, gloss), (sw, sg) in zip(parsed, slot_pairs):
                slots[sw], slots[sg] = word, gloss
                recall_lines.append(
                    f"recall [ {h['provenance']} ] : the word {sw} means {sg} ."
                )
            # which memory does the query ask about?
            qwords = set(query.lower().split())
            asked = next((sw for (h, w, g), (sw, sg) in zip(parsed, slot_pairs)
                          if w in qwords), slot_pairs[0][0])
            prompt = " ".join(recall_lines) + f" q : what does {asked} mean ? a :"
            raw = self._generate(prompt)
            for tok, real in slots.items():
                raw = raw.replace(tok, real)
            prov = parsed[0][0]["provenance"]
            state = parsed[0][0]["state"]
            answer = f"{raw}  [memory:{parsed[0][0]['id']} {prov}/{state}]"
        elif hits:
            h = hits[0]
            answer = (f"i recall : {h['content']} "
                      f"[memory:{h['id']} {h['provenance']}/{h['state']}, tier={h['tier']}]")
        elif self.has_core:
            answer = self._generate(f"q : {query.lower()} a :")
        else:
            answer = "i do not have that yet . (no trained core found — run train.py)"

        self.turns += 1
        self.store.meter("consumption", {
            "turn": self.turns, "tokens_in": len(query.split()),
            "tokens_out": len(answer.split()), "ms": round((time.time() - t0) * 1000, 1),
        })
        return answer

    # -- commands ---------------------------------------------------------------
    def teach(self, fact: str) -> str:
        mid = self.store.remember(fact.strip(), provenance="user", state="captured")
        return f"remembered as memory:{mid} (user/captured)"

    def keep(self, memory_id: int) -> str:
        mid = self.store.remember(f"keep #{memory_id}", provenance="verdict", state="confirmed")
        return f"verdict recorded as memory:{mid}; memory:{memory_id} -> permanent"

    def report(self) -> str:
        lines = ["memory report (chain ok: %s)" % self.store.verify_chain()]
        for r in self.store.report():
            lines.append(
                f"  #{r['id']:<3} [{r['tier']:<10}] recalls={r['recalls']:<3} "
                f"strength={r['strength']:<8} ({r['provenance']}/{r['state']}) "
                f"{r['content']}"
            )
        return "\n".join(lines)

    def handle(self, line: str) -> str:
        line = line.strip()
        if line.startswith("/teach "):
            return self.teach(line[len("/teach "):])
        if line.startswith("/keep "):
            return self.keep(int(line[len("/keep "):].strip()))
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
    print("> /teach the word glorp means happy")
    print(s.teach("the word glorp means happy"))
    print("> /teach the word brindle means a small brave boat")
    print(s.teach("the word brindle means a small brave boat"))
    print("> what does glorp mean ?")
    print(s.ask("what does glorp mean ?"))
    print(f"demo memory file: {path}  (kept for --demo2)")


def demo2():
    import tempfile, os
    path = os.path.join(tempfile.gettempdir(), "sapling_demo_memory.db")
    s = Session(memory_path=path)
    print("== session 2 (new process, same memory) ==")
    print("> what does glorp mean ?")
    print(s.ask("what does glorp mean ?"))
    print("> what does brindle mean ?")
    print(s.ask("what does brindle mean ?"))
    print("> what does spect mean ?")
    print(s.ask("what does the root spect mean ?"))
    print("> /report")
    print(s.report())


def main():
    import sys
    if "--demo2" in sys.argv:
        demo2()
        return
    if "--demo" in sys.argv:
        demo()
        return
    s = Session()
    print("sapling session. commands: /teach <fact>  /keep <id>  /report  /quit")
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
