# Sapling — next steps (after v2)

v1's three steps: (1) runs-without-you is DONE (the heartbeat); (2) record -> consolidation
is DONE (no gate; `/confirm` and `/invalidate` verdicts are the selection events); (3) a world
to practice in is still open. Ordered by dependency.

## 0. After level 2 (v3, 10/14 evening) - what step 2 (his documents) needs to be accurate

Level 2 gave it sentences from generated lessons + children's stories. Before it reads Jaron's own
documents (his call, after he checks this step) it needs, in order:
1. **A bridge lesson** between the two spellings: "report is re + port" - so the morpheme school and the
   whole-word reader are one language, not two (DESIGN-CHOICES D11).
2. **A bigger reading vocabulary** than a first reader's 3,000 words: his documents use words no children's
   story has. Either raise the story cap and read the TinyStories train split (2.2 GB, same license), or
   move to sub-word tokens (BPE) - the second makes old checkpoints and E1 incomparable, so it is his call.
3. **Longer context** than 256 tokens for a document (512-1024), and a retrieval path that feeds a passage
   into the prompt the way `recall [ user ]` feeds a fact today (the E7 "passage : ... q : ... a :" frame is
   that path in the small).
4. **Provenance per document** in memory.db (which file, which line) so an answer can cite it - the
   `retrieval/captured` provenance already exists for this.
5. **An exam from his documents that he writes himself** (cold, like E7) - the only honest check.

## 1. Let it run for days, then read the record (no build)

The 20-minute run proved the loop; what Jaron asked for is behavior over time. Start it
(`python -m sapling.heartbeat start`), leave it, and read: how it spends a full day's budget,
whether it ever self-stops and why (stop reports), whether consolidations drift the exams,
what its explore ticks produce. `PROPOSALS.md` holds the questions the first run raised.
Smallest extra: a Windows scheduled task that starts it at boot (owner's hand = parole).

## 2. One real reference per day (the Live-Reference feed)

One live source (Wikipedia summary or one RSS feed) pulled by an `explore` tick, stored as
`retrieval/captured`, so consolidation has something newer than the school corpus. Success:
an answer that cites a fact newer than training - the Live-Reference test, passed in the small.

## 3. A world to practice in (the third leg)

A tiny deterministic grid world (10x10, a survival budget of its own) where the core proposes
one action per tick, the simulator validates it, and the outcome becomes a `self` memory and
a practice goal. The ECIH move: real constraint, real feedback, labels free by construction.
This is where the budget pressure (clock.py) and the goals (goals.py) meet a body.

## 4. The organ layout

Supabase (pgvector) as working memory behind the same `MemoryStore` interface; GitHub export of
the ledger (hash chain as commits); Arweave only for `keep` verdicts. Adapters, not foundations.

## Open with Jaron (not built on purpose)
- outbound_message and backup approval rules: keep or drop (membrane.yaml marks them pending).
- the six items in `PROPOSALS.md`.
