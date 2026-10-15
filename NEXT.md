# Sapling — next steps (after v2)

v1's three steps: (1) runs-without-you is DONE (the heartbeat); (2) record -> consolidation
is DONE (no gate; `/confirm` and `/invalidate` verdicts are the selection events); (3) a world
to practice in is still open. Ordered by dependency.

## 0. Step 2 - Jaron's documents (his call; the prerequisites are DONE in v3.1, 10/14 night)

Everything the list below asked for is built (CHANGELOG v3.1, DESIGN-CHOICES D16-D19): copy lesson (E6 12 -> 21/48),
bridge lesson (E1b 81/86 cold), vocabulary 17,660 with word pieces (any adult word spellable, E1 still comparable),
context 512 + the passage path (`/read`, `reader.py`, cited answers), provenance per document in memory.db (chain
intact), and the E9 slot (`owner_exam.json`). What step 2 still needs, in order:
1. **Jaron's go**, after he reads the swap rows and the retrieval demo in README "Run record - v3.1".
2. **His documents as plain text files on the MSI** (.txt; a .docx/.pdf needs a text export first - the reader takes
   text only). Then from the phone: `/read C:\path\to\file.txt`, one per document. Each read is a `documents` row
   + its passages as memories; the heartbeat consolidates them on its next draw (no gate) - **read E1-E7 on the page
   after the first consolidation with his text in it** (D14/D18: the big core memorises fast; the share cap is set,
   the drift is still the thing to watch).
3. **E9 filled by him**: questions about his documents + the answers he expects, into `owner_exam.json`. It runs
   after every consolidation from then on; the bar is his.
4. **Long reading is the weak leg** (E7L on a cold essay - see the run record for the drill result): fluent prose
   comes easier to it than copying a sentence out of a 150-word passage it has never seen. More task passages it has
   not read as prose, and a copy task with *long* spans, are the next lessons if E7L stays low; a bar for E7L is
   declared only after it has a level.
5. **Context 1024** is one `grow_context` call away (the same in-place growth) once 512 is used well; at 512 one
   passage + one distractor + a question fit.

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
