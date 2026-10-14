# Sapling — Architecture

One trained core, knowledge pulled live, memory that accumulates, judgment by
exam, and a membrane the owner holds. v1 implements the left column end-to-end;
the right column is the growth path.

```
                 ┌──────────────────── MEMBRANE (human-owned, read-only to system) ──┐
                 │  law: backup on owner's machines with permission; never spread;  │
                 │  never hide. forbidden: self_shutdown, rewrite_history, ...       │
                 └───────────────────────────────────────────────────────────────────┘
 senses                    ┌─────────────┐                 ┌────────────────────────┐
 user text ─────────────► │  THE LOOP   │  observe/meter  │  LEDGER (append-only,  │
 live fetch (later) ─────► │  session.py ├────────────────►│  hash-chained SQLite)  │
                           └─────┬───────┘                 └────────────────────────┘
                                 │ recall
                                 ▼
                    ┌─────────────────────────┐   provenance + state travel
                    │  MEMORY (append-only,   │   with every fact; promotion
                    │  hash-chained, promoted │   moves retrieval priority,
                    │  by recall, decays on   │   never truth status.
                    │  wall-clock half-life)  │
                    └───────────┬─────────────┘
                                │ recall [ provenance ] : context
                                ▼
                    ┌─────────────────────────┐    cold exams, predeclared bars,
                    │  CORE (tiny transformer,│───►│  JUDGMENT: exams.py      │
                    │  ~0.9M params, trained  │    │  E1 composition (held-   │
                    │  school-style on a      │    │  out), E2 affix recall,  │
                    │  morphology curriculum) │    │  E3 recall echo,         │
                    └─────────────────────────┘    │  E4 deferral. FAIL kept. │
                                                   └──────────────────────────┘
```

## The three loops (two built, one stubbed)

1. **Session loop** (built): observe → meter → recall → answer → record.
   Every turn is metered (tokens in/out, ms) into the ledger; every answer
   records its path (memory id, provenance, state) in its text.
2. **School loop** (built): curriculum lessons in order → per-lesson loss on
   the report card → cold exams → results hash-chained into memory.
3. **Consolidation loop** (stubbed, honestly): record rows marked
   `confirmed` would fine-tune the core, owner-gated, on a schedule.
   `NEXT.md` carries the smallest version.

## Answer layering (fed-not-filled, implemented)

1. **Deterministic** — parseable memory hit that the core can echo via slots;
   unparseable hits are cited verbatim with provenance (never paraphrased
   silently).
2. **Retrieval** — memory search by morpheme overlap × decayed strength.
3. **Core generation** — the tiny model composes over the recall context.
4. **Deferral** — "i do not have that yet ." is a trained, exam-tested
   behavior (E4), not a failure mode.
5. **Frontier model** — not in v1; when added, it sits *above* this stack as
   a rented tool the loop calls rarely, with the call recorded.

## Memory semantics

- `captured / confirmed / invalidated` — truth status, set only by events.
- `short_term → working → long_term → permanent` — retrieval tier, derived
  from decayed recall evidence (half-lives: 6h / 1d / 7d / ∞), or a human
  `keep` verdict for permanent. Derived, not stored — history is never
  rewritten, only re-read.
- Hash chain over `memories`, `recalls`, `ledger`; `verify_chain()` re-derives
  every hash. The tamper test proves a silent edit breaks the chain.

## What runs where

Everything v1 runs on the RTX 4060 PC (or any CPU): Python 3.10+, torch.
Training the full curriculum is minutes on the 4060, ~10 minutes on CPU.
No external service is required; Supabase/GitHub/Arweave adapters are growth
path, not foundation.
