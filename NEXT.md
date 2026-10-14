# Sapling — the next three steps

Ordered by dependency. Each is the smallest version that still counts.

## 1. Go live: one real reference per day (the "runs without you" step)

A scheduled `python -m sapling.loop --once` (Windows Task Scheduler on the
RTX 4060 machine) that: pulls ONE live source (a Wikipedia "on this day"
summary or one RSS feed), stores it as `retrieval/captured`, meters itself,
answers one queued question using it, appends a heartbeat block, and stops.
This turns Sapling from session-driven into accumulating under
accumulation-sort's own criterion, and gives fed-not-filled's "today layer"
its first real feed. Success looks like: a week of dated heartbeats in the
ledger and at least one answer that cites a fact newer than the training
corpus — the Live-Reference test, passed in the small.

## 2. Close the selection loop: resolution → consolidation

Add `resolved` events (a later confirmed source or a human verdict marks a
memory resolved/unresolved), then the consolidation loop: a nightly,
owner-gated fine-tune of the core on `confirmed + resolved` rows only, with
the exam suite re-run after every consolidation and the delta kept on the
report card. This is fed-not-filled's third term — selection — made of code,
and prediction-research-system's "freeze the forecast, score it later"
applied to the system's own memory. Success looks like: consolidation run 1
does not regress any predeclared exam, and at least one taught fact survives
in weights after its memory row is (hypothetically) unreachable.

## 3. A world to practice in (the third leg)

Give the loop a tiny deterministic world instead of only text: the simplest
version is a grid foraging sim (unfinished-island's Island Language lineage,
or a 10×10 nutrient grid à la livi-organism) where the core proposes one
action per tick, a deterministic simulator validates and applies it, and the
outcome — success or failure — is recorded and becomes training signal.
This is the ECIH move: real constraint (a survival budget), real feedback,
labels free by construction. Success looks like: the core's action choices
improve across epochs *measured by the sim*, with the predeclared-metric
discipline from fly-school (bar written before the run; FAIL kept).

After these three: the packing-geometry memory slot gets its first honest
attempt (arrangement strategy plug-in behind the existing interface), and
the GitHub/Supabase export adapters turn the local record into the
multi-service organ layout.
