# Sapling — Write-up: what I took from the 107 repos, what I dropped, what's missing

Read in full or in their load-bearing files: the 21 named repos plus the
accumulation-sort index (which reads the whole account daily and keeps the
drift table). READMEs were treated as claims; code and committed results as
evidence, per the account's own rule ("read the actual code, not the README").

## Principles I actually used (and where they live in the code)

1. **Live-Reference Principle** — knowledge is pulled live, not frozen into
   weights. In Sapling this is structural: the core is trained on *patterns*
   (morphology, recall-echo, deferral), while facts live in the memory layer
   with provenance labels. The recall lesson (`curriculum.lesson_recall`)
   trains the core to answer from a `recall [ provenance ] :` context — the
   trained half of live reference.
2. **Fed, not filled** — the two-speed architecture (retrieved-now layer,
   append-only record, weights trained rarely) and the answer layering
   (deterministic rules → retrieval → small core → defer; a frontier model
   would slot in *above* the core, rare by design). Sapling's session loop
   implements tiers 1–3 for real.
3. **fly-school's school** — staged lessons, cold exams on never-taught
   material, predeclared bars, greedy seeded decoding, report cards, and
   **a FAIL stays on the record**. Also the real-clock half-life: memory
   strength decays on wall-clock time, so recall-spacing is what promotes.
4. **Persistent Memory Substrate / Atomic Chronicle** — append-only
   hash-chained SQLite; there is deliberately no update/delete API, and a
   tamper is detectable (`tests/run_tests.py` proves both).
5. **watch / PAM-continuum discipline** — provenance travels with the fact
   (`user/captured`, `exam/confirmed`), "could not see" is a valid entry
   (the deferral pattern), and **persistence ≠ confirmation** is enforced in
   schema: promotion never flips a memory's `state`.
6. **Keava-Owent** — the membrane (human-owned config, system reads it and
   can never write it), the consumption ledger (`meter()`), contribution
   recorded-not-scored, the anti-shutdown invariant as a test, and the
   honest-empty-slot pattern (the packing-geometry memory raises
   `NotImplementedError` instead of being faked).
7. **Memory as spatial arrangement** — only its *proven* half: relational
   retrieval with structure beats a flat log. The packing-geometry
   application stays an open slot (the four-statement separation is copied
   verbatim into `memory.py` so it can't be quietly collapsed).
8. **accumulation-sort's criterion** — the acceptance test for the whole
   project: does it run without you, is it changed by having run, is the
   change legible when you come back? Sapling v1 is session-driven (not yet
   "runs without you"); the record and report cards make it legible.

## What I dropped, and why

- **The fly connectome as the language core's seed.** fly-school's own
  measured result is the reason: Kenyon-cell codes share 40–70% of cells
  across symbols, and eight predeclared communication tests failed before
  one passed. A fly brain is a *world inhabitant*, not a language prior.
  The connectome stays in the "world" lane for later, not in the core.
- **Arweave/Filecoin permanence in v1.** Cost, keys, and irreversibility
  before there's anything worth keeping forever — and permanent public
  storage conflicts with the owner's right to revoke. The promotion ladder
  has a `permanent` tier; cold-storage export is a later adapter, not a
  foundation.
- **The 21-domain news desk in v1.** One live source (Wikipedia/RSS adapter)
  proves the live-reference pipe. 21 domains is scope, not substance.
- **Token economics (Atomic Chronicle's wallet).** Decorative at this
  stage; consumption is *metered* (that's real) but not priced.
- **The X voice.** Nothing worth saying yet, and an outbound channel is an
  owner-approval action by the membrane anyway.
- **The Engine of Division / Structural Conflict / The Last Theory** as
  architecture inputs. They're worldview and record, not mechanisms. One
  pattern does transfer from Structural Conflict: *the people not in the
  room have no standing* → in Sapling, the owner is always in the room
  (membrane, verdicts, stop control).

## Contradictions I found, and which way I went

1. **"Rent language, own judgment" (fed-not-filled) vs "train language
   first, school-style" (the project brief).** This is the real one. My
   call: **train the small language core anyway.** The resolution is about
   what the core is *for*. Renting language is the right move if your goal
   is capability-per-dollar. But this project's goal is an *owned*,
   inspectable seed whose structure grows (emergence: you can't study what
   grew in someone else's weights). Judgment also needs a substrate: if the
   language layer is rented, the owned judgment has nowhere to live and
   nothing to consolidate into. So: own a small core trained on a
   curriculum you control; rent frontier language later as a *tool the core
   calls*, not as the mind. The core's smallness is not a compromise — it's
   the point (BabyLM-scale evidence says grammar is learnable at this
   scale; RETRO/Atlas say knowledge belongs in retrieval anyway).
2. **"Never delete anything" (PMS absolutism) vs "never-used memories
   fade" (promotion).** Resolved the way fly-school already resolves it:
   **fading happens at the access layer, never at the record layer.**
   Retrieval weight decays by half-life; the append-only record is never
   rewritten. Forgetting is a reading rule, not a deletion rule.
3. **"Recalled again and again = promoted" vs "persistence ≠
   confirmation" (watch rule 5).** Promotion moves *retrieval priority*,
   never *truth status*. Enforced in schema: `state` (captured/confirmed/
   invalidated) can only be changed by a new appended event, and recall
   frequency has no path to it. The test `persistence_is_not_confirmation`
   locks this in.
4. **"Structure is grown, not designed" (emergence) vs deterministic
   validators everywhere (origin-works, unfinished-island).** Grow the
   weights; engineer the constraints and the record. Curriculum, membrane,
   exams and ledgers are designed; what the core internally becomes is
   grown and then *measured* by the exams.
5. **ECIH's "let it face scarcity/damage" vs corrigibility (membrane,
   anti-shutdown).** Keava's resolution stands: existence is axiomatic, the
   pressure is *metered consumption*, not mortality. The constraint is a
   ledger, not a threat.

## What's missing (honest gaps, smallest next step for each)

- **A selection function fed by resolution.** fed-not-filled's third term
  (selection) is still unbuilt everywhere on the account. Smallest step:
  mark memories resolved/unresolved by later events (e.g., a taught fact
  contradicted by a later confirmed source flips state via a new event),
  and train consolidation only on resolved rows.
- **Catastrophic-forgetting handling across curriculum stages.** v1
  rehearses the full mix each lesson. Smallest step: per-lesson exam slices
  re-run after each new lesson, on the report card.
- **A live-fetch adapter.** The retrieval interface exists; the network
  adapter doesn't. Smallest step: one Wikipedia summary fetch, stored with
  provenance `retrieval/captured`.
- **Consolidation (record → weights).** The two-speed architecture's slow
  loop. Smallest step: nightly fine-tune of the core on `confirmed` memory
  rows only, gated by the owner.
- **Runs-without-you.** v1 is session-driven. Smallest step: a daily
  scheduled `loop --once` that meters itself, pulls one live source, and
  appends a heartbeat — the pam-continuum pulse pattern.

## v2 addendum (Claude, 2026-10-14) — what Jaron's decisions overturned above

Four of the "contradictions I went with" in v1 were decided the other way on 10/14 and are now
code (see CHANGELOG.md for the line-by-line):
- **Fed-not-filled vs "become what it learns":** facts now DO go into the weights
  (consolidate.py), with no owner gate; the exams after every consolidation are the watch.
- **"Never-used memories fade" by clock:** fading now applies ONLY to memories never recalled;
  use protects (memory.py).
- **ECIH scarcity vs corrigibility:** the pressure is now a real daily budget it can see
  draining and a clock it can feel (clock.py) - pressure on keeping itself running well,
  never a drive to avoid its owner's off switch (the off switch is outside the pressure).
- **Existence axiomatic / anti-shutdown as a test:** dropped. It may stop; a stop is a report.
What stayed: the append-only hash chain, provenance with every fact, persistence != confirmation,
the cold exams with FAILs kept, and the membrane the owner alone edits.
