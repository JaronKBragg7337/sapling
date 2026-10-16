# Sapling — next steps (after v2)

Tool note (10/15): `tools/gpu_yield.py stop|start` pauses the heartbeat for the news render (now 3:50 AM) and restarts it after; it uses `tools/scheduled_parole.py`.

v1's three steps: (1) runs-without-you is DONE (the heartbeat); (2) record -> consolidation
is DONE (no gate; `/confirm` and `/invalidate` verdicts are the selection events); (3) a world
to practice in is still open. Ordered by dependency.

## 0000. Lesson retention (10/16 morning)
D33 slowed the wear (E7L 17 -> 13 in 5 folds, instead of -> 8 in 27 without replay) but did not stop it. Next levers,
cheapest first, each measured the same way (roll forward gen 319, 5 folds, tools/measure_core.py): (1) consolidation lr
1.5e-4 -> 7.5e-5 for a core past 50M params (D14 was tuned for 12-17M); (2) lesson replay 8 -> 12 of 44; (3) fewer
steps per fold (200 -> 120). Then the next cloud lesson starts from a core that keeps it.

## 000. The cloud long-reading lesson (built 10/16, not run) - the morning command

```bash
python tools/runpod_run.py            # bundle (reused) -> upload (reused) -> H100 pod -> ~55 min lesson -> core back -> pod deleted -> swap
```
~75 min door to door, ~$3.40 on community cloud ($5.00 secure); hard cap $6.32. Watch: results/cloud/<stamp>/runner.jsonl,
AI-Shared/state/cloud-gpu-log.md, and the phone (ntfy at start and end). After it: E7L / E10 on the card, and the held-out
lesson score in the runner log. If anything looks stuck: `python tools/runpod_run.py --reap` deletes every sapling-* pod.

## 00. v4 (10/15, Claude as teacher) - running tonight; what comes after it

Stage A (a fresh 60.5M core reading 1.2B tokens) and Stage B (conversation lessons) run unattended tonight via
`tools/teach_v4.py`; the swap and a re-measure after 5 of its own consolidations follow automatically (CHANGELOG v4).
After it, in order:
1. **Read the before / after table and the ten chat samples** on the control room card "Teaching - before and after"
   (results/teacher/, results/chat_samples/). If anything drops after the 5 consolidations, that is the first fix
   (the consolidation mix, D28) - before anything new.
2. **Write the results into README / CHANGELOG v4** (numbers, the samples) - the run itself only writes results files.
3. If E10 / E7L switched on: Jaron's documents (step 2 below) become worth trying. If not: the supervised long-copy
   lesson (0a item 2) on the v4 core.
4. Communication next: multi-turn lessons built from its own memories and taught facts (questions about what it was
   taught, asked in several wordings), and a bigger conversation set if oasst1 + dolly are not enough.
5. **Sapling's own words (Jaron):** the ten answers about itself are his to write on the control room card "Sapling's own
   words". Tonight's Stage B trains none (none written yet). When he has written them, a short top-up on just those
   (~300 steps, a few minutes; reading + school replay inside, heartbeat stopped for it):
   ```bash
   python tools/gpu_yield.py stop "self-lesson top-up" && python -m sapling.talk train --self-only --core results/core_checkpoint.pt
   python -c "from sapling.grow import swap; swap(staged='results/core_self_topup_staged.pt')"
   python tools/gpu_yield.py start "self-lesson top-up"
   ```
   (then `python -m sapling.talk build` so the heartbeat's consolidations replay his answers too).
6. **Archive (decided 10/15 6:06 PM, built):** generation files in bf16, archived to GitHub Releases (D5). Later moves:
   a Heartbeat Observatory GitHub organisation (transfer the repo or point `archive.GH_REPO` at the org's repo), then
   Hugging Face (Jaron sets it up himself; then an `hf upload` target beside `gh_upload`).
7. Items 0a.3-0a.5 below still apply to v4 (E5 refresh, rehearsal share; the reader-route fix is done for talking
   cores - coverage 0.6, D27).

## 0a. After v3.2 (10/15): the reading stage did not switch copying on - what to try next (lab order holds)

(10/15 afternoon: answered in part by v4 - the 17.6M core had no copy circuit at all, so the next try is a bigger, fresh
core with rotary positions; the list below is kept for the v4 core.)

400M tokens of plain reading (CHANGELOG v3.2): E7L 0/40 at all 21 probes, E10 0 -> 9/2016 (chance ~0.1, so above noise
but no circuit), while E6 went 28 -> 46/48 and E7 30 -> 33/40. The hypothesis (induction heads appear with enough text)
is not refuted - 400M tokens may simply be short of it for a 6-layer, 6-head core at lr 3e-4 - but it did not pay in this
budget. Candidates, cheapest first, each one `stage`/`grow` call and each measured by the same E10 / E7L probes:
1. **A second pass** (`python -m sapling.reading stage --tokens 400e6` again; shards and `state.json` resume where it left:
   ~170M story tokens unread + the rest of adult, then a second epoch). Cost ~65 min GPU. Tells whether E10's creep
   (0 -> 12) continues or plateaus.
2. **The tutor lesson Jaron's order names next**: a copy task with LONG spans and many never-seen passages (the v3.1
   docs/longread lessons but on the Gutenberg set from v3.2, task passages never seen as prose, spans of 10-40 tokens).
   Supervised copying is the direct road to E7L; `grow school --base live` with a `copy2` stage.
3. **E5 refresh**: the five taught words washed out (7/7 -> 0/7). The heartbeat re-trains all facts at every
   consolidation; read E5 on the card after the next few. If it stays 0, raise the fact weight for already-consolidated
   user facts in `consolidate.py` (now `max(1, w - 1)`).
4. **Rehearsal share**: E1 oscillated 68-81 (bar 52) on 3 of 32 sequences; if level 1 matters more than that, 4 of 32.
5. Bug seen in chat (10/15 03:06): "once upon a time there was a little girl who" went through the reader and cited the
   Declaration of Independence - `reader.retrieve`'s coverage (>= 0.3 of the question's content words) is met by common
   words; require >= 2 matching content words of length > 3, or raise the threshold. One-line fix in `session.py`.

## 0. Step 2 - Jaron's documents (his call; the MACHINERY is done in v3.1, the reading ability is not - E7L 0/40, unchanged by v3.2)

The machinery the list below asked for is built (CHANGELOG v3.1, DESIGN-CHOICES D16-D19): copy lesson (E6 12 -> 21/48),
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
