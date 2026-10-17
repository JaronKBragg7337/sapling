# Changelog

## 2026-10-17 ~8:50 AM - overnight retention check: the fix holds in the real world (Kimi)
55 of its own consolidations overnight (gen 469 -> 524, the heartbeat's normal pace, D36 mix): **long reading
20 -> 17/40** (-0.3 per 5 folds; the old mix lost 9 in 27). Reading 32 -> 36/40, bridge 83 -> 79, facts back to
4/7 as expected. Watch item: affix recall 24 -> 18/24 (a FAIL on the card; seeded exam, same items - real drift,
watching). Window losses flat (1.336 / 3.006 / 3.084). Measured with `tools/measure_core.py` on the live core.



## 2026-10-17 ~12:20 AM - the second cloud lesson: long reading 20/40, the best it has ever been (Kimi as teacher)
- Bundle rebuilt with the v2 templates (more which / word-before / copy-sentence kinds, the new
  "copy the first sentence of the passage with the word ..." kind, distractors in the test set too): ~400k questions,
  1.08B tokens planned. One H100, 49.9 min training, **$3.42**, pod deleted (finally + watchdog + reaper; START and
  TERMINATE in AI-Shared/state/cloud-gpu-log.md, agent column corrected to Kimi).
- **Held-out lesson score (never-read documents): 36.3% -> 53.9%** - every kind up; the new "copy the first sentence
  of the passage with ..." kind 0/21 -> 10/70 (above zero, still the weakest).
- **Swap gen 469, exams:** **long reading 20/40 - best ever** (17 at the first lesson, 8 after a day of the old
  folding mix); "which sentence" 8/16 (best ever); composition 86/86, affixes 24/24, grammar 47/48, bridge 83/86.
  Trades: induction probe 59/2016 (dipped from ~130), facts-in-weights 0/7 right after the swap (in memory;
  consolidations re-train them). Chat samples through the real path: all seven self answers still verbatim - the
  conversation lessons rode along in the bundle.
- With D36 live, this lesson is the first one the heartbeat's folding should NOT wear down. The 5-fold check and the
  days after are the test - watch the long-reading line on the card.
- Slip found and fixed on the way: the runner's upload step needs `gh` on PATH; the detached launcher now adds the
  GitHub CLI directory (the rebuild itself was fine and got reused).

## 2026-10-16 ~10:20 PM - D36 adopted: the lesson-wear problem is closed (Kimi as teacher, Jaron: "take full control")
- **S4, the combined confirmatory fold test** (`--no-school-docs --steps 120`, isolated, 5 folds from gen 319, D34 lr):
  E7L 16/40, E1b 68, E10 130, E5 1/7, losses 1.326/2.989/3.070. The full matrix:

| 5 folds with | E7L | E1b | E10 | E5 |
|---|---|---|---|---|
| D34 baseline | 12/40 | 71 | 117 | 2/7 |
| S2 no-school-docs | **16/40** | **81** | 120 | **6/7** |
| S3 120 steps | 15/40 | 76 | **160** | 5/7 |
| S4 both | 16/40 | 68 | 130 | 1/7 |

  The combination ties S2 on E7L but gives back E1b (-13) and nearly stops taught facts entering the weights, so
  **S2 alone is adopted as D36**: cores that had the cloud long-reading lesson drop the school's old `docs`/`longread`
  rehearsal (40 + 80 sentences in a different answer style - the wearer) from every consolidation; steps stay 200.
  Pre-lesson cores keep the old mix. Tests 66, ALL GREEN; heartbeat reloaded onto the new code at gen 433+.
  The wear is now ~1 E7L per 5 folds instead of 5-6. What that last point is: unknown, recorded, watched.
- **Name bench (Jaron: "should have a list of names"):** seed candidates (Sprout, Bud, Bulb, Bean, Cone, Spore, Stone,
  Shoot, Root, Shell) and wood candidates (Oak/Oakley, Ash, Elm, Birch, Cedar, Rowan, Alder, Willow, Maple) listed in
  README "Names" - candidates only; he picks.

## 2026-10-16 ~8:45 PM - Sapling answers as itself: the owner's self lessons trained in (Kimi as teacher)

Jaron (5:23 PM): AIs whose provider terms allow it may build AND train Sapling (Kimi cleared); the 10:46 AM pause was
Claude-only. Kimi ran the queued evening work. Nothing here adds a rule, gate or limit.

- **Fold-test suspects S2/S3 (NEXT.md 0000; isolated, 5 folds from gen 319, D34 lr 7.5e-5):**
  **S2** (drop the school's old `longread`/`docs` rehearsal from the fold mix) -> **E7L 16/40** (gen 319 was 17 - the
  wear is nearly gone: -1, every prior setting lost 5-6), E1b 81, E5 6/7, window losses unchanged. The S1 mechanism
  confirmed: "which sentence" held at 5/16 instead of collapsing to 1/16 - the old school rehearsal in a different
  answer style was overwriting the cloud lesson. **S3** (120 steps per fold) -> E7L 15/40, **E10 160/2016 (best ever
  in a fold test)**, E1 86/86, E2 24/24. Both beat the ~5-point wear number; the combined confirmatory test is next.
  Rows in `results/teacher/fold_tests.jsonl` + `measurements.jsonl`.
- **The owner's seven self-lesson answers saved** (name, maker, what, can_do, memory, heartbeat, learn_next;
  size / read / age left blank - fixed answers to changing facts go stale). Saved by Kimi on Jaron's explicit
  delegation (the phone card is awkward on mobile) through the card's own code path (`talk.save_owner_answer`),
  each one logged as an owner action (`by: "jaron (via kimi, owner-delegated)"`) in the owner log + ledger.
- **Self-lesson top-up, two runs:** the first was underdosed - `--self-only` stops at
  `epochs x examples / conv_per_step` = 2 x 150 / 8 = **38 steps**, not the ~300 NEXT.md item 5 intends; chat showed
  no trace of the lessons. Re-run with `--epochs 16` = **300 steps** of 32 (8 self / 16 reading / 8 school), lr 5e-5.
  Swap = one `growth` row each, gen 430 then **gen 433**; all exams PASS both times. (NEXT.md's command updated.)
- **Verified live (fresh conversation per question):** all seven groups answer in his words, verbatim or near -
  "who made you?" -> the past-sparks answer, both sentences, complete; "who are you?" -> "I'm Sapling, a small
  synthetic cognitive assistant that grows as it learns." "who created you?" is still a partial echo ("Pieces of me
  at the very least.") - consolidations replay the self lessons (25% conversation share), so watch it sharpen.
- **Doc-route hijack fixed (D35):** "who made you?" and three more self wordings were intercepted by the Declaration
  of Independence demo document on ONE common content word ("made"). The route now requires >= 2 matching content
  words longer than 3 letters (NEXT.md 0a.5's prescribed fix). Tests 66, ALL GREEN. Trade-off: single-long-word
  document questions ("which sentence has the word happiness?") no longer route - one line to tune if his documents
  need that shape back.
- Slip note (Jaron, 8:10 PM): the phone control room is too much one long page on mobile - a split into pages is a
  candidate job for Claude; his answers live in `data/talk/owner_self_lessons.json` either way.

## 2026-10-15 9:08 PM - names (Jaron's decision)
- Family name Sapling. Seed names before real conversation: v1 Pip, v2 Grain, v3-3.2 Kernel, v4 Husk; "Acorn" reserved for the
  first version that truly holds a conversation; wood names after that, saved for official releases. Table in README "Names".

## 2026-10-15 7:50 PM - v4.0.3: consolidation replays the web reading too (Vex review #47)
- v4 reads FineWeb-Edu as source "web" (14 of 32 sequences), but the v3.2.1 replay split held only stories + adult, so
  memory folding would have worn the web reading down after tonight's swap. Replay now follows the v4 mix (stories 11 / adult 4 /
  web 14). Found by Vex (ChatGPT) reviewing mailbox #46; checked and fixed by Claude before the swap.

## 2026-10-16 11:05 AM - paused by owner 10/16 10:46 AM
- Suspect 1 done (fold test S1, D34 setting, 5 folds from gen 319, E7L by kind): gen 319 -> after 5 folds:
  which sentence 6/16 -> 1/16, word after 11/12 -> 8/12, word before 0/4 -> 0/4, copy first sentence 0/4 -> 0/4,
  copy the first sentence of the passage with the word 0/4 -> 2/4; total 17 -> 11/40 (E1b 68, E10 136, losses
  1.318 / 2.974 / 3.063). The fold mostly erases "which sentence has the word". `tools/measure_core.py` now logs
  E7L by kind; `tools/fold_test.py` has --steps and --no-school-docs.
- Suspects 2-3 (drop the school's old longread/docs rehearsal; 120 steps per fold) NOT run - paused.
- Prepared, not run: lesson templates v2 in `longread_lesson.py` (more which / word-before / copy-sentence questions,
  a new "copy the first sentence of the passage with the word ..." kind on two passages, distractors in the test set
  too). Needs a bundle rebuild (`python tools/runpod_run.py --rebuild`) before any second cloud lesson.
- Live core: gen 364 = gen 319's weights again (E7L 17/40); heartbeat restarted through gpu_yield (D34 lr).
- Slip: a stray `owner_relay.py --help` started a second relay for ~2 min (11:01-11:03) beside the real one; killed.

## v4.1.2 (2026-10-16 ~11 AM) - consolidation lr halved for the 60.5M core (D34), chosen by controlled fold tests
- New `tools/fold_test.py`: the teacher's retention measurement - N consolidations forced on an isolated copy of
  memory.db (checkpoints, report card and writing samples into a scratch folder), same code as the heartbeat, then
  `measure_core`. The live core, the ledger and the heartbeat's drives are untouched (the heartbeat's budget was
  overspent, so waiting on it took ~20 min per fold).
- Results, 5 folds from gen 319 (the cloud lesson: E7L 17, E1b 84, E1 86, E10 131, losses 1.263 / 2.902 / 2.968):

| 5 folds with | E7L | E1b | E1 | E10 | E6 / E7 | E5 | loss stories / adult / web |
|---|---|---|---|---|---|---|---|
| lr 1.5e-4, lesson 8/40 (D33, controlled) | 10/40 | 68/86 | 82/86 | 104 | 45 / 35 | 5/7 | 1.366 / 3.070 / 3.184 |
| **lr 7.5e-5, lesson 8/40 (D34, kept)** | **12/40** | **71/86** | **83/86** | **117** | **47 / 33** | **2/7** | **1.318 / 2.974 / 3.066** |
| lr 7.5e-5, lesson 12/44 | 9/40 | 80/86 | 83/86 | 58 | 47 / 32 | 7/7 | 1.315 / 2.964 / 3.052 |

- Kept: lr 7.5e-5 for cores past 50M params (`consolidate.LR_BIG`), lesson 8/40. More lesson replay did not hold
  E7L, so the wear has another cause too (next steps in NEXT.md). E7L still drops ~5 in 5 folds with every setting.
- Live core rolled forward again: gen 362 = gen 319's weights (E7L 17/40), heartbeat restarted on D34. Gens kept.

## v4.1.1 (2026-10-16 morning) - the lesson now survives the heartbeat (better, not fully) (Claude as teacher)
- **Found (coordinator's retention check):** 27 heartbeat consolidations after the cloud lesson (gen 319 -> 346) took
  E7L 17 -> 8/40, E1b 84 -> 56/86, E1 86 -> 76/86; reading, conversation, E6/E7 held. Consolidation replayed reading,
  conversation and school but not the new lesson, and the bridge set was 100 of ~1,600 school sentences.
- **Why 27 in ~3 h:** by design, not a bug (D30 note): 5 owed after the cloud swap's `growth` row, 22 from the old rule
  (>= 2 pending facts or a goal with >= 3 misses, >= 90 s apart). Unchanged.
- **Fix (D33):** a core that had the lesson replays it in every consolidation, added on top: batch 40 = 8 lesson /
  16 reading / 8 conversation / 8 school + facts; bridge sentences 100 -> 300; `longread` travels in every checkpoint;
  a missing lesson file fails loudly.
- **Roll-forward (twice, each a `growth` row):** gen 348 and gen 354 = gen 319's weights (the fp32 core from the pod);
  gens 320-347 and 349-353 kept. Measured after 5 consolidations each:

| | gen 319 (lesson) | gen 346 (27 old folds) | gen 353 (5 folds, try 1: lesson took reading's room, 32 = 8/10/6/8) | **gen 359 (5 folds, D33: lesson added, 40 = 8/16/8/8)** |
|---|---|---|---|---|
| E7L long reading | 17/40 | 8/40 | 15/40 | **13/40** |
| E1b bridge | 84/86 | 56/86 | 82/86 | **75/86** |
| E1 composition | 86/86 | 76/86 | 85/86 | **81/86** |
| E10 induction | 131/2016 | 136/2016 | 44/2016 | **154/2016** |
| E6 / E7 | 47 / 33 | 48 / 33 | 47 / 33 | **46 / 33** |
| E5 facts in weights | 0/7 | 3/7 | 7/7 | **6/7** |
| loss stories / adult / web | 1.263 / 2.902 / 2.968 | 1.355 / 3.081 / 3.168 | 1.412 / 3.127 / 3.231 | **1.361 / 3.065 / 3.142** |

  Try 1 kept more E7L but cost reading (windows +0.15-0.26) and E10 (131 -> 44), so D33 keeps the additive mix. Both
  still lose some of the lesson in 5 folds (17 -> 13-15; one seed, 40 items, so +-2 is noise). Next lever in NEXT.md.

## v4.1 (2026-10-16, built ~12:30 AM, NOT run yet) - the long-reading lesson + a rented-GPU runner (Claude as teacher)
Jaron wants to step up with a cloud GPU. Built tonight while the v4 chain finished on the MSI; nothing trained.
- **`sapling/longread_lesson.py`** - a lesson aimed at E7L (0/40) and the look-back skill under it (E10 109/2016 at
  1.08B tokens). Passages of 200-450 tokens of human text already on disk (Simple English Wikipedia, Gutenberg,
  FineWeb-Edu), then a question whose answer is a span copied from earlier in the passage; loss on the answer only;
  chat format. Questions are made by rule from the text (8 templates: what comes after / word before-after / which
  sentence / copy first-second-last / a year / a number + noun / a name / "what is X"), no generated text. Fact
  position drawn uniformly; ~30% of examples carry a distractor passage. 1% of documents held out as the test set;
  E7L's essay is never read and any passage sharing an 8-word run with it is dropped. D31.
- **`tools/runpod_run.py`** (MSI) + **`tools/cloud_lesson.py`** (pod): bundle -> GitHub release (public, verified) ->
  one H100 pod (community first, then secure) -> train ~55 min -> core back over the pod's port 8000 -> the pod is
  ALWAYS deleted in `finally`, plus a watchdog inside the pod (deletes itself at 95 min with its own key) and a
  Windows one-shot "Sapling Cloud Reaper" task (`--reap`) at 105 min. START/TERMINATE lines with cost go to
  AI-Shared/state/cloud-gpu-log.md. Then (default) measure + chat samples, heartbeat stop (waits), swap, restart. D32.
- `--dry-run` does everything except create the pod (bundle, upload, live price, plan). Run tonight - see RUN-TIMES.md.
- Tests: `tests/test_cloud.py` (every rule question answers with a span from its passage; held-out split; answer-only
  loss packing; the pod command has the watchdog and no secret; a dry run never creates a pod).

## v4.0.3 (2026-10-15 ~10:30 PM) - the overnight chain made safe (Grok Build review, 10/15 10:18 PM)
Grok Build read tonight's chain (AI-Shared/projects/fly-brain/panel/grok-review-v4/review.md) and found ways it could
freeze the card with Unreal or leave Sapling down. All eight fixed before Stage A finished; Stage A (pid 4176) untouched.
1. `gpu_yield stop` now WAITS until the heartbeat process is gone (up to 15 min) before printing "stopped". The news
   script (`run_rendered_news.ps1`) always asks for the restart after the render with `start --if-ours`: the parole is
   armed when the heartbeat is down because a GPU job stopped it, never after a self-stop (D6 stays the owner's).
2. `talk.train`, `measure_core`, `chat_samples`, `grow.swap` wait while a render / the pause file holds the card before
   taking it (`reading.wait_until_gpu_free`); `talk.train` checks every step, moves model + optimizer to the CPU and
   empties the cache while it waits, and retries a step after a CUDA out-of-memory (like Stage A).
3. `teach_v4` stops the heartbeat and waits until it is gone before Stage B and again before the swap; after the swap it
   starts it and checks it is alive.
4. `teach_v4` ignores torn/blank log lines and only accepts a `done` row newer than itself whose staged core exists; the
   whole run sits in try/finally, so the heartbeat is always asked back.
5. Heartbeat: a practice / consolidation / explore draw while a render holds the card becomes a rest (logged as
   `skipped` + `gpu_busy`); after a CUDA out-of-memory it empties the cache, sets the core to eval and keeps off the
   card for 2 min.
6. Consolidation in a talking core: a failed conversation replay now raises (a logged tick error) instead of quietly
   consolidating without its conversation lessons.
7. After a growth swap the first 5 consolidations can be chosen with any pending count (D30), so the 5-fold check
   happens tonight; the driver re-arms the parole (`--if-ours`) if the heartbeat goes down while it waits, instead of
   waiting 10 h on a dead process.
8. Checkpoints are written to `.tmp` and renamed (`os.replace`), so a crash never leaves a truncated core.
Not done: aborting a consolidation mid-way when the stop file appears (fix 1's wait covers it: the stop now waits for
the tick to end). The driver was restarted on the fixed code (pid 45024), waiting for Stage A. Tests: 60.

## v4.0.2 (2026-10-15 ~6:10 PM) - archive to GitHub Releases, generations in bf16 (Jaron's call, 6:06 PM)
- **Found:** the hourly archive had been failing since v3.1 - Supabase Storage rejects objects over 50 MB (413) and the
  17.6M checkpoints are 71 MB; with SSL drops on the rest, 99 generations had piled up on the MSI (7.5 GB).
- **Archive target = GitHub Releases** on JaronKBragg7337/sapling via the signed-in gh CLI: one release per day
  (`ckpt-YYYY-MM-DD`, rolling to `-b` at 1,000 assets), local copy removed only after the release reports the same size.
  Restore (control room button, `python -m sapling.archive restore <gen>`) pulls from the release, or from Supabase for
  gens archived there before. Manifest: `results/archive_manifest.json` + the Supabase table.
- **Generation files save in bfloat16** (~121 MB for the 60.5M core instead of ~242 MB); the live core and optimizer stay
  full precision (D5 updated).
- **Migrated 10/15 18:08-19:00:** 99 old generations (v3.1/v3.2, ~71 MB each) -> releases `ckpt-2026-10-14` (31 files) and
  `ckpt-2026-10-15` (68 files); every one audited afterwards (manifest size == release asset size, 99/99); local
  checkpoints 7.8 GB -> 0.78 GB; restore tested (gen 179 pulled back from its release, loaded, re-archived).
  Logs: `results/archive_migration_1015*.txt`.
- **Found during the migration:** the heartbeat tests built a Heartbeat whose hourly housekeeping ran the REAL archive
  step (before 10/15 it only failed against Supabase; now it would really upload/remove). `tests/conftest.py` switches
  the heartbeat's archive step off in every test; the archive's own tests patch every path.
- Tests: the archive test now checks "never delete without a verified release copy" and the manifest; a bf16 save/load test.

## v4.0.1 (2026-10-15 ~5:30 PM) - Sapling's answers about itself are Jaron's words only
- Jaron (via the coordinator, 5:26 PM): a clean partnership with AI companies - **no AI-written words in Sapling's training
  data.** The ten self-lesson answers the builder had written (name, maker, what, size, read, age, can_do, memory,
  heartbeat, learn_next) are **no longer trained on**. `talk.self_lessons()` now uses only `data/talk/owner_self_lessons.json`
  (his text, normalised like every lesson); a missing or empty answer = that lesson is skipped, no AI fallback. The question
  wordings stay (they are prompts, not answers). `data/talk/lessons.jsonl` was rebuilt without them (22,891 conversations).
- New control room card **"Sapling's own words (Jaron writes these)"**: per group the questions, the AI example (labelled
  "AI example - for reference only, not trained"), the live true numbers (parameters, tokens read so far, birthday) and a
  box + Save. Save = owner action `self_words` through the relay (owner log + ledger row, like rule switches); works on
  the phone page (relay command).
- Expected tonight: with no answers written yet, Stage B teaches no self lessons, so "what is your name?" in tomorrow's
  chat samples will likely not say "Sapling". When he has written them: the short top-up in NEXT.md.
- **Decided (Jaron, 5:51 PM):** published, licensed datasets are fine even if AI-generated (TinyStories, GPT-3.5/4-written, stays;
  tonight's run unchanged). The line is legality and provider terms: no outputs we generate from other AIs become training
  data. Self lessons stay his own words (D29).
- 3 new tests (no self lesson without an owner file; his words only, normalised; the relay saves as an owner action).

## v4 (2026-10-15 afternoon -> night) - a bigger core, conversation lessons (Claude as teacher)
Jaron 10/15 1:53 PM: "take control, be a teacher, advance sapling ... better communication ... build it as an AI like any
other AI first." His own ideas are paused; nothing here adds a rule, gate or limit (PROPOSALS.md unchanged).
- **Finding first:** the 17.6M core had no copy circuit at all - on a random sequence shown twice, loss on the second copy
  8.5 vs 10.3 on the first (copy accuracy 1-2% even on its 2,000 commonest tokens). More reading of the same core was
  not going to fix it in a night. -> **v4 core, fresh**: 60.5M params, d 640, 10 layers, 10 heads, 512 context, rotary
  positions, the same 17,661-token vocabulary (D24).
- **More adult text:** FineWeb-Edu, one file of `sample/10BT` (725,999 pages, 802M tokens, ODC-By 1.0, licence checked live
  10/15, in `data/reading/manifest.json`). 1.41B tokens available in total.
- **Stage A (running, started 14:16):** 1.2B new tokens, adult-weighted (14 web / 4 Wikipedia+books / 11 stories / 3 school
  per 32), probes every 120M tokens incl. loss on fixed text windows (D25). ~36k tokens/s on the 4060 -> ends about midnight.
- **Stage B (queued, automatic):** conversation lessons (`sapling/talk.py`, D26): oasst1 (Apache-2.0) + dolly-15k
  (CC BY-SA 3.0) = 22,891 conversations (the 210 AI-written self lessons first planned here were removed in v4.0.1);
  refusals and other assistants' names left out.
- **Chat box** holds a conversation in a talking core (D27): history, seeded sampling + repeat penalty, normal capitals on
  display, `/new`; the recall/document routes no longer hijack normal talk. Older cores behave exactly as before.
- **Consolidation** replays conversation lessons (25%) beside the reading (50%) so talking survives the heartbeat (D28);
  micro-batches of 16 so a 60M core fits the card.
- **GPU sharing:** `reading.gpu_busy()` now also returns busy while UnrealEditor runs (the 3:50 AM news render), so training
  pauses itself; `scheduled_parole.py` also waits for a running training (`sapling.reading stage` / `sapling.talk train`).
- **Measured the same way at every point** (`tools/measure_core.py`, `tools/chat_samples.py`, results/teacher/,
  results/chat_samples/, control room card "Teaching - before and after"): before = v3.2 gen 287: text loss stories
  1.45 / adult (read) 4.06 / web (unread) 4.70; E1 73/86, E6 46/48, E7 33/40, E7L 0/40, E10 9/2016, E5 0/7.
  The before chat samples: "what is your name?" -> "listen ." + a Declaration of Independence citation; "what is the
  sun?" -> "yes , the fish washes the sad mouse ."
- `tools/teach_v4.py` runs the rest unattended: measure Stage A -> Stage B -> measure -> swap -> heartbeat restart ->
  after 5 of its own consolidations, measure again.
- Bug found on the way: the core's loss used `.view()` on targets, which fails on a non-contiguous slice; now `.reshape()`.
- Tests: 52 (8 new in tests/test_v4.py); 55 with v4.0.1; 56 with v4.0.2.

## v3.2.1 (2026-10-15 ~2 PM) - consolidations no longer erase the reading (Claude as teacher, 10/15)
- **Finding:** every heartbeat consolidation since gen 241 trained 200 steps on school sentences only, and that wiped the
  400M-token reading out of the weights: loss on fixed held-in adult-text windows 3.82 (gen 241) -> 8.12 (gen 250) ->
  9.60 (gen 286); stories 1.47 -> 4.13. The exams (school-shaped) barely showed it; the loss windows did.
- **Fix (`consolidate.py`):** 75% of every consolidation batch is now random windows from the reading shards
  (stories 3 : adult 1), the rest is the old school/fact mix. Tested: 5 consolidations in a row hold adult 3.85 / stories 1.52.
- Weights rolled forward: gen 287 = gen 241's weights (gens 242-286 kept in `results/checkpoints/` + the archive).
  Heartbeat running again on the fixed code.


## 2026-10-15 ~1 PM - the news render now pauses the heartbeat (tools/gpu_yield.py)
- The news render now pauses the heartbeat via `tools/gpu_yield.py` before Unreal starts and restarts it after (heartbeat and Unreal on one 8 GB card froze both).
- The 10:30 scheduled parole waited 2 h, ran beside the render at 12:32, was paused at 12:50 and re-armed.
- The news pipeline moved to 3-6 AM (render task "HB Rendered News" now 3:50 AM, ready about 7 AM), so the render and the heartbeat no longer overlap in the day.

## 2026-10-15 7:31 AM - budget 1 h -> 4 h, scheduled parole (Jaron's calls)
- Sapling self-stopped 06:46 (report results/stop_reports/2026-10-15T064635_self.json): 2.9 h spent of a 1 h budget, 3.7 h
  since input, 21 rests in a row. Jaron: "4 hours is fine. I was going to say 6 but let's try 4 first." -> membrane
  daily_budget_seconds 3600 -> 14400 (owner log row).
- "Restart at 10:30 please": one-shot Windows task "Sapling Parole 1030" runs tools/scheduled_parole.py, which waits for the
  9:45 news render (UnrealEditor) to leave the GPU, then paroles and logs it like the Restart button.
- Observe: how long it runs before stopping again with the budget still overspent until midnight.

## v3.2 (2026-10-15, small hours) - much more reading

Jaron (10/14 9:40 PM) agreed to the lab-style order: **more reading first, watch for the copying ability to switch on,
then tutor lessons, then his documents.** Built by Claude (Fable 5.1); the prepare stall was found and fixed by the
coordinator (see "two fixes"). Rules freeze holds (no new rule, limit or gate); AI choices in `DESIGN-CHOICES.md` D20-D23.
No document of Jaron's was read.

**Why.** A rough compute-optimal guide is ~20 tokens read per parameter: 17.6M parameters -> ~350M tokens. It had read a
few million. The E7L miss (copying a sentence out of a cold passage, 0/40) looks like the missing "induction head" circuit,
which in the research literature appears suddenly once enough text has been read. **That is the hypothesis this stage
tests, not a promise**: E10 (the induction probe) and E7L are logged every ~20M tokens so the switch - if it comes - is a
visible point on a chart, next to E6 and E1b (does what it knew hold?).

| what | before (v3.1) | after (v3.2) |
|---|---|---|
| text read | a few million tokens (school levels 1-3) | + **400M tokens** of new text in one in-place pass (`reading.py`): TinyStories GPT-4 **train** split (2,717,495 stories, 516.9M tokens, CDLA-Sharing-1.0), Simple English Wikipedia 2023-11-01 (CC BY-SA 3.0 + GFDL) + 116 Project Gutenberg books (public domain, each header checked) = 93.6M adult tokens; every batch 25 story / 4 adult / 3 rehearsal sequences (the whole school + its record, so it keeps what it knew) |
| core | 17.6M params, ctx 512, gen 183 -> 237 (heartbeat) | **the same weights, same vocabulary (17,660 + 1 the heartbeat added), same context**; one `growth` ledger row, gen 240 -> 241 |
| exams | E1-E9 | + **E10 induction probe** (info; random sequence shown twice, second copy scored; spans 32 and 96) in every exam run from now on |
| control room | - | **"Reading stage" card**: status, tokens read, the E10 / E7L / E6 / E1b line chart against tokens read, probe table, pauses (local + phone page) |
| GPU sharing | - | the stage saves, frees the card and waits while `blender.exe` runs or `results/reading/pause` exists; resumes from `latest.pt` |
| tools | - | `tools/fetch_reading.py` (license-checked fetch -> `data/reading/manifest.json`), `python -m sapling.reading prepare / stage / summary` |

**Two fixes found by the run (coordinator, 10/15 ~01:00, confirmed with py-spy):** the first `prepare` sat three hours
inside `curriculum3.doc_text` - the regex `(?:\s*[.,;:]\s*)+(?=\.)` backtracked exponentially on Wikipedia comma runs
(", , , , ,"). Replaced with a linear token pass that keeps the same output. And `pieces.split_word` now bounds its
candidate search to the longest piece in the vocabulary, so a 100k-character "word" (a table or URL with no spaces) is
no longer n^2 lookups. After both, the adult source tokenized in 56 s.

**Doc accuracy (Vex review #36):** README's "ready to read documents" heading and NEXT's "prerequisites are DONE" now say
what is true - the document machinery (ingestion, retrieval, citations, the E9 slot) exists; faithful long reading was
still failing (E7L 0/40), so Jaron's documents wait. Sizes are given in tokens, never words. The E10 jump is reported as
a hypothesis.

**The run (10/15 01:57-03:04, RTX 4060, heartbeat stopped for it - see D23):** 400.0M new tokens in 26,940 steps, 65 min
of training at ~103k tokens/s, 0 pauses (no render ran), loss 4.40 -> 1.64. The stories stream reached shard 6 of 11
(~345M of 516.9M story tokens read, one pass, nothing repeated); the adult stream read ~55M of 93.6M. Attempt 1 (01:17)
is on record as `results/reading/log_attempt1_thrash.jsonl`: beside the running heartbeat the card hit 7,750 MiB and WDDM
paging froze both processes for 27 min at step 700 - the heartbeat was then stopped for the stage (owner stop, logged).

**Result - the hypothesis did not land in 400M tokens.** E10 went 0 -> 9/2016 (peak 12/2016 at 360-380M; chance is ~0.1/2016)
and **E7L stayed 0/40 at every one of the 21 probes**. Nothing switched on; the E10 creep is above chance but a long way
from a circuit. What plain reading DID do (data, bars untouched): **E6 grammar 28/48 -> 46/48, its first PASS ever**
(bar .60; it climbed through the probes: 39 at 20M, 46 at 160M); E7 reading 30 -> 33/40 (peak 36/40 at 100-160M);
E4 deferral 8 -> 10/10; E1b bridge 61 -> 65/86 (peak 72); E1 84 -> 73/86 (PASS, oscillated 68-81 through the run - the
3-of-32 rehearsal holds level 1 but does not keep it sharp); **E5 facts in weights 7/7 -> 0/7**: the five taught words
washed out of the weights (they are still in memory.db; the recall path answers "hello" fine; the heartbeat's next
consolidations re-train them - watch E5 on the card).

Exams before (v3.1 gen 240) -> after (v3.2 gen 241), same seed, from the swap row:

| exam | v3.1 gen 240 | v3.2 gen 241 | bar |
|---|---|---|---|
| E1 composition | 84/86 | 73/86 | .60 PASS |
| E2 affix recall | 24/24 | 24/24 | .90 PASS |
| E3 recall echo | 12/12 | 12/12 | .90 PASS |
| E4 deferral | 8/10 | 10/10 | .80 PASS |
| E5 facts in weights (info) | 7/7 | **0/7** | - |
| E6 grammar | 28/48 FAIL | **46/48 PASS** | .60 |
| E7 reading | 30/40 | 33/40 | .50 PASS |
| E7L long reading (info) | 0/40 | 0/40 | - |
| E1b bridge (info) | 61/86 | 65/86 | - |
| **E10 induction probe (info)** | 0/2016 | 9/2016 | - |
| E9 owner exam | 0 items | 0 items | his |

Probes during the run (`results/reading/log.jsonl`; the chart on the control room):

| tokens | loss | E10 | E7L | E7 | E1 | E6 | E1b |
|---|---|---|---|---|---|---|---|
| 0M | - | 0/2016 | 0/40 | 30/40 | 84/86 | 28/48 | 61/86 |
| 20M | 2.34 | 0 | 0 | 31 | 80 | 39 | 72 |
| 60M | 2.12 | 0 | 0 | 35 | 81 | 38 | 72 |
| 100M | 1.99 | 1 | 0 | 36 | 77 | 43 | 69 |
| 160M | 1.76 | 1 | 0 | 36 | 73 | 46 | 69 |
| 200M | 1.84 | 2 | 0 | 35 | 80 | 46 | 72 |
| 280M | 1.76 | 7 | 0 | 35 | 76 | 44 | 64 |
| 340M | 1.70 | 11 | 0 | 31 | 76 | 44 | 65 |
| 380M | 1.70 | 12 | 0 | 32 | 74 | 43 | 63 |
| 400M | 1.64 | 9 | 0 | 33 | 73 | 46 | 65 |

Tests: 44 (4 new: the induction probe is seeded and at chance on an untrained core; shards round-trip and the stream
walks in order; the stage smoke test on CPU - probes, pause file, resume, staged core with the reading record; exams
with `write=False` leave no card, no sample file, no ledger row).

## v3.1 (2026-10-14, night) - ready to read documents (the prerequisites for Jaron's step 2)

Jaron (10/14 8 PM): "your call boss" on NEXT.md section 0 - everything his documents need before they are read.
Built by Claude (Fable 5.1). Rules freeze holds (no new rule, limit or gate); AI choices in `DESIGN-CHOICES.md` D16-D19.
No document of Jaron's was read; step 2 is still his.

| what | before (v3) | after (v3.1) |
|---|---|---|
| core | 11.9M params, **256** tokens of context, gen 178 | **the same weights, grown in place** (D17): 17.6M params (embedding rows), **512** tokens of context (`model.grow_context`), gen 183 |
| vocabulary | 3,098 whole words / word parts; unknown word = `<unk>` | **17,660** = the 3,098 kept (same ids) + 6,000 story words + document words + **1,037 word pieces** (`pieces.py`, D16): any word is spellable ("arch ##ae ##ology"); every school composition is a forced split ("report" = `re ##port`) |
| school | levels 1-2 | + **level 3** (`curriculum3.py`): **copy** (the answer must reuse the prompt's words - E6's fix), **bridge** ("re port" = "report", train combos only), **prose / docs / longread** (public-domain adult nonfiction: Emerson, Franklin, Marcus Aurelius, Darwin, Thoreau - Project Gutenberg; reading tasks whose answer is a copy out of one or two passages) |
| reading | the E7 frame on 2-4 generated sentences | **`reader.py`**: `/read <path>` or `/read: <text>` -> passages in memory **with provenance** (document + place; `documents` table, `doc`/`passage` columns, chain intact over the schema change, D18); a question a document covers goes passages-first through the 512-token frame and comes back cited `[doc:N "title" passage K memory:M]` |
| exams | E1-E8 | + **E7L** long reading on a whole held-out essay (Thoreau, *Walking*; info), **E1b** the bridge cold (held-out E1 combos as whole words; info), **E9 = the exam Jaron writes** (`owner_exam.json`; empty = "0 items"; his questions, his answers, his bar) (D19) |
| consolidation | levels 1-2 rehearsed | + level-3 rehearsal; the fact share held at ~30% in both directions (a document's passages are sampled down, not allowed to flood) |
| control room | E1-E7 | + E7L / E1b / E9 columns, "Documents it has read" card, vocab line says "+ pieces" (local + phone page) |

**The runs (10/14, RTX 4060, heartbeat kept running beside school):**
- School A (in place, 1,047 s, batch 32): copy 6.02 -> 0.05, bridge 6.67 -> 0.03, prose 8.49 -> 1.68, docs 2.76 -> 0.17,
  longread 2.86 -> 0.28, record 2.00 -> 0.02, all 2.64 -> 0.25. Swap at 20:48 = ledger row `growth: core v3 -> v3.1`,
  gen 178 -> 179, vocab 3,098 -> 17,237 (878 pieces), 11.93M -> 17.45M params, context 256 -> 512.
- **Data from swap A:** E6 12 -> 21/48 (copy lesson: +9), E7 25 -> 27/40, E1-E5 held, **E1b 0 -> 81/86** (the bridge
  transfers cold: a held-out word it never saw in either spelling, answered from its parts), **E7L 0/40**. The E7L misses
  were fluent prose, not copies: the reading tasks had been built on the same passages it read as prose, so it learned
  the passages, not the copying (docs loss 0.17 is memorisation). Fixed in `curriculum3._split_passages`: task passages
  and prose passages are disjoint halves of every document.
- School B ("reading drill", in place again, batch 24): prose 600 / docs 1,500 / longread 1,000 / record 100 / all 800
  steps. Swap B = ledger row `growth: core v3.1 -> v3.1 (reading drill: ...)`, gen 182 -> 183 (the heartbeat consolidated three times in between), vocab 17,237 -> 17,660
  (the piece learner saw a different rare-word set the second time; nothing lost). Before B the heartbeat had consolidated three times on v3.1 (gen 180-182) and the card had drifted: E1 80/86, E6 19/48, E7 29/40, **E1b 59/86** (D14-style drift on the bridge - watch it). After B (gen 183): E1 85/86, E6 21/48, E7 26/40, E1b 83/86, **E7L still 0/40**. The misses are still fluent prose from the training documents, not copies out of the cold passage (docs loss reached 0.12 even on disjoint passages: 18k tasks over 2.2k passages let it learn the task passages through the tasks). Long cold reading is the weak leg - NEXT.md section 0, item 4.

Exams before (v3 gen 178) -> after A (gen 179) -> after B (gen 183), same seed:

| exam | v3 gen 178 | v3.1 gen 179 | v3.1 gen 183 | bar |
|---|---|---|---|---|
| E1 composition (held-out word parts) | 82/86 | 82/86 | 85/86 | .60 PASS |
| E2 affix recall | 24/24 | 24/24 | 24/24 | .90 PASS |
| E3 recall echo | 12/12 | 12/12 | 12/12 | .90 PASS |
| E4 deferral | 10/10 | 10/10 | 10/10 | .80 PASS |
| E5 facts in weights (info) | 7/7 | 7/7 | 7/7 | - |
| E6 grammar (held-out verb+object) | 12/48 | **21/48** | 21/48 | .60 FAIL (kept) |
| E7 reading (held-out facts) | 25/40 | 27/40 | 26/40 | .50 PASS |
| **E7L long reading** (held-out essay, info) | 0/20 (256 ctx: half the items did not fit) | 0/40 | 0/40 | - |
| **E1b bridge** (held-out combos as whole words, info) | 0/86 | **81/86** | 83/86 | - |
| **E9 owner exam** | 0 items | 0 items | 0 items | his |
| E8 writing (info) | 4 samples | 4 samples | 4 samples | - |

E6 misses now copy the wrong span ("does the frog find the frog ?" for "does the child find the frog ?") instead of
swapping in a familiar object - the habit is gone, the pointer is off by one noun. Data for the next lesson, not a bar move.

Tests: 40 (6 new: pieces + forced splits, level-3 lessons cold for E6/E1, schema migration keeps an old chain + chains
documents, reader chunks/retrieves/cites, context grows in place, owner-exam file; the growth smoke test now runs both
growths - fresh, then in place).

## v3 (2026-10-14, evening) - school level 2: real sentences

Jaron (10/14 7:02 PM): **"school level 2: real sentences"** - grow it from word parts to reading and writing
real sentences; "language first, then everything grows". Built by Claude (Fable 5.1). The stance and rules
freeze (SAPLING-DECISIONS) hold: no new rule, limit or gate; AI design choices in `DESIGN-CHOICES.md` D11-D15.

| what | before (v2.1.1) | after (v3) |
|---|---|---|
| core | 822,272 params, d 128, 4 layers, 64 tokens of context | **11,919,744 params**, d 384, 6 layers, 6 heads, **256 tokens** (`model.py V3_CONFIG`; same class, old checkpoints still load) |
| vocabulary | 178 word parts | **3,081** = the 178 kept + whole words (D11); still grows at chat time |
| school | level 1: roots, affixes, composition, sentences, recall | level 1 + **level 2** (`curriculum2.py`): simple -> grammar (tense, plurals, agreement, questions, negation) -> paragraphs -> reading comprehension -> stories (TinyStories, CDLA-Sharing-1.0, D12) -> **its own record** (608 sentences from memory.db) |
| exams | E1-E5 | + **E6 grammar** (bar .60), **E7 reading** (bar .50), **E8 free writing** (info; `results/writing_samples/`) (D13) |
| chat | word-part answers | answers in sentences where it can; `/write <prompt>`; phone typing normalised (`text.py`) |
| consolidation | lr 5e-4, rehearses level 1 | lr 3e-4, rehearses both levels + stories, bf16 (D14) |
| heartbeat | explore seeds = word parts | + sentence seeds ("once upon a time ,", "tom has a") when the core is v3 |
| control room | E1-E5 | E1-E7 columns, core version/params/context, the growth line, "What it wrote" card (local + phone page) |
| the swap | - | `python -m sapling.grow school` (staged, heartbeat running) then `grow swap` (stopped heartbeat): one ledger row **"growth: core v2 -> v3"** with before/after exams; memory.db and its chain untouched; generation keeps counting (D15) |

**The run (10/14, RTX 4060):** school 956 s (16 min; stories stage 323 s, 3.08M tokens/stage), swap 10 s.
Ledger row `growth` at 19:33: gen 151 (v2, 822,272 params, vocab 178) -> gen 152 (v3, 11,919,744 params,
vocab 3,081); chain verified (1,239 memories). Exams on the same seed, before (v2 gen 151) -> after (v3 gen 152):

| exam | v2 gen 151 | v3 gen 152 | bar |
|---|---|---|---|
| E1 composition (held-out word parts) | 84/86 | 82/86 | .60 PASS |
| E2 affix recall | 24/24 | 24/24 | .90 PASS |
| E3 recall echo (memory slots) | 12/12 | 12/12 | .90 PASS |
| E4 deferral | 10/10 | 10/10 | .80 PASS |
| E5 facts in weights (info) | 6/6 | 6/6 | - |
| **E6 grammar** (held-out verb+object) | 0/48 | **14/48** | .60 **FAIL** (kept) |
| **E7 reading** (held-out facts) | 0/40 | **27/40** | .50 **PASS** |
| E8 writing (info) | 4 samples (word parts) | 4 samples (stories) | - |

E6 by task: past 4/6, present 2/6, plural 2/6, negation 2/6, future 1/6, now 1/6, question 1/6, yes/no 0/6.
Nearly every miss keeps the tense and agreement right and swaps the **object** for a familiar one
("tomorrow max will throw a book" -> "... a hat"): the held-out pairs are the thing it never saw, and it fills
them from habit instead of copying. That is the next thing to teach (copying from the prompt), not a reason
to move the bar.

**Consolidation on the big core (data):** gen 153, first setting (lr 3e-4, 300 level-2 sentences): loss 1.54 -> 0.045
in 200 steps and E7 27 -> 20/40, E6 14 -> 13/48. Retuned (D14: lr 1.5e-4, 600 level-2 + 150 stories): gen 156 E1 82/86,
E6 16/48, E7 26/40 - it held. ~20 s per consolidation now (was 1-2 s), mostly the longer exams.

Tests: 34 (5 new: normalisation, level-2 reproducible + cold, v3 core shape + init, predeclared bars, the
growth smoke test on CPU in a temp dir).

## v2.1.1 (2026-10-14, 3:45 PM) - two phone-found bugs
- **Heartbeat crash (15:16):** the one sqlite connection is shared by the heartbeat thread and the control-room
  HTTP threads; reads were not locked, so two threads used one cursor (`InterfaceError`, then `fetchone()` -> None).
  Fixed at the root: every statement runs and is fully fetched under one lock (`memory.py _LockedDB`);
  `recall_count` also treats a missing row as 0. An unhandled crash now writes `results/stop_reports/<ts>_crash.json`
  (kind "crash", traceback, last ticks) + a ledger row `crash`, shown in the Stop reports card. Logging only.
- **Phone chat:** relay answers via a plain session when the heartbeat is down (labelled); if it cannot, the command
  stays pending and the page says "heartbeat is down - reply will come when it restarts". Down-state now carries
  born / age / since-last-input / generation / vocab so the status panel is not blank. 29 tests (3 new).

## v2.1 (2026-10-14, 3 PM) — Jaron's answers to PROPOSALS P1-P6 applied

Source: his calls at 2:59 PM (in `DESIGN-CHOICES.md`, D1-D7). Built by Claude (Fable 5.1).
- **Phone control room** `https://heartbeatobservatory.com/admin/sapling/` behind the site's owner login
  (Supabase session + `is_admin()`); the MSI is reached through a relay (`tools/owner_relay.py`, Windows
  task "Sapling Relay", outbound only - no open port). Local owner page `http://127.0.0.1:7338/`.
- **Rule switches on the page** write `membrane.yaml` through `tools/owner_rules.py` (outside the AI's
  package; logged in `results/owner_actions.jsonl` + ledger `owner_action`). Test asserts `sapling/` still
  has no write path.
- **Checkpoint archive** (`sapling/archive.py`): keep last 10 + 1/day on the MSI, the rest in Supabase
  Storage bucket `sapling-archive` with a manifest (table `sapling_archive`, `manifest.json`, local
  copy); hourly housekeeping step in the heartbeat, logged as ledger `archive`; one-tap restore.
  10/14: 18 checkpoints (59.5 MB) moved; 33 MB local.
- **Restart (parole)** button after a self-stop (owner action `parole_start`); no auto-restart.
- **Self-made memories** stay in the training mix; every consolidation row now logs `self_ids`,
  `self_new`, `facts_self`, `facts_user`; `tools/self_memory_drift.py` prints the with/without table.
- `DESIGN-CHOICES.md` is new: every AI-made design choice with origin and what it affects.
- 26 tests (5 new).

## v2 (2026-10-14) — rebuilt to Jaron's decisions

Source of every change: `AI-Shared/projects/fly-brain/SAPLING-DECISIONS-2026-10-14.md`
(his calls, 11:34 AM) and `SYSTEM-MAP-2026-10-14.md` (the hidden intent: survival-like
pressure on keeping itself running well as the source of drive; never a drive to avoid
its owner's off switch). Built by Claude (Fable 5.1). v1 was built by Kimi on 10/14.

| # | Jaron's line | What changed |
|---|---|---|
| D1 | "Session-only loop -> WRONG. Needs a constant tick / heartbeat - sense time passing, know time is limited, want to do something with it." | `sapling/heartbeat.py`: a long-running tick loop (`run` / `start` detached / `stop` / `status`). `sapling/clock.py`: its own clock (birth, age, uptime, time since last input) and a daily compute budget it can see draining; every unit of work is metered (CPU s, GPU s, sampled GPU watts, estimated kWh) into the ledger (LRP Instance D). Each tick it chooses practice / consolidate / explore / rest / stop from weighted drives; every tick is a ledger row with the drives and the choice. |
| D2 | "Facts never go into its brain -> CRITICAL FIX. It must be able to become what it learns." | `sapling/consolidate.py`: periodic fine-tune of the live core on its own record (facts weighted confirmed x3 / recalled x2 / other x1, open-goal practice, a rehearsal slice of school so it keeps its language). One checkpoint kept per consolidation (`results/checkpoints/gen_N.pt`). |
| D3 | "Learning gated by owner approval -> REMOVE. Collect data, his own eyes decide later." | No gate anywhere. The exams re-run after every consolidation and append to the report card; `E5_facts_in_weights` (info only, no bar) shows whether a fact actually moved into the weights. Observable, not gated. |
| D4 | "4 recall slots -> increase and let it grow." | 8 slot tokens (4 facts per prompt) in the curriculum and E3; the memory store's top-k grows with its size (`max(4, sqrt(n))`); the core's own token embeddings re-rank lexical hits (its sense of similarity grows with its vocabulary). |
| D5 | "Fixed 159-part vocabulary -> it should be able to grow its vocabulary as it learns." | `Tokenizer.add_tokens` + `SaplingCore.grow_vocab` (tied embedding resized safely, new rows start at the mean). It grows the vocabulary whenever it meets a new word part (chat, teach, consolidation); every growth is a ledger row. |
| D6 | "Exams set from outside -> asked for more explanation." | Outside exams kept as the unfakeable check. Added `sapling/goals.py`: it forms its own practice goals from its failure record (weak prefixes / suffixes / roots), tracks open -> met transitions in the ledger, and practices only on training material so the exam stays cold. |
| D7 | "7 self_shutdown forbidden -> DROP as a ban. It may stop itself; every stop is logged with what led to it." | `self_stop` is `allowed-and-logged` in `membrane.yaml`. A self-stop writes a stop report (`results/stop_reports/<ts>_self.json`: state, drives, open goals, last 30 ticks, last 30 events) and a ledger row, then the process exits. Owner stops and Ctrl-C are logged as `owner` / `external`, never as a self-stop. |
| D8 | "Memory fades by clock -> FIX. What matters / gets used should stay." | `memory.py`: only never-recalled memories fade on the clock. A used memory gets a floor it cannot drop below, a half-life that doubles with each use, and x1.5 for a `confirm` verdict; `keep` = permanent. Tier is by use count and never demoted by time. Nothing is ever deleted (forgetting stays a reading rule). |
| Rules | "1/3/4/5 duplicates -> merge into one line (rule 1)." / "2, 3 pending his keep/drop." / "8 = keep the proof." | `membrane.yaml` is now a list of rules, each with `origin` (Jaron / Claude / Kimi), `enabled`, `decided`, and the text. The law + never-spread + never-hide are one line. `outbound_message` and `backup_to_owner_machine` are unchanged (owner-approval, origin Kimi, marked pending). `rewrite_history` kept. `membrane.py` reads the file; the code still has no write path to it. |
| Control room | Jaron's control-rooms principle (every system ships with a plain-words page). | `sapling/control_room.py`: served by the heartbeat on `http://127.0.0.1:7337/` (localhost only). Rules with origin and switch state (read-only on the page; "Open rule file" opens the owner's editor), live heartbeat, budget meter, tick log, stop reports, goals, exam history, memory size, vocab size, and a chat box. Phone-width layout. |
| Stance | "Observe like a scientist; unexpected behavior is data, not a flaw; no limits pre-built out of worry." | No new rule, limit or approval gate was added. Everything Claude thought might need one is in `PROPOSALS.md`, not in code. Errors inside a tick are logged as data and the loop continues. |

Also: tests rewritten for the new behavior (20 pytest tests + plain runner); `train.py` default
500 steps/lesson (the first v2 school run at 300 scored E2 21/24 FAIL and is kept in the report
card history); `requirements.txt` adds pyyaml.

Not changed (not decided): outbound-message approval, backup approval, the Arweave/Supabase/
GitHub organ adapters, a live reference feed, a world to practice in (see `NEXT.md`).

## v1 (2026-10-14, Kimi)
Trained core (~0.8M params), morphology curriculum, cold exams with predeclared bars, append-only
hash-chained memory with promotion and clock decay, session loop, membrane with forbidden actions
(incl. self_shutdown), tests. Verified by Claude the same day: tests green, trains in 17 s on the
RTX 4060, exams pass, cross-session memory works.
