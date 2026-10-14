# Changelog

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
