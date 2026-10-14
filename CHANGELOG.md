# Changelog

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
