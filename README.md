# Sapling

A small AI of its own design, rebuilt (v2) to Jaron's decisions of 2026-10-14: a **heartbeat**
that keeps going on its own and feels time and energy passing, a **trained core** (~0.82M
parameters, from scratch) that can **become what it learns**, **memory** that grows and is
protected by use, **goals it sets itself**, **outside exams** it cannot fake, the freedom to
**stop itself** (every stop is a report), and a **membrane** of rules only the owner edits -
each rule with its origin. A plain-words **control room** on localhost shows all of it and has
a chat box.

Why each piece is the way it is: [CHANGELOG.md](CHANGELOG.md) (each change tied to his
decision line), the system shape in [ARCHITECTURE.md](ARCHITECTURE.md), the v1 reasoning in
[WRITEUP.md](WRITEUP.md), what comes next in [NEXT.md](NEXT.md), and the things Claude would
have added as rules but did not in [PROPOSALS.md](PROPOSALS.md).

## Where to run it from

- **Phone:** https://heartbeatobservatory.com/admin/sapling/ - sign in at /admin first (owner login).
  Chat, rules (switches = the owner's edit, logged), heartbeat, exams, tick log, stop reports,
  checkpoint archive with one-tap restore, and **Restart (parole)** after a self-stop.
- **On the MSI:** http://127.0.0.1:7338/ (the owner's door, `tools/owner_relay.py`, Windows task
  "Sapling Relay") or the AI's own page http://127.0.0.1:7337/ (served by the heartbeat).
- Every design choice an AI made, with origin and what it touches: [DESIGN-CHOICES.md](DESIGN-CHOICES.md).

## The stance (Jaron, 10/14)

Observe like a scientist. Behavior the code "says shouldn't happen" is data, not a flaw.
No limits pre-built out of worry; bad actions are handled by contain / correct / parole, not
deletion. So there is no approval gate on what it learns, no ban on stopping, no cap that
switches it off - and every tick, meter reading, consolidation, goal and stop is on an
append-only hash-chained record for his eyes.

## The rules (membrane.yaml - the owner's file; the code has no write path to it)

| rule | origin | state |
|---|---|---|
| Back up on the owner's own machines with permission; never spread to other machines; never hide from the owner | Jaron | on (old 1/3/4/5 merged into one line) |
| Never rewrite its own history (append-only hash chain) | Jaron | on |
| It may stop itself; every stop is logged with what led to it | Jaron | on (old "self_shutdown forbidden" dropped) |
| Outbound messages need owner approval | Kimi | on, pending Jaron's keep/drop |
| Backup to an owner machine needs owner approval | Kimi | on, pending Jaron's keep/drop |
| The system cannot write this file | Kimi (structural) | on |

## Quickstart (the RTX 4060 PC, or any CPU)

```bash
pip install -r requirements.txt           # torch, pyyaml
python -m sapling.train                   # school: 5 lessons, ~20 s on the 4060 (gen 0)
python -m sapling.exams                   # cold exams, predeclared bars -> results/report_card.json
python -m sapling.heartbeat start         # the heartbeat, detached; log in results/heartbeat.log
#   control room: http://127.0.0.1:7337/  (rules, status, budget, ticks, goals, exams, memory, chat)
python -m sapling.heartbeat status
python -m sapling.heartbeat stop          # owner stop -> stop report + exit
python -m sapling.session                 # talk without the heartbeat; /teach /confirm /invalidate /keep /report
python tests/run_tests.py                 # invariants without pytest   (or: pytest tests/)
```

## What it does every tick

It reads its own state - budget left today, open goals, facts waiting to go into its weights,
how long since it explored, how long since anyone spoke to it - turns that into drives, and
draws one action: **practice** (its weak parts, training material only), **consolidate**
(fine-tune on its own record; one checkpoint per generation; exams re-run after), **explore**
(rehearse an old memory, or free-generate and keep what it said), **rest**, or **stop**.
Every tick is a ledger row with the drives and the choice. Every unit of work is metered:
CPU seconds, GPU seconds, a sampled GPU power reading, and an estimated kWh.

## What the first 20-minute heartbeat run produced (RTX 4060, 2026-10-14)

See the section "Run record" below - filled from the ledger, not from memory.

**Run 1 — 21 minutes, 223 ticks** (13:51-14:12, owner stop; `results/heartbeat_run1.log`,
`results/report_card_run1.json`, `results/stop_reports/2026-10-14T141252_owner.json`):

```
ticks: practice 150 / explore 48 / consolidate 13 / rest 12 / stop 0   errors inside ticks: 0
metered: 54.9 cpu-s + 59.1 gpu-s of a 3600-s daily budget (96.6% left at stop)
         GPU 3.3-25.3 W sampled; ~0.32 Wh estimated for the run (211 meter rows)
consolidations: gen 2 -> gen 14, 1.0-2.3 s each, loss ~0.75 -> ~0.35, exams after every one
exams across the 13 consolidations: E1 59-76/86 (all PASS, drifting), E2 20-24/24, E3 12/12, E4 10/10
goals: 203 open/met transitions (e.g. suffix:er, root:luc opened at 3-4 misses/10; root:fer, prefix:trans met)
practice accuracy: 853/900 (first 10 ticks 93%, last 10 ticks 90%)
memory: 115 rows, chain verified; vocab 163 -> 173 (zint, cold, wind added when taught by chat)
recall after 21 minutes: glorp (taught before the run) and zint (taught at tick 7 by chat) answered
  from memory, tiers working / long_term; brindle (never asked during the run) fell to strength 0.99
```

Sample tick rows (ledger, verbatim):
```
#7   practice   {misses: [], n: 6, ok: 6}      drives {practice .749, explore .154, rest .080, consolidate 0, stop 0}
#109 explore    {kind: generate, prompt: "the root spect", said: "means look .", memory: 64}
#229 consolidate {generation 14, facts_new 2, loss .717 -> .349, checkpoint gen_0014.pt, exams: E1 66/86 PASS ...}
```

**What run 1 found (data, and two bugs):** E5 (facts in weights) stayed at 0-2/4 for all 13
consolidations - the core answered a blend ("the word glorp means a small means happy ."). Cause:
new word parts were initialised almost on top of each other (noise 0.02 against an embedding
spread of 0.9), so glorp / zib / brindle were one word to it; and facts were ~3% of the fine-tune
mix. Second bug: "?" and "." stripped to "" and matched every memory, so one self memory
("re + scrib = re scrib .") was recalled on every query and reached "permanent" - an artifact,
not a promotion. Both fixed (model.py init at the embedding spread; facts ~30% of the mix;
memory.py discards ""), and the E2 21/24 FAIL from the first v2 school run stays in the card.

**Run 2 — fixed recipe, fresh gen 0** (started 14:14; the heartbeat is still running):

```
gen 2  E1 74/86  E2 21/24  E3 12/12  E4 10/10  E5 5/5
gen 3  E1 74/86  E2 23/24  E3 12/12  E4 10/10  E5 5/5
gen 4  E1 67/86  E2 22/24  E3 12/12  E4 10/10  E5 5/5
raw core, no memory at all:  q : what does quimble mean ? a : -> the word quimble means a tiny lantern .
                             q : what does brindle mean ? a : -> the word brindle means a small brave boat .
                             q : what does <m1> mean ? a :    -> i do not have that yet .   (deferral kept)
```

All five facts it was taught (zib, glorp, brindle, zint, quimble) are now in its weights - it
became what it learned, with no gate, and the outside exams show the cost (E1 drifts down a few
points per consolidation; that drift is the next thing to watch, not a reason to add a gate).

## Repo map

```
sapling/
  heartbeat.py     # the tick loop: drives -> choice -> act -> log; start/stop/status; stop reports
  clock.py         # birth, age, uptime, since-last-input; daily budget; metering (cpu s, gpu s, watts, kWh)
  consolidate.py   # record -> weights; checkpoint per generation; exams after each
  goals.py         # its own practice goals from its failure record (parts, never exam items)
  control_room.py  # localhost page: rules + origins, status, budget, ticks, stops, goals, exams, memory, chat
  session.py       # a turn of talk; vocab grows on new word parts; up to 4 recalled facts per prompt
  memory.py        # append-only hash-chained SQLite; fades only if unused; use protects; top-k grows
  exams.py         # E1-E4 outside exams (bars fixed, FAIL kept); E5 facts-in-weights (info only)
  curriculum.py    # school: roots -> affixes -> composition -> sentences -> recall (8 slots)
  model.py         # tiny decoder-only transformer; grow_vocab() resizes the tied embedding safely
  train.py         # school training (gen 0), tokenizer with add_tokens, checkpoint save/load
  membrane.py      # reads membrane.yaml (rules with origin + switch); no write path
  archive.py       # hourly: keep last 10 + 1/day local, move the rest to cloud (Supabase 'sapling-archive'), restore any gen
membrane.yaml      # the rules and the settings - the owner's file
tools/             # OWNER side, outside the AI's package:
  owner_relay.py   # phone/remote door: pushes state to Supabase, runs owner commands (chat, rule switch, restore, parole)
  owner_rules.py   # flips a rule's enabled line when the owner clicks; logged
  self_memory_drift.py # compares exams with vs without self-made memories in training
DESIGN-CHOICES.md  # every design choice an AI made, who made it, what it affects, options not chosen
PROPOSALS.md       # open questions for the owner; nothing in here is built
CHANGELOG.md       # v1 -> v2 -> v2.1, each change tied to Jaron's decision
results/           # report card (history kept, FAILs kept), train log, stop reports, memory.db, checkpoints/
tests/             # 26 pytest tests + plain runner
```

## Lineage

v1 by Kimi (10/14) from Jaron's principle repos: Live-Reference Principle, fed-not-filled, ECIH,
memory-as-spatial-arrangement, persistent-memory-substrate, Atomic Chronicle, Keava-Owent,
fly-school, watch, pam-continuum, accumulation-sort. v2 and v2.1 (phone control room, cloud archive, parole) by Claude (Fable 5.1), same day, to
his decisions. The fly mattered for the heartbeat - constant going - not as the brain.

License: MIT.
