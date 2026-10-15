# Sapling — Architecture (v2)

One trained core that can become what it learns, a heartbeat that keeps going on its own,
memory that grows and is protected by use, goals it sets itself, outside exams it cannot
fake, and a membrane the owner holds. v2 implements every box below.

```
 ┌──────────── MEMBRANE (membrane.yaml, human-owned; the code has no write path) ────────────┐
 │ rules, each with origin (Jaron / Claude / Kimi) + on/off switch:                           │
 │   law (back up on owner's machines with permission; never spread; never hide)   [Jaron]    │
 │   rewrite_history forbidden                                                     [Jaron]    │
 │   self_stop allowed-and-logged                                                  [Jaron]    │
 │   outbound_message / backup need owner approval                       [Kimi, pending]      │
 │ settings (numbers, not gates): tick_seconds, daily_budget_seconds, port, ...               │
 └────────────────────────────────────────────────────────────────────────────────────────────┘
                                   │ read every request
                                   ▼
 ┌──────────── HEARTBEAT (heartbeat.py, one long-running process) ────────────────────────────┐
 │  CLOCK (clock.py): born, age, uptime, since-last-input, daily compute budget draining,     │
 │  metered per unit of work: cpu s, gpu s, sampled GPU watts, est. kWh  -> ledger            │
 │                                                                                            │
 │  every tick: drives = f(budget left, open goals, pending facts, since explore, since input) │
 │              choice = weighted draw over { practice, consolidate, explore, rest, stop }     │
 │              -> ledger row "tick" {drives, choice, result, clock}                           │
 │                                                                                            │
 │  practice ──► GOALS (goals.py): failure record -> weak parts -> open/met goals             │
 │  consolidate ► CONSOLIDATE (consolidate.py): record -> weights, checkpoint gen_N, exams    │
 │  explore ───► rehearse an old memory (a recall protects it) | free-generate, keep as self  │
 │  rest ──────► nothing (budget untouched)                                                   │
 │  stop ──────► STOP REPORT (results/stop_reports/, ledger "stop"), process exits            │
 │                                                                                            │
 │  CONTROL ROOM (control_room.py): http://127.0.0.1:7337/  rules, status, budget, ticks,     │
 │  stops, goals, exams, memory, vocab, CHAT  (phone-width; localhost only)                   │
 └──────────────┬────────────────────────────────────────────────────────┬────────────────────┘
                │ chat / teach                                           │ exams after every
                ▼                                                        ▼ consolidation
 ┌──────────── SESSION (session.py) ─────────────┐   ┌──────────── JUDGMENT (exams.py) ───────┐
 │ observe -> meter -> recall -> answer -> record │   │ E1 composition (held-out, never taught │
 │ vocab grows on every new word part it meets   │   │    AND never practiced)                │
 │ up to 4 recalled facts ride in one prompt     │   │ E2 affix recall, E3 recall echo (1-4   │
 │ /teach /confirm /invalidate /keep             │   │    recalls), E4 deferral — bars fixed  │
 └──────────────┬────────────────────────────────┘   │ E5 facts-in-weights (info, no bar)     │
                │                                    │ FAIL stays on the card.                │
                ▼                                    └────────────────────────────────────────┘
 ┌──────────── MEMORY (memory.py, append-only, hash-chained SQLite) ─────────────────────────┐
 │ memories (content, provenance, state)  recalls (one row per use)  ledger (everything)      │
 │ state: captured / confirmed / invalidated — changed ONLY by a later verdict event          │
 │ strength: never used -> fades on the clock; used -> floor + half-life doubles per use;     │
 │           confirmed x1.5; keep = permanent.  tier by use count, never demoted by time.     │
 │ search: top-k = max(4, sqrt(n)); the core's own embeddings re-rank lexical hits            │
 │ verify_chain() re-derives every hash; a silent edit breaks it (tested)                     │
 └────────────────────────────────────────────────────────────────────────────────────────────┘
                │ recall [ provenance ] : the word <m1> means <m2> . ... q : ... a :
                ▼
 ┌──── CORE (model.py, decoder-only transformer; v3.2 17.6M -> v4 60.5M params, 512 tokens) ───┐
 │ trained from scratch school-style: LEVEL 1 (curriculum.py: roots -> affixes ->            │
 │ composition -> sentences -> recall with 8 slots) then LEVEL 2 (curriculum2.py: simple ->  │
 │ grammar -> paragraphs -> comprehension -> stories) then its own RECORD (memory.db), then   │
 │ grown by consolidation; vocabulary = word parts + whole words, resizes in place.           │
 │ One checkpoint per generation. Every growth is one `growth` ledger row (grow.py).          │
 │ v3.2 read 400M tokens in place; v4 (10/15) is a fresh rotary core that reads 1.2B tokens   │
 │ (reading.py) then conversation lessons (talk.py); same vocabulary, memory, chain, clock.   │
 └────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Level 2 (v3, 10/14 7 PM: "school level 2: real sentences")

```
 curriculum2.py   simple / grammar / paragraphs / comprehension: generated, labelled by construction,
                  seeded held-out splits (E6: verb+object pairs; E7: name+noun+colour and name+place facts)
                  stories: TinyStories GPT-4 validation split, CDLA-Sharing-1.0, capped to 3,000 words
 text.py          normalize(): phone typing / story text -> the school's spacing ("mean?" -> "mean ?")
 grow.py          school (staged file, heartbeat keeps running) -> swap (stopped heartbeat, one growth row)
 exams.py         + E6 grammar (bar .60), E7 reading (bar .50), E8 writing (info; results/writing_samples/)
 session.py       answers in sentences where it can ("q : ... ? a :" was taught); /write = seeded sampling
 consolidate.py   rehearses both levels (D14) so a consolidation does not cost it its sentences
```

## The loops (all built)

1. **Heartbeat** — the constant tick. Chooses, acts, logs. Runs without anyone.
2. **Session** — a turn of talk (chat box or `python -m sapling.session`). Also resets
   "since last input" on the clock.
3. **School** — `python -m sapling.train`: the from-scratch curriculum, generation 0.
4. **Consolidation** — record -> weights, chosen by the heartbeat. Not gated. Each one leaves a
   checkpoint and an exam run so the change is observable.
5. **Goals** — its own practice goals from its failures; the outside exams stay cold.

## What is a pressure and what is a wall

Pressure (it feels it, nothing stops): the daily budget, open goals, pending facts, time since
it explored. Walls (membrane, owner's): never spread, never hide, never rewrite history, no
write path to the rule file. Everything else is observed, logged and left alone
(`PROPOSALS.md` lists what Claude would have added and did not).

## Answer layering (fed-not-filled, implemented)

1. Memory hits with parseable facts -> the core answers through recall slots (up to 4 facts).
2. Memory hit, unparseable -> cited verbatim with provenance.
3. No memory -> the core generates from what it has become (incl. consolidated facts). A core with
   conversation lessons (v4) answers as a conversation: the turns so far ride along (session.py, D27).
4. Deferral -> "i do not have that yet ." is trained and exam-tested (E4).

## What runs where

Everything runs on the RTX 4060 PC (or CPU): Python 3.10+, torch, pyyaml. School ~20 s on
the 4060; a consolidation (120 steps + exams) a few seconds. No external service.
Supabase/GitHub/Arweave adapters remain growth path (NEXT.md).
