# Sapling

A small, independent AI of own design: a **trained core** (~820K parameters,
trained from scratch on a school-style morphology curriculum), **live knowledge**
kept out of the weights, and **its own memory** that persists across sessions,
promotes what gets re-used, fades what doesn't, and never rewrites its history.

This is v1: the first working piece, built for real, with the exams and the
fails kept on the record. Design reasoning is in [WRITEUP.md](WRITEUP.md),
the system shape in [ARCHITECTURE.md](ARCHITECTURE.md), and the road forward
in [NEXT.md](NEXT.md).

## The one law

> The system may copy and back itself up on its owner's own machines, with
> permission. It must never spread to other people's machines or hide from
> its owner.

Enforced in `membrane.yaml` (human-owned; the system has no write path to it)
and locked by tests.

## Quickstart (the RTX 4060 PC, or any CPU)

```bash
pip install -r requirements.txt          # torch only
python -m sapling.train                  # school: 5 lessons, ~5 min on CPU, faster on the 4060
python -m sapling.exams                  # cold exams, predeclared bars -> results/report_card.json
python -m sapling.session                # talk to it; /teach /keep /report /quit
python tests/run_tests.py                # invariants, no pytest needed (or: pytest tests/)
```

## What it actually produced (this repo's own runs, CPU, seed 7337)

**School** — 5 lessons, 819,840 params, 159-token morpheme vocabulary,
~4.4 minutes total on 3 CPU cores (`results/train_log.jsonl`,
`results/train_run*_stdout.txt`):

```
lesson=roots        loss 84.946 -> 0.084
lesson=affixes      loss  7.263 -> 0.094
lesson=composition  loss 12.794 -> 0.633
lesson=sentences    loss  3.544 -> 0.700
lesson=recall       loss  6.545 -> 0.532
```

**Exams** (cold, held-out, predeclared bars — `results/report_card.json`):

```
E1_composition   85/86  acc=0.988  bar=0.60  PASS   # meanings of NEVER-TAUGHT word combos
E2_affix_recall  24/24  acc=1.000  bar=0.90  PASS
E3_recall_echo   10/10  acc=1.000  bar=0.90  PASS   # answer from injected recall slots
E4_deferral      10/10  acc=1.000  bar=0.80  PASS   # "i do not have that yet ." when no recall
```

The one E1 miss: asked `what does vis able mean ?` it answered the composition
form `vis + able = vis able .` — a format confusion, not a semantic one.

**The FAIL that is kept:** run 1 spelled composed words as atomic tokens
(`cograph`), so held-out words had untrained embeddings and E1 scored **0/86**.
That run is preserved in `report_card.json → history` and in
`results/train_run1_stdout.txt`. The fix (visible morpheme boundaries:
`co graph`) is why run 2 passes. The fail is the finding — a model cannot
generalize over tokens it cannot represent.

**Memory across sessions** (`results/demo_session1.txt`, `demo_session2.txt`):

```
== session 1 ==                        == session 2 (a NEW process) ==
> /teach the word glorp means happy    > what does glorp mean ?
remembered as memory:1                 the word glorp means happy . [memory:1 user/captured]
                                       > what does spect mean ?
                                       the root spect means look .   # from the trained core
```

`glorp` was never in the training vocabulary. The core answered through the
memory layer's recall slots — the Live-Reference Principle working as
architecture: the fact lives in the record, the skill lives in the weights.

## Repo map

```
sapling/
  curriculum.py   # school: roots -> affixes -> composition -> sentences -> recall (deterministic, seeded)
  model.py        # tiny decoder-only transformer (~0.8M params)
  train.py        # staged training, report card, checkpoint
  exams.py        # E1-E4, predeclared bars, FAILs kept
  memory.py       # append-only hash-chained SQLite: memories, recalls, ledger; promotion + half-life decay
  session.py      # the loop: observe -> meter -> recall -> answer -> record
  membrane.py     # human-owned permissions; the system can never write them
membrane.yaml     # the law and the forbidden actions
tests/            # pytest suite + plain-python runner (tamper, promotion, anti-shutdown, ...)
results/          # train logs, report card (with the kept FAIL), demo transcripts
```

## Lineage (the account this grew from)

Live-Reference Principle · fed-not-filled · ECIH · memory-as-spatial-arrangement ·
persistent-memory-substrate · Atomic Chronicle · Keava-Owent · fly-school · watch ·
pam-continuum · accumulation-sort. What each contributed — and what was dropped and
why — is in [WRITEUP.md](WRITEUP.md).

License: MIT.
