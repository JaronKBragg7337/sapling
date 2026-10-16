# Sapling - how long each training run took (wall clock, from the run logs)
Kept so cloud vs MSI can be compared on real numbers. Add a row after every run. MSI = laptop RTX 4060 8 GB.

| Run (name) | Machine | Model size | Tokens | Start | End | Wall time | Speed (tokens/s) | Notes |
|---|---|---|---|---|---|---|---|---|
| v1/v2 school (Pip/Grain) | MSI | 0.82M | ~130K | 10/14 13:48 | 10/14 13:49 | ~12-20 s | - | 2-5 s per lesson |
| v3 growth (Kernel) | MSI | 11.9M | - | 10/14 19:17 | 10/14 19:33 | 16 min | - | from README run record |
| v3.1 growth (Kernel 3.1) | MSI | 17.6M | - | 10/14 20:25 | 10/14 21:05 | 40 min | - | from README run record |
| v3.2 reading, attempt 1 (Kernel 3.2) | MSI | 17.6M | 10.4M | 10/15 01:15 | 01:20 then froze | stuck 27 min | 102,700 | heartbeat + training overflowed GPU memory; stopped |
| v3.2 reading (Kernel 3.2) | MSI | 17.6M | 400M | 10/15 01:56 | 10/15 03:04 | 1 h 08 min | 102,800 | |
| v4 Stage A reading (Husk) | MSI | 60.5M | 1.2B | 10/15 14:16 | ~23:50 (projected) | ~9 h 35 min | 35,500 | 830M done at 20:53 |
| v4 Stage B conversation (Husk) | MSI | 60.5M | - | after Stage A | - | ~30 min (est.) | - | runs automatically |

From now on every run logs start, end and tokens/s.

## Speed test: MSI vs cloud (10/15-16 night, tools/bench_speed.py, same 60.5M v4 model, random tokens)
| Machine | Settings | Tokens/s | 1.2B tokens would take | Cost per 1.2B run |
|---|---|---|---|---|
| MSI laptop RTX 4060 | micro 16 (Stage A, real) | 35,500 | 9 h 35 min | electricity |
| RunPod H100 SXM (secure, $3.99/h) | micro 16 (same as MSI) | 261,267 | 1 h 17 min | ~$5.10 |
| RunPod H100 SXM | micro 32 (fits on 80 GB) | 303,153 | 1 h 06 min | ~$4.40 |
Test: pod cvnd3rc2lle3cs, started 11:57 PM, terminated 12:02 AM, logged in AI-Shared/state/cloud-gpu-log.md.

## Cloud long-reading lesson - prepared 10/16 00:26-00:40 (dry run, no pod)
| Step | Machine | Time | Notes |
|---|---|---|---|
| bundle build (400k lesson examples, 162M tokens; 150M reading tokens; talk + school) | MSI CPU | 8 min 42 s | ~770 examples/s |
| upload to GitHub release cloud-20261016-0034 (1,052 MB) | MSI -> GitHub | 4 min 42 s | 3.3-4.0 MB/s; anonymous download back verified (urllib) |
| planned pod time (boot + download + 55 min lesson + fetch) | H100 SXM | ~70 min | $2.69/h community -> ~$3.15; $3.99/h secure -> ~$4.65; watchdog cap 95 min |

## Cloud run 1 - long-reading lesson (10/16, tools/runpod_run.py)
| Run | Machine | Tokens | Start | End | Wall | Speed | Cost | Result |
|---|---|---|---|---|---|---|---|---|
| Long-reading lesson (Husk -> gen 319) | RunPod H100 SXM secure, $3.99/h | 953M | 01:22 | 02:31 (pod 01:23-02:28) | 69 min | ~338k tok/s | $4.34 | E7L 0/40 -> 17/40; held-out copy-answer test 1/300 -> 498/1000; E10 109 -> 131/2016 |

## Lesson retention fix (10/16 morning, MSI, heartbeat consolidations)
| Run | Machine | Time | Notes |
|---|---|---|---|
| roll-forward 1 (gen 319 -> 348) + 5 consolidations (try 1 mix, batch 32) | MSI | 08:28-08:52 (24 min) | ~105 s per consolidation incl. exams |
| roll-forward 2 (gen 319 -> 354) + 5 consolidations (D33 mix, batch 40) | MSI | 08:55-09:38 (43 min) | ~110 s per consolidation; the 5th waited ~20 min (budget overspent -> rest drive 0.98) |
