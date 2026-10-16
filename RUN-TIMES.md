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
