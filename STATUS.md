# Sapling current status

Automatically regenerated on **2026-11-21** from the aggregate report card. The daily sync commits only this file and pushes committed work; it does not auto-add source, raw results, secrets, databases, or weights.

- **Core:** v5, **98,584,320 parameters** (98.6M), 512-token context.
- **Generation:** **1673**.
- **Vocabulary:** 17,663 tokens.
- **Latest full exam run:** 2026-11-15T00:10:54.

## Exams

| Exam | Score | Verdict |
|---|---:|---|
| Composition | 85/86 | PASS |
| Affix recall | 22/24 | PASS |
| Recall echo | 12/12 | PASS |
| Deferral | 9/10 | PASS |
| Grammar | 48/48 | PASS |
| Reading | 38/40 | PASS |
| Long reading (informational) | 19/40 | INFO |
| Bridge (informational) | 48/86 | INFO |
| Induction (informational) | 31/2016 | INFO |
| Owner exam | 0 items | INFO |
| Writing | 4 samples | INFO |
| Facts in weights (informational) | 7/7 | INFO |

The six scored exams with declared bars are composition, affix recall, recall echo, deferral, grammar, and reading. Long reading, bridge, induction, and facts-in-weights are informational probes without declared pass bars.

## Heartbeat health (2026-11-21)

The memory-read stall is fixed: three bulk queries replaced per-fact state/recall queries. After the heartbeat restart, **82 ticks ran in 11 minutes** (median gap **6.649 s**, mean **7.992 s**, including the configured 5-s pause and real work). Both control-room APIs reported `running:true`; the memory chain verified. No training or live consolidation was run because consolidation was not due.

On 2026-11-22, malformed ledger JSON was guarded after a `JSONDecodeError` crash: valid rows still read, a bad row is skipped with a logged warning and counted in `ledger_health`. The full live/backup audit found **zero stored malformed rows**, so no history was edited or repaired. The one-off decode trigger was not reproduced. After news published at 05:25:32, the fixed heartbeat started and ran **84 ticks in 11 minutes** (median **6.041 s**, mean **7.906 s**), with both APIs `running:true` and zero skipped rows.

## What is next

1. Improve sustained multi-turn conversation, especially the name-thread check, and reading/bridge retention.
2. Prepare and validate the bounded fresh ~350M-parameter core pilot on rebuilt inputs. Cloud spending remains subject to Jaron's approval.
3. Keep heartbeat tick rate and control-room latency visible after the bulk memory-read fix.

## Evidence and privacy

See [the public results policy](results/PUBLIC-RESULTS.md) for selected samples and exclusions. Raw chat/memory context, stop reports, operational logs, databases, and weights are not bulk-published in the source repository.
