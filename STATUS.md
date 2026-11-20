# Sapling current status

Automatically regenerated on **2026-11-20** from the aggregate report card. The daily sync commits only this file and pushes committed work; it does not auto-add source, raw results, secrets, databases, or weights.

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

## What is next

1. Improve sustained multi-turn conversation, especially the name-thread check, and reading/bridge retention.
2. Prepare and validate the bounded fresh ~350M-parameter core pilot on rebuilt inputs. Cloud spending remains subject to Jaron's approval.
3. Investigate heartbeat delays outside measured GPU work before treating process liveness as proof of healthy ticking.

## Evidence and privacy

See [the public results policy](results/PUBLIC-RESULTS.md) for selected samples and exclusions. Raw chat/memory context, stop reports, operational logs, databases, and weights are not bulk-published in the source repository.
