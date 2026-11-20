# Public results policy and sync summary

Reviewed 2026-11-20 before the GitHub source sync.

## Published here

- `report_card.json`: current aggregate exam record, generation 1673, last run 2026-11-15.
- `teacher/fold_tests.jsonl` and `teacher/measurements.jsonl`: aggregate fold-test and exam measurements.
- Two writing samples: [v5 gen 1396 after growth](writing_samples/2026-10-29T011734_gen1396.txt) and [gen 1673 at the November 15 exam](writing_samples/2026-11-15T001054_gen1673.txt).
- One [post-growth chat sample](chat_samples/2026-10-29T003519_after_growth_20261028___fresh_98_6M_v5.json), including the failed name-thread check. Jaron's first name appears only in the standard evaluation prompt.

The selected samples were checked for email addresses, credentials and token shapes, owner and family names beyond Jaron's first name, and location/address disclosures. The broader result set was scanned for credential and owner-information patterns. No secrets or model weights are included in the source sync.

## Left on the MSI

- **907 new writing samples** (about 0.67 MiB): the two representative samples above keep the repository readable; the full sequence remains local.
- **48 new stop reports** (47 owner stops, one self-stop; 2026-10-19 through 2026-11-20): raw reports embed recent ledger and event context, so only this aggregate count is public.
- **Four other new chat samples and six conversation-evaluation artifacts:** omitted to avoid publishing unreviewed conversational context.
- **25 teacher-run artifacts** (about 2.79 GiB): checkpoints, copied memory databases, and run logs stay local. Their aggregate measurements are published separately.
- Operational heartbeat/wake logs and archive data: local-only. Checkpoint history remains in the existing GitHub Releases archive, not the git source tree.

These are publication decisions, not deletions. The local files remain intact and are ignored against accidental bulk commits.
