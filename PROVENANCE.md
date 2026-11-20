# Sapling training data provenance audit

2026-10-29 audit; 2026-11-01 swap by Kimi (Moonshot). Trainer and replacement-text author: Kimi. Claude: orchestrator only (schedules jobs, reads results; writes no training data).

## Findings and limits

- On 2026-11-01 Jaron approved the swap. TinyStories train/validation, all 11 tokenized story shards, the old mixed reading replay, old school/talk replay, old mixed conversations, old self-only lessons, and OpenAssistant raw input were MOVED to `data/quarantine/`; nothing was deleted. They are not active training inputs.
- Active school lessons come from Kimi's `sapling/kimi_school.py`, with fixed exam generators retained for evaluation only. The full story replacement completed on 2026-11-02: **2,717,495 train stories / 516,895,754 tokenized tokens** and **27,630 validation stories / 6,221,698 tokens**. Exhaustive QA passed all counts, lengths, hashes, exact uniqueness, split separation and token-budget checks. The sampled near-duplicate limitations remain in `stories_full/qa_report.json`.
- `data/talk/lessons.jsonl` was rebuilt from 8,000 Kimi practice rows, 75 Kimi lead-in/Jaron-answer rows, 150 new self-only rows, and 14,962 fresh Dolly rows. `conversation_train_with_longread.jsonl` contains 8,000 Kimi practice rows, the same 225 self rows, 4,000 new Kimi/Gutenberg exact-span tasks, and 2,578 fresh Dolly rows. No rows were salvaged from the old mixed files and no OpenAssistant rows are included.
- Active cloud `school.npy`, `talk.npz`, `lesson.npz`, and `reading.npy` use rebuilt/approved inputs. Reading replay has **292,396 uint16 windows of 513 tokens** (149,999,148 stored tokens) in the 11:4:14 story/adult/web proportions. The bundle is COMPLETE for the bounded fresh-350M pilot. `test.jsonl` is historical evaluation-only; `core_gen1590.pt` supplies the tokenizer and has historical exposure.
- Jaron accepted the balanced-practice candidate on 2026-11-02, including the measured small drops. It installed as generation 1626; the previous live checkpoint moved to `results/checkpoints/core_v5_gen_1625.pt`. Level-1 x2, bridge x2, and fresh Kimi definition forms are now the default live recipe. `SAPLING_KIMI_REPAIR=0` is the diagnostic opt-out.
- Kimi conversation manifest states Kimi authored practice and Jaron supplied exact self answers. Owner self lessons are attributed to Jaron in the data and git commit 0195b83; modified local content was not independently authenticated.
- Dolly is a licensed dataset, but individual authorship and any embedded AI-generated text were not verified. FineWeb is crawled web text; it is not possible to prove every document human-authored from the available manifest. Treat those approved sources as unknown at row level, not certified human-only.
- Every existing checkpoint, archived bundle, cloud release copy and previous run has historical exposure to the flagged sources. The swap changes future inputs only; it does not make existing weights clean. A fresh-core run is required for a core with no historical exposure.
- Git history inspected in the original audit: data tracked only owner_self_lessons.json (0195b83); most datasets and generated files are gitignored. Commit authorship is not proof of each text author. Gutenberg mentions of Claude are not evidence of AI authorship.

## File inventory

Each path is relative to this repository. Quarantine paths preserve the original path beneath data/quarantine. Metadata and checkpoints are listed to show scope; they are not themselves training text.

| File | Source / status |
|---|---|
| `data/cloud/bundle/bundle.json` | Active COMPLETE manifest for the bounded fresh-350M pilot |
| `data/cloud/bundle/core_gen1590.pt` | Baseline checkpoint; supplies tokenizer only to fresh pilot and has historical exposure |
| `data/quarantine/2026-11-01-before-full-reading/cloud/bundle/core_gen0458.pt` | Superseded historical core; moved backup |
| `data/cloud/bundle/lesson.npz` | Active 4,000 Kimi/Gutenberg exact-span long-reading examples; answer-only masks |
| `data/quarantine/cloud/bundle/lesson.npz` | Old 400,002-example mixed-template replay; quarantined and replaced 2026-11-01 |
| `data/cloud/bundle/test.jsonl` | Historical rule-generated evaluation examples; template authorship unverified |
| `data/cloud/bundle/school.npy` | Active Kimi school replay without memory-record text |
| `data/cloud/bundle/talk.npz` | Active answer-masked replay, rebuilt from clean 23,187-row lessons mix |
| `data/cloud/bundle/reading.npy` | Active 292,396-window replay from full Kimi stories plus adult/web shards |
| `data/kimi_rebuild/school_level1.json` | Staged Kimi school source; generator includes normal and test-only repair modes |
| `data/kimi_rebuild/school_level2.json` | Staged Kimi school source with original story batch 001 |
| `data/kimi_rebuild/school_level3.json` | Staged Kimi school source with Gutenberg passages; pg1022 held out |
| `data/kimi_rebuild/stories_full/` | Complete validated Kimi corpus; see build_manifest.json and qa_report.json |
| `data/kimi_rebuild/stories/train_batch_001.txt` | Immutable original 110-story Kimi batch, included in full build |
| `data/kimi_rebuild/stories/valid_batch_001.txt` | Immutable 20-story Kimi validation batch, included in full build |
| `data/talk/lessons.jsonl` | Active rebuilt 23,187-row mix; exact sources and hashes in `data/kimi_rebuild/mix_manifest.json` |
| `data/talk/conversation_train_with_longread.jsonl` | Active rebuilt 14,803-row mix with 4,000 Kimi/Gutenberg exact-span tasks |
| `data/talk/self_only.jsonl` | Active 150 fresh Kimi-question/Jaron-answer rows |
| `sapling/self_questions.json` | Package copy of Kimi's prompt bank so cloud clones do not depend on gitignored data |
| `sapling/goals.py` | Weak-part practice now uses the same Kimi school question wording |
| `data/gutenberg/pg1.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/gutenberg/pg1022.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/gutenberg/pg1228.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/gutenberg/pg20203.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/gutenberg/pg205.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/gutenberg/pg2680.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/gutenberg/pg2944.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/quarantine/cloud/bundle/reading.npy` | Old mixed replay; quarantined and replaced 2026-11-02 |
| `data/quarantine/cloud/bundle/school.npy` | Old mixed replay; quarantined and replaced 2026-11-01 |
| `data/quarantine/cloud/bundle/talk.npz` | Old mixed replay; quarantined and replaced 2026-11-01 |
| `data/quarantine/reading/raw/TinyStoriesV2-GPT4-train.txt` | GPT-4 TinyStories; quarantined |
| `data/quarantine/reading/shards/stories_000.npy` | GPT-4 TinyStories; quarantined |
| `data/quarantine/reading/shards/stories_001.npy` | GPT-4 TinyStories; quarantined |
| `data/quarantine/reading/shards/stories_002.npy` | GPT-4 TinyStories; quarantined |
| `data/quarantine/reading/shards/stories_003.npy` | GPT-4 TinyStories; quarantined |
| `data/quarantine/reading/shards/stories_004.npy` | GPT-4 TinyStories; quarantined |
| `data/quarantine/reading/shards/stories_005.npy` | GPT-4 TinyStories; quarantined |
| `data/quarantine/reading/shards/stories_006.npy` | GPT-4 TinyStories; quarantined |
| `data/quarantine/reading/shards/stories_007.npy` | GPT-4 TinyStories; quarantined |
| `data/quarantine/reading/shards/stories_008.npy` | GPT-4 TinyStories; quarantined |
| `data/quarantine/reading/shards/stories_009.npy` | GPT-4 TinyStories; quarantined |
| `data/quarantine/reading/shards/stories_010.npy` | GPT-4 TinyStories; quarantined |
| `data/quarantine/talk/conversation_train_with_longread.jsonl` | Old mixed data; quarantined and replaced 2026-11-01 |
| `data/quarantine/talk/lessons.jsonl` | Old mixed data; quarantined and replaced 2026-11-01 |
| `data/quarantine/talk/self_only.jsonl` | Old self-prompt data; quarantined and replaced 2026-11-01 |
| `data/quarantine/reading/raw/oasst1_ready.trees.jsonl.gz` | OpenAssistant source; quarantined and excluded from all rebuilt mixes |
| `data/quarantine/tinystories/TinyStoriesV2-GPT4-valid.txt` | GPT-4 TinyStories; quarantined |
| `data/reading/manifest.json` | Manifest or index; see source records, not training text |
| `data/reading/raw/databricks-dolly-15k.jsonl` | Databricks Dolly, CC BY-SA; individual authorship unverified |
| `data/reading/raw/fineweb_edu_000_00000.parquet` | FineWeb-Edu / Common Crawl; mixed web authorship unverified |
| `data/reading/raw/gutenberg/pg10.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg1023.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg103.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg105.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg1065.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg1080.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg11.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg113.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg1155.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg1184.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg12.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg120.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg121.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg1232.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg1250.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg1251.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg1257.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg1260.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg1322.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg1342.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg135.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg1399.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg140.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg1400.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg1404.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg141.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg147.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg1497.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg1513.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg1524.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg1526.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg15399.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg158.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg16.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg161.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg164.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg1661.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg17135.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg1727.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg174.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg18857.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg1946.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg1952.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg19942.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg1998.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg203.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg20417.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg2097.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg2147.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg2148.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg215.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg2160.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg2199.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg2246.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg2267.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg23.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg2300.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg236.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg23700.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg24.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg2413.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg244.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg2446.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg2500.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg25344.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg2554.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg2591.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg2641.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg2680.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg2701.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg2708.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg28054.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg2814.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg2852.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg29870.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg3090.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg3177.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg3207.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg3268.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg3296.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg33.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg3300.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg345.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg35.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg36.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg3600.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg408.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg4085.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg41.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg4217.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg4280.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg43.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg4300.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg4363.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg46.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg4705.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg521.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg55.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg5827.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg600.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg6400.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg6593.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg7142.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg730.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg7370.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg74.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg76.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg766.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg768.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg829.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg834.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg84.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg8800.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg910.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg98.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/gutenberg/pg996.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/reading/raw/simplewiki.parquet` | Simple Wikipedia and Gutenberg; mixed human-authored source per manifest; row-level provenance unverified |
| `data/reading/shards/adult_000.npy` | Simple Wikipedia and Gutenberg; mixed human-authored source per manifest; row-level provenance unverified |
| `data/reading/shards/adult_001.npy` | Simple Wikipedia and Gutenberg; mixed human-authored source per manifest; row-level provenance unverified |
| `data/reading/shards/index.json` | Manifest or index; see source records, not training text |
| `data/quarantine/reading/shards/index.v32.json` | Historical story/adult index; quarantined with the old story shards |
| `data/reading/shards/web_000.npy` | FineWeb-Edu / Common Crawl; mixed web authorship unverified |
| `data/reading/shards/web_001.npy` | FineWeb-Edu / Common Crawl; mixed web authorship unverified |
| `data/reading/shards/web_002.npy` | FineWeb-Edu / Common Crawl; mixed web authorship unverified |
| `data/reading/shards/web_003.npy` | FineWeb-Edu / Common Crawl; mixed web authorship unverified |
| `data/reading/shards/web_004.npy` | FineWeb-Edu / Common Crawl; mixed web authorship unverified |
| `data/reading/shards/web_005.npy` | FineWeb-Edu / Common Crawl; mixed web authorship unverified |
| `data/reading/shards/web_006.npy` | FineWeb-Edu / Common Crawl; mixed web authorship unverified |
| `data/reading/shards/web_007.npy` | FineWeb-Edu / Common Crawl; mixed web authorship unverified |
| `data/reading/shards/web_008.npy` | FineWeb-Edu / Common Crawl; mixed web authorship unverified |
| `data/reading/shards/web_009.npy` | FineWeb-Edu / Common Crawl; mixed web authorship unverified |
| `data/reading/shards/web_010.npy` | FineWeb-Edu / Common Crawl; mixed web authorship unverified |
| `data/reading/shards/web_011.npy` | FineWeb-Edu / Common Crawl; mixed web authorship unverified |
| `data/reading/shards/web_012.npy` | FineWeb-Edu / Common Crawl; mixed web authorship unverified |
| `data/reading/shards/web_013.npy` | FineWeb-Edu / Common Crawl; mixed web authorship unverified |
| `data/reading/shards/web_014.npy` | FineWeb-Edu / Common Crawl; mixed web authorship unverified |
| `data/reading/shards/web_015.npy` | FineWeb-Edu / Common Crawl; mixed web authorship unverified |
| `data/reading/shards/web_016.npy` | FineWeb-Edu / Common Crawl; mixed web authorship unverified |
| `data/talk/conversation_heldout.jsonl` | Kimi-authored held-out evaluation, per local manifest |
| `data/talk/conversation_manifest.json` | Manifest or index; see source records, not training text |
| `data/talk/conversation_train.jsonl` | Kimi-authored dialogue and Jaron words, per local manifest; not independently verified |
| `data/talk/owner_self_lessons.json` | Jaron words / drafts, per repository records; local edits unverified |
| `data/talk/owner_self_lessons_drafts.json` | Jaron words / drafts, per repository records; local edits unverified |
