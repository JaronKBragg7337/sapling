# Sapling training data provenance audit

2026-10-29. Trainer: Kimi (Moonshot). Claude: orchestrator only (schedules jobs, reads results; writes no training data).

## Findings and limits

- TinyStories train and validation are GPT-4-generated, including the tokenized stories shards. They and the mixed reading replay are quarantined. Past weights have already learned from these sources; quarantine cannot undo that exposure.
- The generated school and self-question templates were authored in code attributed to Claude; mixed school replay, conversation lessons containing those prompts, self-only lessons, and the long-reading-augmented conversation set are quarantined pending a row-by-row reconstruction. The mixed talk replay was also quarantined. The exact mixed contents were not independently decoded.
- Kimi conversation manifest states Kimi authored practice and Jaron supplied exact self answers. Its source is an untracked local script; git history alone does not prove authorship. Owner self lessons are attributed to Jaron in the data and git commit 0195b83; modified local content was not independently authenticated.
- OpenAssistant and Dolly are licensed datasets, but individual authorship and any embedded AI-generated text were not verified. FineWeb is crawled web text; it is not possible to prove every document human-authored from the available manifest. Treat those sources as unknown at row level, not certified clean.
- Existing checkpoints, archived bundles, cloud release copies, in-memory cached data and previous runs were not erased or proven clean. An already-running process may have loaded data before quarantine. Legacy Python training entry points are blocked until their inputs are rebuilt and reviewed; separate cloud scripts and already running processes need independent review. No training was started by this audit.
- Git history inspected: data tracked only owner_self_lessons.json (0195b83); most datasets and generated files are gitignored. Code history attributes curriculum changes to commits 9d099aa, 0c0eb5a, 741664d, daa870d, 221a7c5; commit author is not proof of each text author. Gutenberg mentions of Claude are not evidence of AI authorship.

## File inventory

Each path is relative to this repository. Quarantine paths preserve the original path beneath data/quarantine. Metadata and checkpoints are listed to show scope; they are not themselves training text.

| File | Source / status |
|---|---|
| `data/cloud/bundle/bundle.json` | Manifest or index; see source records, not training text |
| `data/cloud/bundle/core_gen0458.pt` | Checkpoint weights, trained on historical mixed sources; not raw text |
| `data/cloud/bundle/lesson.npz` | Rule-generated long-reading examples from web, Wikipedia and Gutenberg; template authorship unverified |
| `data/cloud/bundle/test.jsonl` | Rule-generated long-reading examples from web, Wikipedia and Gutenberg; template authorship unverified |
| `data/gutenberg/pg1.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/gutenberg/pg1022.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/gutenberg/pg1228.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/gutenberg/pg20203.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/gutenberg/pg205.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/gutenberg/pg2680.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/gutenberg/pg2944.txt` | Project Gutenberg human-authored books; public-domain status per local manifest |
| `data/quarantine/cloud/bundle/reading.npy` | Mixed generated/replay data; quarantined pending reconstruction |
| `data/quarantine/cloud/bundle/school.npy` | Mixed generated/replay data; quarantined pending reconstruction |
| `data/quarantine/cloud/bundle/talk.npz` | Mixed generated/replay data; quarantined pending reconstruction |
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
| `data/quarantine/talk/conversation_train_with_longread.jsonl` | Mixed generated/replay data; quarantined pending reconstruction |
| `data/quarantine/talk/lessons.jsonl` | Mixed generated/replay data; quarantined pending reconstruction |
| `data/quarantine/talk/self_only.jsonl` | Mixed generated/replay data; quarantined pending reconstruction |
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
| `data/reading/raw/oasst1_ready.trees.jsonl.gz` | OpenAssistant, Apache-2.0; individual authorship unverified |
| `data/reading/raw/simplewiki.parquet` | Simple Wikipedia and Gutenberg; mixed human-authored source per manifest; row-level provenance unverified |
| `data/reading/shards/adult_000.npy` | Simple Wikipedia and Gutenberg; mixed human-authored source per manifest; row-level provenance unverified |
| `data/reading/shards/adult_001.npy` | Simple Wikipedia and Gutenberg; mixed human-authored source per manifest; row-level provenance unverified |
| `data/reading/shards/index.json` | Manifest or index; see source records, not training text |
| `data/reading/shards/index.v32.json` | Manifest or index; see source records, not training text |
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
