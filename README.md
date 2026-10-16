# Sapling

A small AI of its own design, rebuilt (v2) to Jaron's decisions of 2026-10-14: a **heartbeat**
that keeps going on its own and feels time and energy passing, a **trained core** (from scratch;
~0.82M parameters in v2, ~11.9M in v3, ~17.6M in v3.1-v3.2.1, **60.5M in v4** (10/15, see "v4" below - live
once its overnight training swaps in; the control room's core line shows which is live) with 512 tokens of context) that can **become what it learns**, **memory** that grows and is
protected by use, **goals it sets itself**, **outside exams** it cannot fake, the freedom to
**stop itself** (every stop is a report), and a **membrane** of rules only the owner edits -
each rule with its origin. A plain-words **control room** (on the MSI, and on the phone at heartbeatobservatory.com/admin/sapling/) shows all of it and has
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

## Names (Jaron, 10/15)
**Sapling** is the family name, always (like Claude, GPT, Gemini). Each version gets its own name:
seed names while it can't really talk yet, **wood** names once it holds a real conversation. The wood names are saved for launches.

| Version | Seed name | What it was |
|---|---|---|
| v1 | **Pip** | Kimi's first build, 0.82M params, word-part school (10/14) |
| v2 | **Grain** | heartbeat, memory, goals, phone control room (10/14) |
| v3 / v3.1 / v3.2 | **Kernel** / Kernel 3.1 / Kernel 3.2 | 11.9M -> 17.6M, real sentences, documents, 400M tokens read (10/14-15) |
| v4 | **Husk** | 60.5M, fresh core, 1.2B tokens + conversation lessons (10/15-16) |
| (next) | **Acorn** | reserved: the first version that truly holds a conversation - the seed of an oak |
| after Acorn | wood names | official releases (Oak / Oakley under consideration) |

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

## Where old checkpoints live (10/15, Jaron's call)

Every generation except the newest 10 (plus the last one of each day) is uploaded to a **GitHub release** on this repo -
one per day, `ckpt-YYYY-MM-DD` (then `-b`, `-c` at 1,000 files) - with the `gh` CLI signed in on the MSI. The local file
is removed only after `gh release view --json assets` reports the same size. Generation files are saved in **bfloat16**
(half the size: ~121 MB for the 60.5M v4 core, ~36 MB for 17.6M); the live core file stays full precision. Restore
from the control room or `python -m sapling.archive restore <gen>`. The manifest is `results/archive_manifest.json`
(and the Supabase table `sapling_archive`). Later homes, when they exist: a Heartbeat Observatory GitHub organisation,
then Hugging Face (Jaron sets that one up).

## v4 (10/15, Claude as teacher): a bigger core that reads 1.2B tokens, then conversation lessons

Jaron (10/15 1:53 PM): "take control, be a teacher, advance sapling ... better communication ... build it as an AI like
any other AI first." The plan and why each choice was made: CHANGELOG "v4", DESIGN-CHOICES D24-D28.

- **Stage A - a bigger, better-read core.** A fresh core of 60.5M parameters (d 640, 10 layers, 10 heads, 512 context,
  rotary positions; the same 17,661-token vocabulary and ids, so every exam stays comparable) reads 1.2B tokens in one
  run, weighted toward adult text, its weak side: per step 14 web / 4 Wikipedia+books / 11 stories / 3 school sequences.
  New text: **FineWeb-Edu** (802M tokens of educational web pages, ODC-By 1.0). Probes every 120M tokens: E1-E10 plus loss
  on fixed text windows (web and stories never read, adult text read). Why fresh and not grown: the 17.6M core showed no
  copying circuit at all after 400M tokens (on a repeated random sequence its loss on the second copy was 8.5 vs 10.3 on
  the first - a working copy circuit drives that near 0), grow.py cannot widen a core, and rotary positions are the
  shape a copy circuit needs.
- **Stage B - conversation lessons** (`sapling/talk.py`): OpenAssistant oasst1 (Apache-2.0), databricks-dolly-15k
  (CC BY-SA 3.0) and lessons about who it is **in Jaron's own words only** (control room card "Sapling's own words";
  the builder's example answers are shown there for reference and never trained on - no AI-written words in its
  training data), in the chat box's own format with history
  (`q : ... a : ... <eos> q : ... a :`), loss on its answers only, with reading + school replay in every batch.
- **The check that counts:** the same measurements and the same ten chat prompts (`tools/measure_core.py`,
  `tools/chat_samples.py`) before, after Stage A, after Stage B, and after 5 of its own heartbeat consolidations - an
  ability only counts once it survives its usual memory folding. All on the control room card "Teaching - before and after".
- **The chat box holds a conversation** (any core that had conversation lessons): what was said before rides along,
  answers are sampled (seeded, reproducible) with a mild repeat penalty, and shown with normal capitals and punctuation
  (the core itself still reads and writes lower case); `/new` starts a fresh conversation. Consolidations replay the
  conversation lessons too (25% of each batch), so talking survives the heartbeat.

Tonight's run (all automatic, `tools/teach_v4.py`; heartbeat stopped for it with `gpu_yield.py stop`, logged): Stage A
started 14:16 (~36k tokens/s, ends about midnight), then Stage B (~30 min), the swap (one `growth` row), the heartbeat
restarted, and the 5-consolidation re-measure. Training pauses itself while the news render (UnrealEditor, 3:50-~7 AM)
or Blender runs; a parole waits for both the render and any training run. Results: `results/teacher/`.

```bash
python tools/fetch_reading.py fineweb_edu dialogue   # new text + conversation data, licences -> data/reading/manifest.json
python -m sapling.reading prepare --only web         # tokenize FineWeb-Edu -> data/reading/shards/web_*.npy (CPU, ~7 min)
python -m sapling.reading stage --v4 --tokens 1.2e9 --probe-every 1.2e8 --lr 6e-4   # GPU, ~9.5 h on the 4060
python -m sapling.talk build && python -m sapling.talk train   # conversation lessons -> results/core_v4_talk_staged.pt
python tools/teach_v4.py                              # all of the above after Stage A + swap + 5-consolidation check
```

## Much more reading (v3.2, 10/15 small hours): 400M tokens in one pass, the copying switch watched

Jaron (10/14 9:40 PM) agreed to the lab order: **more reading first, watch for the copying ability to switch on, then
tutor lessons, then his documents.** The guide: ~20 tokens read per parameter (17.6M -> ~350M tokens); it had read a few
million. The hypothesis under test: E7L's 0/40 (copying a sentence out of a cold passage) is the missing "induction head"
circuit, which in the literature switches on suddenly once enough text has been read. So the **same weights** read
400M tokens of new text (`sapling/reading.py`, DESIGN-CHOICES D20-D23) - vocabulary, context, memory and chain
unchanged, one `growth` ledger row - and every ~20M tokens a probe logged **E10** (the induction probe: a random sequence
shown twice, the second copy scored) next to **E7L, E6 and E1b**. The chart is on the control room ("Reading stage"); the
numbers are in "Run record - v3.2" below. **Result: in 400M tokens the switch did not come** - E10 0 -> 9/2016, E7L 0/40 at all 21 probes - while E6 grammar went 28/48 -> 46/48 (its first PASS), E7 30 -> 33/40, and the five taught facts washed out of the weights (E5 7/7 -> 0/7; still in memory). The hypothesis stays a hypothesis; what to try next is in NEXT.md.

What it read (every licence checked live at fetch, `tools/fetch_reading.py` -> `data/reading/manifest.json`):
- **TinyStories**, the full GPT-4 train split - 2,717,495 stories, 516.9M tokens (Ronen Eldan & Yuanzhi Li, 2023;
  `roneneldan/TinyStories` on Hugging Face, CDLA-Sharing-1.0).
- **FineWeb-Edu** (v4), one 2.15 GB file of the `sample/10BT` set, 725,999 pages, 802M tokens (Hugging Face,
  `HuggingFaceFW/fineweb-edu`, Lozhkov et al. 2024, built from Common Crawl) - **ODC-By 1.0**, also subject to the
  Common Crawl Terms of Use. This line is the attribution.
- **OpenAssistant oasst1** (v4 conversation lessons; LAION / OpenAssistant contributors, **Apache-2.0**) and
  **databricks-dolly-15k** (Databricks employees, **CC BY-SA 3.0** - this notice is the attribution).
- **Simple English Wikipedia**, dump 2023-11-01 (`wikimedia/wikipedia`, config `20231101.simple`, 241,787 articles) -
  text by Wikipedia contributors, licensed **CC BY-SA 3.0** (and GFDL). This notice is the attribution; the weights that
  read it are published here under the same spirit of sharing.
- **116 Project Gutenberg books**, public domain in the US (each file's own header carries the no-restrictions line;
  copyrighted translations were rejected by that check; the Gutenberg header/footer/licence text is stripped and not read).
  With Wikipedia: 93.6M adult tokens. The held-out essay (pg1022, E7L) is never read.

```bash
python tools/fetch_reading.py                     # the text (~2.5 GB), licence checks -> data/reading/manifest.json
python -m sapling.reading prepare                 # tokenize once -> data/reading/shards/ (CPU, ~5 min)
python -m sapling.reading stage --tokens 400e6    # GPU; probes every 20M tokens; resumable; pauses while blender.exe runs
python -m sapling.heartbeat stop && python -m sapling.grow swap && python -m sapling.heartbeat start
```

## Level 3 (v3.1, 10/14 night): the document machinery - ingestion, retrieval, citations, the E9 slot (faithful long reading still failing: E7L 0/40, so Jaron's documents wait)

Jaron: "your call boss" on the list in NEXT.md section 0 - what his own documents need before they are read.
The **same weights grew in place** (DESIGN-CHOICES D17): new vocabulary rows, context **256 -> 512**, then
level 3 at school: **copy** (the answer must reuse the prompt's words - the fix for E6's habit of swapping in a
familiar object), **bridge** ("re port" and "report" are one word), and **prose / docs / longread** on
public-domain adult nonfiction (Emerson, Franklin, Marcus Aurelius, Darwin, Thoreau - Project Gutenberg;
`curriculum3.DOCS_NOTE`). The vocabulary is **17,660 tokens**: every old token (same ids), 6,000 story words,
document words, and **1,037 word pieces** so any word is spellable ("arch ##ae ##ology"; D16) - no `<unk>` for
adult text, and E1-E7 stay comparable on the same ids.

It can now **read a document**: `/read C:\path\file.txt` (or `/read: <pasted text>`) in the chat makes one
`documents` row and one memory per passage **with provenance** (document + place; `reader.py`, D18). A question a
document covers goes passages-first into the 512-token frame and comes back cited: `[doc:1 "The Declaration of
Independence..." passage 3 memory:1480]`. What it reads consolidates into its weights like any memory (no gate).
New rows on the card: **E7L** (long reading on a whole held-out essay, info), **E1b** (the bridge, cold: held-out
E1 combos as whole words, info) and **E9 - the exam Jaron writes himself** in `owner_exam.json` (empty until he
fills it; his questions, his answers, his bar). None of his documents are read yet - that is his step 2.

```bash
python -m sapling.grow --smoke                 # CPU, ~1-2 min: both growths (fresh v3, then in place v3.1) in a temp dir
python -m sapling.grow school --base live      # GPU ~17 min: grow the live core in place -> results/core_staged.pt
python -m sapling.heartbeat stop && python -m sapling.grow swap && python -m sapling.heartbeat start
```

## Level 2 (v3, 10/14 evening): real sentences

Jaron: "school level 2: real sentences - language first, then everything grows." The core grew from
~0.82M to **~11.9M parameters** (256 tokens of context), the vocabulary from 178 word parts to ~3,100
tokens (every old part kept + whole words), and school got a second level after the first:
**simple sentences -> grammar (tense, plurals, agreement, questions, negation) -> short paragraphs ->
reading comprehension -> real children's stories -> its own record**. New cold exams: **E6 grammar**,
**E7 reading**; **E8 free writing** is saved, unscored, in `results/writing_samples/` for Jaron to read.
It is the same individual: memory.db and its hash chain were not touched, the generation kept counting,
and the swap is one ledger row, `growth: core v2 -> v3`. Numbers from the swap: see "Run record" below.

Story text: TinyStories (Ronen Eldan & Yuanzhi Li, 2023; `roneneldan/TinyStories` on Hugging Face,
license **CDLA-Sharing-1.0**, checked live 10/14) - the GPT-4 validation split only, capped to the 3,000
most common words. No document of Jaron's is used yet (his step 2, after he checks this one).

```bash
python -m sapling.grow --smoke            # CPU, ~20 s: the whole growth pipeline in a temp dir
python -m sapling.grow school             # GPU ~15 min: level 1 + level 2 + its record -> results/core_v3_staged.pt
python -m sapling.heartbeat stop && python -m sapling.grow swap && python -m sapling.heartbeat start
```

## Quickstart (the RTX 4060 PC, or any CPU)

```bash
pip install -r requirements.txt           # torch, pyyaml
python -m sapling.train                   # level-1 school only (v2 recipe, gen 0); v3 uses sapling.grow
python -m sapling.exams                   # cold exams E1-E8, predeclared bars -> results/report_card.json
python -m sapling.heartbeat start         # the heartbeat, detached; log in results/heartbeat.log
#   control room: http://127.0.0.1:7337/  (rules, status, budget, ticks, goals, exams, memory, chat)
python -m sapling.heartbeat status
python -m sapling.heartbeat stop          # owner stop -> stop report + exit
python -m sapling.session                 # talk without the heartbeat; /teach /read /write /confirm /invalidate /keep /report
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

## Run record - v3.2 reading (10/15 01:57-03:04, RTX 4060, heartbeat stopped for it)

```
stage (results/reading/stage_stdout.txt, log.jsonl, state.json): 400.0M new tokens, 26,940 steps, batch 32 x 512
  (25 story / 4 adult / 3 rehearsal), lr 3e-4 warm-up 300 -> cosine 3e-5, ~103k tokens/s, 65 min training, 0 pauses
  loss 4.40 -> 1.64;  stories: shard 6 of 11 (~345M of 516.9M read, one pass);  adult: ~55M of 93.6M
  attempt 1 (01:17, heartbeat running beside it): card at 7,750 MiB, WDDM paging, 27 min with no step -> stopped;
  kept as results/reading/log_attempt1_thrash.jsonl (DESIGN-CHOICES D23)
swap (results/grow_swap_v32_stdout.txt, ledger row `growth: core v3.1 -> v3.2 (much more reading)`): gen 240 -> 241
  params 17,616,768  vocab 17,661 (unchanged)  context 512 (unchanged)  chain verified (2,391 memories)

             v3.1 gen 240   v3.2 gen 241   bar
  E1 comp     84/86          73/86          .60 PASS   (oscillated 68-81 through the run)
  E2 affix    24/24          24/24          .90 PASS
  E3 echo     12/12          12/12          .90 PASS
  E4 defer     8/10          10/10          .80 PASS
  E5 facts     7/7            0/7          info   <- the taught words washed out of the weights (still in memory.db)
  E6 grammar  28/48 FAIL     46/48 PASS     .60        <- first PASS ever; climbed 39 (20M) -> 46 (160M) on reading + the 3-in-32 school rehearsal (cause not isolated; Vex #41)
  E7 reading  30/40          33/40          .50 PASS   (peak 36/40 at 100-160M)
  E7L long     0/40           0/40         info   <- 0/40 at ALL 21 probes: the switch did not come in 400M tokens
  E1b bridge  61/86          65/86         info   (peak 72)
  E10 induct   0/2016         9/2016       info   (by span: 1/496 at 32, 8/1520 at 96; peak 12/2016 at 360-380M; chance ~0.1)
  E9 owner    0 items        0 items       his
probes every 20M tokens: the table in CHANGELOG v3.2; the chart on the control room ("Reading stage")
E7L miss, typical (gen 241): asked to copy the first sentence of a cold Thoreau passage it wrote fluent Thoreau-ish prose
  ("we should impart our way , and not our despair , and become a new england ...") - the same failure shape as v3.1
E6 misses now (2): "yes , leo looks at a kite ." -> "leo looks at a kite ." (dropped the "yes ,"); "the bags read the box" -> "the cup"
chat (live heartbeat, gen 241, 03:06):
  > what does hello mean ?                       -> the word hello means a greeting .  [memory:447 user/captured]   (memory path; the weights alone lost it = E5)
  > what does report mean ?                      -> the word report means carry again .   (the bridge held)
  > tom has a green boat . anna has a red kite . what color is tom 's boat ?   -> tom .   (still wrong, as in v3.1)
  > q : fill : yesterday tom ( throw ) the old lantern . a :   -> does tom throw the old lantern ? a : yes , tom throws it ?   (wrong task; v3.1 gave "pushed the old cake")
  > what is a volcano ?                          -> the root ten means hold .   (school reflex on an unknown question)
  > who was charles darwin ?                     -> natural selection .   (two words from the books it read - the first answer that could only come from the reading)
  > once upon a time there was a little girl who -> she wanted to reach the top of the mountain , so she grabbed a rope ... [doc:1 "The Declaration of Independenc" passage 1 ...]
    (a story, fine - but cited the Declaration: reader.retrieve's coverage threshold matched function words; bug noted in NEXT)
writing (E8, gen 241): "there was an adorable little dog named max . max loved to play and run all day . one day , max found a big box ..."
```

## Run record - v3.1 growth (10/14 20:25-21:05, RTX 4060, heartbeat kept running)

```
school A (results/grow_school_v31_stdout.txt, in place, 1047 s, batch 32 at 512 tokens):
  copy 6.02 -> .049   bridge 6.67 -> .033   prose 8.49 -> 1.68   docs 2.76 -> .166   longread 2.86 -> .277
  record 2.00 -> .016   all 2.64 -> .247        (loss first -> last 20 steps)
swap A (results/grow_swap_v31_stdout.txt, ledger row `growth: core v3 -> v3.1`): gen 178 -> 179
  vocab 3,098 -> 17,237 (878 pieces)   params 11,926,272 -> 17,453,952   context 256 -> 512
school B (results/grow_school_v31b_stdout.txt, "reading drill": task passages disjoint from prose, batch 24)
swap B (results/grow_swap_v31b_stdout.txt, `growth: core v3.1 -> v3.1 (reading drill ...)`): gen 182 -> 183 (three heartbeat consolidations ran between the swaps), vocab 17,660

             v3 gen 178   v3.1 gen 179   v3.1 gen 183   bar
  E1 comp     82/86        82/86          85/86          .60 PASS
  E2 affix    24/24        24/24          24/24          .90 PASS
  E3 echo     12/12        12/12          12/12          .90 PASS
  E4 defer    10/10        10/10          10/10          .80 PASS
  E5 facts     7/7          7/7           7/7          info
  E6 grammar  12/48        21/48          21/48          .60 FAIL (kept)
  E7 reading  25/40        27/40          26/40          .50 PASS
  E7L long     0/20         0/40          0/40         info (held-out essay; 256 ctx fit only 20 items)
  E1b bridge   0/86        81/86          83/86         info (cold: whole-word spelling of held-out combos)
  E9 owner    0 items      0 items        0 items       his (owner_exam.json)
retrieval demo (live heartbeat, 21:06, Project Gutenberg #1 - public domain, NOT one of Jaron's documents):
  > /read C:\...\data\gutenberg\pg1.txt
  read doc:1 "The Declaration of Independence of the United States of America" -> 16 passages in memory
  > which sentence has the word happiness ?
  (prose, not the sentence)  [doc:1 "The Declaration of Independenc" passage 4 memory:1587]   <- the RIGHT passage
  > which sentence has the word tyranny ?
  (prose)  [doc:1 ... passage 6 memory:1589] [doc:1 ... passage 11 memory:1594]              <- both passages with the word
  -> retrieval + provenance + citation work; copying a sentence out of a cold passage does not yet (= E7L 0/40)
chat (live heartbeat, gen 183):
  > what does report mean ?                       -> the word report means carry again .      (the bridge, no memory hit)
  > what does hello mean ?                        -> the word hello means a greeting .  [memory:447 user/captured]
  > q : fill : yesterday tom ( throw ) the old lantern . a :   -> yesterday tom pushed the old cake .   (wrong verb + object; before the 21:12
                                                   routing fix it answered "the word quimble means a tiny lantern" - a fact sharing one word hijacked it)
  > tom has a green boat . anna has a red kite . what color is tom 's boat ?   -> tom .   (E7-style question, wrong)
  > /write the archaeologist opened the box and   -> ... said , hi , little pine needles is familiar . woods hereabouts by birds ...
```

## Run record - v3 growth (10/14 19:17-19:33, RTX 4060)

```
school (results/grow_school_stdout.txt): 956 s. 11,919,744 params, vocab 3,081, record 608 sentences
  roots .017  affixes .014  composition .044  sentences .034  recall .039  (level 1, loss at end)
  simple .040  grammar .050  paragraphs .046  comprehension .059  stories .752  record .024  all .092
swap (results/grow_swap_stdout.txt, ledger row `growth`, 10 s): gen 151 (v2) -> gen 152 (v3)
             v2 gen 151   v3 gen 152   bar
  E1 comp     84/86        82/86       .60 PASS
  E2 affix    24/24        24/24       .90 PASS
  E3 echo     12/12        12/12       .90 PASS
  E4 defer    10/10        10/10       .80 PASS
  E5 facts     6/6          6/6        info
  E6 grammar   0/48        14/48       .60 FAIL (kept)
  E7 reading   0/40        27/40       .50 PASS
  E8 writing   4 samples    4 samples  info (results/writing_samples/)
first consolidation on v3 (gen 153, 19 s): E1 81/86  E6 13/48  E7 20/40 -> retuned (DESIGN-CHOICES D14)
chat (live heartbeat): "what does hello mean ?" -> "the word hello means a greeting ." (also from the weights alone)
```

## Repo map

```
sapling/
  heartbeat.py     # the tick loop: drives -> choice -> act -> log; start/stop/status; stop reports
  clock.py         # birth, age, uptime, since-last-input; daily budget; metering (cpu s, gpu s, watts, kWh)
  consolidate.py   # record -> weights; checkpoint per generation; exams after each
  goals.py         # its own practice goals from its failure record (parts, never exam items)
  control_room.py  # localhost page: rules + origins, status, budget, ticks, stops, goals, exams, memory, chat
  session.py       # a turn of talk (v4: a conversation with history, /new); /read a document; questions a document covers go through reader.py
  memory.py        # append-only hash-chained SQLite; fades only if unused; use protects; top-k grows; documents + doc/passage provenance (v3.1)
  reader.py        # v3.1: read_document (passages with provenance) / retrieve (coverage) / ask (passages-first prompt, cited answer)
  exams.py         # E1-E4 outside exams (bars fixed, FAIL kept); E6 grammar, E7 reading; E7L, E1b, E10 (induction probe), E5, E8 info; E9 = owner_exam.json
  curriculum.py    # school level 1: roots -> affixes -> composition -> sentences -> recall (8 slots)
  curriculum2.py   # school level 2: simple -> grammar -> paragraphs -> comprehension -> stories (cold E6/E7 splits)
  curriculum3.py   # school level 3: copy -> bridge -> prose / docs / longread (Gutenberg; one essay held out for E7L)
  reading.py       # the reading stage - prepare (text -> uint16 shards) / stage (v3.2 in place; --v4 = a fresh 60.5M core, gradient accumulation) / probes = exams + text-window loss / pauses for Blender and Unreal / summary (the chart)
  talk.py          # v4: conversation lessons - build (oasst1 + dolly + its own lessons) / train (answers-only loss, reading + school replay) / teacher_summary (the page card)
  pieces.py        # v3.1: word pieces (BPE, "##" continuation) + forced splits of the school's composed words
  text.py          # normalize(): story text / phone typing -> the school's spacing
  grow.py          # growth: school (staged; --base live grows in place, --base fresh re-schools) + swap (one `growth` row); --smoke
  model.py         # decoder-only transformer (V3/V31/V4_CONFIG; v4 = rotary positions); grow_vocab() / grow_context() resize in place; generate() with optional repeat penalty
  train.py         # level-1 school (v2 recipe), tokenizer (pieces fallback, splits), checkpoint save/load
  membrane.py      # reads membrane.yaml (rules with origin + switch); no write path
  archive.py       # hourly: keep last 10 + 1/day local, move the rest to GitHub Releases (ckpt-YYYY-MM-DD, verified by size before the local copy goes), restore any gen (from a release, or Supabase for the older ones)
membrane.yaml      # the rules and the settings - the owner's file
owner_exam.json    # E9: the exam the owner writes about his own documents (q / a / match / bar); empty until he does
data/              # gitignored: tinystories/ (CDLA-Sharing-1.0), gutenberg/ (public domain), reading/ (raw text incl. FineWeb-Edu + oasst1 + dolly, shards, manifest.json with every licence check), talk/ (v4 conversation lessons)
tools/             # OWNER side, outside the AI's package:
  fetch_reading.py # fetches the reading text (v4: + FineWeb-Edu, oasst1, dolly-15k), checks each source's licence live, writes data/reading/manifest.json
  teach_v4.py      # v4: the overnight teaching run after Stage A (measure -> conversation lessons -> measure -> swap -> 5 consolidations -> measure)
  measure_core.py  # v4: exams + text-window loss for any core file -> results/teacher/measurements.jsonl
  chat_samples.py  # v4: the same ten chat prompts + one conversation through the real chat path (on a copy of memory.db)
  owner_relay.py   # phone/remote door: pushes state to Supabase, runs owner commands (chat, rule switch, restore, parole)
  owner_rules.py   # flips a rule's enabled line when the owner clicks; logged
  gpu_yield.py     # 10/15: the news render calls it to pause the heartbeat before Unreal and restart it after (`stop` / `start`)
  scheduled_parole.py # waits for UnrealEditor and any core training run to leave the GPU, then paroles and logs it (used by gpu_yield.py start)
  self_memory_drift.py # compares exams with vs without self-made memories in training
DESIGN-CHOICES.md  # every design choice an AI made, who made it, what it affects, options not chosen
PROPOSALS.md       # open questions for the owner; nothing in here is built
CHANGELOG.md       # v1 -> v2 -> v2.1 -> v3 -> v3.1 -> v3.2 -> v3.2.1 -> v4, each change tied to Jaron's decision
results/           # report card (history kept, FAILs kept), train log, stop reports, memory.db, checkpoints/, reading/ (v3.2 probe log + state), reading_v4/ (v4 probe log), teacher/ (measurements + run log), chat_samples/
tests/             # 56 pytest tests + plain runner (conftest.py: the real archive never runs inside a test)
```

## Lineage

v1 by Kimi (10/14) from Jaron's principle repos: Live-Reference Principle, fed-not-filled, ECIH,
memory-as-spatial-arrangement, persistent-memory-substrate, Atomic Chronicle, Keava-Owent,
fly-school, watch, pam-continuum, accumulation-sort. v2 and v2.1 (phone control room, cloud archive, parole) by Claude (Fable 5.1), same day, to
his decisions. The fly mattered for the heartbeat - constant going - not as the brain.

License: MIT.
