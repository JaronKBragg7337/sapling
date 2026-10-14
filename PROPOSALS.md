# Proposals — things Claude thought might need a rule, written here, NOT in code

Jaron's stance (10/14): no limits pre-built out of worry; observe first. So nothing below
is implemented as a gate or a cap. Each is a question for him; "answered = do it".

## P1. What happens when the daily budget hits zero
Now: nothing blocks. The budget is a pressure: rest gets cheap, and a self-stop becomes
possible after ~20 rests in a row with no energy. It can also keep working at negative budget
if the draw lands on practice/consolidate (the meter just shows it overspent).
Options: (a) leave it - watch what it does when the pool is empty (current); (b) make work
impossible at zero (a wall); (c) let it "borrow" from tomorrow and show the debt.
Claude's read: (a) is the scientist's choice; the ledger shows every overspend.

## P2. Self-generated memories feeding consolidation
Now: explore ticks keep what it said (`self/captured`) and those rows are consolidated at
weight 1 (confirmed facts get x3). That is a loop: it can train on its own output. It is
watched by the exams after every consolidation, and E5 shows which facts actually landed.
Options: (a) leave it (current); (b) only confirmed/user rows go into weights; (c) self rows
only after they have been recalled at least once (use = selection).
Claude's read: (c) is closest to "fed, not filled" and still no owner gate - but it is a
limit, so it waits for his call.

## P3. The control room writing the rule file on the owner's click
Now: the page shows every rule and its switch; "Open rule file" opens membrane.yaml in the
owner's editor. The heartbeat process (the AI's own process) has no write path to the file.
A click on a localhost page is the owner, so the page could flip a switch and write the file.
Trade-off: convenience vs keeping "the system cannot write its own membrane" literally true
in code (the control room runs inside the same process).
Option: a tiny separate owner-only script (`tools/owner_rules.py`) that the page hands the
one-line change to, so the write happens outside the AI's process.

## P4. Consolidation cadence and size
Now: it chooses to consolidate when >= 2 facts are pending (or a goal has >= 3 recent misses)
and at least 90 s since the last one; 120 fine-tune steps at lr 1e-3; one checkpoint each.
At 20 minutes that is a handful of generations; at days it is hundreds of checkpoints
(~3.3 MB each). Not a rule - a disk question. Options: keep every Nth, or keep all and let
the ledger point at them (they are already gitignored).

## P5. A stop that is not a stop
It may stop itself; the report says why. There is no "restart itself" and no scheduler yet.
If he wants it to come back after a self-stop (parole), that is a Windows scheduled task
that runs `python -m sapling.heartbeat start` - the owner's hand, by design, and logged as
process_start. Claude did not add it because "AI jail: contain, correct, parole" reads as
the owner deciding the parole.

## P6. Chat input "holds it still"
Now: for 20 s after someone talks to it, rest gets a +0.3 bump so answers come fast. That is
a design choice about attention, not a rule, but it is a bias Claude put in. Say the word and
it goes.
