from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "STATUS.md"
REPORT_CARD = ROOT / "results" / "report_card.json"
LOG = ROOT / "results" / "github-sync.log"
CORE_PARAMS = 98_584_320
EXAM_NAMES = {
    "E1_composition": "Composition",
    "E2_affix_recall": "Affix recall",
    "E3_recall_echo": "Recall echo",
    "E4_deferral": "Deferral",
    "E6_grammar": "Grammar",
    "E7_reading": "Reading",
    "E7L_longread": "Long reading (informational)",
    "E1b_bridge": "Bridge (informational)",
    "E10_induction": "Induction (informational)",
    "E9_owner_exam": "Owner exam",
    "E8_writing": "Writing",
    "E5_facts_in_weights": "Facts in weights (informational)",
}


def log(message: str) -> None:
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    line = f"{stamp} {message}"
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if check and result.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {(result.stderr or result.stdout).strip()}")
    return result


def render_status(card: dict, date: str) -> str:
    rows = []
    for exam in card.get("exams", []):
        name = EXAM_NAMES.get(exam.get("exam"), str(exam.get("exam", "Unknown")))
        rows.append(f"| {name} | {exam.get('score', '-')} | {exam.get('verdict', '-')} |")
    table = "\n".join(rows) or "| No exam rows | - | - |"
    return f"""# Sapling current status

Automatically regenerated on **{date}** from the aggregate report card. The daily sync commits only this file and pushes committed work; it does not auto-add source, raw results, secrets, databases, or weights.

- **Core:** v5, **{CORE_PARAMS:,} parameters** (98.6M), 512-token context.
- **Generation:** **{card.get('generation', 'unknown')}**.
- **Vocabulary:** {card.get('vocab_size', 'unknown'):,} tokens.
- **Latest full exam run:** {card.get('ts', 'unknown')}.

## Exams

| Exam | Score | Verdict |
|---|---:|---|
{table}

The six scored exams with declared bars are composition, affix recall, recall echo, deferral, grammar, and reading. Long reading, bridge, induction, and facts-in-weights are informational probes without declared pass bars.

## What is next

1. Improve sustained multi-turn conversation, especially the name-thread check, and reading/bridge retention.
2. Prepare and validate the bounded fresh ~350M-parameter core pilot on rebuilt inputs. Cloud spending remains subject to Jaron's approval.
3. Investigate heartbeat delays outside measured GPU work before treating process liveness as proof of healthy ticking.

## Evidence and privacy

See [the public results policy](results/PUBLIC-RESULTS.md) for selected samples and exclusions. Raw chat/memory context, stop reports, operational logs, databases, and weights are not bulk-published in the source repository.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate STATUS.md, commit it alone, and push main.")
    parser.add_argument("--no-push", action="store_true", help="Generate and commit locally without pushing")
    args = parser.parse_args(argv)
    try:
        card = json.loads(REPORT_CARD.read_text(encoding="utf-8"))
        date = datetime.now().date().isoformat()
        STATUS.write_text(render_status(card, date), encoding="utf-8")
        staged = git("diff", "--cached", "--name-only").stdout.splitlines()
        unrelated = [path for path in staged if path != "STATUS.md"]
        if unrelated:
            raise RuntimeError(f"refusing to commit unrelated staged paths: {', '.join(unrelated)}")
        git("add", "--", "STATUS.md")
        changed = git("diff", "--cached", "--quiet", "--", "STATUS.md", check=False).returncode != 0
        if changed:
            git("commit", "-m", f"Status {date}: generation {card.get('generation', 'unknown')}")
            log(f"committed STATUS.md for generation {card.get('generation')}")
        else:
            log("STATUS.md unchanged")
        if not args.no_push:
            git("push", "origin", "main")
            log("push origin main completed")
        return 0
    except Exception as exc:
        log(f"ERROR {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
