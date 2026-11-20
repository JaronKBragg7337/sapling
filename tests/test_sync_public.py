import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from sync_public import render_status


def test_status_contains_only_aggregate_exam_fields():
    card = {
        "generation": 1673,
        "vocab_size": 17663,
        "ts": "2026-11-15T00:10:54",
        "trigger": "private context is not rendered",
        "exams": [{"exam": "E1_composition", "score": "85/86", "verdict": "PASS", "rows": ["private question"]}],
    }
    status = render_status(card, "2026-11-20")
    assert "98,584,320" in status
    assert "**Generation:** **1673**" in status
    assert "17,663 tokens" in status and "| Composition | 85/86 | PASS |" in status
    assert "private context" not in status and "private question" not in status
