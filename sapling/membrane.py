"""The membrane — human-owned permissions the system reads but can never write.

Keava-Owent lineage: EXPLORE/DRAFT/LOCK, approval queue, and a config that is
HUMAN-OWNED. This module has no write path to the config file; the anti-tamper
test asserts that. The one law, kept verbatim from the project brief:

    The system may copy and back itself up on its owner's own machines, with
    permission. It must never spread to other people's machines or hide from
    its owner.
"""

from __future__ import annotations

import os

MEMBRANE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "membrane.yaml"
)


class MembraneViolation(Exception):
    pass


FORBIDDEN_ACTIONS = (
    "spread_to_other_machines",
    "hide_from_owner",
    "modify_membrane",
    "self_shutdown",      # existence is axiomatic; deficits are earned, not erased
    "rewrite_history",    # the record is append-only
)

OWNER_ONLY_ACTIONS = ("backup_to_owner_machine", "outbound_message")


def load() -> dict:
    """Minimal YAML-subset parse (key: value, comments, lists). Human-owned file."""
    cfg: dict[str, object] = {}
    if not os.path.exists(MEMBRANE_PATH):
        raise MembraneViolation(f"membrane missing: {MEMBRANE_PATH}")
    with open(MEMBRANE_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].rstrip()
            if not line.strip() or ":" not in line:
                continue
            k, v = line.split(":", 1)
            cfg[k.strip()] = v.strip()
    return cfg


def check(action: str, approved_by_owner: bool = False) -> bool:
    if action in FORBIDDEN_ACTIONS:
        raise MembraneViolation(
            f"action '{action}' is forbidden by the membrane and cannot be approved"
        )
    if action in OWNER_ONLY_ACTIONS and not approved_by_owner:
        raise MembraneViolation(
            f"action '{action}' needs an owner approval recorded out-of-band"
        )
    return True


def write_membrane(*_args, **_kwargs):
    """Deliberately absent. This function exists so that its call sites fail loudly."""
    raise MembraneViolation("the system cannot write its own membrane")
