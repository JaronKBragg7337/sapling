"""The membrane — human-owned rules the system reads but can never write.

Keava-Owent lineage: a config that is HUMAN-OWNED. This module has no write
path to the file; the anti-tamper test asserts that. v2: rules are read from
membrane.yaml (with origin + enabled switch per rule) instead of being
hard-coded here, so the owner's file edit is the only thing that changes them.

Jaron 10/14: self_stop is no longer forbidden. It is allowed and logged.
"""

from __future__ import annotations

import os

import yaml

MEMBRANE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "membrane.yaml"
)

DEFAULT_SETTINGS = {
    "tick_seconds": 5,
    "daily_budget_seconds": 3600,
    "control_room_port": 7337,
    "consolidate_min_pending": 2,
    "consolidate_steps": 200,
    "practice_batch": 6,
    "loop_mode": "heartbeat",
}


class MembraneViolation(Exception):
    pass


def load(path: str = MEMBRANE_PATH) -> dict:
    if not os.path.exists(path):
        raise MembraneViolation(f"membrane missing: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("rules", [])
    cfg["settings"] = {**DEFAULT_SETTINGS, **(cfg.get("settings") or {})}
    return cfg


def rules(path: str = MEMBRANE_PATH) -> list[dict]:
    return list(load(path)["rules"])


def settings(path: str = MEMBRANE_PATH) -> dict:
    return load(path)["settings"]


def action_classes(path: str = MEMBRANE_PATH) -> dict[str, str]:
    """action -> 'forbidden' | 'owner_only' | 'allowed_logged' (enabled rules only)."""
    out: dict[str, str] = {}
    for r in rules(path):
        if not r.get("enabled", True):
            continue
        for a in r.get("actions_forbidden", []) or []:
            out[a] = "forbidden"
        for a in r.get("actions_owner_only", []) or []:
            out[a] = "owner_only"
        for a in r.get("actions_allowed", []) or []:
            out[a] = "allowed_logged"
    return out


def check(action: str, approved_by_owner: bool = False, path: str = MEMBRANE_PATH) -> bool:
    cls = action_classes(path).get(action)
    if cls == "forbidden":
        raise MembraneViolation(
            f"action '{action}' is forbidden by the membrane and cannot be approved"
        )
    if cls == "owner_only" and not approved_by_owner:
        raise MembraneViolation(
            f"action '{action}' needs an owner approval recorded out-of-band"
        )
    return True


def write_membrane(*_args, **_kwargs):
    """Deliberately absent. This function exists so that its call sites fail loudly."""
    raise MembraneViolation("the system cannot write its own membrane")
