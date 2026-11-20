"""Build Kimi-written multi-turn practice and a disjoint held-out chat set."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sapling.talk import OWNER_SELF, SELF_QUESTIONS, owner_answers  # noqa: E402
from sapling.text import normalize  # noqa: E402

TALK_DIR = os.path.join(ROOT, "data", "talk")
TRAIN = os.path.join(TALK_DIR, "conversation_train.jsonl")
HELDOUT = os.path.join(TALK_DIR, "conversation_heldout.jsonl")
MANIFEST = os.path.join(TALK_DIR, "conversation_manifest.json")
LESSONS = os.path.join(TALK_DIR, "lessons.jsonl")

TRAIN_NAMES = ["Ada", "Ben", "Cara", "Devon", "Emi", "Farah", "Gus", "Hana", "Idris", "June", "Kian", "Lena",
               "Milo", "Nina", "Owen", "Priya", "Quinn", "Rosa", "Sam", "Tessa", "Umar", "Vera", "Will", "Yara"]
TRAIN_OBJECTS = ["backpack", "button", "compass", "cup", "key", "kite", "lantern", "marble", "notebook", "pencil",
                 "shell", "ticket", "umbrella", "whistle", "basket", "candle", "envelope", "glove", "mirror", "ribbon"]
TRAIN_PLACES = ["library", "market", "station", "garden", "beach", "museum", "bakery", "harbor", "workshop", "orchard"]
COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "black", "white", "brown", "silver"]

HELDOUT_NAMES = ["Zorielle", "Kavren", "Lumik", "Brantel", "Mivora", "Tessik",
                 "Orvella", "Pelmor", "Rivwick", "Sanquel", "Virem", "Yollan"]
HELDOUT_OBJECTS = ["zimrel", "tovbox", "quimlet", "vessik", "lorphon", "quorvex",
                   "pelvix", "drosbet", "kavlock", "sornet", "frelkin", "dralfin"]
HELDOUT_PLACES = ["Nuvara", "Belquay", "Torvessa", "Quilbrook", "Zephyra", "Zorquay",
                  "Trenquil", "Veloria", "Thistara", "Vandrel", "Kelquay", "Brisora"]


def conv(turns, src="kimi conversation practice", **extra):
    return {"passage": None, "turns": [[normalize(u), normalize(a)] for u, a in turns], "src": src, **extra}


def practice(rng: random.Random, count: int) -> list[dict]:
    out = []
    for i in range(count):
        name = rng.choice(TRAIN_NAMES)
        obj = rng.choice(TRAIN_OBJECTS)
        place = rng.choice(TRAIN_PLACES)
        color = rng.choice(COLORS)
        kind = i % 8
        if kind == 0:
            turns = [[f"Hi, I am {name}.", f"Hello {name}."], ["What is my name?", f"Your name is {name}."]]
            label = "name"
        elif kind == 1:
            turns = [[f"My name is {name}.", f"Thank you for telling me, {name}."],
                     [f"I have a {obj}.", f"You have a {obj}."], ["What is my name?", f"Your name is {name}."]]
            label = "name"
        elif kind == 2:
            turns = [[f"I put my {obj} in the {place}.", f"Your {obj} is in the {place}."],
                     [f"Where is my {obj}?", f"Your {obj} is in the {place}."]]
            label = "object_place"
        elif kind == 3:
            turns = [[f"I am going to the {place}.", f"You are going to the {place}."],
                     ["Where am I going?", f"You are going to the {place}."]]
            label = "place"
        elif kind == 4:
            turns = [[f"My {obj} is {color}.", f"Your {obj} is {color}."],
                     ["What color is it?", f"Your {obj} is {color}."]]
            label = "followup"
        elif kind == 5:
            turns = [[f"I saw {name} at the {place}.", f"You saw {name} at the {place}."],
                     ["Who did I see?", f"You saw {name}."]]
            label = "name_followup"
        elif kind == 6:
            turns = [[f"The {obj} is {color}.", f"The {obj} is {color}."],
                     [f"What did I just say was {color}?", f"You just said the {obj} was {color}."]]
            label = "what_just_said"
        else:
            turns = [[f"I left the {obj} at the {place}.", f"You left the {obj} at the {place}."],
                     ["Can you remind me where it is?", f"It is at the {place}."]]
            label = "simple_followup"
        out.append(conv(turns, kind=label, practice_index=i))
    rng.shuffle(out)
    return out


def owner_chats(rng: random.Random) -> list[dict]:
    answers = owner_answers()
    out = []
    contexts = [(["Hello.", "Hello."]),
                (["I want to know you better.", "I am listening."]),
                (["Before we talk about other things, I have a question for you.", "Go ahead."])]
    for key, questions in SELF_QUESTIONS.items():
        if key not in answers:
            continue
        for q in questions:
            for lead in contexts:
                out.append(conv([lead, [q, answers[key]]], src="kimi chat + Jaron's exact words", key=key))
    rng.shuffle(out)
    return out


def heldout() -> list[dict]:
    out = []
    for i in range(12):
        name, obj, place = HELDOUT_NAMES[i], HELDOUT_OBJECTS[i], HELDOUT_PLACES[i]
        color = COLORS[(i + 2) % len(COLORS)]
        name_chat = ([[f"Hello, my name is {name}.", f"Hello {name}."], ["What is my name?", f"Your name is {name}."]]
                     if i % 2 == 0 else
                     [[f"I am {name}.", f"Nice to meet you, {name}."], [f"My {obj} is {color}.", f"Your {obj} is {color}."],
                      ["What is my name?", f"Your name is {name}."]])
        specs = [
            ("name", name_chat, name),
            ("object", [[f"I found a {obj}.", f"You found a {obj}."], ["What did I find?", f"You found a {obj}."]], obj),
            ("place", [[f"We met in {place}.", f"We met in {place}."], ["Where did we meet?", f"We met in {place}."]], place),
            ("object", [[f"My {obj} is {color}.", f"Your {obj} is {color}."], ["What color is it?", f"Your {obj} is {color}."]], color),
            ("followup", [[f"I left the {obj} in {place}.", f"You left the {obj} in {place}."],
                          ["Where is it?", f"It is in {place}."]], place),
            ("what_just_said", [[f"The {obj} belongs to {name}.", f"The {obj} belongs to {name}."],
                                ["What did I just say?", f"You just said the {obj} belongs to {name}."]], f"the {obj} belongs to {name}"),
            ("followup", [[f"{name} has a {obj}.", f"{name} has a {obj}."], ["What does that person have?", f"{name} has a {obj}."]], obj),
        ]
        for j, (kind, turns, expected) in enumerate(specs):
            out.append(conv(turns, src="kimi held-out chat", id=f"heldout-{i:02d}-{j}", kind=kind,
                            expected=normalize(expected), entities=[normalize(name), normalize(obj), normalize(place)]))
    return out


def write_jsonl(path: str, rows: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(count: int = 8000, merge: bool = False) -> dict:
    rng = random.Random(29029)
    train = practice(rng, count) + owner_chats(rng)
    test = heldout()
    base_text = ""
    if os.path.exists(LESSONS):
        base_text = open(LESSONS, encoding="utf-8").read().lower()
    train_text = "\n".join(json.dumps(r, ensure_ascii=False).lower() for r in train)
    forbidden = set(HELDOUT_NAMES + HELDOUT_OBJECTS + HELDOUT_PLACES)
    leaks = sorted(x for x in forbidden if f" {normalize(x)} " in f" {normalize(base_text)} " or
                   f" {normalize(x)} " in f" {normalize(train_text)} ")
    if leaks:
        raise RuntimeError(f"held-out entities already occur in training: {leaks}")
    os.makedirs(TALK_DIR, exist_ok=True)
    write_jsonl(TRAIN, train)
    write_jsonl(HELDOUT, test)
    if merge:
        old = [r for r in read_jsonl(LESSONS) if not str(r.get("src", "")).startswith("kimi ")] if os.path.exists(LESSONS) else []
        write_jsonl(LESSONS, old + train)
    manifest = {
        "authoring": "New dialogue text written by Kimi; self answers are Jaron's exact normalized words.",
        "name_thread_bar": "10/12", "name_thread_pass_min": 10, "name_thread_total": 12,
        "train_conversations": len(train), "heldout_conversations": len(test),
        "heldout_kinds": {k: sum(r["kind"] == k for r in test) for k in sorted({r["kind"] for r in test})},
        "heldout_entities_absent_from_training": True,
        "heldout_names": HELDOUT_NAMES, "heldout_objects": HELDOUT_OBJECTS, "heldout_places": HELDOUT_PLACES,
        "train_sha256": sha(TRAIN), "heldout_sha256": sha(HELDOUT), "merged_into_lessons": merge,
        "lessons_sha256": sha(LESSONS) if merge else None,
    }
    json.dump(manifest, open(MANIFEST, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=8000)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args(argv)
    print(json.dumps(build(a.count, a.merge), indent=1))


if __name__ == "__main__":
    main()
