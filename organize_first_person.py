#!/usr/bin/env python3
"""Convert first_person.json to a compact, analysis-friendly structure.

Usage:
    python organize_first_person.py
    python organize_first_person.py input.json -o output.json

The output is a JSON array. Each element represents one source record.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


RATING_SCALE = {
    "Strongly disagree": 0,
    "Somewhat disagree": 1,
    "Neither agree nor disagree": 2,
    "Somewhat agree": 3,
    "Strongly agree": 4,
}

# (new field name, question in the source file)
APPRAISAL_FIELDS = (
    ("A1_relevance_general", "This situation matters to me."),
    (
        "A2_relevance_urgency",
        "I need to do something about this situation right away.",
    ),
    ("A3_relevance_goals", "This situation relates to my goals and plans."),
    (
        "A4_relevance_physical_wellbeing",
        "This situation relates to my physical well-being.",
    ),
    (
        "A5_relevance_people",
        "This situation involves people who matter to me.",
    ),
    (
        "A6_relevance_identity",
        "This situation concerns who I am and what I stand for.",
    ),
    (
        "B1_certainty_clarity",
        "It is clear to me what is going on in this situation.",
    ),
    (
        "B2_certainty_next",
        "I know what will come next in this situation.",
    ),
    ("B3_certainty_expected", "I saw this situation coming."),
    ("B4_certainty_novelty", "This is a new kind of situation for me."),
    ("C1_congruence_good", "This is a good situation."),
    (
        "C2_congruence_improvement",
        "This situation will get better with time.",
    ),
    (
        "C3_congruence_better_than_expected",
        "This situation is better than I expected.",
    ),
    (
        "C4_congruence_worse_than_expected",
        "This situation is worse than I expected.",
    ),
    ("D1_control_general", "This situation is under my control."),
    (
        "D2_control_stay_or_leave",
        "I can decide whether to stay in this situation or leave it.",
    ),
    ("D3_control_others", "Someone can handle this situation for me."),
    ("D4_control_effort", "I have to exert effort in this situation."),
    (
        "E1_accountability_self",
        "I am responsible for this situation.",
    ),
    (
        "E2_accountability_others",
        "Someone else is responsible for this situation.",
    ),
    (
        "E3_accountability_intentional",
        "This situation was caused intentionally.",
    ),
    (
        "E4_accountability_fairness",
        "This situation is fair and deserved.",
    ),
)

EMOTION_LABELS = {
    "Amused, entertained": "amused",
    "Angry, frustrated, annoyed": "angry",
    "Ashamed, humiliated, embarrassed": "ashamed",
    "Calm, peaceful, relaxed": "calm",
    "Confused, disoriented, surprised": "confused",
    "Despair, hopelessness, sorrow": "despair",
    "Disgust, distaste, revulsion": "disgust",
    "Excited, enthusiastic, elated": "excited",
    "Glad, happy, joyful": "glad",
    "Grateful, appreciative, thankful": "grateful",
    "Guilty, blameworthy, repentant": "guilty",
    "Hopeful, optimistic, encouraged": "hopeful",
    "Inspiration, admiration, appreciation": "inspiration",
    "Lonely, isolated, disconnected from others": "lonely",
    "Love, closeness, trust": "love",
    "Panicked, alarmed, freaked out": "panicked",
    "Proud, confident, determined": "proud",
    "Sad, downhearted, unhappy": "sad",
    "Sorrow": "sorrow",
    "Sympathy, concern, compassion": "sympathy",
    "Worried, nervous, fearful": "worried",
}

REASONING_FIELDS = ("relevance", "congruence", "accountability", "control", "certainty")
INTENSITY_PATTERN = re.compile(r"^\s*([0-6])\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Organize first-person appraisal and emotion records."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("first_person.json"),
        help="source JSON file (default: first_person.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("first_person_organized.json"),
        help="destination JSON file (default: first_person_organized.json)",
    )
    return parser.parse_args()


def parse_intensity(value: Any, record_id: str, field: str) -> int | None:
    """Read the leading 0-6 score; preserve a missing score as JSON null."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if not isinstance(value, str):
        raise ValueError(f"{record_id}: {field} must be a string, got {type(value).__name__}")

    match = INTENSITY_PATTERN.match(value)
    if match is None:
        raise ValueError(f"{record_id}: cannot parse {field} value {value!r}")
    return int(match.group(1))


def normalize_labels(values: Any, record_id: str, field: str) -> list[str]:
    if not isinstance(values, list):
        raise ValueError(f"{record_id}: {field} must be an array")

    normalized: list[str] = []
    for value in values:
        try:
            normalized.append(EMOTION_LABELS[value])
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"{record_id}: unknown label in {field}: {value!r}"
            ) from exc
    return normalized


def convert_ratings(source: Any, record_id: str) -> dict[str, int]:
    if not isinstance(source, dict):
        raise ValueError(f"{record_id}: appraisal_ratings must be an object")

    result: dict[str, int] = {}
    for output_name, question in APPRAISAL_FIELDS:
        if question not in source:
            raise ValueError(f"{record_id}: missing appraisal question {question!r}")
        answer = source[question]
        try:
            result[output_name] = RATING_SCALE[answer]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"{record_id}: unknown answer {answer!r} for {question!r}"
            ) from exc
    return result


def convert_record(record_id: str, source: Any) -> dict[str, Any]:
    if not isinstance(source, dict):
        raise ValueError(f"{record_id}: record must be an object")

    try:
        situation = source["story_collection"]["final_scenario"]
        reasoning_source = source["cognitive_questions"]["summary_answers"]
        emotion_source = source["emotion_labels"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{record_id}: required source field is missing") from exc

    if not isinstance(situation, str):
        raise ValueError(f"{record_id}: story_collection.final_scenario must be a string")
    if not isinstance(reasoning_source, dict):
        raise ValueError(
            f"{record_id}: cognitive_questions.summary_answers must be an object"
        )
    if not isinstance(emotion_source, dict):
        raise ValueError(f"{record_id}: emotion_labels must be an object")

    missing_reasoning = [key for key in REASONING_FIELDS if key not in reasoning_source]
    if missing_reasoning:
        raise ValueError(
            f"{record_id}: missing reasoning fields: {', '.join(missing_reasoning)}"
        )

    return {
        "id": record_id,
        "perspective": "first_person",
        "situation": situation,
        "appraisal_reasoning": {
            key: reasoning_source[key] for key in REASONING_FIELDS
        },
        "appraisal_ratings": convert_ratings(source.get("appraisal_ratings"), record_id),
        "emotion": {
            "negative_intensity": parse_intensity(
                emotion_source.get("negative_level"), record_id, "negative_level"
            ),
            "negative_labels": normalize_labels(
                emotion_source.get("negative_emotion_labels"),
                record_id,
                "negative_emotion_labels",
            ),
            "positive_intensity": parse_intensity(
                emotion_source.get("positive_level"), record_id, "positive_level"
            ),
            "positive_labels": normalize_labels(
                emotion_source.get("positive_emotion_labels"),
                record_id,
                "positive_emotion_labels",
            ),
        },
    }


def main() -> None:
    args = parse_args()
    with args.input.open("r", encoding="utf-8") as source_file:
        source_data = json.load(source_file)

    if not isinstance(source_data, dict):
        raise ValueError("the top level of the source JSON must be an object")

    output_data = [
        convert_record(record_id, record)
        for record_id, record in source_data.items()
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(output_data, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")

    missing_intensities = sum(
        item["emotion"][field] is None
        for item in output_data
        for field in ("negative_intensity", "positive_intensity")
    )
    print(f"Converted {len(output_data)} records to {args.output}")
    if missing_intensities:
        print(f"Kept {missing_intensities} missing emotion intensities as null")


if __name__ == "__main__":
    main()
