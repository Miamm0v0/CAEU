#!/usr/bin/env python3
"""Shared schema, I/O, and parsing helpers for CHIARO evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


POSITIVE_EMOTIONS = ("joy", "pride", "relief", "gratitude", "excitement")
NEGATIVE_EMOTIONS = ("anger", "sadness", "fear", "disgust", "embarrassment")
ALL_EMOTIONS = POSITIVE_EMOTIONS + NEGATIVE_EMOTIONS

APPRAISAL_DIMENSIONS = (
    "relevance",
    "epistemic",
    "goal_congruence",
    "agency_accountability",
    "control_coping_potential",
)

REQUIRED_ITEM_FIELDS = (
    "id",
    "sentence",
    "agent_a_role",
    "agent_b_role",
    "human_gold_a",
    "human_gold_b",
    "options_a",
    "options_b",
)


def load_chiaro_items(
    path: Path,
    *,
    split: str = "all",
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Load and validate CHIARO scenes while preserving file order."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"CHIARO input must be a JSON list: {path}")

    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_item in enumerate(payload):
        if not isinstance(raw_item, dict):
            raise ValueError(f"Item {index} is not a JSON object")
        missing = [field for field in REQUIRED_ITEM_FIELDS if field not in raw_item]
        if missing:
            raise ValueError(f"Item {index} is missing fields: {', '.join(missing)}")
        item = dict(raw_item)
        item_id = str(item["id"])
        if item_id in seen:
            raise ValueError(f"Duplicate CHIARO id: {item_id}")
        seen.add(item_id)
        if split != "all" and str(item.get("split", "")) != split:
            continue
        for slot in ("a", "b"):
            gold = normalize_emotion(item[f"human_gold_{slot}"])
            if gold is None:
                raise ValueError(f"{item_id}: invalid human_gold_{slot}")
            options = item[f"options_{slot}"]
            if not isinstance(options, dict) or not options:
                raise ValueError(f"{item_id}: options_{slot} must be a mapping")
            normalized_options = {
                str(letter).upper(): normalize_emotion(label)
                for letter, label in options.items()
            }
            if any(label is None for label in normalized_options.values()):
                raise ValueError(f"{item_id}: options_{slot} contains an invalid label")
            item[f"options_{slot}"] = normalized_options
        items.append(item)

    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("--max_samples must be positive")
        items = items[:max_samples]
    return items


def normalize_emotion(value: Any) -> str | None:
    label = str(value or "").strip().lower()
    return label if label in ALL_EMOTIONS else None


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object, tolerating surrounding model chatter."""
    start = text.find("{")
    if start < 0:
        raise ValueError("Model output does not contain a JSON object")
    try:
        parsed, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON output: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Model output JSON must be an object")
    return parsed


def parse_appraisal_reasoning(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("appraisal_reasoning must be an object")
    actual = set(value)
    expected = set(APPRAISAL_DIMENSIONS)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"appraisal_reasoning keys mismatch; missing={missing}, extra={extra}"
        )
    result: dict[str, str] = {}
    for dimension in APPRAISAL_DIMENSIONS:
        answer = value[dimension]
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"appraisal_reasoning.{dimension} must be non-empty text")
        result[dimension] = answer.strip()
    return result


def _parse_intensity(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer from 0 to 6")
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if not isinstance(value, int) or not 0 <= value <= 6:
        raise ValueError(f"{field} must be an integer from 0 to 6")
    return value


def _parse_label_list(
    value: Any,
    *,
    allowed: tuple[str, ...],
    field: str,
) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON array")
    result: list[str] = []
    for raw_label in value:
        label = normalize_emotion(raw_label)
        if label is None or label not in allowed:
            raise ValueError(f"{field} contains invalid label: {raw_label!r}")
        if label not in result:
            result.append(label)
    return result


def parse_free_emotion(value: Any) -> dict[str, Any]:
    """Parse the CAREBench-style, unrestricted CHIARO emotion object."""
    if not isinstance(value, dict):
        raise ValueError("emotion must be an object")
    required = {
        "positive_intensity",
        "negative_intensity",
        "positive_labels",
        "negative_labels",
    }
    actual = set(value)
    if actual != required:
        raise ValueError(
            "emotion keys mismatch; missing={}, extra={}".format(
                sorted(required - actual), sorted(actual - required)
            )
        )
    positive_labels = _parse_label_list(
        value["positive_labels"],
        allowed=POSITIVE_EMOTIONS,
        field="emotion.positive_labels",
    )
    negative_labels = _parse_label_list(
        value["negative_labels"],
        allowed=NEGATIVE_EMOTIONS,
        field="emotion.negative_labels",
    )
    return {
        "positive_intensity": _parse_intensity(
            value["positive_intensity"], "emotion.positive_intensity"
        ),
        "negative_intensity": _parse_intensity(
            value["negative_intensity"], "emotion.negative_intensity"
        ),
        "positive_labels": positive_labels,
        "negative_labels": negative_labels,
    }


def select_free_emotion(emotion: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Select a single CHIARO label from a parsed CAREBench emotion object."""
    positive_labels = list(emotion["positive_labels"])
    negative_labels = list(emotion["negative_labels"])
    positive_intensity = int(emotion["positive_intensity"])
    negative_intensity = int(emotion["negative_intensity"])
    if not positive_labels and not negative_labels:
        raise ValueError("At least one positive or negative emotion label is required")
    if positive_labels and negative_labels:
        if positive_intensity == negative_intensity:
            raise ValueError(
                "Cannot select one label when both valences have labels and equal intensity"
            )
        selected_valence = (
            "positive" if positive_intensity > negative_intensity else "negative"
        )
    elif positive_labels:
        selected_valence = "positive"
    else:
        selected_valence = "negative"
    selected_label = (
        positive_labels[0] if selected_valence == "positive" else negative_labels[0]
    )
    return selected_label, {
        "rule": "higher-intensity-valence-then-first-ranked-label",
        "selected_valence": selected_valence,
        "selected_label": selected_label,
        "intensity_tie": positive_intensity == negative_intensity,
    }


def parse_constrained_emotion(value: Any, allowed: Iterable[str]) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"label"}:
        raise ValueError("emotion must contain exactly one key: label")
    label = normalize_emotion(value["label"])
    allowed_set = set(allowed)
    if label is None or label not in allowed_set:
        raise ValueError(f"emotion.label must be one of {sorted(allowed_set)}")
    return {"label": label}


def letter_for_emotion(options: Mapping[str, str], emotion: str) -> str | None:
    for letter, label in options.items():
        if label == emotion:
            return letter
    return None


def read_prediction_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".jsonl":
            records = [json.loads(line) for line in handle if line.strip()]
        else:
            records = json.load(handle)
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError(f"Predictions must be a JSON list or JSONL objects: {path}")
    return records


def write_prediction_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        if path.suffix.lower() == ".jsonl":
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            json.dump(records, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    os.replace(temporary, path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)
