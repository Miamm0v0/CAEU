#!/usr/bin/env python3
"""CAREBench-aligned output schema and projection helpers.

The external CovidET and crowd-enVent evaluations consume their own native
appraisal/emotion fields. This module lets their generators ask the Policy for
a CAREBench-style intermediate output first, then project it back into the
native fields required by the existing metrics.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


POLICY_APPRAISAL_DIMENSIONS = (
    "relevance",
    "epistemic",
    "goal_congruence",
    "agency_accountability",
    "control_coping_potential",
)

CAREBENCH_RATING_STATEMENTS = (
    ("relevance.general", "This situation matters to me."),
    ("relevance.urgency", "I need to do something about this situation right away."),
    ("relevance.goals", "This situation relates to my goals and plans."),
    ("relevance.bodily motives", "This situation relates to my physical well-being."),
    ("relevance.social motives", "This situation involves people who matter to me."),
    ("relevance.identity motives", "This situation concerns who I am and what I stand for."),
    ("certainty.construal", "It is clear to me what is going on in this situation."),
    ("certainty.outlook", "I know what will come next in this situation."),
    ("certainty.predictability", "I saw this situation coming."),
    ("certainty.novelty", "This is a new kind of situation for me."),
    ("congruence.general", "This is a good situation."),
    ("congruence.outlook", "This situation will get better with time."),
    ("congruence.positive prediction error", "This situation is better than I expected."),
    ("congruence.negative prediction error", "This situation is worse than I expected."),
    ("control.general", "This situation is under my control."),
    ("control.select", "I can decide whether to stay in this situation or leave it."),
    ("control.vicarious", "Someone can handle this situation for me."),
    ("control.effortful", "I have to exert effort in this situation."),
    ("accountability.self", "I am responsible for this situation."),
    ("accountability.other", "Someone else is responsible for this situation."),
    ("accountability.intentionality", "This situation was caused intentionally."),
    ("accountability.fairness", "This situation is fair and deserved."),
)

CAREBENCH_RATING_KEYS = tuple(key for key, _ in CAREBENCH_RATING_STATEMENTS)
CAREBENCH_NEUTRAL_RATING = 3

COVIDET_POSITIVE_LABELS = ("anticipation", "joy", "trust")
COVIDET_NEGATIVE_LABELS = ("anger", "disgust", "fear", "sadness")

CROWD_ENVENT_NO_EMOTION = "no-emotion"
CROWD_ENVENT_POSITIVE_LABELS = (
    "joy",
    "pride",
    "relief",
    "trust",
    CROWD_ENVENT_NO_EMOTION,
)
CROWD_ENVENT_NEGATIVE_LABELS = (
    "anger",
    "boredom",
    "disgust",
    "fear",
    "guilt",
    "sadness",
    "shame",
    "surprise",
    CROWD_ENVENT_NO_EMOTION,
)

Projection = tuple[tuple[str, bool], ...]


COVIDET_CAREBENCH_PROJECTIONS: dict[str, Projection] = {
    "dim1": (("accountability.self", False),),
    "dim2": (("accountability.other", False),),
    "dim3": (
        ("accountability.self", True),
        ("accountability.other", True),
        ("accountability.intentionality", True),
    ),
    "dim4": (("control.general", False), ("control.select", False)),
    "dim5": (("relevance.general", False), ("relevance.goals", False)),
    "dim6": (("relevance.urgency", False),),
    "dim7": (("control.general", False), ("control.select", False)),
    "dim8": (("control.general", False), ("control.select", False)),
    "dim9": (("control.vicarious", False), ("accountability.other", False)),
    "dim10": (
        ("control.general", True),
        ("control.vicarious", True),
        ("accountability.intentionality", True),
    ),
    "dim11": (("certainty.outlook", False), ("certainty.predictability", False)),
    "dim12": (
        ("relevance.general", False),
        ("congruence.general", True),
        ("congruence.negative prediction error", False),
    ),
    "dim13": (("congruence.general", False),),
    "dim14": (("certainty.construal", False), ("certainty.outlook", False)),
    "dim15": (("congruence.general", False), ("congruence.outlook", False)),
    "dim17": (("congruence.outlook", False),),
    "dim19": (
        ("congruence.general", True),
        ("congruence.negative prediction error", False),
    ),
    "dim20": (("certainty.novelty", True),),
    "dim21": (("control.effortful", False),),
    "dim22": (("control.effortful", False), ("control.general", True)),
    "dim24": (("certainty.predictability", False), ("certainty.novelty", True)),
}


CROWD_ENVENT_CAREBENCH_PROJECTIONS: dict[str, Projection] = {
    "suddenness": (("certainty.novelty", False),),
    "familiarity": (("certainty.novelty", True),),
    "predict_event": (("certainty.predictability", False),),
    "pleasantness": (("congruence.general", False),),
    "unpleasantness": (
        ("congruence.general", True),
        ("congruence.negative prediction error", False),
    ),
    "goal_relevance": (("relevance.general", False), ("relevance.goals", False)),
    "chance_responsblt": (
        ("accountability.self", True),
        ("accountability.other", True),
        ("accountability.intentionality", True),
    ),
    "self_responsblt": (("accountability.self", False),),
    "other_responsblt": (("accountability.other", False),),
    "predict_conseq": (("certainty.outlook", False), ("certainty.predictability", False)),
    "goal_support": (("congruence.general", False), ("congruence.outlook", False)),
    "urgency": (("relevance.urgency", False),),
    "self_control": (("control.general", False), ("control.select", False)),
    "other_control": (("control.vicarious", False), ("accountability.other", False)),
    "chance_control": (
        ("control.general", True),
        ("control.vicarious", True),
        ("accountability.intentionality", True),
    ),
    "accept_conseq": (("control.general", False), ("control.select", False)),
    "standards": (("relevance.identity motives", False),),
    "social_norms": (("relevance.social motives", False), ("accountability.fairness", False)),
    "attention": (("relevance.general", False), ("relevance.urgency", False)),
    "not_consider": (("control.general", True), ("control.effortful", False)),
    "effort": (("control.effortful", False),),
}


def carebench_appraisal_statement_lines() -> str:
    return "\n".join(
        f"- {key}: {statement} Rate 1 = Strongly disagree, "
        "2 = Somewhat disagree, 3 = Neither agree nor disagree, "
        "4 = Somewhat agree, 5 = Strongly agree."
        for key, statement in CAREBENCH_RATING_STATEMENTS
    )


def carebench_appraisal_rating_schema() -> dict[str, int]:
    return {key: CAREBENCH_NEUTRAL_RATING for key in CAREBENCH_RATING_KEYS}


def split_emotion_schema(
    positive_labels: tuple[str, ...],
    negative_labels: tuple[str, ...],
    *,
    include_intensity: bool = False,
) -> dict[str, Any]:
    positive_example = [label for label in positive_labels if label != CROWD_ENVENT_NO_EMOTION]
    negative_example = [label for label in negative_labels if label != CROWD_ENVENT_NO_EMOTION]
    schema: dict[str, Any] = {
        "positive_labels": positive_example[:1],
        "negative_labels": negative_example[:1],
    }
    if include_intensity:
        schema = {
            "positive_intensity": 3,
            "negative_intensity": 3,
            **schema,
        }
    return schema


def crowd_envent_valenced_emotion_schema() -> dict[str, dict[str, Any]]:
    return {
        "positive": {
            "label": "joy",
            "intensity": 3,
        },
        "negative": {
            "label": "sadness",
            "intensity": 3,
        },
    }


def _require_int(value: Any, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer in [{minimum}, {maximum}]")
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{field} must be an integer in [{minimum}, {maximum}]")
    return value


def normalize_carebench_appraisal_ratings(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError("carebench_appraisal_ratings must be a JSON object")
    expected = set(CAREBENCH_RATING_KEYS)
    found = set(value)
    if found != expected:
        raise ValueError(
            "carebench_appraisal_ratings keys mismatch: "
            f"missing={sorted(expected - found)}, extra={sorted(found - expected)}"
        )
    return {
        key: _require_int(value[key], f"carebench_appraisal_ratings.{key}", 1, 5)
        for key in CAREBENCH_RATING_KEYS
    }


def normalize_carebench_appraisal_reasoning(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("appraisal_reasoning must be a JSON object")
    expected = set(POLICY_APPRAISAL_DIMENSIONS)
    found = set(value)
    if found != expected:
        raise ValueError(
            "appraisal_reasoning keys mismatch: "
            f"missing={sorted(expected - found)}, extra={sorted(found - expected)}"
        )
    normalized: dict[str, str] = {}
    for dimension in POLICY_APPRAISAL_DIMENSIONS:
        text = value[dimension]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"appraisal_reasoning.{dimension} must be non-empty text")
        normalized[dimension] = text.strip()
    return normalized


def normalize_split_emotion_labels(
    value: Any,
    *,
    positive_labels: tuple[str, ...],
    negative_labels: tuple[str, ...],
    field: str = "emotion",
    require_intensity: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    label_keys = {"positive_labels", "negative_labels"}
    intensity_keys = {"positive_intensity", "negative_intensity"}
    expected = label_keys | intensity_keys if require_intensity else label_keys
    found = set(value)
    if require_intensity:
        keys_valid = found == expected
    else:
        keys_valid = (
            label_keys.issubset(found)
            and found <= label_keys | intensity_keys
            and (not (found & intensity_keys) or intensity_keys.issubset(found))
        )
    if not keys_valid:
        raise ValueError(
            f"{field} keys mismatch: missing={sorted(expected - found)}, "
            f"extra={sorted(found - (label_keys | intensity_keys))}"
        )
    normalized: dict[str, Any] = {
        "positive_labels": _normalize_label_list(
            value["positive_labels"],
            positive_labels,
            f"{field}.positive_labels",
            allow_no_emotion=False,
        ),
        "negative_labels": _normalize_label_list(
            value["negative_labels"],
            negative_labels,
            f"{field}.negative_labels",
            allow_no_emotion=False,
        ),
    }
    if intensity_keys.issubset(found):
        normalized["positive_intensity"] = _require_int(
            value["positive_intensity"],
            f"{field}.positive_intensity",
            0,
            6,
        )
        normalized["negative_intensity"] = _require_int(
            value["negative_intensity"],
            f"{field}.negative_intensity",
            0,
            6,
        )
    return normalized


def _normalize_label_list(
    value: Any,
    allowed: tuple[str, ...],
    field: str,
    *,
    allow_no_emotion: bool,
) -> list[str]:
    if isinstance(value, str) and value.strip().lower() in {"none", "no emotion", "no-emotion"}:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON list")
    allowed_set = set(allowed)
    normalized: list[str] = []
    for index, raw_label in enumerate(value):
        if not isinstance(raw_label, str) or not raw_label.strip():
            raise ValueError(f"{field}[{index}] must be a non-empty string")
        label = raw_label.strip().lower()
        if label in {"none", "no emotion"}:
            label = CROWD_ENVENT_NO_EMOTION
        if label == CROWD_ENVENT_NO_EMOTION and not allow_no_emotion:
            continue
        if label not in allowed_set:
            raise ValueError(
                f"{field}[{index}] has invalid label {raw_label!r}; "
                f"allowed={list(allowed)}"
            )
        if label != CROWD_ENVENT_NO_EMOTION and label not in normalized:
            normalized.append(label)
    return [label for label in allowed if label in normalized]


def merged_split_labels(
    split_emotion: Mapping[str, list[str]],
    ordered_labels: tuple[str, ...] | list[str],
) -> list[str]:
    selected = set(split_emotion.get("positive_labels", [])) | set(
        split_emotion.get("negative_labels", [])
    )
    return [label for label in ordered_labels if label in selected]


def normalize_crowd_envent_valenced_emotion(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        raise ValueError("emotion must be a JSON object")
    expected = {"positive", "negative"}
    found = set(value)
    if found != expected:
        raise ValueError(
            f"emotion keys mismatch: missing={sorted(expected - found)}, "
            f"extra={sorted(found - expected)}"
        )
    return {
        "positive": _normalize_single_valenced_emotion(
            value["positive"],
            CROWD_ENVENT_POSITIVE_LABELS,
            "emotion.positive",
        ),
        "negative": _normalize_single_valenced_emotion(
            value["negative"],
            CROWD_ENVENT_NEGATIVE_LABELS,
            "emotion.negative",
        ),
    }


def _normalize_single_valenced_emotion(
    value: Any,
    allowed: tuple[str, ...],
    field: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    expected = {"label", "intensity"}
    found = set(value)
    if found != expected:
        raise ValueError(
            f"{field} keys mismatch: missing={sorted(expected - found)}, "
            f"extra={sorted(found - expected)}"
        )
    label = value["label"]
    if not isinstance(label, str) or not label.strip():
        raise ValueError(f"{field}.label must be non-empty text")
    normalized_label = label.strip().lower()
    if normalized_label in {"none", "no emotion"}:
        normalized_label = CROWD_ENVENT_NO_EMOTION
    if normalized_label not in set(allowed):
        raise ValueError(
            f"{field}.label {label!r} is invalid; allowed={list(allowed)}"
        )
    intensity = _require_int(value["intensity"], f"{field}.intensity", 1, 5)
    if normalized_label == CROWD_ENVENT_NO_EMOTION:
        intensity = 1
    return {
        "label": normalized_label,
        "intensity": intensity,
    }


def select_crowd_envent_emotion(
    valenced_emotion: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    positive = valenced_emotion["positive"]
    negative = valenced_emotion["negative"]
    positive_intensity = int(positive["intensity"])
    negative_intensity = int(negative["intensity"])
    positive_label = str(positive["label"])
    negative_label = str(negative["label"])

    if positive_intensity > negative_intensity:
        selected = positive
    elif negative_intensity > positive_intensity:
        selected = negative
    elif positive_label == CROWD_ENVENT_NO_EMOTION and negative_label != CROWD_ENVENT_NO_EMOTION:
        selected = negative
    elif negative_label == CROWD_ENVENT_NO_EMOTION and positive_label != CROWD_ENVENT_NO_EMOTION:
        selected = positive
    else:
        selected = positive

    return {
        "label": str(selected["label"]),
        "intensity": int(selected["intensity"]),
    }


def project_covidet_rating(
    source_dimension: str,
    carebench_ratings: Mapping[str, int],
) -> int:
    return project_rating(
        carebench_ratings,
        COVIDET_CAREBENCH_PROJECTIONS.get(source_dimension, ()),
        minimum=1,
        maximum=9,
    )


def project_crowd_envent_rating(
    field: str,
    carebench_ratings: Mapping[str, int],
) -> int:
    return project_rating(
        carebench_ratings,
        CROWD_ENVENT_CAREBENCH_PROJECTIONS.get(field, ()),
        minimum=1,
        maximum=5,
    )


def project_rating(
    carebench_ratings: Mapping[str, int],
    projection: Projection,
    *,
    minimum: int,
    maximum: int,
) -> int:
    score = projected_carebench_score(carebench_ratings, projection)
    scaled = minimum + (score - 1.0) * ((maximum - minimum) / 4.0)
    return max(minimum, min(maximum, int(math.floor(scaled + 0.5))))


def projected_carebench_score(
    carebench_ratings: Mapping[str, int],
    projection: Projection,
) -> float:
    if not projection:
        return float(CAREBENCH_NEUTRAL_RATING)
    values: list[float] = []
    for key, invert in projection:
        if key not in carebench_ratings:
            raise ValueError(f"Missing CAREBench appraisal rating: {key}")
        rating = int(carebench_ratings[key])
        values.append(float(6 - rating if invert else rating))
    return sum(values) / len(values)
