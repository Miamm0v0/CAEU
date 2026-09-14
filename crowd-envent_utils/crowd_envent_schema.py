#!/usr/bin/env python3
"""Shared schema, prompts, and strict parsing for crowd-enVent evaluation."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

UTILS_DIR = Path(__file__).resolve().parent
FIRST_PERSON_SCRIPTS = UTILS_DIR.parent / "FirstPersonMethod" / "scripts"
if str(FIRST_PERSON_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(FIRST_PERSON_SCRIPTS))

from carebench_aligned_projection import (
    carebench_appraisal_rating_schema,
    carebench_appraisal_statement_lines,
    normalize_carebench_appraisal_ratings,
    project_crowd_envent_rating,
)


METHOD_APPRAISAL_DIMENSIONS = (
    "relevance",
    "epistemic",
    "goal_congruence",
    "agency_accountability",
    "control_coping_potential",
)

METHOD_APPRAISAL_DEFINITIONS = {
    "relevance": (
        "Why the event matters or does not matter to my goals, needs, "
        "well-being, relationships, identity, or urgent concerns."
    ),
    "epistemic": (
        "What I know and do not know about the event: its clarity, certainty, "
        "expectedness or novelty, and what I believe is likely to happen next."
    ),
    "goal_congruence": (
        "How the actual or expected outcome helps, obstructs, or leaves "
        "unaffected my goals, needs, wishes, and preferred state."
    ),
    "agency_accountability": (
        "Who or what caused the event, including my own or others' agency, "
        "intentionality, responsibility, fairness, blame, or credit."
    ),
    "control_coping_potential": (
        "How much control I have and how able I am to act, change the outcome, "
        "seek help, tolerate it, adapt, withdraw, or otherwise cope."
    ),
}

EMOTION_LABELS = (
    "anger",
    "boredom",
    "disgust",
    "fear",
    "guilt",
    "joy",
    "pride",
    "relief",
    "sadness",
    "shame",
    "surprise",
    "trust",
    "no-emotion",
)

# All fields are native crowd-enVent 1--5 ratings.  ``method_dimension`` is
# used only for grouped reporting; metrics always retain all 21 source fields.
APPRAISAL_SPECS = (
    ("suddenness", "epistemic", "The event occurred suddenly or unexpectedly."),
    ("familiarity", "epistemic", "The situation felt familiar to me."),
    ("predict_event", "epistemic", "I could predict that the event would happen."),
    ("pleasantness", "goal_congruence", "The event was pleasant for me."),
    ("unpleasantness", "goal_congruence", "The event was unpleasant for me."),
    ("goal_relevance", "relevance", "The event was relevant to my goals or needs."),
    ("chance_responsblt", "agency_accountability", "Chance or circumstances were responsible."),
    ("self_responsblt", "agency_accountability", "I was responsible for the event."),
    ("other_responsblt", "agency_accountability", "Another person was responsible for the event."),
    ("predict_conseq", "epistemic", "I could predict the consequences of the event."),
    ("goal_support", "goal_congruence", "The event supported my goals."),
    ("urgency", "relevance", "The situation required an urgent response."),
    ("self_control", "control_coping_potential", "I could control or influence the situation."),
    ("other_control", "control_coping_potential", "Another person could control or influence it."),
    ("chance_control", "control_coping_potential", "Chance or circumstances controlled the outcome."),
    ("accept_conseq", "control_coping_potential", "I could accept or live with the consequences."),
    ("standards", "auxiliary_norm_value", "The event concerned my personal standards or values."),
    ("social_norms", "auxiliary_norm_value", "The event concerned social norms or expectations."),
    ("attention", "relevance", "The event demanded my attention."),
    ("not_consider", "control_coping_potential", "I tried not to think about or consider the event."),
    ("effort", "control_coping_potential", "Dealing with the event required effort from me."),
)

APPRAISAL_FIELDS = tuple(item[0] for item in APPRAISAL_SPECS)
APPRAISAL_TO_METHOD_DIMENSION = {item[0]: item[1] for item in APPRAISAL_SPECS}


CHAIN_SYSTEM_PROMPT = """You model the first-person cognitive appraisal process
that connects an experienced situation to emotion. First reason using exactly
five appraisal dimensions, then predict every requested crowd-enVent appraisal
rating and finally predict one emotion and its intensity. Adopt the experiencer's
perspective, ground every claim only in the situation, and do not invent goals,
beliefs, intentions, causes, coping resources, consequences, or emotions. Return
exactly one valid JSON object without markdown or commentary."""

DIRECT_SYSTEM_PROMPT = """You predict first-person cognitive appraisal ratings
and emotion from an experienced situation. Adopt the experiencer's perspective
and use only evidence in the situation. Return exactly one valid JSON object
without markdown or commentary."""

CHAIN_APPRAISALS_SYSTEM_PROMPT = """You model the first-person cognitive
appraisal process. First reason using exactly five appraisal dimensions, then
use that reasoning to predict all 21 crowd-enVent appraisal ratings. Adopt the
experiencer's perspective, ground every claim only in the situation, and do not
invent goals, beliefs, intentions, causes, coping resources, or consequences.
Return exactly one valid JSON object without markdown or commentary."""

CHAIN_EMOTION_SYSTEM_PROMPT = """You model the first-person cognitive appraisal
process that connects an experienced situation to emotion. First reason using
exactly five appraisal dimensions, then use that reasoning to predict one
crowd-enVent emotion and its intensity. Adopt the experiencer's perspective,
ground every claim only in the situation, and do not invent goals, beliefs,
intentions, causes, coping resources, consequences, or emotions. Return exactly
one valid JSON object without markdown or commentary."""

CAREBENCH_APPRAISALS_SYSTEM_PROMPT = """You model the first-person cognitive
appraisal process. First write the shared core appraisals, then use those
appraisals to rate the CAREBench appraisal statements. Adopt the experiencer's
perspective, ground every claim only in the situation, and do not invent goals,
beliefs, intentions, causes, coping resources, consequences, or emotions. Do
not predict emotion in this call. Return exactly one valid JSON object without
markdown or commentary."""

CAREBENCH_EMOTION_SYSTEM_PROMPT = """You model the first-person cognitive
appraisal process that connects an experienced situation to emotion. First
write the shared core appraisals, then use those appraisals to predict one
emotion in the native crowd-enVent label and intensity format. Adopt the
experiencer's perspective, ground every claim only in the situation, and do not
invent goals, beliefs, intentions, causes, coping resources, consequences, or
emotions. Do not rate appraisal statements in this call. Return exactly one
valid JSON object without markdown or commentary."""

CAREBENCH_ALIGNED_SYSTEM_PROMPT = CAREBENCH_APPRAISALS_SYSTEM_PROMPT

CHAIN_TARGETS = ("appraisals", "emotion")


def appraisal_definition_lines() -> str:
    return "\n".join(
        f"- {name}: {METHOD_APPRAISAL_DEFINITIONS[name]}"
        for name in METHOD_APPRAISAL_DIMENSIONS
    )


def rating_definition_lines() -> str:
    return "\n".join(
        f"- {field}: {description} Rate 1 = not at all, 5 = extremely."
        for field, _, description in APPRAISAL_SPECS
    )


def output_schema(include_reasoning: bool = True) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if include_reasoning:
        payload["appraisal_reasoning"] = {
            dimension: f"A concise first-person {dimension} appraisal."
            for dimension in METHOD_APPRAISAL_DIMENSIONS
        }
    payload["appraisal_ratings"] = {
        field: 3 for field in APPRAISAL_FIELDS
    }
    payload["emotion"] = {
        "label": "joy",
        "intensity": 3,
    }
    return payload


def carebench_aligned_output_schema(target: str | None = None) -> dict[str, Any]:
    if target is not None and target not in CHAIN_TARGETS:
        raise ValueError(f"Unknown carebench target {target!r}; allowed={list(CHAIN_TARGETS)}")
    payload: dict[str, Any] = {
        "core_appraisals": {
            dimension: f"A concise first-person {dimension} appraisal."
            for dimension in METHOD_APPRAISAL_DIMENSIONS
        },
    }
    if target is None or target == "appraisals":
        payload["carebench_appraisal_ratings"] = carebench_appraisal_rating_schema()
    if target is None or target == "emotion":
        payload["emotion"] = {
            "label": "joy",
            "intensity": 3,
        }
    return payload


def build_policy_prompt(
    situation: str,
    *,
    include_reasoning: bool = True,
    correction: str = "",
) -> str:
    sections = [
        "Situation (first-person experiencer):\n" + situation.strip(),
    ]
    if include_reasoning:
        sections.append(
            "Appraisal reasoning dimensions:\n" + appraisal_definition_lines()
        )
    sections.extend(
        [
            "Required crowd-enVent ratings:\n" + rating_definition_lines(),
            "Allowed emotion labels:\n" + ", ".join(EMOTION_LABELS),
            (
                "Emotion intensity is an integer from 1 (not at all intense) "
                "to 5 (extremely intense)."
            ),
            "Return exactly this JSON structure and exactly these keys:\n"
            + json.dumps(output_schema(include_reasoning), ensure_ascii=False, indent=2),
        ]
    )
    if correction:
        sections.append(
            "Your previous response was invalid. Return a fresh complete JSON "
            f"object. Validation error: {correction[:1000]}"
        )
    return "\n\n".join(sections)


def build_carebench_aligned_prompt(
    situation: str,
    target: str,
    *,
    correction: str = "",
) -> str:
    if target not in CHAIN_TARGETS:
        raise ValueError(f"Unknown carebench target {target!r}; allowed={list(CHAIN_TARGETS)}")
    if target == "appraisals":
        target_sections = [
            (
                "Step 2 - Appraisal ratings:\n"
                "Use the core appraisals above to rate these CAREBench "
                "appraisal statements.\n"
                + carebench_appraisal_statement_lines()
            ),
            "Do not predict emotion in this call.",
        ]
    else:
        target_sections = [
            (
                "Step 2 - Emotion and intensity:\n"
                "Use the core appraisals above to predict emotion using the "
                "native crowd-enVent format. Choose exactly one label and give "
                "that same emotion an intensity from 1 (not at all intense) "
                "to 5 (extremely intense). Do not generate separate positive "
                "and negative candidates."
            ),
            "Allowed emotion labels:\n" + ", ".join(EMOTION_LABELS),
            "Do not rate CAREBench appraisal statements in this call.",
        ]
    sections = [
        "Situation (first-person experiencer):\n" + situation.strip(),
        "Step 1 - Core appraisals:\n" + appraisal_definition_lines(),
        *target_sections,
        "Return exactly this JSON structure and exactly these keys:\n"
        + json.dumps(
            carebench_aligned_output_schema(target),
            ensure_ascii=False,
            indent=2,
        ),
    ]
    if correction:
        sections.append(
            "Your previous response was invalid. Return a fresh complete JSON "
            f"object. Validation error: {correction[:1000]}"
        )
    return "\n\n".join(sections)


def chain_target_output_schema(target: str) -> dict[str, Any]:
    if target not in CHAIN_TARGETS:
        raise ValueError(f"Unknown chain target {target!r}; allowed={list(CHAIN_TARGETS)}")
    payload: dict[str, Any] = {
        "appraisal_reasoning": {
            dimension: f"A concise first-person {dimension} appraisal."
            for dimension in METHOD_APPRAISAL_DIMENSIONS
        }
    }
    if target == "appraisals":
        payload["appraisal_ratings"] = {
            field: 3 for field in APPRAISAL_FIELDS
        }
    else:
        payload["emotion"] = {"label": "joy", "intensity": 3}
    return payload


def build_chain_target_prompt(
    situation: str,
    target: str,
    *,
    correction: str = "",
) -> str:
    if target not in CHAIN_TARGETS:
        raise ValueError(f"Unknown chain target {target!r}; allowed={list(CHAIN_TARGETS)}")
    sections = [
        "Situation (first-person experiencer):\n" + situation.strip(),
        "Appraisal reasoning dimensions:\n" + appraisal_definition_lines(),
    ]
    if target == "appraisals":
        sections.append("Required crowd-enVent ratings:\n" + rating_definition_lines())
    else:
        sections.extend(
            [
                "Allowed emotion labels:\n" + ", ".join(EMOTION_LABELS),
                (
                    "Emotion intensity is an integer from 1 (not at all intense) "
                    "to 5 (extremely intense)."
                ),
            ]
        )
    sections.append(
        "Return exactly this JSON structure and exactly these keys:\n"
        + json.dumps(chain_target_output_schema(target), ensure_ascii=False, indent=2)
    )
    if correction:
        sections.append(
            "Your previous response was invalid. Return a fresh complete JSON "
            f"object. Validation error: {correction[:1000]}"
        )
    return "\n\n".join(sections)


def extract_json_object(text: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Policy returned empty content")
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL | re.IGNORECASE)
    if fenced:
        stripped = fenced.group(1).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Policy response does not contain a JSON object")
        try:
            value = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Policy response contains invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("Policy response root must be a JSON object")
    return value


def _normalize_integer(value: Any, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer in [{minimum}, {maximum}]")
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{field} must be an integer in [{minimum}, {maximum}]")
    return value


def normalize_reasoning(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("appraisal_reasoning must be a JSON object")
    expected = set(METHOD_APPRAISAL_DIMENSIONS)
    found = set(value)
    if found != expected:
        raise ValueError(
            "appraisal_reasoning keys mismatch: "
            f"missing={sorted(expected - found)}, extra={sorted(found - expected)}"
        )
    normalized: dict[str, str] = {}
    for dimension in METHOD_APPRAISAL_DIMENSIONS:
        text = value[dimension]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"appraisal_reasoning.{dimension} must be non-empty text")
        normalized[dimension] = text.strip()
    return normalized


def normalize_ratings(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError("appraisal_ratings must be a JSON object")
    expected = set(APPRAISAL_FIELDS)
    found = set(value)
    if found != expected:
        raise ValueError(
            "appraisal_ratings keys mismatch: "
            f"missing={sorted(expected - found)}, extra={sorted(found - expected)}"
        )
    return {
        field: _normalize_integer(value[field], f"appraisal_ratings.{field}", 1, 5)
        for field in APPRAISAL_FIELDS
    }


def normalize_emotion(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("emotion must be a JSON object")
    expected = {"label", "intensity"}
    found = set(value)
    if found != expected:
        raise ValueError(
            f"emotion keys mismatch: missing={sorted(expected - found)}, "
            f"extra={sorted(found - expected)}"
        )
    label = value["label"]
    if not isinstance(label, str):
        raise ValueError("emotion.label must be text")
    label = label.strip().lower()
    if label not in EMOTION_LABELS:
        raise ValueError(
            f"emotion.label {label!r} is invalid; allowed={list(EMOTION_LABELS)}"
        )
    return {
        "label": label,
        "intensity": _normalize_integer(
            value["intensity"], "emotion.intensity", 1, 5
        ),
    }


def parse_policy_response(text: str, *, include_reasoning: bool = True) -> dict[str, Any]:
    payload = extract_json_object(text)
    expected = {"appraisal_ratings", "emotion"}
    if include_reasoning:
        expected.add("appraisal_reasoning")
    found = set(payload)
    if found != expected:
        raise ValueError(
            f"Policy root keys mismatch: missing={sorted(expected - found)}, "
            f"extra={sorted(found - expected)}"
        )
    normalized: dict[str, Any] = {
        "appraisal_ratings": normalize_ratings(payload["appraisal_ratings"]),
        "emotion": normalize_emotion(payload["emotion"]),
    }
    if include_reasoning:
        normalized["appraisal_reasoning"] = normalize_reasoning(
            payload["appraisal_reasoning"]
        )
    return normalized


def parse_chain_target_response(text: str, target: str) -> dict[str, Any]:
    if target not in CHAIN_TARGETS:
        raise ValueError(f"Unknown chain target {target!r}; allowed={list(CHAIN_TARGETS)}")
    payload = extract_json_object(text)
    outcome_key = "appraisal_ratings" if target == "appraisals" else "emotion"
    expected = {"appraisal_reasoning", outcome_key}
    found = set(payload)
    if found != expected:
        raise ValueError(
            f"chain-{target} root keys mismatch: missing={sorted(expected-found)}, "
            f"extra={sorted(found-expected)}"
        )
    normalized: dict[str, Any] = {
        "appraisal_reasoning": normalize_reasoning(payload["appraisal_reasoning"])
    }
    if target == "appraisals":
        normalized["appraisal_ratings"] = normalize_ratings(
            payload["appraisal_ratings"]
        )
    else:
        normalized["emotion"] = normalize_emotion(payload["emotion"])
    return normalized


def parse_carebench_aligned_response(text: str, target: str) -> dict[str, Any]:
    if target not in CHAIN_TARGETS:
        raise ValueError(f"Unknown carebench target {target!r}; allowed={list(CHAIN_TARGETS)}")
    payload = extract_json_object(text)
    outcome_key = (
        "carebench_appraisal_ratings" if target == "appraisals" else "emotion"
    )
    reasoning_keys = {"core_appraisals", "appraisal_reasoning"}
    found_reasoning_keys = set(payload) & reasoning_keys
    if len(found_reasoning_keys) != 1:
        raise ValueError(
            f"carebench-{target} response must contain exactly one core_appraisals object"
        )
    reasoning_key = next(iter(found_reasoning_keys))
    expected = {reasoning_key, outcome_key}
    found = set(payload)
    if found != expected:
        raise ValueError(
            f"carebench-{target} response root keys mismatch: missing={sorted(expected-found)}, "
            f"extra={sorted(found-expected)}"
        )
    reasoning = normalize_reasoning(payload[reasoning_key])
    if target == "appraisals":
        carebench_ratings = normalize_carebench_appraisal_ratings(
            payload["carebench_appraisal_ratings"]
        )
        return {
            "appraisal_reasoning": reasoning,
            "carebench_appraisal_ratings": carebench_ratings,
            "appraisal_ratings": {
                field: project_crowd_envent_rating(field, carebench_ratings)
                for field in APPRAISAL_FIELDS
            },
        }
    return {
        "appraisal_reasoning": reasoning,
        "emotion": normalize_emotion(payload["emotion"]),
    }
