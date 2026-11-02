"""Strict policy/Judge JSON parsing and deterministic reward aggregation."""

from __future__ import annotations

import json
from typing import Any, Sequence

try:
    from ..sft_common import canonicalize_emotion_labels
except (ImportError, ValueError):
    from sft_common import canonicalize_emotion_labels

from .spec import (
    APPRAISAL_CRITERIA,
    EXPECTED_EMOTION_KEYS,
    EXPECTED_TOP_LEVEL_KEYS,
    resolve_appraisal_dimensions,
)


def object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        content = completion.get("content")
        return content if isinstance(content, str) else ""
    if isinstance(completion, list):
        parts = [
            item.get("content", "")
            for item in completion
            if isinstance(item, dict) and item.get("role") == "assistant"
        ]
        if not parts and len(completion) == 1 and isinstance(completion[0], dict):
            parts = [completion[0].get("content", "")]
        return "".join(part for part in parts if isinstance(part, str))
    return ""


def require_exact_keys(
    value: Any,
    expected: set[str],
    location: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{location} keys mismatch: missing={missing}, extra={extra}"
        )
    return value


def require_bounded_int(
    value: Any,
    minimum: int,
    maximum: int,
    location: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{location} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{location} must be in [{minimum}, {maximum}]")
    return value


def require_label_list(
    value: Any,
    valence: str,
    location: str,
) -> list[str]:
    try:
        return canonicalize_emotion_labels(
        value,
        valence,
        field=location,
        reject_duplicates=False,
        )
    except ValueError as exc:
        raise ValueError(str(exc)) from exc


def parse_policy_output(
    text: str,
    appraisal_dimensions: Sequence[str] | str | None = None,
) -> dict[str, Any]:
    selected_dimensions = resolve_appraisal_dimensions(
        appraisal_dimensions
    )
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty completion")
    try:
        payload = json.loads(
            stripped,
            object_pairs_hook=object_without_duplicate_keys,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc.msg}") from exc
    require_exact_keys(payload, EXPECTED_TOP_LEVEL_KEYS, "root")

    reasoning = require_exact_keys(
        payload["appraisal_reasoning"],
        set(selected_dimensions),
        "appraisal_reasoning",
    )
    for dimension in selected_dimensions:
        value = reasoning[dimension]
        location = f"appraisal_reasoning.{dimension}"
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"appraisal_reasoning.{dimension} must be a non-empty string"
            )
        else:
            reasoning[dimension] = value.strip()

    emotion = require_exact_keys(
        payload["emotion"],
        EXPECTED_EMOTION_KEYS,
        "emotion",
    )
    require_bounded_int(
        emotion["positive_intensity"],
        0,
        6,
        "emotion.positive_intensity",
    )
    require_bounded_int(
        emotion["negative_intensity"],
        0,
        6,
        "emotion.negative_intensity",
    )
    emotion["positive_labels"] = require_label_list(
        emotion["positive_labels"],
        "positive",
        "emotion.positive_labels",
    )
    emotion["negative_labels"] = require_label_list(
        emotion["negative_labels"],
        "negative",
        "emotion.negative_labels",
    )
    if emotion["positive_intensity"] == 0 and emotion["positive_labels"]:
        raise ValueError(
            "emotion.positive_labels must be empty when positive_intensity is 0"
        )
    if emotion["negative_intensity"] == 0 and emotion["negative_labels"]:
        raise ValueError(
            "emotion.negative_labels must be empty when negative_intensity is 0"
        )
    return payload


def _validate_score_item(
    value: Any,
    location: str,
    rationale_mode: str = "none",
) -> dict[str, Any]:
    if rationale_mode not in {"required", "none"}:
        raise ValueError("rationale_mode must be 'required' or 'none'")
    expected_keys = (
        {"score", "rationale"}
        if rationale_mode == "required"
        else {"score"}
    )
    item = require_exact_keys(value, expected_keys, location)
    require_bounded_int(item["score"], 0, 4, f"{location}.score")
    if rationale_mode == "required":
        if (
            not isinstance(item["rationale"], str)
            or not item["rationale"].strip()
        ):
            raise ValueError(f"{location}.rationale must be non-empty")
        item["rationale"] = item["rationale"].strip()
    return item


def parse_judge_output(
    text: str,
    appraisal_dimensions: Sequence[str] | str | None = None,
    rationale_mode: str = "none",
) -> dict[str, Any]:
    selected_dimensions = resolve_appraisal_dimensions(
        appraisal_dimensions
    )
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty judge response")
    try:
        payload = json.loads(
            stripped,
            object_pairs_hook=object_without_duplicate_keys,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid judge JSON: {exc.msg}") from exc
    if rationale_mode not in {"required", "none"}:
        raise ValueError("rationale_mode must be 'required' or 'none'")
    root_keys = {"appraisals", "coherence", "transition"}
    if rationale_mode == "required":
        root_keys.add("overall_feedback")
    require_exact_keys(payload, root_keys, "judge.root")
    appraisals = require_exact_keys(
        payload["appraisals"],
        set(selected_dimensions),
        "judge.appraisals",
    )
    for dimension in selected_dimensions:
        criteria = require_exact_keys(
            appraisals[dimension],
            set(APPRAISAL_CRITERIA),
            f"judge.appraisals.{dimension}",
        )
        for criterion in APPRAISAL_CRITERIA:
            _validate_score_item(
                criteria[criterion],
                f"judge.appraisals.{dimension}.{criterion}",
                rationale_mode,
            )

    _validate_score_item(payload["coherence"], "judge.coherence", rationale_mode)
    _validate_score_item(payload["transition"], "judge.transition", rationale_mode)
    if rationale_mode == "required":
        if (
            not isinstance(payload["overall_feedback"], str)
            or not payload["overall_feedback"].strip()
        ):
            raise ValueError("judge.overall_feedback must be a non-empty string")
        payload["overall_feedback"] = payload["overall_feedback"].strip()
    return payload


def aggregate_judgment(
    judgment: dict[str, Any],
    appraisal_dimensions: Sequence[str] | str | None = None,
) -> dict[str, Any]:
    """Aggregate appraisal quality and situation grounding separately.

    Outcome correctness is deliberately absent here. It is calculated from the
    gold CAREBench emotion by :mod:`reward_components` rather than by the LLM
    Judge.
    """
    selected_dimensions = resolve_appraisal_dimensions(
        appraisal_dimensions
    )
    dimension_scores: dict[str, float] = {}
    ungrounded_dimension_scores: dict[str, float] = {}
    grounding_dimension_scores: dict[str, float] = {}
    effective_validity_scores: dict[str, float] = {}
    for dimension in selected_dimensions:
        validity = (
            judgment["appraisals"][dimension]["dimension_specific_validity"]["score"]
            / 4.0
        )
        fidelity = (
            judgment["appraisals"][dimension]["experiencer_fidelity"]["score"]
            / 4.0
        )
        grounding = (
            judgment["appraisals"][dimension]["situation_grounding"]["score"] / 4.0
        )
        effective_validity_scores[dimension] = validity
        ungrounded_score = (validity + fidelity) / 2.0
        ungrounded_dimension_scores[dimension] = ungrounded_score
        dimension_scores[dimension] = grounding * ungrounded_score
        grounding_dimension_scores[dimension] = grounding
    appraisal_score = sum(dimension_scores.values()) / len(dimension_scores)
    grounding_values = list(grounding_dimension_scores.values())
    weakest = sorted(grounding_values)[:2]
    grounding_mean = sum(grounding_values) / len(grounding_values)
    grounding_tail = sum(weakest) / len(weakest)
    grounding_reward = 0.7 * grounding_mean + 0.3 * grounding_tail
    appraisal_criterion_scores = {
        criterion: sum(
            judgment["appraisals"][dimension][criterion]["score"]
            for dimension in selected_dimensions
        )
        / (4.0 * len(selected_dimensions))
        for criterion in APPRAISAL_CRITERIA
    }
    appraisal_criterion_scores["dimension_specific_validity"] = (
        sum(effective_validity_scores.values()) / len(effective_validity_scores)
    )

    coherence_reward = judgment["coherence"]["score"] / 4.0
    raw_transition_reward = judgment["transition"]["score"] / 4.0
    transition_reward = raw_transition_reward
    return {
        "appraisal_reward": float(appraisal_score),
        "grounding_reward": float(grounding_reward),
        "grounding_mean": float(grounding_mean),
        "grounding_tail": float(grounding_tail),
        "grounding_dimension_scores": grounding_dimension_scores,
        "coherence_reward": float(coherence_reward),
        "transition_reward": float(transition_reward),
        "raw_transition_reward": float(raw_transition_reward),
        "dimension_scores": dimension_scores,
        "ungrounded_dimension_scores": ungrounded_dimension_scores,
        "effective_validity_scores": effective_validity_scores,
        "appraisal_criterion_scores": appraisal_criterion_scores,
        "transition_gate_components": {
            "raw_appraisal_emotion_linkage": float(raw_transition_reward),
            "mean_dimension_specific_validity": (
                appraisal_criterion_scores["dimension_specific_validity"]
            ),
            "mean_situation_grounding": appraisal_criterion_scores[
                "situation_grounding"
            ],
            "effective_transition_reward": float(transition_reward),
        },
    }
