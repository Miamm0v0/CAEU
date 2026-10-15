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
    APPRAISAL_INSUFFICIENT_VALUE,
    EVIDENCE_TOP_LEVEL_KEYS,
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
    *,
    require_evidence: bool = False,
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
    has_evidence = isinstance(payload, dict) and bool(
        EVIDENCE_TOP_LEVEL_KEYS.intersection(payload)
        or (
            isinstance(payload.get("appraisal_reasoning"), dict)
            and any(isinstance(value, dict) for value in payload["appraisal_reasoning"].values())
        )
    )
    expected_keys = EXPECTED_TOP_LEVEL_KEYS | (
        EVIDENCE_TOP_LEVEL_KEYS if require_evidence or has_evidence else set()
    )
    require_exact_keys(payload, expected_keys, "root")

    reasoning = require_exact_keys(
        payload["appraisal_reasoning"],
        set(selected_dimensions),
        "appraisal_reasoning",
    )
    for dimension in selected_dimensions:
        value = reasoning[dimension]
        location = f"appraisal_reasoning.{dimension}"
        if require_evidence or has_evidence:
            item = require_exact_keys(value, {"evidence", "appraisal"}, location)
            quotes = require_quote_list(item["evidence"], f"{location}.evidence")
            if len(quotes) > 2:
                raise ValueError(f"{location}.evidence allows at most two quotes")
            appraisal = item["appraisal"]
            if not isinstance(appraisal, str) or not appraisal.strip():
                raise ValueError(f"{location}.appraisal must be a non-empty string")
            reasoning[dimension] = {"evidence": quotes, "appraisal": appraisal.strip()}
        elif not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"appraisal_reasoning.{dimension} must be a non-empty string"
            )
        else:
            reasoning[dimension] = value.strip()

    if require_evidence or has_evidence:
        emotion_evidence = require_exact_keys(
            payload["emotion_evidence"],
            {"situation_quotes", "appraisal_dimensions", "inference"},
            "emotion_evidence",
        )
        emotion_evidence["situation_quotes"] = require_quote_list(
            emotion_evidence["situation_quotes"], "emotion_evidence.situation_quotes"
        )
        used = emotion_evidence["appraisal_dimensions"]
        if (
            not isinstance(used, list)
            or any(not isinstance(d, str) or d not in selected_dimensions for d in used)
            or len(set(used)) != len(used)
        ):
            raise ValueError("emotion_evidence.appraisal_dimensions must list unique selected dimensions")
        inference = emotion_evidence["inference"]
        if not isinstance(inference, str) or not inference.strip():
            raise ValueError("emotion_evidence.inference must be non-empty text")
        emotion_evidence["inference"] = inference.strip()

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


def require_quote_list(value: Any, location: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(quote, str) or not quote.strip() for quote in value
    ):
        raise ValueError(f"{location} must be an array of non-empty quotes")
    return [quote.strip() for quote in value]


def verify_evidence(situation: str, candidate: dict[str, Any]) -> dict[str, Any]:
    """Check quotation provenance; semantic entailment remains the Judge's task."""
    source = " ".join(situation.split())

    def quotes_match(quotes: list[str]) -> bool:
        return all(" ".join(quote.split()) in source for quote in quotes)

    dimensions: dict[str, dict[str, Any]] = {}
    for dimension, answer in candidate["appraisal_reasoning"].items():
        appraisal = answer.get("appraisal", "") if isinstance(answer, dict) else answer
        quotes = answer.get("evidence", []) if isinstance(answer, dict) else []
        abstained = appraisal == APPRAISAL_INSUFFICIENT_VALUE
        valid = (not quotes if abstained else bool(quotes)) and quotes_match(quotes)
        dimensions[dimension] = {
            "valid": valid,
            "abstained": abstained,
            "quotes_match": quotes_match(quotes),
            "quote_count": len(quotes),
        }
    emotion_evidence = candidate.get("emotion_evidence", {})
    emotion_quotes = emotion_evidence.get("situation_quotes", [])
    used = emotion_evidence.get("appraisal_dimensions", [])
    has_emotion = any(candidate["emotion"][key] for key in EXPECTED_EMOTION_KEYS)
    emotion_valid = (
        bool(emotion_evidence)
        and quotes_match(emotion_quotes)
        and all(
            dimensions[d]["valid"] and not dimensions[d]["abstained"] for d in used
        )
        and (not has_emotion or bool(emotion_quotes or used))
    )
    return {"dimensions": dimensions, "emotion": {"valid": emotion_valid}}


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
    evidence_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate appraisal quality and evidence grounding separately.

    Outcome correctness is deliberately absent here. It is calculated from the
    gold CAREBench emotion by :mod:`reward_components` rather than by the LLM
    Judge.
    """
    selected_dimensions = resolve_appraisal_dimensions(
        appraisal_dimensions
    )
    dimension_scores: dict[str, float] = {}
    grounding_dimension_scores: dict[str, float] = {}
    for dimension in selected_dimensions:
        raw_scores = [
            judgment["appraisals"][dimension][criterion]["score"]
            for criterion in ("dimension_specific_validity", "experiencer_fidelity")
        ]
        dimension_scores[dimension] = sum(raw_scores) / (
            4.0 * len(raw_scores)
        )
        grounding = (
            judgment["appraisals"][dimension]["situation_grounding"]["score"] / 4.0
        )
        if (
            evidence_diagnostics is not None
            and not evidence_diagnostics["dimensions"][dimension]["valid"]
        ):
            grounding = 0.0
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

    coherence_reward = judgment["coherence"]["score"] / 4.0
    raw_transition_reward = judgment["transition"]["score"] / 4.0
    transition_reward = raw_transition_reward
    if (
        evidence_diagnostics is not None
        and not evidence_diagnostics["emotion"]["valid"]
    ):
        transition_reward = 0.0
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
        "appraisal_criterion_scores": appraisal_criterion_scores,
        # Quote provenance caps linkage; grounding multiplication is applied later.
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
