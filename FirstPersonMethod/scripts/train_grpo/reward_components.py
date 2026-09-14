"""Deterministic outcome rewards and component-wise GRPO normalization."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


REWARD_COMPONENT_FIELDS = (
    "appraisal_reward",
    "coherence_reward",
    "transition_reward",
    "outcome_label_reward",
    "outcome_intensity_reward",
    "outcome_reward",
)

# Label/intensity rewards are retained as separately logged diagnostics. Their
# process-gated combination is the outcome component used for optimization, so
# adding the two diagnostics again would count the outcome twice.
OPTIMIZATION_REWARD_FIELDS = (
    "appraisal_reward",
    "coherence_reward",
    "transition_reward",
    "outcome_reward",
)


def label_set_f1(predicted: Sequence[str], gold: Sequence[str]) -> float:
    """Example-level set F1, with an exact empty/empty prediction worth 1."""
    predicted_set = set(predicted)
    gold_set = set(gold)
    if not predicted_set and not gold_set:
        return 1.0
    if not predicted_set or not gold_set:
        return 0.0
    true_positive = len(predicted_set & gold_set)
    return (2.0 * true_positive) / (len(predicted_set) + len(gold_set))


def intensity_similarity(predicted: int, gold: int) -> float:
    """Convert CAREBench's 0--6 absolute intensity error to a [0, 1] score."""
    return 1.0 - abs(int(predicted) - int(gold)) / 6.0


def score_gold_emotion(
    candidate_emotion: Mapping[str, Any],
    gold_emotion: Mapping[str, Any],
) -> dict[str, float]:
    """Score the four requested outcome fields directly against gold."""
    positive_label_score = label_set_f1(
        candidate_emotion["positive_labels"],
        gold_emotion["positive_labels"],
    )
    negative_label_score = label_set_f1(
        candidate_emotion["negative_labels"],
        gold_emotion["negative_labels"],
    )
    positive_intensity_score = intensity_similarity(
        candidate_emotion["positive_intensity"],
        gold_emotion["positive_intensity"],
    )
    negative_intensity_score = intensity_similarity(
        candidate_emotion["negative_intensity"],
        gold_emotion["negative_intensity"],
    )
    return {
        "positive_label_score": float(positive_label_score),
        "negative_label_score": float(negative_label_score),
        "positive_intensity_score": float(positive_intensity_score),
        "negative_intensity_score": float(negative_intensity_score),
        "outcome_label_reward": float(
            (positive_label_score + negative_label_score) / 2.0
        ),
        "outcome_intensity_reward": float(
            (positive_intensity_score + negative_intensity_score) / 2.0
        ),
    }


def compute_process_gate(
    appraisal_reward: float,
    transition_reward: float,
    mode: str,
) -> float:
    """Return a soft [0, 1] gate controlled by process and transition quality."""
    appraisal = min(1.0, max(0.0, float(appraisal_reward)))
    transition = min(1.0, max(0.0, float(transition_reward)))
    if mode == "min":
        return min(appraisal, transition)
    if mode == "product":
        return appraisal * transition
    if mode == "geometric_mean":
        return math.sqrt(appraisal * transition)
    raise ValueError(f"Unknown process gate mode: {mode}")


def build_reward_components(
    process_scores: Mapping[str, Any],
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    outcome_label_weight: float,
    outcome_intensity_weight: float,
    use_process_gate: bool,
    process_gate_mode: str,
) -> tuple[dict[str, float], dict[str, float]]:
    """Build the six persisted components plus atomic outcome diagnostics."""
    gold_emotion = reference.get("gold_emotion")
    if not isinstance(gold_emotion, dict):
        raise ValueError("internal reference is missing gold_emotion")
    candidate_emotion = candidate.get("emotion")
    if not isinstance(candidate_emotion, dict):
        raise ValueError("parsed candidate is missing emotion")

    outcome = score_gold_emotion(candidate_emotion, gold_emotion)
    denominator = outcome_label_weight + outcome_intensity_weight
    if denominator <= 0:
        raise ValueError("Outcome label + intensity weights must be positive")
    ungated_outcome_reward = (
        outcome_label_weight * outcome["outcome_label_reward"]
        + outcome_intensity_weight * outcome["outcome_intensity_reward"]
    ) / denominator
    process_gate = (
        compute_process_gate(
            float(process_scores["appraisal_reward"]),
            float(process_scores["transition_reward"]),
            process_gate_mode,
        )
        if use_process_gate
        else 1.0
    )
    components = {
        "appraisal_reward": float(process_scores["appraisal_reward"]),
        "coherence_reward": float(process_scores["coherence_reward"]),
        "transition_reward": float(process_scores["transition_reward"]),
        "outcome_label_reward": outcome["outcome_label_reward"],
        "outcome_intensity_reward": outcome["outcome_intensity_reward"],
        "outcome_reward": float(process_gate * ungated_outcome_reward),
    }
    diagnostics = {
        key: float(value) for key, value in outcome.items()
    }
    diagnostics.update(
        {
            "ungated_outcome_reward": float(ungated_outcome_reward),
            "process_gate": float(process_gate),
            "process_gate_enabled": float(bool(use_process_gate)),
        }
    )
    return components, diagnostics


def invalid_reward_components(value: float) -> dict[str, float]:
    return {field: float(value) for field in REWARD_COMPONENT_FIELDS}


def group_normalize_components(
    component_rows: Sequence[Mapping[str, float] | None],
    group_size: int,
    optimization_weights: Mapping[str, float],
    process_gates: Sequence[float | None] | None = None,
    outcome_label_weight: float = 0.5,
    outcome_intensity_weight: float = 0.5,
    epsilon: float = 1.0e-4,
) -> tuple[list[dict[str, float] | None], list[float | None]]:
    """Build component-wise normalized signals for each contiguous GRPO group.

    Missing rows represent Judge/API failures and remain excluded. Sample
    standard deviation matches the usual GRPO group-normalization convention.
    Process rewards and both atomic outcome subrewards are standardized first;
    the process gate is applied to normalized outcome afterward so its magnitude
    cannot be canceled by standardization. The trainer receives the weighted
    average of the four non-duplicative optimization components; no raw
    ``total_reward`` is persisted.
    """
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if epsilon <= 0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be positive and finite")
    if len(component_rows) % group_size != 0:
        raise ValueError(
            f"reward batch size {len(component_rows)} is not divisible by "
            f"num_generations={group_size}"
        )
    if process_gates is None:
        process_gates = [
            None if row is None else 1.0 for row in component_rows
        ]
    if len(process_gates) != len(component_rows):
        raise ValueError("process_gates must align with component_rows")
    outcome_weight_denominator = (
        outcome_label_weight + outcome_intensity_weight
    )
    if outcome_weight_denominator <= 0:
        raise ValueError("Outcome label + intensity weights must be positive")
    weight_denominator = sum(
        float(optimization_weights[field])
        for field in OPTIMIZATION_REWARD_FIELDS
    )
    if weight_denominator <= 0:
        raise ValueError("Optimization reward weights must sum to a positive value")

    normalized_rows: list[dict[str, float] | None] = [
        None for _ in component_rows
    ]
    for group_start in range(0, len(component_rows), group_size):
        group_indices = range(group_start, group_start + group_size)
        # The two atomic outcome components are normalized before gating. This
        # ordering is essential: normalizing a pre-gated outcome would cancel a
        # gate that is similar across all candidates in a group.
        for field in REWARD_COMPONENT_FIELDS[:-1]:
            available = [
                (index, float(component_rows[index][field]))
                for index in group_indices
                if component_rows[index] is not None
            ]
            if not available:
                continue
            mean = sum(value for _, value in available) / len(available)
            if len(available) <= 1:
                standard_deviation = 0.0
            else:
                variance = sum(
                    (value - mean) ** 2 for _, value in available
                ) / (len(available) - 1)
                standard_deviation = math.sqrt(variance)
            for index, value in available:
                if normalized_rows[index] is None:
                    normalized_rows[index] = {}
                normalized_rows[index][field] = (
                    0.0
                    if standard_deviation <= epsilon
                    else (value - mean) / (standard_deviation + epsilon)
                )

    for index, normalized in enumerate(normalized_rows):
        if normalized is None:
            continue
        gate = process_gates[index]
        if gate is None:
            raise ValueError("A scored reward row must have a process gate")
        normalized_base_outcome = (
            outcome_label_weight * normalized["outcome_label_reward"]
            + outcome_intensity_weight
            * normalized["outcome_intensity_reward"]
        ) / outcome_weight_denominator
        normalized["outcome_reward"] = float(
            float(gate) * normalized_base_outcome
        )

    training_signals: list[float | None] = []
    for normalized in normalized_rows:
        if normalized is None:
            training_signals.append(None)
            continue
        signal = sum(
            float(optimization_weights[field]) * normalized[field]
            for field in OPTIMIZATION_REWARD_FIELDS
        ) / weight_denominator
        training_signals.append(float(signal))
    return normalized_rows, training_signals
