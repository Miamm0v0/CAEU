"""Dependency-free CAREBench diagnostics for GRPO generation batches."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Mapping, Sequence


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def _ngrams(tokens: Sequence[str], size: int) -> list[tuple[str, ...]]:
    if size <= 0 or len(tokens) < size:
        return []
    return [
        tuple(tokens[index : index + size])
        for index in range(len(tokens) - size + 1)
    ]


def bleu_score(reference: str, prediction: str, max_n: int = 4) -> float:
    reference_tokens = _tokens(reference)
    prediction_tokens = _tokens(prediction)
    if not reference_tokens or not prediction_tokens:
        return 0.0
    precisions: list[float] = []
    for size in range(1, max_n + 1):
        prediction_ngrams = _ngrams(prediction_tokens, size)
        reference_ngrams = _ngrams(reference_tokens, size)
        if not prediction_ngrams:
            return 0.0
        prediction_counts = Counter(prediction_ngrams)
        reference_counts = Counter(reference_ngrams)
        overlap = sum(
            min(count, reference_counts[gram])
            for gram, count in prediction_counts.items()
        )
        precision = _safe_div(overlap, sum(prediction_counts.values()))
        if precision <= 0.0:
            return 0.0
        precisions.append(precision)
    brevity_penalty = (
        1.0
        if len(prediction_tokens) > len(reference_tokens)
        else math.exp(1.0 - len(reference_tokens) / len(prediction_tokens))
    )
    return brevity_penalty * math.exp(
        sum(math.log(value) for value in precisions) / max_n
    )


def rouge_l_f1(reference: str, prediction: str) -> float:
    reference_tokens = _tokens(reference)
    prediction_tokens = _tokens(prediction)
    if not reference_tokens or not prediction_tokens:
        return 0.0
    previous = [0] * (len(prediction_tokens) + 1)
    for reference_token in reference_tokens:
        current = [0]
        for index, prediction_token in enumerate(prediction_tokens, start=1):
            if reference_token == prediction_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    longest = previous[-1]
    precision = longest / len(prediction_tokens)
    recall = longest / len(reference_tokens)
    return _safe_div(2.0 * precision * recall, precision + recall)


def _label_f1(gold: Sequence[str], prediction: Sequence[str]) -> float:
    gold_set = set(gold)
    prediction_set = set(prediction)
    if not gold_set and not prediction_set:
        return 1.0
    return _safe_div(
        2.0 * len(gold_set & prediction_set),
        len(gold_set) + len(prediction_set),
    )


def carebench_metric_summary(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Compute metrics conditional on strict-format-valid Policy outputs."""
    appraisal_bleu: list[float] = []
    appraisal_rouge_l: list[float] = []
    positive_f1: list[float] = []
    negative_f1: list[float] = []
    positive_exact: list[float] = []
    negative_exact: list[float] = []
    positive_intensity_errors: list[float] = []
    negative_intensity_errors: list[float] = []
    valid_candidates = 0

    for result in results:
        candidate = result.get("candidate")
        reference = result.get("reference")
        if not result.get("valid") or not isinstance(candidate, Mapping):
            continue
        if not isinstance(reference, Mapping):
            continue
        valid_candidates += 1

        prediction_reasoning = candidate.get("appraisal_reasoning")
        gold_reasoning = reference.get(
            "metric_human_appraisal_reasoning",
            reference.get("legacy_human_appraisal_reasoning"),
        )
        if isinstance(prediction_reasoning, Mapping) and isinstance(
            gold_reasoning, Mapping
        ):
            for dimension, gold_text in gold_reasoning.items():
                prediction_text = prediction_reasoning.get(dimension)
                if (
                    isinstance(gold_text, str)
                    and gold_text.strip()
                    and isinstance(prediction_text, str)
                    and prediction_text.strip()
                ):
                    appraisal_bleu.append(bleu_score(gold_text, prediction_text))
                    appraisal_rouge_l.append(
                        rouge_l_f1(gold_text, prediction_text)
                    )

        prediction_emotion = candidate.get("emotion")
        gold_emotion = reference.get("gold_emotion")
        if not isinstance(prediction_emotion, Mapping) or not isinstance(
            gold_emotion, Mapping
        ):
            continue
        predicted_positive = prediction_emotion.get("positive_labels", [])
        predicted_negative = prediction_emotion.get("negative_labels", [])
        gold_positive = gold_emotion.get("positive_labels", [])
        gold_negative = gold_emotion.get("negative_labels", [])
        positive_f1.append(_label_f1(gold_positive, predicted_positive))
        negative_f1.append(_label_f1(gold_negative, predicted_negative))
        positive_exact.append(
            float(set(gold_positive) == set(predicted_positive))
        )
        negative_exact.append(
            float(set(gold_negative) == set(predicted_negative))
        )
        positive_intensity_errors.append(
            float(
                prediction_emotion["positive_intensity"]
                - gold_emotion["positive_intensity"]
            )
        )
        negative_intensity_errors.append(
            float(
                prediction_emotion["negative_intensity"]
                - gold_emotion["negative_intensity"]
            )
        )

    metrics: dict[str, float] = {
        "carebench_valid_candidate_coverage": _safe_div(
            valid_candidates, len(results)
        ),
        "carebench_appraisal_reference_pairs": float(len(appraisal_rouge_l)),
        "carebench_emotion_reference_samples": float(len(positive_f1)),
    }
    optional_metrics = {
        "carebench_appraisal_bleu": _mean(appraisal_bleu),
        "carebench_appraisal_rouge_l": _mean(appraisal_rouge_l),
        "carebench_positive_example_f1": _mean(positive_f1),
        "carebench_negative_example_f1": _mean(negative_f1),
        "carebench_positive_label_exact": _mean(positive_exact),
        "carebench_negative_label_exact": _mean(negative_exact),
        "carebench_positive_intensity_rmse": (
            math.sqrt(
                sum(value * value for value in positive_intensity_errors)
                / len(positive_intensity_errors)
            )
            if positive_intensity_errors
            else None
        ),
        "carebench_negative_intensity_rmse": (
            math.sqrt(
                sum(value * value for value in negative_intensity_errors)
                / len(negative_intensity_errors)
            )
            if negative_intensity_errors
            else None
        ),
    }
    metrics.update(
        {
            name: float(value)
            for name, value in optional_metrics.items()
            if value is not None
        }
    )
    return metrics
