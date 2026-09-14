#!/usr/bin/env python3
"""Automatic metrics for crowd-enVent Policy predictions."""

from __future__ import annotations

import math
import statistics
import warnings
from collections import defaultdict
from typing import Any, Iterable, Mapping

from crowd_envent_schema import (
    APPRAISAL_FIELDS,
    APPRAISAL_TO_METHOD_DIMENSION,
    EMOTION_LABELS,
    METHOD_APPRAISAL_DIMENSIONS,
)


RATING_RANGE = 4.0  # native 1--5 scale
MISSING_PREDICTION_POLICIES = ("error", "skip", "penalize")
CROWD_ENVENT_MISSING_RATING = 3
CROWD_ENVENT_MISSING_EMOTION = "no-emotion"
CROWD_ENVENT_MISSING_INTENSITY = 1
CROWD_ENVENT_POSITIVE_EMOTIONS = frozenset({"joy", "pride", "relief", "trust"})
CROWD_ENVENT_NEGATIVE_EMOTIONS = frozenset(
    {"anger", "boredom", "disgust", "fear", "guilt", "sadness", "shame", "surprise"}
)


def mean_or_none(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else None


def average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for original_index, _ in indexed[index:end]:
            ranks[original_index] = average_rank
        index = end
    return ranks


def pearson_correlation(left: list[float], right: list[float]) -> float | None:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    left_denominator = math.sqrt(
        sum((left_value - left_mean) ** 2 for left_value in left)
    )
    right_denominator = math.sqrt(
        sum((right_value - right_mean) ** 2 for right_value in right)
    )
    denominator = left_denominator * right_denominator
    if denominator == 0:
        return None
    return numerator / denominator


def fallback_spearman(gold: list[float], predicted: list[float]) -> float | None:
    return pearson_correlation(average_ranks(gold), average_ranks(predicted))


def compute_spearman(gold: list[float], predicted: list[float]) -> dict[str, Any]:
    if len(gold) != len(predicted):
        raise ValueError("Spearman inputs have different lengths")
    if len(gold) < 2:
        return {
            "correlation": None,
            "p_value": None,
            "count": len(gold),
            "defined": False,
            "reason": "fewer than two pairs",
        }
    try:
        from scipy.stats import spearmanr
    except ImportError as exc:
        correlation = fallback_spearman(gold, predicted)
        if correlation is None or math.isnan(correlation):
            return {
                "correlation": None,
                "p_value": None,
                "count": len(gold),
                "defined": False,
                "reason": "constant gold or prediction ranks",
            }
        return {
            "correlation": float(correlation),
            "p_value": None,
            "count": len(gold),
            "defined": True,
            "reason": "computed without p-value because scipy is unavailable",
        }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = spearmanr(gold, predicted)
    correlation = getattr(result, "statistic", getattr(result, "correlation", None))
    p_value = getattr(result, "pvalue", None)
    correlation = float(correlation)
    if math.isnan(correlation):
        return {
            "correlation": None,
            "p_value": None,
            "count": len(gold),
            "defined": False,
            "reason": "constant gold or prediction ranks",
        }
    numeric_p = float(p_value) if p_value is not None else None
    return {
        "correlation": correlation,
        "p_value": None if numeric_p is not None and math.isnan(numeric_p) else numeric_p,
        "count": len(gold),
        "defined": True,
        "reason": None,
    }


def regression_metrics(
    gold: list[float], predicted: list[float], *, normalize_range: float = RATING_RANGE
) -> dict[str, Any]:
    if len(gold) != len(predicted):
        raise ValueError("Regression inputs have different lengths")
    if not gold:
        return {
            "count": 0,
            "mae": None,
            "rmse": None,
            "normalized_mae": None,
            "normalized_rmse": None,
            "exact_accuracy": None,
            "spearman": compute_spearman([], []),
        }
    absolute = [abs(reference - prediction) for reference, prediction in zip(gold, predicted)]
    squared = [(reference - prediction) ** 2 for reference, prediction in zip(gold, predicted)]
    mae = sum(absolute) / len(absolute)
    rmse = math.sqrt(sum(squared) / len(squared))
    return {
        "count": len(gold),
        "mae": mae,
        "rmse": rmse,
        "normalized_mae": mae / normalize_range,
        "normalized_rmse": rmse / normalize_range,
        "exact_accuracy": sum(reference == prediction for reference, prediction in zip(gold, predicted)) / len(gold),
        "spearman": compute_spearman(gold, predicted),
    }


def classification_metrics(gold: list[str], predicted: list[str]) -> dict[str, Any]:
    if len(gold) != len(predicted):
        raise ValueError("Classification inputs have different lengths")
    if not gold:
        raise ValueError("No emotion pairs are available")
    allowed = set(EMOTION_LABELS)
    invalid = sorted((set(gold) | set(predicted)) - allowed)
    if invalid:
        raise ValueError(f"Emotion labels outside crowd-enVent label space: {invalid}")
    per_label: dict[str, dict[str, Any]] = {}
    macro_f1: list[float] = []
    weighted_f1_sum = 0.0
    correct = sum(reference == prediction for reference, prediction in zip(gold, predicted))
    for label in EMOTION_LABELS:
        tp = sum(r == label and p == label for r, p in zip(gold, predicted))
        fp = sum(r != label and p == label for r, p in zip(gold, predicted))
        fn = sum(r == label and p != label for r, p in zip(gold, predicted))
        support = tp + fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        macro_f1.append(f1)
        weighted_f1_sum += f1 * support
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted": tp + fp,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
        }
    accuracy = correct / len(gold)
    return {
        "samples": len(gold),
        "accuracy": accuracy,
        "micro_f1": accuracy,
        "macro_f1": statistics.mean(macro_f1),
        "weighted_f1": weighted_f1_sum / len(gold),
        "per_label": per_label,
        "note": "For single-label multiclass prediction, micro-F1 equals accuracy.",
    }


def emotion_candidate_hit_metrics(
    gold: list[str], predictions: list[Mapping[str, Any]]
) -> dict[str, Any]:
    """Measure whether the original label candidate set contains author gold."""
    if len(gold) != len(predictions):
        raise ValueError("Candidate-hit inputs have different lengths")
    hits = 0
    explicit_candidate_samples = 0
    single_label_fallback_samples = 0
    candidate_counts: list[float] = []
    per_label_counts = {
        label: {"support": 0, "hits": 0} for label in EMOTION_LABELS
    }
    for gold_label, prediction in zip(gold, predictions):
        ranked_candidates, source = _candidate_ranking(prediction)
        candidates = set(ranked_candidates)
        if source in {
            "direct_multilabel_labels",
            "chain_evaluation_only_ranking",
        }:
            explicit_candidate_samples += 1
        elif source == "single_label":
            single_label_fallback_samples += 1
        invalid = sorted(candidates - set(EMOTION_LABELS))
        if invalid:
            raise ValueError(
                f"Emotion candidate labels outside crowd-enVent label space: {invalid}"
            )
        hit = gold_label in candidates
        hits += int(hit)
        candidate_counts.append(float(len(candidates)))
        per_label_counts[gold_label]["support"] += 1
        per_label_counts[gold_label]["hits"] += int(hit)

    per_gold_label = {}
    for label, counts in per_label_counts.items():
        support = counts["support"]
        per_gold_label[label] = {
            **counts,
            "hit_rate": counts["hits"] / support if support else None,
        }
    return {
        "samples": len(gold),
        "hits": hits,
        "hit_rate": hits / len(gold) if gold else None,
        "mean_candidate_count": mean_or_none(candidate_counts),
        "explicit_candidate_samples": explicit_candidate_samples,
        "single_label_fallback_samples": single_label_fallback_samples,
        "per_gold_label": per_gold_label,
        "definition": (
            "Author gold label is contained anywhere in the same explicit "
            "model ranking used by Hit@k. Single-label predictions contain only "
            "their predicted label; missing predictions are always misses."
        ),
    }


def _validated_candidate_sequence(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    candidates: list[str] = []
    for label in value:
        if not isinstance(label, str) or label not in EMOTION_LABELS:
            raise ValueError(
                f"{field} contains a label outside the crowd-enVent space: {label!r}"
            )
        if label in candidates:
            raise ValueError(f"{field} contains duplicate label {label!r}")
        candidates.append(label)
    return candidates


def _candidate_ranking(prediction: Mapping[str, Any]) -> tuple[list[str], str]:
    if prediction.get("missing_prediction"):
        return [], "missing"
    primary = prediction["emotion"]["label"]
    raw_candidates = prediction.get("emotion_candidate_labels")
    schema = prediction.get("generation_schema")

    raw_direct_emotion = prediction.get("multilabel_emotion")
    if raw_direct_emotion is not None:
        if schema != "direct" or not isinstance(raw_direct_emotion, Mapping):
            raise ValueError(
                "multilabel_emotion ranking requires a direct prediction"
            )
        ranked = _validated_candidate_sequence(
            raw_direct_emotion.get("labels"), "multilabel_emotion.labels"
        )
        if ranked[0] != primary:
            raise ValueError(
                "First direct ranked label does not match the top-1 emotion"
            )
        if raw_candidates is not None:
            candidates = _validated_candidate_sequence(
                raw_candidates, "emotion_candidate_labels"
            )
            if set(candidates) != set(ranked):
                raise ValueError("Direct ranking and emotion candidate sets differ")
        return ranked, "direct_multilabel_labels"

    raw_chain_ranking = prediction.get("evaluation_only_emotion_ranking")
    if raw_chain_ranking is not None:
        if schema != "carebench":
            raise ValueError(
                "evaluation_only_emotion_ranking requires a converted chain "
                "prediction"
            )
        ranked = _validated_candidate_sequence(
            raw_chain_ranking, "evaluation_only_emotion_ranking"
        )
        if raw_candidates is not None:
            candidates = _validated_candidate_sequence(
                raw_candidates, "emotion_candidate_labels"
            )
            if set(candidates) != set(ranked):
                raise ValueError("Chain ranking and emotion candidate sets differ")
        return ranked, "chain_evaluation_only_ranking"

    if schema == "carebench" and prediction.get("carebench_emotion") is not None:
        raise ValueError(
            "Chain multi-label prediction has no evaluation_only_emotion_ranking; "
            "regenerate its chain-emotion output before computing Hit@k"
        )
    return [primary], "single_label"


def _emotion_for_metrics(prediction: Mapping[str, Any]) -> dict[str, Any]:
    ranking, source = _candidate_ranking(prediction)
    if prediction.get("missing_prediction"):
        return {
            "label": prediction["emotion"]["label"],
            "intensity": prediction["emotion"]["intensity"],
            "source": source,
        }

    label = ranking[0]
    intensity = prediction["emotion"]["intensity"]
    if source == "chain_evaluation_only_ranking" and label != "no-emotion":
        if label in CROWD_ENVENT_POSITIVE_EMOTIONS:
            valence = "positive"
        elif label in CROWD_ENVENT_NEGATIVE_EMOTIONS:
            valence = "negative"
        else:
            raise ValueError(f"Cannot determine valence for emotion label {label!r}")
        raw_valenced = prediction.get("carebench_emotion")
        if not isinstance(raw_valenced, Mapping):
            raise ValueError(
                "Chain multi-label prediction has no native valenced emotion "
                "intensities"
            )
        raw_side = raw_valenced.get(valence)
        if not isinstance(raw_side, Mapping) or "intensity" not in raw_side:
            raise ValueError(
                f"Chain multi-label prediction has no {valence} intensity"
            )
        intensity = raw_side["intensity"]
    return {"label": label, "intensity": intensity, "source": source}


def emotion_hit_at_k_metrics(
    gold: list[str], predictions: list[Mapping[str, Any]]
) -> dict[str, Any]:
    """Compute Hit@k over each prediction's ranked emotion candidates."""
    if len(gold) != len(predictions):
        raise ValueError("Hit@k inputs have different lengths")
    k_values = [1, 2, 3]
    hits_by_k = {k: 0 for k in k_values}
    source_counts = {
        "direct_multilabel_labels": 0,
        "chain_evaluation_only_ranking": 0,
        "single_label": 0,
        "missing": 0,
    }
    for gold_label, prediction in zip(gold, predictions):
        ranked_candidates, source = _candidate_ranking(prediction)
        source_counts[source] += 1
        for k in k_values:
            hits_by_k[k] += int(gold_label in ranked_candidates[:k])
    return {
        "samples": len(gold),
        "k_values": k_values,
        "by_k": {
            str(k): {
                "hits": hits_by_k[k],
                "hit_rate": hits_by_k[k] / len(gold) if gold else None,
            }
            for k in k_values
        },
        "ranking_sources": source_counts,
        "definition": (
            "Hit@k = (1/N) * sum_i 1[gold_i is in the first k ranked "
            "emotion candidates]. Candidate lists shorter than k are not padded; "
            "missing predictions are misses."
        ),
        "ranking_fields": {
            "direct_multi_label": "multilabel_emotion.labels",
            "chain_multi_label": "evaluation_only_emotion_ranking",
            "single_label": "emotion.label",
        },
        "ranking_note": (
            "Hit@k uses only model-generated explicit rankings for multi-label "
            "outputs; it never infers a ranking from valence intensity or the "
            "canonical dataset label order."
        ),
    }


def reasoning_diagnostics(predictions: list[Mapping[str, Any]]) -> dict[str, Any]:
    traces: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    samples_with_reasoning = 0
    for prediction in predictions:
        raw = prediction.get("appraisal_reasoning")
        if not isinstance(raw, dict):
            continue
        if set(raw) == set(METHOD_APPRAISAL_DIMENSIONS):
            traces["joint"].append(raw)
            samples_with_reasoning += 1
        elif set(raw) == {"appraisals", "emotion"} and all(
            isinstance(raw[target], dict) for target in ("appraisals", "emotion")
        ):
            traces["appraisals"].append(raw["appraisals"])
            traces["emotion"].append(raw["emotion"])
            samples_with_reasoning += 1
    by_trace: dict[str, dict[str, Any]] = {}
    for trace_name, reasoning_items in traces.items():
        per_dimension: dict[str, dict[str, Any]] = {}
        for dimension in METHOD_APPRAISAL_DIMENSIONS:
            texts = [
                str(reasoning.get(dimension, "")).strip()
                for reasoning in reasoning_items
            ]
            lengths = [len(text.split()) for text in texts if text]
            per_dimension[dimension] = {
                "nonempty": len(lengths),
                "mean_words": mean_or_none(float(value) for value in lengths),
            }
        by_trace[trace_name] = {
            "samples": len(reasoning_items),
            "per_dimension": per_dimension,
        }
    return {
        "samples_with_reasoning": samples_with_reasoning,
        "coverage": samples_with_reasoning / len(predictions) if predictions else 0.0,
        "by_trace": by_trace,
        "quality_note": (
            "crowd-enVent has no gold natural-language rationales; use the "
            "optional API rubric Judge for reasoning quality."
        ),
    }


def resolve_missing_prediction_policy(
    allow_incomplete: bool,
    missing_prediction_policy: str | None,
) -> str:
    if missing_prediction_policy is None:
        missing_prediction_policy = "skip" if allow_incomplete else "error"
    if missing_prediction_policy not in MISSING_PREDICTION_POLICIES:
        raise ValueError(
            f"Invalid missing prediction policy {missing_prediction_policy!r}; "
            f"allowed={list(MISSING_PREDICTION_POLICIES)}"
        )
    return missing_prediction_policy


def penalized_prediction(sample_id: str) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "generation_schema": "missing",
        "missing_prediction": True,
        "appraisal_ratings": {
            field: CROWD_ENVENT_MISSING_RATING for field in APPRAISAL_FIELDS
        },
        "emotion": {
            "label": CROWD_ENVENT_MISSING_EMOTION,
            "intensity": CROWD_ENVENT_MISSING_INTENSITY,
        },
    }


def _ratings_for_scope(
    samples: list[Mapping[str, Any]],
    predictions: list[Mapping[str, Any]],
    fields: Iterable[str],
    *,
    reference_key: str,
) -> dict[str, Any]:
    gold: list[float] = []
    predicted: list[float] = []
    for sample, prediction in zip(samples, predictions):
        if reference_key == "author":
            references = sample["reference"]["appraisal_ratings"]
        else:
            references = sample.get("validation_reference", {}).get(
                "mean_appraisal_ratings", {}
            )
        for field in fields:
            if field in references:
                gold.append(float(references[field]))
                predicted.append(float(prediction["appraisal_ratings"][field]))
    return regression_metrics(gold, predicted)


def appraisal_report(
    samples: list[Mapping[str, Any]],
    predictions: list[Mapping[str, Any]],
    *,
    reference_key: str,
) -> dict[str, Any]:
    by_source = {
        field: _ratings_for_scope(
            samples, predictions, [field], reference_key=reference_key
        )
        for field in APPRAISAL_FIELDS
    }
    grouped: dict[str, list[str]] = defaultdict(list)
    for field in APPRAISAL_FIELDS:
        grouped[APPRAISAL_TO_METHOD_DIMENSION[field]].append(field)
    by_method = {
        dimension: _ratings_for_scope(
            samples, predictions, fields, reference_key=reference_key
        )
        for dimension, fields in grouped.items()
    }
    primary_dimensions = {
        dimension: by_method[dimension]
        for dimension in METHOD_APPRAISAL_DIMENSIONS
    }
    return {
        "overall_21_ratings": _ratings_for_scope(
            samples, predictions, APPRAISAL_FIELDS, reference_key=reference_key
        ),
        "five_method_dimensions": primary_dimensions,
        "auxiliary_norm_value": by_method["auxiliary_norm_value"],
        "by_source_rating": by_source,
        "aggregation_note": (
            "Grouped metrics pool native rating pairs; standards and social_norms "
            "are reported separately because the Policy uses five dimensions."
        ),
    }


def build_metrics_report(
    selected_samples: list[Mapping[str, Any]],
    prediction_by_id: Mapping[str, Mapping[str, Any]],
    *,
    allow_incomplete: bool,
    missing_prediction_policy: str | None = None,
) -> dict[str, Any]:
    missing_prediction_policy = resolve_missing_prediction_policy(
        allow_incomplete,
        missing_prediction_policy,
    )
    missing = [sample["id"] for sample in selected_samples if sample["id"] not in prediction_by_id]
    if missing and missing_prediction_policy == "error":
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"Missing {len(missing)} prediction(s): {preview}. "
            "Pass --missing_prediction_policy penalize to keep them in the "
            "denominator, or pass --allow_incomplete/--missing_prediction_policy "
            "skip for debugging valid outputs only."
        )
    samples: list[Mapping[str, Any]] = []
    predictions: list[Mapping[str, Any]] = []
    valid_predictions = 0
    penalized_missing = 0
    for sample in selected_samples:
        prediction = prediction_by_id.get(sample["id"])
        if prediction is None:
            if missing_prediction_policy == "skip":
                continue
            prediction = penalized_prediction(sample["id"])
            penalized_missing += 1
        else:
            valid_predictions += 1
        samples.append(sample)
        predictions.append(prediction)
    if not samples:
        raise ValueError("No valid crowd-enVent predictions are available")

    author_gold_labels = [sample["reference"]["emotion"]["label"] for sample in samples]
    evaluated_emotions = [_emotion_for_metrics(prediction) for prediction in predictions]
    predicted_labels = [emotion["label"] for emotion in evaluated_emotions]
    author_intensities = [float(sample["reference"]["emotion"]["intensity"]) for sample in samples]
    predicted_intensities = [
        float(emotion["intensity"]) for emotion in evaluated_emotions
    ]
    emotion_prediction_source_counts: dict[str, int] = defaultdict(int)
    for emotion in evaluated_emotions:
        emotion_prediction_source_counts[str(emotion["source"])] += 1

    validator_gold: list[str] = []
    validator_predictions: list[str] = []
    majority_correct = 0
    majority_count = 0
    tie_count = 0
    for sample, predicted_label in zip(samples, predicted_labels):
        validation = sample.get("validation_reference", {})
        labels = validation.get("emotion_labels", [])
        majority = validation.get("majority_emotion_labels", [])
        if labels:
            validator_gold.extend(labels)
            validator_predictions.extend([predicted_label] * len(labels))
        if majority:
            majority_count += 1
            majority_correct += int(predicted_label in majority)
            tie_count += int(len(majority) > 1)

    observer_appraisal_available = any(
        sample.get("validation_reference", {}).get("mean_appraisal_ratings")
        for sample in samples
    )
    return {
        "dataset": "crowd-enVent-2023",
        "coverage": {
            "selected_samples": len(selected_samples),
            "valid_predictions": valid_predictions,
            "evaluated_samples": len(samples),
            "missing_predictions": len(missing),
            "penalized_missing_predictions": penalized_missing,
            "skipped_missing_predictions": (
                len(missing) if missing_prediction_policy == "skip" else 0
            ),
            "coverage": valid_predictions / len(selected_samples),
            "metric_sample_coverage": len(samples) / len(selected_samples),
            "allow_incomplete": allow_incomplete,
            "missing_prediction_policy": missing_prediction_policy,
            "missing_sample_ids": missing,
            "penalized_missing_sample_ids": (
                missing if missing_prediction_policy == "penalize" else []
            ),
            "penalty_defaults": (
                {
                    "appraisal_rating": CROWD_ENVENT_MISSING_RATING,
                    "emotion_label": CROWD_ENVENT_MISSING_EMOTION,
                    "emotion_intensity": CROWD_ENVENT_MISSING_INTENSITY,
                }
                if missing_prediction_policy == "penalize"
                else None
            ),
        },
        "author_self_report": {
            "appraisals": appraisal_report(samples, predictions, reference_key="author"),
            "emotion_classification": classification_metrics(
                author_gold_labels, predicted_labels
            ),
            "emotion_candidate_hit": emotion_candidate_hit_metrics(
                author_gold_labels, predictions
            ),
            "emotion_hit_at_k": emotion_hit_at_k_metrics(
                author_gold_labels, predictions
            ),
            "emotion_intensity": regression_metrics(
                author_intensities, predicted_intensities
            ),
            "emotion_prediction_source_counts": dict(
                emotion_prediction_source_counts
            ),
        },
        "observer_robustness": {
            "appraisals": (
                appraisal_report(samples, predictions, reference_key="observer")
                if observer_appraisal_available
                else None
            ),
            "emotion_against_individual_annotations": (
                classification_metrics(validator_gold, validator_predictions)
                if validator_gold
                else None
            ),
            "majority_set_accuracy": (
                majority_correct / majority_count if majority_count else None
            ),
            "majority_samples": majority_count,
            "majority_ties": tie_count,
            "note": (
                "Observer annotations are secondary because they do not represent "
                "the first-person experiencer's own appraisal or emotion."
            ),
        },
        "appraisal_reasoning": reasoning_diagnostics(predictions),
    }
