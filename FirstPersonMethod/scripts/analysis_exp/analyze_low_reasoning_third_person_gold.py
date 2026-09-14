#!/usr/bin/env python3
"""Compare low-scoring CAREBench reasoning with third-person references.

The same model-authored five-dimensional appraisal profile is scored against
the original participant's first-person answers and against every available
third-person annotator answer. The primary third-person result averages over
references; best-reference scores are reported separately as an optimistic
upper bound.

This tests whether reference mismatch may explain low CAREBench reasoning
scores. CAREBench third-person annotators were instructed to imagine being the
person in the story, so the experiment concerns annotator perspective rather
than grammatical first-person versus third-person wording.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Sequence

try:
    from .analyze_emotion_correct_appraisals import (
        bootstrap_mean_ci,
        extract_situation,
        gold_core_appraisal,
        lcs_length,
        load_gold,
        load_json,
        safe_div,
        safe_mean,
        tokenize_text,
        write_csv,
    )
except ImportError:
    from analyze_emotion_correct_appraisals import (
        bootstrap_mean_ci,
        extract_situation,
        gold_core_appraisal,
        lcs_length,
        load_gold,
        load_json,
        safe_div,
        safe_mean,
        tokenize_text,
        write_csv,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_DIMENSIONS = (
    "relevance",
    "congruence",
    "accountability",
    "control",
    "certainty",
)
LEXICAL_METRICS = ("bleu", "rouge-1", "rouge-2", "rouge-l")
POLICY_DIMENSION_FOR_CORE = {
    "relevance": "relevance",
    "congruence": "goal_congruence",
    "accountability": "agency_accountability",
    "control": "control_coping_potential",
    "certainty": "epistemic",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rescore low CAREBench appraisal reasoning examples against "
            "third-person annotations."
        )
    )
    parser.add_argument(
        "--first_person_gold",
        type=Path,
        required=True,
        help="First-person JSON file or per-sample folder.",
    )
    parser.add_argument(
        "--third_person_gold",
        type=Path,
        required=True,
        help="third_person.json or a folder of per-sample annotator lists.",
    )
    parser.add_argument(
        "--pred_root",
        type=Path,
        required=True,
        help="Prediction run containing core-appraisals/ or chain-emotion/.",
    )
    parser.add_argument(
        "--prediction_source",
        choices=("auto", "core-appraisals", "chain-emotion"),
        default="auto",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Defaults to <pred_root>/analysis_low_reasoning_third_person_gold.",
    )
    parser.add_argument(
        "--selection_mode",
        choices=("threshold", "bottom-quantile"),
        default="threshold",
        help="How first-person low-scoring samples are selected.",
    )
    parser.add_argument(
        "--selection_metric",
        choices=(*LEXICAL_METRICS, "bertscore"),
        default="rouge-l",
    )
    parser.add_argument(
        "--low_threshold",
        type=float,
        default=0.2,
        help="Select sample-level first-person scores below this value.",
    )
    parser.add_argument(
        "--bottom_fraction",
        type=float,
        default=0.25,
        help="Fraction selected when --selection_mode bottom-quantile.",
    )
    parser.add_argument(
        "--min_dimension_coverage",
        type=float,
        default=1.0,
        help="Required fraction of the five core dimensions per sample.",
    )
    parser.add_argument(
        "--min_third_person_references",
        type=int,
        default=1,
        help="Minimum usable third-person references for every dimension.",
    )
    parser.add_argument(
        "--bertscore_model",
        type=str,
        default="",
        help=(
            "Optional BERTScore model path/id. Required when selection_metric "
            "is bertscore."
        ),
    )
    parser.add_argument(
        "--bertscore_num_layers",
        type=int,
        default=17,
        help="Explicit layer count, useful for local roberta-large paths.",
    )
    parser.add_argument("--bertscore_batch_size", type=int, default=32)
    parser.add_argument(
        "--bertscore_device",
        type=str,
        default="auto",
        help="auto, cpu, cuda, or cuda:N.",
    )
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--permutation_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def make_ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def bleu_score(reference: str, prediction: str, max_n: int = 4) -> float:
    reference_tokens = tokenize_text(reference)
    prediction_tokens = tokenize_text(prediction)
    if not reference_tokens or not prediction_tokens:
        return 0.0
    precisions: list[float] = []
    for n in range(1, max_n + 1):
        prediction_ngrams = make_ngrams(prediction_tokens, n)
        reference_ngrams = make_ngrams(reference_tokens, n)
        if not prediction_ngrams:
            return 0.0
        prediction_counts = collections.Counter(prediction_ngrams)
        reference_counts = collections.Counter(reference_ngrams)
        overlap = sum(
            min(count, reference_counts[gram])
            for gram, count in prediction_counts.items()
        )
        precision = safe_div(overlap, sum(prediction_counts.values()))
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


def rouge_n_f1(reference: str, prediction: str, n: int) -> float:
    reference_ngrams = make_ngrams(tokenize_text(reference), n)
    prediction_ngrams = make_ngrams(tokenize_text(prediction), n)
    if not reference_ngrams or not prediction_ngrams:
        return 0.0
    reference_counts = collections.Counter(reference_ngrams)
    prediction_counts = collections.Counter(prediction_ngrams)
    overlap = sum(
        min(count, prediction_counts[gram])
        for gram, count in reference_counts.items()
    )
    precision = safe_div(overlap, sum(prediction_counts.values()))
    recall = safe_div(overlap, sum(reference_counts.values()))
    return safe_div(2.0 * precision * recall, precision + recall)


def rouge_l_f1(reference: str, prediction: str) -> float:
    reference_tokens = tokenize_text(reference)
    prediction_tokens = tokenize_text(prediction)
    if not reference_tokens or not prediction_tokens:
        return 0.0
    overlap = lcs_length(reference_tokens, prediction_tokens)
    precision = safe_div(overlap, len(prediction_tokens))
    recall = safe_div(overlap, len(reference_tokens))
    return safe_div(2.0 * precision * recall, precision + recall)


def lexical_scores(reference: str, prediction: str) -> dict[str, float]:
    return {
        "bleu": bleu_score(reference, prediction),
        "rouge-1": rouge_n_f1(reference, prediction, 1),
        "rouge-2": rouge_n_f1(reference, prediction, 2),
        "rouge-l": rouge_l_f1(reference, prediction),
    }


def load_third_person(path: Path) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    if path.is_file():
        payload = load_json(path)
        if not isinstance(payload, dict):
            raise ValueError("third_person_gold JSON must be sample_id -> items")
        candidates = payload.items()
    elif path.is_dir():
        candidates = (
            (item_path.stem, load_json(item_path))
            for item_path in sorted(path.glob("*.json"))
        )
    else:
        raise FileNotFoundError(f"third_person_gold not found: {path}")

    for sample_id, raw_items in candidates:
        if isinstance(raw_items, dict):
            raw_items = [raw_items]
        if not isinstance(raw_items, list):
            continue
        items = [item for item in raw_items if isinstance(item, dict)]
        if items:
            output[str(sample_id)] = items
    if not output:
        raise ValueError(f"No third-person annotations found in: {path}")
    return output


def available_prediction_source(root: Path, requested: str) -> str | None:
    if requested != "auto":
        return requested if (root / requested).is_dir() else None
    for source in ("core-appraisals", "chain-emotion"):
        if (root / source).is_dir():
            return source
    return None


def resolve_prediction_root(path: Path, requested: str) -> tuple[Path, str]:
    source = available_prediction_source(path, requested)
    if source:
        return path, source
    if not path.is_dir():
        raise FileNotFoundError(f"pred_root not found: {path}")
    candidates: list[tuple[Path, str]] = []
    for child in sorted(path.iterdir()):
        child_source = available_prediction_source(child, requested)
        if child_source:
            candidates.append((child, child_source))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No core-appraisals/ or chain-emotion/ predictions under: {path}"
        )
    raise ValueError(
        "Multiple prediction runs found; select one --pred_root: "
        + ", ".join(str(candidate[0]) for candidate in candidates)
    )


def core_prediction(payload: Any, source: str, dimension: str) -> str | None:
    if not isinstance(payload, dict):
        return None
    if source == "core-appraisals":
        value = payload.get(dimension)
        if isinstance(value, dict):
            value = value.get("answer")
    else:
        reasoning = payload.get("appraisal_reasoning")
        if not isinstance(reasoning, dict):
            return None
        value = reasoning.get(
            POLICY_DIMENSION_FOR_CORE[dimension], reasoning.get(dimension)
        )
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def annotator_id(item: dict[str, Any], index: int) -> str:
    value = item.get("participant_id", item.get("id"))
    if isinstance(value, str) and value.strip():
        return value.strip()
    return f"annotator_{index}"


def build_observations(
    *,
    first_person: dict[str, dict[str, Any]],
    third_person: dict[str, list[dict[str, Any]]],
    prediction_root: Path,
    prediction_source: str,
    min_third_references: int,
    min_dimension_coverage: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    coverage = {
        "first_person_samples": len(first_person),
        "third_person_samples": len(third_person),
        "matched_complete_samples": 0,
        "missing_third_person_sample": 0,
        "missing_prediction_file": 0,
        "insufficient_dimension_coverage": 0,
    }
    required_dimensions = max(
        1, math.ceil(len(CORE_DIMENSIONS) * min_dimension_coverage)
    )
    for sample_id, first_item in sorted(first_person.items()):
        third_items = third_person.get(sample_id)
        if not third_items:
            coverage["missing_third_person_sample"] += 1
            continue
        prediction_path = (
            prediction_root / prediction_source / f"{sample_id}.json"
        )
        if not prediction_path.is_file():
            coverage["missing_prediction_file"] += 1
            continue
        prediction = load_json(prediction_path)
        observations: list[dict[str, Any]] = []
        for dimension in CORE_DIMENSIONS:
            first_text = gold_core_appraisal(first_item, dimension)
            predicted_text = core_prediction(
                prediction, prediction_source, dimension
            )
            if first_text is None or predicted_text is None:
                continue
            third_references = []
            for index, third_item in enumerate(third_items):
                third_text = gold_core_appraisal(third_item, dimension)
                if third_text is None:
                    continue
                third_references.append(
                    {
                        "annotator_id": annotator_id(third_item, index),
                        "text": third_text,
                        "metrics": lexical_scores(third_text, predicted_text),
                    }
                )
            if len(third_references) < min_third_references:
                continue
            observations.append(
                {
                    "dimension": dimension,
                    "prediction": predicted_text,
                    "first_person_reference": first_text,
                    "first_person_metrics": lexical_scores(
                        first_text, predicted_text
                    ),
                    "third_person_references": third_references,
                }
            )
        if len(observations) < required_dimensions:
            coverage["insufficient_dimension_coverage"] += 1
            continue
        records.append(
            {
                "sample_id": sample_id,
                "situation": extract_situation(first_item),
                "dimension_coverage": safe_div(
                    len(observations), len(CORE_DIMENSIONS)
                ),
                "observations": observations,
            }
        )
        coverage["matched_complete_samples"] += 1
    return records, coverage


def add_bertscore(
    records: Sequence[dict[str, Any]],
    *,
    model_type: str,
    num_layers: int,
    batch_size: int,
    device: str,
) -> None:
    try:
        from bert_score import BERTScorer
    except ImportError as exc:
        raise RuntimeError(
            "BERTScore requested but bert-score is not installed"
        ) from exc

    resolved_device = device
    if device == "auto":
        try:
            import torch

            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            resolved_device = "cpu"
    kwargs: dict[str, Any] = {
        "model_type": model_type,
        "lang": "en",
        "device": resolved_device,
    }
    if num_layers > 0:
        kwargs["num_layers"] = num_layers
    scorer = BERTScorer(**kwargs)

    pending: list[tuple[dict[str, Any], str, str]] = []
    for record in records:
        for observation in record["observations"]:
            pending.append(
                (
                    observation["first_person_metrics"],
                    observation["prediction"],
                    observation["first_person_reference"],
                )
            )
            for reference in observation["third_person_references"]:
                pending.append(
                    (
                        reference["metrics"],
                        observation["prediction"],
                        reference["text"],
                    )
                )
    for start in range(0, len(pending), max(1, batch_size)):
        batch = pending[start : start + max(1, batch_size)]
        candidates = [item[1] for item in batch]
        references = [item[2] for item in batch]
        _, _, f1 = scorer.score(
            candidates,
            references,
            verbose=False,
            batch_size=max(1, batch_size),
        )
        for (target, _, _), score in zip(batch, f1):
            target["bertscore"] = float(score)


def metric_names(include_bertscore: bool) -> tuple[str, ...]:
    return (*LEXICAL_METRICS, "bertscore") if include_bertscore else LEXICAL_METRICS


def observation_values(
    observation: dict[str, Any], metric: str
) -> dict[str, float]:
    first_score = float(observation["first_person_metrics"][metric])
    third_scores = [
        float(reference["metrics"][metric])
        for reference in observation["third_person_references"]
    ]
    return {
        "first": first_score,
        "third_mean": float(safe_mean(third_scores) or 0.0),
        "third_best": max(third_scores),
        "third_median": float(statistics.median(third_scores)),
    }


def add_sample_scores(
    records: Sequence[dict[str, Any]], metrics: Sequence[str]
) -> None:
    for record in records:
        scores: dict[str, dict[str, float]] = {}
        for metric in metrics:
            values = [
                observation_values(observation, metric)
                for observation in record["observations"]
            ]
            first = float(safe_mean([value["first"] for value in values]) or 0.0)
            third_mean = float(
                safe_mean([value["third_mean"] for value in values]) or 0.0
            )
            third_best = float(
                safe_mean([value["third_best"] for value in values]) or 0.0
            )
            scores[metric] = {
                "first_person": first,
                "third_person_mean_reference": third_mean,
                "third_person_best_reference": third_best,
                "mean_reference_uplift": third_mean - first,
                "best_reference_uplift": third_best - first,
            }
        record["scores"] = scores


def select_low_records(
    records: Sequence[dict[str, Any]],
    *,
    metric: str,
    mode: str,
    threshold: float,
    bottom_fraction: float,
) -> tuple[list[dict[str, Any]], float | None]:
    ordered = sorted(
        records,
        key=lambda record: (
            record["scores"][metric]["first_person"],
            record["sample_id"],
        ),
    )
    if mode == "threshold":
        selected = [
            record
            for record in ordered
            if record["scores"][metric]["first_person"] < threshold
        ]
        return selected, threshold
    count = min(len(ordered), max(1, math.ceil(len(ordered) * bottom_fraction)))
    selected = ordered[:count]
    cutoff = (
        selected[-1]["scores"][metric]["first_person"] if selected else None
    )
    return selected, cutoff


def sign_flip_p_value(
    differences: Sequence[float], samples: int, rng: random.Random
) -> float | None:
    if not differences or samples <= 0:
        return None
    observed = abs(sum(differences) / len(differences))
    extreme = 0
    for _ in range(samples):
        estimate = abs(
            sum(value if rng.random() < 0.5 else -value for value in differences)
            / len(differences)
        )
        if estimate + 1e-12 >= observed:
            extreme += 1
    return (extreme + 1.0) / (samples + 1.0)


def summarize_metric(
    records: Sequence[dict[str, Any]],
    metric: str,
    bootstrap_samples: int,
    permutation_samples: int,
    rng: random.Random,
) -> dict[str, Any]:
    first_scores = [record["scores"][metric]["first_person"] for record in records]
    third_means = [
        record["scores"][metric]["third_person_mean_reference"]
        for record in records
    ]
    third_bests = [
        record["scores"][metric]["third_person_best_reference"]
        for record in records
    ]
    mean_differences = [
        third - first for first, third in zip(first_scores, third_means)
    ]
    best_differences = [
        third - first for first, third in zip(first_scores, third_bests)
    ]
    pair_values = [
        observation_values(observation, metric)
        for record in records
        for observation in record["observations"]
    ]
    return {
        "samples": len(records),
        "dimension_pairs": len(pair_values),
        "first_person_mean": safe_mean(first_scores),
        "third_person_mean_reference": safe_mean(third_means),
        "third_person_best_reference": safe_mean(third_bests),
        "mean_reference_uplift": safe_mean(mean_differences),
        "mean_reference_uplift_bootstrap_95_ci": bootstrap_mean_ci(
            mean_differences, bootstrap_samples, rng
        ),
        "mean_reference_two_sided_sign_flip_p": sign_flip_p_value(
            mean_differences, permutation_samples, rng
        ),
        "best_reference_uplift": safe_mean(best_differences),
        "sample_third_mean_better_rate": safe_div(
            sum(int(value > 0.0) for value in mean_differences), len(records)
        ),
        "sample_any_third_better_rate": safe_div(
            sum(int(value > 0.0) for value in best_differences), len(records)
        ),
        "pair_third_mean_better_rate": safe_div(
            sum(
                int(value["third_mean"] > value["first"])
                for value in pair_values
            ),
            len(pair_values),
        ),
        "pair_any_third_better_rate": safe_div(
            sum(
                int(value["third_best"] > value["first"])
                for value in pair_values
            ),
            len(pair_values),
        ),
    }


def summarize_subset(
    records: Sequence[dict[str, Any]],
    metrics: Sequence[str],
    args: argparse.Namespace,
    rng: random.Random,
) -> dict[str, Any]:
    return {
        metric: summarize_metric(
            records,
            metric,
            args.bootstrap_samples,
            args.permutation_samples,
            rng,
        )
        for metric in metrics
    }


def dimension_summary(
    records: Sequence[dict[str, Any]], metric: str, dimension: str
) -> dict[str, Any]:
    values = [
        observation_values(observation, metric)
        for record in records
        for observation in record["observations"]
        if observation["dimension"] == dimension
    ]
    differences = [value["third_mean"] - value["first"] for value in values]
    best_differences = [value["third_best"] - value["first"] for value in values]
    return {
        "pairs": len(values),
        "first_person_mean": safe_mean([value["first"] for value in values]),
        "third_person_mean_reference": safe_mean(
            [value["third_mean"] for value in values]
        ),
        "third_person_best_reference": safe_mean(
            [value["third_best"] for value in values]
        ),
        "mean_reference_uplift": safe_mean(differences),
        "best_reference_uplift": safe_mean(best_differences),
        "third_mean_better_rate": safe_div(
            sum(int(value > 0.0) for value in differences), len(values)
        ),
        "any_third_better_rate": safe_div(
            sum(int(value > 0.0) for value in best_differences), len(values)
        ),
    }


def write_outputs(
    *,
    output_dir: Path,
    records: Sequence[dict[str, Any]],
    low_records: Sequence[dict[str, Any]],
    metrics: Sequence[str],
    selection_metric: str,
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    low_ids = {record["sample_id"] for record in low_records}
    sample_rows = []
    for record in records:
        row: dict[str, Any] = {
            "sample_id": record["sample_id"],
            "situation": record["situation"],
            "selected_low": record["sample_id"] in low_ids,
            "dimension_coverage": record["dimension_coverage"],
        }
        for metric in metrics:
            for name, value in record["scores"][metric].items():
                row[f"{metric}_{name}"] = value
        sample_rows.append(row)
    sample_fields = list(sample_rows[0]) if sample_rows else ("sample_id",)
    write_csv(output_dir / "samples.csv", sample_fields, sample_rows)

    dimension_rows = []
    for metric in metrics:
        for dimension in CORE_DIMENSIONS:
            row = {"metric": metric, "dimension": dimension}
            for subset_name, subset_records in (
                ("all_matched", records),
                ("low_first_person", low_records),
            ):
                values = dimension_summary(subset_records, metric, dimension)
                for name, value in values.items():
                    row[f"{subset_name}_{name}"] = value
            dimension_rows.append(row)
    dimension_fields = list(dimension_rows[0]) if dimension_rows else ("metric",)
    write_csv(output_dir / "dimensions.csv", dimension_fields, dimension_rows)

    low_pair_rows = []
    reference_rows = []
    for record in low_records:
        for observation in record["observations"]:
            row = {
                "sample_id": record["sample_id"],
                "situation": record["situation"],
                "dimension": observation["dimension"],
                "prediction": observation["prediction"],
                "first_person_reference": observation["first_person_reference"],
                "third_person_reference_count": len(
                    observation["third_person_references"]
                ),
            }
            for metric in metrics:
                values = observation_values(observation, metric)
                for name, value in values.items():
                    row[f"{metric}_{name}"] = value
                row[f"{metric}_mean_uplift"] = (
                    values["third_mean"] - values["first"]
                )
                row[f"{metric}_best_uplift"] = (
                    values["third_best"] - values["first"]
                )
            low_pair_rows.append(row)
            for reference in observation["third_person_references"]:
                reference_row = {
                    "sample_id": record["sample_id"],
                    "dimension": observation["dimension"],
                    "annotator_id": reference["annotator_id"],
                    "third_person_reference": reference["text"],
                }
                for metric in metrics:
                    reference_row[metric] = reference["metrics"][metric]
                reference_rows.append(reference_row)
    low_pair_rows.sort(
        key=lambda row: (
            -float(row[f"{selection_metric}_mean_uplift"]),
            row["sample_id"],
            row["dimension"],
        )
    )
    pair_fields = list(low_pair_rows[0]) if low_pair_rows else ("sample_id",)
    write_csv(output_dir / "low_examples.csv", pair_fields, low_pair_rows)
    reference_fields = (
        list(reference_rows[0]) if reference_rows else ("sample_id",)
    )
    write_csv(
        output_dir / "third_person_reference_details.csv",
        reference_fields,
        reference_rows,
    )

    headline = summary["headline"]
    write_csv(
        output_dir / "headline.csv",
        ("metric", "value"),
        ({"metric": key, "value": value} for key, value in headline.items()),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0.0 <= args.low_threshold <= 1.0:
        raise ValueError("--low_threshold must be between 0 and 1")
    if not 0.0 < args.bottom_fraction <= 1.0:
        raise ValueError("--bottom_fraction must be in (0, 1]")
    if not 0.0 <= args.min_dimension_coverage <= 1.0:
        raise ValueError("--min_dimension_coverage must be between 0 and 1")
    if args.min_third_person_references <= 0:
        raise ValueError("--min_third_person_references must be positive")
    if args.bootstrap_samples < 0 or args.permutation_samples < 0:
        raise ValueError("Resample counts must be non-negative")
    if args.selection_metric == "bertscore" and not args.bertscore_model:
        raise ValueError(
            "--selection_metric bertscore requires --bertscore_model"
        )

    prediction_root, prediction_source = resolve_prediction_root(
        args.pred_root, args.prediction_source
    )
    output_dir = args.output_dir or (
        prediction_root / "analysis_low_reasoning_third_person_gold"
    )
    first_person = load_gold(args.first_person_gold)
    third_person = load_third_person(args.third_person_gold)
    records, coverage = build_observations(
        first_person=first_person,
        third_person=third_person,
        prediction_root=prediction_root,
        prediction_source=prediction_source,
        min_third_references=args.min_third_person_references,
        min_dimension_coverage=args.min_dimension_coverage,
    )
    if not records:
        raise ValueError("No samples have complete first/third/prediction reasoning")

    include_bertscore = bool(args.bertscore_model)
    if include_bertscore:
        add_bertscore(
            records,
            model_type=args.bertscore_model,
            num_layers=args.bertscore_num_layers,
            batch_size=args.bertscore_batch_size,
            device=args.bertscore_device,
        )
    metrics = metric_names(include_bertscore)
    add_sample_scores(records, metrics)
    low_records, cutoff = select_low_records(
        records,
        metric=args.selection_metric,
        mode=args.selection_mode,
        threshold=args.low_threshold,
        bottom_fraction=args.bottom_fraction,
    )

    rng = random.Random(args.seed)
    all_summary = summarize_subset(records, metrics, args, rng)
    low_summary = summarize_subset(low_records, metrics, args, rng)
    selected_metric_summary = low_summary[args.selection_metric]
    summary = {
        "meta": {
            "first_person_gold": str(args.first_person_gold),
            "third_person_gold": str(args.third_person_gold),
            "prediction_root": str(prediction_root),
            "prediction_source": prediction_source,
            "selection_mode": args.selection_mode,
            "selection_metric": args.selection_metric,
            "low_threshold": args.low_threshold,
            "bottom_fraction": args.bottom_fraction,
            "effective_cutoff": cutoff,
            "min_dimension_coverage": args.min_dimension_coverage,
            "min_third_person_references": args.min_third_person_references,
            "bertscore_model": args.bertscore_model or None,
            "bertscore_num_layers": (
                args.bertscore_num_layers if include_bertscore else None
            ),
            "bootstrap_unit": "sample",
            "third_person_primary_aggregation": "mean over annotators",
        },
        "coverage": {
            **coverage,
            "low_first_person_samples": len(low_records),
        },
        "headline": {
            "selected_low_samples": len(low_records),
            "selection_metric": args.selection_metric,
            "low_subset_first_person_mean": selected_metric_summary[
                "first_person_mean"
            ],
            "low_subset_third_person_mean_reference": selected_metric_summary[
                "third_person_mean_reference"
            ],
            "low_subset_mean_reference_uplift": selected_metric_summary[
                "mean_reference_uplift"
            ],
            "low_subset_mean_reference_uplift_95_ci": selected_metric_summary[
                "mean_reference_uplift_bootstrap_95_ci"
            ],
            "low_subset_mean_reference_sign_flip_p": selected_metric_summary[
                "mean_reference_two_sided_sign_flip_p"
            ],
            "low_subset_third_mean_better_sample_rate": selected_metric_summary[
                "sample_third_mean_better_rate"
            ],
            "all_samples_mean_reference_uplift": all_summary[
                args.selection_metric
            ]["mean_reference_uplift"],
        },
        "all_matched": all_summary,
        "low_first_person_subset": low_summary,
        "dimension": {
            metric: {
                dimension: {
                    "all_matched": dimension_summary(
                        records, metric, dimension
                    ),
                    "low_first_person": dimension_summary(
                        low_records, metric, dimension
                    ),
                }
                for dimension in CORE_DIMENSIONS
            }
            for metric in metrics
        },
        "interpretation_guardrails": [
            (
                "A positive mean-reference uplift supports alignment with "
                "third-person annotator interpretations, not proof that "
                "grammatical perspective caused the original low score."
            ),
            (
                "Selecting samples for low first-person similarity creates "
                "selection bias; report all_matched results alongside the low subset."
            ),
            (
                "Best-reference scores are optimistic because they select the "
                "most favorable annotator; mean-reference is the primary result."
            ),
            (
                "Lexical overlap metrics can penalize valid paraphrases; use "
                "BERTScore or human/rubric judging as a robustness analysis."
            ),
        ],
    }
    write_outputs(
        output_dir=output_dir,
        records=records,
        low_records=low_records,
        metrics=metrics,
        selection_metric=args.selection_metric,
        summary=summary,
    )
    uplift = summary["headline"]["low_subset_mean_reference_uplift"]
    uplift_text = "n/a" if uplift is None else f"{float(uplift):.6f}"
    print(
        "[done] matched={} low={} metric={} mean_reference_uplift={} output={}".format(
            len(records),
            len(low_records),
            args.selection_metric,
            uplift_text,
            output_dir,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
