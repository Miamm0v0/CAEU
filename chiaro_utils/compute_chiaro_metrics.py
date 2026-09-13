#!/usr/bin/env python3
"""Compute official-style and pair-level metrics for CHIARO predictions."""

from __future__ import annotations

import argparse
import collections
import math
import re
from pathlib import Path
from typing import Any, Iterable

from chiaro_common import (
    ALL_EMOTIONS,
    NEGATIVE_EMOTIONS,
    POSITIVE_EMOTIONS,
    load_chiaro_items,
    normalize_emotion,
    read_prediction_records,
    write_json,
)


UTILS_DIR = Path(__file__).resolve().parent
REPO_ROOT = UTILS_DIR.parent
DEFAULT_EVAL_FILE = REPO_ROOT / "Chiaro-main" / "data" / "chiaro_test.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute CHIARO emotion metrics against adjudicated human gold"
    )
    parser.add_argument("--eval_file", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--predictions_file", type=Path, required=True)
    parser.add_argument("--results_file", type=Path, required=True)
    parser.add_argument(
        "--split", choices=["all", "train", "val", "test"], default="all"
    )
    parser.add_argument("--max_samples", type=int, default=None)
    return parser


def _safe_percent(numerator: int | float, denominator: int | float) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def _classification_metrics(
    pairs: Iterable[tuple[str, str]],
) -> dict[str, Any]:
    pair_list = list(pairs)
    tp: collections.Counter[str] = collections.Counter()
    fp: collections.Counter[str] = collections.Counter()
    fn: collections.Counter[str] = collections.Counter()
    support: collections.Counter[str] = collections.Counter()
    confusion = {
        gold: {prediction: 0 for prediction in ALL_EMOTIONS}
        for gold in ALL_EMOTIONS
    }
    correct = 0
    for prediction, gold in pair_list:
        support[gold] += 1
        confusion[gold][prediction] += 1
        if prediction == gold:
            correct += 1
            tp[gold] += 1
        else:
            fp[prediction] += 1
            fn[gold] += 1

    per_emotion: dict[str, dict[str, Any]] = {}
    f1_values: list[float] = []
    for emotion in ALL_EMOTIONS:
        precision = tp[emotion] / (tp[emotion] + fp[emotion]) if tp[emotion] + fp[emotion] else 0.0
        recall = tp[emotion] / (tp[emotion] + fn[emotion]) if tp[emotion] + fn[emotion] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_emotion[emotion] = {
            "precision": 100.0 * precision,
            "recall": 100.0 * recall,
            "f1": 100.0 * f1,
            "support": support[emotion],
        }
    return {
        "n": len(pair_list),
        "accuracy": _safe_percent(correct, len(pair_list)),
        "macro_f1": 100.0 * sum(f1_values) / len(ALL_EMOTIONS),
        "per_emotion": per_emotion,
        "confusion_matrix": confusion,
    }


def _normalize_sentence(text: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def _prediction_indexes(
    records: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_id: dict[str, dict[str, Any]] = {}
    by_sentence: dict[str, dict[str, Any]] = {}
    for row_number, record in enumerate(records, 1):
        item_id = str(record.get("mcq_id_a", record.get("id", ""))).strip()
        if not item_id:
            raise ValueError(f"Prediction row {row_number} has no mcq_id_a")
        if item_id in by_id:
            raise ValueError(f"Duplicate prediction id: {item_id}")
        by_id[item_id] = record
        sentence_key = _normalize_sentence(record.get("sentence"))
        if sentence_key:
            if sentence_key in by_sentence:
                raise ValueError(
                    f"Duplicate normalized prediction sentence at row {row_number}"
                )
            by_sentence[sentence_key] = record
    return by_id, by_sentence


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _free_emotion_diagnostics(
    matched_records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    positive_counts: list[float] = []
    negative_counts: list[float] = []
    positive_intensities: list[float] = []
    negative_intensities: list[float] = []
    selected_agrees = 0
    selected_comparable = 0
    intensity_ties = 0
    total = 0
    for record in matched_records:
        for key, slot in (("agent_a_output", "A"), ("agent_b_output", "B")):
            output = record.get(key)
            if not isinstance(output, dict):
                continue
            emotion = output.get("emotion")
            if not isinstance(emotion, dict) or "positive_labels" not in emotion:
                continue
            positive_labels = emotion.get("positive_labels", [])
            negative_labels = emotion.get("negative_labels", [])
            positive_intensity = emotion.get("positive_intensity")
            negative_intensity = emotion.get("negative_intensity")
            if not isinstance(positive_labels, list) or not isinstance(negative_labels, list):
                continue
            if not isinstance(positive_intensity, int) or not isinstance(negative_intensity, int):
                continue
            selected = normalize_emotion(record.get(f"llm_emotion_{slot}"))
            if selected is None:
                continue
            total += 1
            positive_counts.append(float(len(positive_labels)))
            negative_counts.append(float(len(negative_labels)))
            positive_intensities.append(float(positive_intensity))
            negative_intensities.append(float(negative_intensity))
            if positive_intensity == negative_intensity:
                intensity_ties += 1
            else:
                selected_comparable += 1
                selected_positive = selected in POSITIVE_EMOTIONS
                stronger_positive = positive_intensity > negative_intensity
                selected_agrees += int(selected_positive == stronger_positive)
    if total == 0:
        return None
    return {
        "agent_predictions": total,
        "avg_positive_labels": _mean(positive_counts),
        "avg_negative_labels": _mean(negative_counts),
        "avg_total_labels": _mean(
            [a + b for a, b in zip(positive_counts, negative_counts)]
        ),
        "avg_positive_intensity": _mean(positive_intensities),
        "avg_negative_intensity": _mean(negative_intensities),
        "intensity_tie_rate": _safe_percent(intensity_ties, total),
        "selected_label_agrees_with_stronger_valence_rate": _safe_percent(
            selected_agrees, selected_comparable
        ),
        "stronger_valence_comparable_n": selected_comparable,
    }


def build_report(
    items: list[dict[str, Any]], records: list[dict[str, Any]]
) -> dict[str, Any]:
    predictions_by_id, predictions_by_sentence = _prediction_indexes(records)
    all_pairs: list[tuple[str, str]] = []
    slot_pairs: dict[str, list[tuple[str, str]]] = {"A": [], "B": []}
    version_pairs: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    matched_records: list[dict[str, Any]] = []
    parsed_agents = 0
    correct_agents = 0
    complete_pairs = 0
    correct_pairs = 0
    strict_correct_pairs = 0
    missing_scene_ids: list[str] = []
    invalid_prediction_labels: list[dict[str, Any]] = []

    for item in items:
        item_id = str(item["id"])
        record = predictions_by_id.get(item_id)
        if record is None:
            # Released raw predictions use legacy story-based IDs. CHIARO's
            # official scorer joins those records to the current data by sentence.
            record = predictions_by_sentence.get(_normalize_sentence(item["sentence"]))
        if record is None:
            missing_scene_ids.append(item_id)
            continue
        matched_records.append(record)
        pair_is_complete = True
        pair_is_correct = True
        for slot, lower in (("A", "a"), ("B", "b")):
            prediction = normalize_emotion(record.get(f"llm_emotion_{slot}"))
            gold = normalize_emotion(item[f"human_gold_{lower}"])
            if prediction is None:
                pair_is_complete = False
                pair_is_correct = False
                if record.get(f"llm_emotion_{slot}") is not None:
                    invalid_prediction_labels.append(
                        {
                            "id": item_id,
                            "slot": slot,
                            "value": record.get(f"llm_emotion_{slot}"),
                        }
                    )
                continue
            if gold is None:
                raise ValueError(f"{item_id}: invalid human gold for slot {slot}")
            parsed_agents += 1
            correct = prediction == gold
            correct_agents += int(correct)
            pair_is_correct = pair_is_correct and correct
            pair = (prediction, gold)
            all_pairs.append(pair)
            slot_pairs[slot].append(pair)
            version_pairs[str(item.get("version", "unknown"))].append(pair)
        if pair_is_complete:
            complete_pairs += 1
            correct_pairs += int(pair_is_correct)
        if pair_is_complete and pair_is_correct:
            strict_correct_pairs += 1

    total_scenes = len(items)
    total_agents = total_scenes * 2
    overall = _classification_metrics(all_pairs)
    report: dict[str, Any] = {
        "metric_scale": "0-100",
        "dataset": {
            "scenes": total_scenes,
            "agent_targets": total_agents,
            "positive_labels": list(POSITIVE_EMOTIONS),
            "negative_labels": list(NEGATIVE_EMOTIONS),
        },
        "run": {
            "generation_schemas": sorted(
                {str(record.get("generation_schema")) for record in matched_records}
            ),
            "emotion_modes": sorted(
                {str(record.get("emotion_mode")) for record in matched_records}
            ),
            "models": sorted({str(record.get("model")) for record in matched_records}),
        },
        "coverage": {
            "prediction_scenes_matched": len(matched_records),
            "prediction_scenes_total": total_scenes,
            "scene_coverage": _safe_percent(len(matched_records), total_scenes),
            "parsed_agents": parsed_agents,
            "agent_targets": total_agents,
            "agent_parse_coverage": _safe_percent(parsed_agents, total_agents),
            "missing_scene_ids": missing_scene_ids,
            "invalid_prediction_labels": invalid_prediction_labels,
        },
        "overall": {
            "n": overall["n"],
            "accuracy": overall["accuracy"],
            "macro_f1": overall["macro_f1"],
            "strict_accuracy_missing_as_wrong": _safe_percent(
                correct_agents, total_agents
            ),
        },
        "pair": {
            "complete_pairs": complete_pairs,
            "correct_pairs": correct_pairs,
            "pair_accuracy_complete_only": _safe_percent(
                correct_pairs, complete_pairs
            ),
            "pair_accuracy_missing_as_wrong": _safe_percent(
                strict_correct_pairs, total_scenes
            ),
        },
        "by_agent_slot": {
            slot: {
                key: value
                for key, value in _classification_metrics(pairs).items()
                if key in {"n", "accuracy", "macro_f1"}
            }
            for slot, pairs in slot_pairs.items()
        },
        "by_version": {
            version: {
                key: value
                for key, value in _classification_metrics(pairs).items()
                if key in {"n", "accuracy", "macro_f1"}
            }
            for version, pairs in sorted(version_pairs.items())
        },
        "per_emotion": overall["per_emotion"],
        "confusion_matrix": overall["confusion_matrix"],
    }
    free_diagnostics = _free_emotion_diagnostics(matched_records)
    if free_diagnostics is not None:
        report["valence_free_diagnostics"] = free_diagnostics
    return report


def _format_metric(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and math.isnan(value):
            return "nan"
        return f"{value:.2f}"
    return str(value)


def execute(args: argparse.Namespace) -> int:
    if not args.eval_file.is_file():
        raise FileNotFoundError(f"CHIARO evaluation file not found: {args.eval_file}")
    if not args.predictions_file.is_file():
        raise FileNotFoundError(f"Predictions file not found: {args.predictions_file}")
    items = load_chiaro_items(
        args.eval_file, split=args.split, max_samples=args.max_samples
    )
    records = read_prediction_records(args.predictions_file)
    report = build_report(items, records)
    write_json(args.results_file, report)

    overall = report["overall"]
    pair = report["pair"]
    coverage = report["coverage"]
    print(
        "[metrics] scenes={} parsed_agents={}/{} accuracy={} macro_f1={} "
        "pair_accuracy={} strict_pair_accuracy={}".format(
            report["dataset"]["scenes"],
            coverage["parsed_agents"],
            coverage["agent_targets"],
            _format_metric(overall["accuracy"]),
            _format_metric(overall["macro_f1"]),
            _format_metric(pair["pair_accuracy_complete_only"]),
            _format_metric(pair["pair_accuracy_missing_as_wrong"]),
        )
    )
    for version, metrics in report["by_version"].items():
        print(
            f"[metrics:{version}] n={metrics['n']} "
            f"accuracy={_format_metric(metrics['accuracy'])} "
            f"macro_f1={_format_metric(metrics['macro_f1'])}"
        )
    print(f"[metrics] wrote {args.results_file}")
    return 0


def main() -> int:
    return execute(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
