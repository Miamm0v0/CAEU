#!/usr/bin/env python3
"""Analyze v7 training-Judge consistency with blinded human consensus."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train_grpo.reward_components import compute_process_gate  # noqa: E402
from train_grpo.spec import (  # noqa: E402
    APPRAISAL_CRITERIA,
    COHERENCE_CRITERION,
    RUBRIC_VERSION,
    TRANSITION_CRITERION,
    resolve_appraisal_dimensions,
)


DEFAULT_DIMENSIONS = resolve_appraisal_dimensions(
    "relevance,epistemic,goal_congruence,agency_accountability,"
    "control_coping_potential"
)
PROCESS_REWARDS = (
    "appraisal_reward",
    "coherence_reward",
    "transition_reward",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare training-Judge scores with human consensus"
    )
    parser.add_argument("--judge_results", type=Path, required=True)
    parser.add_argument("--human_consensus", type=Path, required=True)
    parser.add_argument("--identity_key", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--appraisal_dimensions",
        type=resolve_appraisal_dimensions,
        default=list(DEFAULT_DIMENSIONS),
    )
    parser.add_argument("--bootstrap_iterations", type=int, default=10000)
    parser.add_argument("--confidence_level", type=float, default=0.95)
    parser.add_argument(
        "--process_gate_mode",
        choices=["min", "product", "geometric_mean"],
        default="min",
        help="Must match the GRPO run; default matches train_grpo/config.py",
    )
    process_gate = parser.add_mutually_exclusive_group()
    process_gate.add_argument(
        "--use_process_gate",
        dest="use_process_gate",
        action="store_true",
        help="Analyze the enabled GRPO process gate (default)",
    )
    process_gate.add_argument(
        "--no_process_gate",
        dest="use_process_gate",
        action="store_false",
        help="Match a GRPO run trained with --no_process_gate",
    )
    parser.set_defaults(use_process_gate=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc.msg}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(row)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            )


def rubric_item_names(dimensions: Sequence[str]) -> list[str]:
    return [
        f"appraisals.{dimension}.{criterion}"
        for dimension in dimensions
        for criterion in APPRAISAL_CRITERIA
    ] + [COHERENCE_CRITERION, TRANSITION_CRITERION]


def extract_rubric_scores(
    judgment: Mapping[str, Any], dimensions: Sequence[str]
) -> dict[str, float]:
    result: dict[str, float] = {}
    for dimension in dimensions:
        for criterion in APPRAISAL_CRITERIA:
            value = judgment["appraisals"][dimension][criterion]["score"]
            result[f"appraisals.{dimension}.{criterion}"] = checked_score(
                value
            )
    result[COHERENCE_CRITERION] = checked_score(
        judgment["coherence"]["score"]
    )
    result[TRANSITION_CRITERION] = checked_score(
        judgment["transition"]["score"]
    )
    return result


def checked_score(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"rubric score must be numeric, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 4.0:
        raise ValueError(f"rubric score must be in [0, 4], got {value!r}")
    return number


def collapsed_criteria(
    scores: Mapping[str, float], dimensions: Sequence[str]
) -> dict[str, float]:
    result = {
        criterion: statistics.mean(
            scores[f"appraisals.{dimension}.{criterion}"]
            for dimension in dimensions
        )
        for criterion in APPRAISAL_CRITERIA
    }
    result[COHERENCE_CRITERION] = scores[COHERENCE_CRITERION]
    result[TRANSITION_CRITERION] = scores[TRANSITION_CRITERION]
    return result


def average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for offset in range(start, end):
            ranks[order[offset]] = rank
        start = end
    return ranks


def pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    left_ss = sum((value - left_mean) ** 2 for value in left)
    right_ss = sum((value - right_mean) ** 2 for value in right)
    if left_ss <= 0 or right_ss <= 0:
        return None
    covariance = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    return float(covariance / math.sqrt(left_ss * right_ss))


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    return pearson(average_ranks(left), average_ranks(right))


def mean_absolute_error(
    left: Sequence[float], right: Sequence[float]
) -> float | None:
    if len(left) != len(right) or not left:
        return None
    return float(
        statistics.mean(
            abs(left_value - right_value)
            for left_value, right_value in zip(left, right)
        )
    )


def percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires values")
    position = probability * (len(sorted_values) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return float(sorted_values[low])
    fraction = position - low
    return float(
        sorted_values[low] * (1.0 - fraction)
        + sorted_values[high] * fraction
    )


def metric_seed(base_seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{base_seed}|{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def bootstrap_ci(
    groups: Sequence[Any],
    statistic: Callable[[Sequence[Any]], float | None],
    iterations: int,
    confidence_level: float,
    seed: int,
) -> tuple[float | None, float | None, int]:
    if not groups or iterations <= 0:
        return None, None, 0
    generator = random.Random(seed)
    estimates: list[float] = []
    for _ in range(iterations):
        sampled = [generator.choice(groups) for _ in groups]
        estimate = statistic(sampled)
        if estimate is not None and math.isfinite(estimate):
            estimates.append(float(estimate))
    if not estimates:
        return None, None, 0
    estimates.sort()
    tail = (1.0 - confidence_level) / 2.0
    return (
        percentile(estimates, tail),
        percentile(estimates, 1.0 - tail),
        len(estimates),
    )


def correlation_result(
    pair_groups: Sequence[Sequence[Mapping[str, Any]]],
    judge_getter: Callable[[Mapping[str, Any]], float],
    human_getter: Callable[[Mapping[str, Any]], float],
    iterations: int,
    confidence_level: float,
    seed: int,
    score_range: float,
) -> dict[str, Any]:
    records = [record for group in pair_groups for record in group]

    def paired_values(
        groups: Sequence[Sequence[Mapping[str, Any]]],
    ) -> tuple[list[float], list[float]]:
        flattened = [record for group in groups for record in group]
        return (
            [judge_getter(record) for record in flattened],
            [human_getter(record) for record in flattened],
        )

    def calculate_spearman(
        groups: Sequence[Sequence[Mapping[str, Any]]],
    ) -> float | None:
        judge_values, human_values = paired_values(groups)
        return spearman(judge_values, human_values)

    def calculate_mae(
        groups: Sequence[Sequence[Mapping[str, Any]]],
    ) -> float | None:
        judge_values, human_values = paired_values(groups)
        return mean_absolute_error(judge_values, human_values)

    if score_range <= 0:
        raise ValueError("score_range must be positive")
    spearman_estimate = calculate_spearman(pair_groups)
    spearman_low, spearman_high, spearman_valid = bootstrap_ci(
        pair_groups,
        calculate_spearman,
        iterations,
        confidence_level,
        seed,
    )
    mae_estimate = calculate_mae(pair_groups)
    mae_low, mae_high, mae_valid = bootstrap_ci(
        pair_groups,
        calculate_mae,
        iterations,
        confidence_level,
        seed + 1,
    )
    return {
        "n_candidates": len(records),
        "n_pairs": len(pair_groups),
        "score_range": score_range,
        "spearman_rho": spearman_estimate,
        "spearman_ci_low": spearman_low,
        "spearman_ci_high": spearman_high,
        "spearman_bootstrap_valid_replicates": spearman_valid,
        "mae": mae_estimate,
        "mae_ci_low": mae_low,
        "mae_ci_high": mae_high,
        "mae_bootstrap_valid_replicates": mae_valid,
        "normalized_mae": (
            None if mae_estimate is None else mae_estimate / score_range
        ),
        "normalized_mae_ci_low": (
            None if mae_low is None else mae_low / score_range
        ),
        "normalized_mae_ci_high": (
            None if mae_high is None else mae_high / score_range
        ),
    }


def process_gate_bottleneck(
    record: Mapping[str, Any], source: str, tolerance: float = 1e-12
) -> str:
    rewards = record[f"{source}_rewards"]
    appraisal = float(rewards["appraisal_reward"])
    transition = float(rewards["transition_reward"])
    if abs(appraisal - transition) <= tolerance:
        return "tie"
    return "appraisal_reward" if appraisal < transition else "transition_reward"


def process_gate_consistency_result(
    pair_groups: Sequence[Sequence[Mapping[str, Any]]],
    iterations: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    records = [record for group in pair_groups for record in group]

    def agreement_stats(
        groups: Sequence[Sequence[Mapping[str, Any]]],
    ) -> dict[str, Any]:
        flattened = [record for group in groups for record in group]
        bottlenecks = [
            (
                process_gate_bottleneck(record, "judge"),
                process_gate_bottleneck(record, "human"),
            )
            for record in flattened
        ]
        non_ties = [
            item for item in bottlenecks if "tie" not in item
        ]
        absolute_errors = [
            abs(record["judge_process_gate"] - record["human_process_gate"])
            for record in flattened
        ]
        return {
            "bottleneck_source_agreement_including_ties": (
                statistics.mean(left == right for left, right in bottlenecks)
                if bottlenecks
                else None
            ),
            "bottleneck_source_agreement_excluding_ties": (
                statistics.mean(left == right for left, right in non_ties)
                if non_ties
                else None
            ),
            "n_non_tie_candidates": len(non_ties),
            "within_0_10_gate_error_rate": (
                statistics.mean(error <= 0.10 for error in absolute_errors)
                if absolute_errors
                else None
            ),
        }

    point = agreement_stats(pair_groups)
    for offset, statistic_name in enumerate(
        (
            "bottleneck_source_agreement_including_ties",
            "bottleneck_source_agreement_excluding_ties",
            "within_0_10_gate_error_rate",
        )
    ):
        low, high, valid = bootstrap_ci(
            pair_groups,
            lambda groups, field=statistic_name: agreement_stats(groups)[field],
            iterations,
            confidence_level,
            seed + offset,
        )
        point[f"{statistic_name}_ci_low"] = low
        point[f"{statistic_name}_ci_high"] = high
        point[f"{statistic_name}_bootstrap_valid_replicates"] = valid
    point.update(
        {
            "n_candidates": len(records),
            "judge_bottleneck_counts": {
                name: sum(
                    process_gate_bottleneck(record, "judge") == name
                    for record in records
                )
                for name in ("appraisal_reward", "transition_reward", "tie")
            },
            "human_bottleneck_counts": {
                name: sum(
                    process_gate_bottleneck(record, "human") == name
                    for record in records
                )
                for name in ("appraisal_reward", "transition_reward", "tie")
            },
        }
    )
    return point


def sign(value: float, tolerance: float = 1e-12) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def ranking_stats(
    rank_pairs: Sequence[Mapping[str, Any]], metric: str
) -> dict[str, Any]:
    signs: list[tuple[int, int]] = []
    for pair in rank_pairs:
        base = pair["base"]
        caeu = pair["caeu"]
        if metric == "process_reward_mean":
            judge_base = statistics.mean(
                base["judge_rewards"][name] for name in PROCESS_REWARDS
            )
            judge_caeu = statistics.mean(
                caeu["judge_rewards"][name] for name in PROCESS_REWARDS
            )
            human_base = statistics.mean(
                base["human_rewards"][name] for name in PROCESS_REWARDS
            )
            human_caeu = statistics.mean(
                caeu["human_rewards"][name] for name in PROCESS_REWARDS
            )
        elif metric == "process_gate":
            judge_base = base["judge_process_gate"]
            judge_caeu = caeu["judge_process_gate"]
            human_base = base["human_process_gate"]
            human_caeu = caeu["human_process_gate"]
        else:
            judge_base = base["judge_rewards"][metric]
            judge_caeu = caeu["judge_rewards"][metric]
            human_base = base["human_rewards"][metric]
            human_caeu = caeu["human_rewards"][metric]
        signs.append(
            (sign(judge_caeu - judge_base), sign(human_caeu - human_base))
        )
    non_ties = [item for item in signs if item[0] != 0 and item[1] != 0]
    return {
        "n_pairs": len(signs),
        "agreement_including_ties": (
            sum(left == right for left, right in signs) / len(signs)
            if signs
            else None
        ),
        "n_non_tie_pairs": len(non_ties),
        "direction_agreement_excluding_ties": (
            sum(left == right for left, right in non_ties) / len(non_ties)
            if non_ties
            else None
        ),
        "judge_tie_count": sum(left == 0 for left, _ in signs),
        "human_tie_count": sum(right == 0 for _, right in signs),
    }


def ranking_result(
    rank_pairs: Sequence[Mapping[str, Any]],
    metric: str,
    iterations: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    result = ranking_stats(rank_pairs, metric)
    for statistic_name in (
        "agreement_including_ties",
        "direction_agreement_excluding_ties",
    ):
        low, high, valid = bootstrap_ci(
            rank_pairs,
            lambda rows, field=statistic_name: ranking_stats(rows, metric)[field],
            iterations,
            confidence_level,
            seed + (0 if statistic_name == "agreement_including_ties" else 1),
        )
        result[f"{statistic_name}_ci_low"] = low
        result[f"{statistic_name}_ci_high"] = high
        result[f"{statistic_name}_bootstrap_valid_replicates"] = valid
    return result


def latest_successful_judge_rows(
    rows: Sequence[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    latest: dict[str, dict[str, Any]] = {}
    successful: dict[str, dict[str, Any]] = {}
    for row in rows:
        candidate_id = str(row.get("candidate_id", ""))
        if not candidate_id:
            continue
        latest[candidate_id] = row
        if row.get("status") == "ok":
            successful[candidate_id] = row
    return successful, latest


def main() -> int:
    args = build_parser().parse_args()
    dimensions = list(args.appraisal_dimensions)
    if args.bootstrap_iterations <= 0:
        raise ValueError("--bootstrap_iterations must be positive")
    if not 0.0 < args.confidence_level < 1.0:
        raise ValueError("--confidence_level must lie in (0, 1)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    table_path = args.output_dir / "paper_table.csv"
    matched_path = args.output_dir / "matched_scores.jsonl"
    if any(path.exists() for path in (summary_path, table_path, matched_path)):
        if not args.overwrite:
            raise FileExistsError("Analysis output exists; pass --overwrite")

    judge_all = read_jsonl(args.judge_results)
    judge_rows, latest_judge = latest_successful_judge_rows(judge_all)
    human_all = read_jsonl(args.human_consensus)
    human_rows = {
        str(row.get("candidate_id", "")): row
        for row in human_all
        if str(row.get("candidate_id", ""))
    }
    if len(human_rows) != len(human_all):
        raise ValueError("human consensus candidate IDs must be non-empty/unique")

    merged: dict[str, dict[str, Any]] = {}
    invalid_merged: list[dict[str, str]] = []
    for candidate_id in sorted(set(judge_rows) & set(human_rows)):
        judge_row = judge_rows[candidate_id]
        human_row = human_rows[candidate_id]
        try:
            judge_scores = extract_rubric_scores(
                judge_row["judgment"], dimensions
            )
            human_scores = extract_rubric_scores(
                human_row["consensus_judgment"], dimensions
            )
            judge_rewards = {
                name: float(judge_row[name]) for name in PROCESS_REWARDS
            }
            human_rewards = {
                name: float(human_row[name]) for name in PROCESS_REWARDS
            }
            judge_process_gate = (
                compute_process_gate(
                    judge_rewards["appraisal_reward"],
                    judge_rewards["transition_reward"],
                    args.process_gate_mode,
                )
                if args.use_process_gate
                else 1.0
            )
            human_process_gate = (
                compute_process_gate(
                    human_rewards["appraisal_reward"],
                    human_rewards["transition_reward"],
                    args.process_gate_mode,
                )
                if args.use_process_gate
                else 1.0
            )
            if not all(
                math.isfinite(value)
                for value in [
                    *judge_rewards.values(),
                    *human_rewards.values(),
                    judge_process_gate,
                    human_process_gate,
                ]
            ):
                raise ValueError("non-finite process reward")
            merged[candidate_id] = {
                "candidate_id": candidate_id,
                "pair_id": str(judge_row.get("pair_id", "")),
                "candidate_slot": judge_row.get("candidate_slot"),
                "source_sample_id": judge_row.get("source_sample_id"),
                "judge_rubric_scores": judge_scores,
                "human_rubric_scores": human_scores,
                "judge_collapsed_criteria": collapsed_criteria(
                    judge_scores, dimensions
                ),
                "human_collapsed_criteria": collapsed_criteria(
                    human_scores, dimensions
                ),
                "judge_rewards": judge_rewards,
                "human_rewards": human_rewards,
                "judge_process_gate": judge_process_gate,
                "human_process_gate": human_process_gate,
            }
        except (KeyError, TypeError, ValueError) as exc:
            invalid_merged.append(
                {"candidate_id": candidate_id, "error": str(exc)}
            )
    if not merged:
        raise ValueError("No valid matched Judge/Human candidates")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in merged.values():
        if record["pair_id"]:
            grouped[record["pair_id"]].append(record)
    pair_groups = list(grouped.values())

    item_correlations: dict[str, Any] = {}
    for item_name in rubric_item_names(dimensions):
        item_correlations[item_name] = correlation_result(
            pair_groups,
            lambda record, key=item_name: record["judge_rubric_scores"][key],
            lambda record, key=item_name: record["human_rubric_scores"][key],
            args.bootstrap_iterations,
            args.confidence_level,
            metric_seed(args.seed, "item:" + item_name),
            score_range=4.0,
        )

    criterion_correlations: dict[str, Any] = {}
    for criterion in [
        *APPRAISAL_CRITERIA,
        COHERENCE_CRITERION,
        TRANSITION_CRITERION,
    ]:
        criterion_correlations[criterion] = correlation_result(
            pair_groups,
            lambda record, key=criterion: record["judge_collapsed_criteria"][
                key
            ],
            lambda record, key=criterion: record["human_collapsed_criteria"][
                key
            ],
            args.bootstrap_iterations,
            args.confidence_level,
            metric_seed(args.seed, "criterion:" + criterion),
            score_range=4.0,
        )

    reward_correlations: dict[str, Any] = {}
    for reward_name in PROCESS_REWARDS:
        reward_correlations[reward_name] = correlation_result(
            pair_groups,
            lambda record, key=reward_name: record["judge_rewards"][key],
            lambda record, key=reward_name: record["human_rewards"][key],
            args.bootstrap_iterations,
            args.confidence_level,
            metric_seed(args.seed, "reward:" + reward_name),
            score_range=1.0,
        )

    process_gate_value_metrics = correlation_result(
        pair_groups,
        lambda record: record["judge_process_gate"],
        lambda record: record["human_process_gate"],
        args.bootstrap_iterations,
        args.confidence_level,
        metric_seed(args.seed, "process_gate:value"),
        score_range=1.0,
    )
    process_gate_decision_metrics = process_gate_consistency_result(
        pair_groups,
        args.bootstrap_iterations,
        args.confidence_level,
        metric_seed(args.seed, "process_gate:decision"),
    )

    identity = read_json(args.identity_key)
    if not isinstance(identity, dict) or not isinstance(identity.get("pairs"), list):
        raise ValueError("identity key has invalid structure")
    rank_pairs: list[dict[str, Any]] = []
    incomplete_rank_pairs: list[str] = []
    for pair in identity["pairs"]:
        pair_id = str(pair.get("pair_id", ""))
        candidates = pair.get("candidates")
        if not isinstance(candidates, dict):
            incomplete_rank_pairs.append(pair_id)
            continue
        by_system: dict[str, dict[str, Any]] = {}
        for value in candidates.values():
            if not isinstance(value, dict):
                continue
            candidate_id = str(value.get("candidate_id", ""))
            system = str(value.get("system", ""))
            if system in {"base", "caeu"} and candidate_id in merged:
                by_system[system] = merged[candidate_id]
        if set(by_system) == {"base", "caeu"}:
            rank_pairs.append(
                {"pair_id": pair_id, "base": by_system["base"], "caeu": by_system["caeu"]}
            )
        else:
            incomplete_rank_pairs.append(pair_id)

    ranking_agreement = {
        metric: ranking_result(
            rank_pairs,
            metric,
            args.bootstrap_iterations,
            args.confidence_level,
            metric_seed(args.seed, "ranking:" + metric),
        )
        for metric in [
            *PROCESS_REWARDS,
            "process_reward_mean",
            "process_gate",
        ]
    }

    summary = {
        "schema_version": "judge-human-consistency-summary-v2",
        "rubric_version": RUBRIC_VERSION,
        "appraisal_dimensions": dimensions,
        "bootstrap": {
            "iterations": args.bootstrap_iterations,
            "confidence_level": args.confidence_level,
            "resampling_unit": "CAREBench situation pair",
            "seed": args.seed,
        },
        "process_gate_enabled": args.use_process_gate,
        "process_gate_mode": args.process_gate_mode,
        "coverage": {
            "judge_result_row_count": len(judge_all),
            "judge_unique_success_count": len(judge_rows),
            "judge_latest_error_count": sum(
                row.get("status") != "ok" for row in latest_judge.values()
            ),
            "human_consensus_count": len(human_rows),
            "matched_valid_candidate_count": len(merged),
            "judge_success_without_human_candidate_ids": sorted(
                set(judge_rows) - set(human_rows)
            ),
            "human_without_judge_success_candidate_ids": sorted(
                set(human_rows) - set(judge_rows)
            ),
            "matched_pair_group_count": len(pair_groups),
            "complete_base_caeu_pair_count": len(rank_pairs),
            "incomplete_base_caeu_pair_ids": incomplete_rank_pairs,
            "invalid_matched_rows": invalid_merged,
        },
        "rubric_item_consistency": item_correlations,
        "criterion_consistency": criterion_correlations,
        "process_reward_consistency": reward_correlations,
        "process_gate_consistency": {
            "enabled": args.use_process_gate,
            "mode": args.process_gate_mode,
            "bottleneck_definition": (
                "Lower of appraisal_reward and transition_reward; reported "
                "as a diagnostic for all modes"
            ),
            "gate_value": process_gate_value_metrics,
            "gate_decision": process_gate_decision_metrics,
        },
        "base_vs_caeu_ranking_agreement": ranking_agreement,
    }
    write_json(summary_path, summary)
    write_jsonl(matched_path, merged.values())

    table_rows: list[dict[str, Any]] = []
    for category, metrics in (
        ("rubric_item_consistency", item_correlations),
        ("criterion_consistency", criterion_correlations),
        ("process_reward_consistency", reward_correlations),
    ):
        for metric, result in metrics.items():
            for statistic_name in ("spearman_rho", "mae"):
                table_rows.append(
                    {
                        "category": category,
                        "metric": metric,
                        "statistic": statistic_name,
                        "score_range": result["score_range"],
                        "n": result["n_candidates"],
                        "estimate": result[statistic_name],
                        "ci_low": result[f"{statistic_name.split('_')[0]}_ci_low"],
                        "ci_high": result[
                            f"{statistic_name.split('_')[0]}_ci_high"
                        ],
                    }
                )

    for statistic_name in ("spearman_rho", "mae"):
        prefix = statistic_name.split("_")[0]
        table_rows.append(
            {
                "category": "process_gate_consistency",
                "metric": "process_gate_value",
                "statistic": statistic_name,
                "score_range": 1.0,
                "n": process_gate_value_metrics["n_candidates"],
                "estimate": process_gate_value_metrics[statistic_name],
                "ci_low": process_gate_value_metrics[f"{prefix}_ci_low"],
                "ci_high": process_gate_value_metrics[f"{prefix}_ci_high"],
            }
        )
    for statistic_name, count_field in (
        ("bottleneck_source_agreement_including_ties", "n_candidates"),
        (
            "bottleneck_source_agreement_excluding_ties",
            "n_non_tie_candidates",
        ),
        ("within_0_10_gate_error_rate", "n_candidates"),
    ):
        table_rows.append(
            {
                "category": "process_gate_consistency",
                "metric": "process_gate_decision",
                "statistic": statistic_name,
                "score_range": 1.0,
                "n": process_gate_decision_metrics[count_field],
                "estimate": process_gate_decision_metrics[statistic_name],
                "ci_low": process_gate_decision_metrics[
                    f"{statistic_name}_ci_low"
                ],
                "ci_high": process_gate_decision_metrics[
                    f"{statistic_name}_ci_high"
                ],
            }
        )
    for metric, result in ranking_agreement.items():
        for statistic_name in (
            "agreement_including_ties",
            "direction_agreement_excluding_ties",
        ):
            table_rows.append(
                {
                    "category": "base_vs_caeu_ranking_agreement",
                    "metric": metric,
                    "statistic": statistic_name,
                    "score_range": 1.0,
                    "n": (
                        result["n_pairs"]
                        if statistic_name == "agreement_including_ties"
                        else result["n_non_tie_pairs"]
                    ),
                    "estimate": result[statistic_name],
                    "ci_low": result[f"{statistic_name}_ci_low"],
                    "ci_high": result[f"{statistic_name}_ci_high"],
                }
            )
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "category",
                "metric",
                "statistic",
                "score_range",
                "n",
                "estimate",
                "ci_low",
                "ci_high",
            ],
        )
        writer.writeheader()
        writer.writerows(table_rows)

    print(
        f"Matched {len(merged)} candidates and {len(rank_pairs)} complete "
        f"Base/CAEU pairs; wrote {summary_path} and {table_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
