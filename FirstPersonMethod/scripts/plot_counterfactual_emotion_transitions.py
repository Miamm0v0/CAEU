#!/usr/bin/env python3
"""Visualize CAREBench appraisal-intervention emotion changes.

The script expects the enriched JSON produced by
``evaluate_counterfactual_emotion.py``. It writes compact CSV tables and, when
matplotlib is available, PNG/PDF/SVG figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable


LABEL_TASKS = ("positive-labels", "negative-labels")
LEVEL_TASKS = ("positive-level", "negative-level")
ACTORS = ("model", "human")
DEFAULT_DIRECTIONS = ("all", "low_to_high", "high_to_low", "mixed", "unknown")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def sanitize_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    value = value.strip("._")
    return value or "plot"


def short_label(label: str) -> str:
    return label.split(",", 1)[0]


def parse_directions(value: str) -> list[str]:
    directions = [item.strip() for item in value.split(",") if item.strip()]
    return directions or list(DEFAULT_DIRECTIONS)


def parse_simple_actors(value: str) -> list[str]:
    if value == "both":
        return list(ACTORS)
    return [value]


def iter_dimensions(results: dict[str, Any]) -> list[str]:
    meta = results.get("meta", {})
    dimensions = meta.get("dimensions") if isinstance(meta, dict) else None
    if isinstance(dimensions, list) and dimensions:
        return [str(item) for item in dimensions]
    dimension_obj = results.get("dimension", {})
    if isinstance(dimension_obj, dict):
        return sorted(str(key) for key in dimension_obj)
    return []


def label_summary(
    results: dict[str, Any],
    dimension: str,
    task: str,
    actor: str,
    direction: str,
) -> dict[str, Any] | None:
    task_obj = (
        results.get("dimension", {})
        .get(dimension, {})
        .get(task, {})
    )
    if not isinstance(task_obj, dict):
        return None
    transitions = task_obj.get("label_transitions", {})
    if not isinstance(transitions, dict):
        return None
    actor_obj = transitions.get(actor, {})
    if not isinstance(actor_obj, dict):
        return None
    if direction == "all":
        summary = actor_obj.get("all")
    else:
        by_direction = actor_obj.get("by_appraisal_direction", {})
        summary = by_direction.get(direction) if isinstance(by_direction, dict) else None
    return summary if isinstance(summary, dict) else None


def intensity_summary(
    results: dict[str, Any],
    dimension: str,
    task: str,
    direction: str,
) -> dict[str, Any] | None:
    task_obj = (
        results.get("dimension", {})
        .get(dimension, {})
        .get(task, {})
    )
    if not isinstance(task_obj, dict):
        return None
    changes = task_obj.get("intensity_changes", {})
    if not isinstance(changes, dict):
        return None
    if direction == "all":
        summary = changes.get("all")
    else:
        by_direction = changes.get("by_appraisal_direction", {})
        summary = by_direction.get(direction) if isinstance(by_direction, dict) else None
    return summary if isinstance(summary, dict) else None


def collect_labels(results: dict[str, Any], dimensions: Iterable[str], task: str) -> list[str]:
    for dimension in dimensions:
        for actor in ACTORS:
            summary = label_summary(results, dimension, task, actor, "all")
            if not summary:
                continue
            per_label = summary.get("per_label")
            if isinstance(per_label, dict) and per_label:
                return [str(label) for label in per_label]
    return []


def collect_label_columns(
    results: dict[str, Any],
    dimensions: Iterable[str],
) -> list[tuple[str, str]]:
    columns: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for task in LABEL_TASKS:
        for label in collect_labels(results, dimensions, task):
            key = (task, label)
            if key in seen:
                continue
            columns.append(key)
            seen.add(key)
    return columns


def label_prevalence_pair(
    summary: dict[str, Any] | None,
    label: str,
) -> tuple[float, float] | None:
    if not summary:
        return None
    items = int(summary.get("items") or 0)
    if items <= 0:
        return None
    per_label = summary.get("per_label", {})
    values = per_label.get(label) if isinstance(per_label, dict) else None
    if not isinstance(values, dict):
        return None
    kept_present = int(values.get("kept_present") or 0)
    removed = int(values.get("removed") or 0)
    added = int(values.get("added") or 0)
    origin_count = kept_present + removed
    counterfactual_count = kept_present + added
    return origin_count / items, counterfactual_count / items


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def build_label_rows(
    results: dict[str, Any],
    dimensions: list[str],
    directions: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dimension in dimensions:
        for task in LABEL_TASKS:
            for actor in ACTORS:
                for direction in directions:
                    summary = label_summary(results, dimension, task, actor, direction)
                    if not summary:
                        continue
                    per_label = summary.get("per_label", {})
                    if not isinstance(per_label, dict):
                        continue
                    for label, values in per_label.items():
                        if not isinstance(values, dict):
                            continue
                        rows.append(
                            {
                                "dimension": dimension,
                                "task": task,
                                "actor": actor,
                                "appraisal_direction": direction,
                                "label": label,
                                "items": summary.get("items"),
                                "added": values.get("added"),
                                "removed": values.get("removed"),
                                "kept_present": values.get("kept_present"),
                                "kept_absent": values.get("kept_absent"),
                                "added_rate": values.get("added_rate"),
                                "removed_rate": values.get("removed_rate"),
                                "net_change": values.get("net_change"),
                                "net_change_rate": values.get("net_change_rate"),
                            }
                        )
    return rows


def build_label_prevalence_rows(
    results: dict[str, Any],
    dimensions: list[str],
    directions: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dimension in dimensions:
        for task in LABEL_TASKS:
            for actor in ACTORS:
                for direction in directions:
                    summary = label_summary(results, dimension, task, actor, direction)
                    if not summary:
                        continue
                    items = int(summary.get("items") or 0)
                    per_label = summary.get("per_label", {})
                    if items <= 0 or not isinstance(per_label, dict):
                        continue
                    for label, values in per_label.items():
                        if not isinstance(values, dict):
                            continue
                        kept_present = int(values.get("kept_present") or 0)
                        removed = int(values.get("removed") or 0)
                        added = int(values.get("added") or 0)
                        origin_count = kept_present + removed
                        counterfactual_count = kept_present + added
                        rows.append(
                            {
                                "dimension": dimension,
                                "task": task,
                                "actor": actor,
                                "appraisal_direction": direction,
                                "label": label,
                                "items": items,
                                "origin_count": origin_count,
                                "counterfactual_count": counterfactual_count,
                                "origin_rate": origin_count / items,
                                "counterfactual_rate": counterfactual_count / items,
                            }
                        )
    return rows


def build_intensity_rows(
    results: dict[str, Any],
    dimensions: list[str],
    directions: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dimension in dimensions:
        for task in LEVEL_TASKS:
            for direction in directions:
                summary = intensity_summary(results, dimension, task, direction)
                if not summary:
                    continue
                rows.append(
                    {
                        "dimension": dimension,
                        "task": task,
                        "appraisal_direction": direction,
                        "items": summary.get("items"),
                        "mean_human_delta": summary.get("mean_human_delta"),
                        "mean_model_delta": summary.get("mean_model_delta"),
                        "same_direction_rate": summary.get("same_direction_rate"),
                        "mean_original_human": summary.get("mean_original_human"),
                        "mean_counterfactual_human": summary.get("mean_counterfactual_human"),
                        "mean_original_model": summary.get("mean_original_model"),
                        "mean_counterfactual_model": summary.get("mean_counterfactual_model"),
                    }
                )
    return rows


def build_transition_rows(
    results: dict[str, Any],
    dimensions: list[str],
    directions: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dimension in dimensions:
        for task in LABEL_TASKS:
            for actor in ACTORS:
                for direction in directions:
                    summary = label_summary(results, dimension, task, actor, direction)
                    if not summary:
                        continue
                    transitions = summary.get("top_transitions", [])
                    if not isinstance(transitions, list):
                        continue
                    for transition in transitions:
                        if not isinstance(transition, dict):
                            continue
                        rows.append(
                            {
                                "dimension": dimension,
                                "task": task,
                                "actor": actor,
                                "appraisal_direction": direction,
                                "items": summary.get("items"),
                                "from": transition.get("from"),
                                "to": transition.get("to"),
                                "count": transition.get("count"),
                                "rate": transition.get("rate"),
                            }
                        )
    return rows


def import_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def plot_label_heatmaps(
    plt: Any,
    results: dict[str, Any],
    dimensions: list[str],
    directions: list[str],
    output_dir: Path,
    image_format: str,
) -> list[Path]:
    written: list[Path] = []
    for task in LABEL_TASKS:
        labels = collect_labels(results, dimensions, task)
        if not labels:
            continue
        for actor in ACTORS:
            for direction in directions:
                matrix: list[list[float]] = []
                active_dimensions: list[str] = []
                for dimension in dimensions:
                    summary = label_summary(results, dimension, task, actor, direction)
                    if not summary or int(summary.get("items") or 0) == 0:
                        continue
                    per_label = summary.get("per_label", {})
                    if not isinstance(per_label, dict):
                        continue
                    row: list[float] = []
                    for label in labels:
                        values = per_label.get(label, {})
                        value = (
                            values.get("net_change_rate")
                            if isinstance(values, dict)
                            else None
                        )
                        row.append(float(value) if value is not None else 0.0)
                    matrix.append(row)
                    active_dimensions.append(dimension)
                if not matrix:
                    continue
                max_abs = max(abs(value) for row in matrix for value in row)
                limit = max(max_abs, 0.01)
                fig_width = max(8.0, 0.55 * len(labels) + 3.0)
                fig_height = max(3.0, 0.55 * len(active_dimensions) + 1.8)
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                image = ax.imshow(
                    matrix,
                    aspect="auto",
                    cmap="coolwarm",
                    vmin=-limit,
                    vmax=limit,
                )
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels([short_label(label) for label in labels], rotation=45, ha="right")
                ax.set_yticks(range(len(active_dimensions)))
                ax.set_yticklabels(active_dimensions)
                ax.set_title(f"{actor} label net-change rate: {task}, {direction}")
                ax.set_xlabel("Emotion label")
                ax.set_ylabel("Intervened appraisal dimension")
                fig.colorbar(image, ax=ax, label="added rate - removed rate")
                fig.tight_layout()
                filename = sanitize_filename(
                    f"label_net_change_{actor}_{task}_{direction}.{image_format}"
                )
                path = output_dir / filename
                fig.savefig(path, dpi=200)
                plt.close(fig)
                written.append(path)
    return written


def plot_origin_counterfactual_label_prevalence(
    plt: Any,
    results: dict[str, Any],
    dimensions: list[str],
    directions: list[str],
    actors: list[str],
    output_dir: Path,
    image_format: str,
    split_by_dimension: bool = False,
) -> list[Path]:
    written: list[Path] = []
    label_columns = collect_label_columns(results, dimensions)
    if not label_columns:
        return written

    x_labels = [
        f"{'P' if task == 'positive-labels' else 'N'}:{short_label(label)}"
        for task, label in label_columns
    ]
    positive_count = sum(1 for task, _ in label_columns if task == "positive-labels")
    dimension_groups: list[tuple[str | None, list[str]]]
    if split_by_dimension:
        dimension_groups = [(dimension, [dimension]) for dimension in dimensions]
    else:
        dimension_groups = [(None, dimensions)]

    for actor in actors:
        for direction in directions:
            for dimension_name, group_dimensions in dimension_groups:
                origin_matrix: list[list[float]] = []
                counterfactual_matrix: list[list[float]] = []
                active_dimensions: list[str] = []

                for dimension in group_dimensions:
                    origin_row: list[float] = []
                    counterfactual_row: list[float] = []
                    has_values = False

                    for task, label in label_columns:
                        summary = label_summary(results, dimension, task, actor, direction)
                        pair = label_prevalence_pair(summary, label)
                        if pair is None:
                            origin_row.append(float("nan"))
                            counterfactual_row.append(float("nan"))
                            continue
                        origin_rate, counterfactual_rate = pair
                        origin_row.append(origin_rate)
                        counterfactual_row.append(counterfactual_rate)
                        has_values = True

                    if has_values:
                        origin_matrix.append(origin_row)
                        counterfactual_matrix.append(counterfactual_row)
                        active_dimensions.append(dimension)

                if not active_dimensions:
                    continue

                cmap = plt.get_cmap("YlGnBu").copy()
                cmap.set_bad("#f2f2f2")
                fig_width = max(10.0, 0.48 * len(label_columns) + 3.8)
                fig_height = max(5.8, 0.72 * len(active_dimensions) * 2 + 2.0)
                fig, axes = plt.subplots(
                    2,
                    1,
                    figsize=(fig_width, fig_height),
                    sharex=True,
                    constrained_layout=True,
                )
                axes_list = list(axes)

                image = None
                for ax, matrix, title in (
                    (axes_list[0], origin_matrix, "Origin"),
                    (axes_list[1], counterfactual_matrix, "Counterfactual"),
                ):
                    image = ax.imshow(
                        matrix,
                        aspect="auto",
                        cmap=cmap,
                        vmin=0.0,
                        vmax=1.0,
                    )
                    ax.set_title(title)
                    ax.set_yticks(range(len(active_dimensions)))
                    ax.set_yticklabels(active_dimensions)
                    ax.set_ylabel("Appraisal")
                    if 0 < positive_count < len(label_columns):
                        ax.axvline(positive_count - 0.5, color="white", linewidth=1.4)

                axes_list[-1].set_xticks(range(len(label_columns)))
                axes_list[-1].set_xticklabels(x_labels, rotation=45, ha="right")
                axes_list[-1].set_xlabel("Emotion label")
                title_parts = [f"{actor} label prevalence before/after intervention"]
                if dimension_name is not None:
                    title_parts.append(dimension_name)
                title_parts.append(direction)
                fig.suptitle(" | ".join(title_parts))
                if image is not None:
                    fig.colorbar(
                        image,
                        ax=axes_list,
                        label="Label prevalence",
                        fraction=0.025,
                        pad=0.02,
                    )
                filename_parts = [
                    "origin_vs_counterfactual_label_prevalence",
                    actor,
                ]
                if dimension_name is not None:
                    filename_parts.append(dimension_name)
                filename_parts.append(direction)
                filename = sanitize_filename(
                    f"{'_'.join(filename_parts)}.{image_format}"
                )
                path = output_dir / filename
                fig.savefig(path, dpi=240)
                plt.close(fig)
                written.append(path)

    return written


def plot_intensity_bars(
    plt: Any,
    results: dict[str, Any],
    dimensions: list[str],
    directions: list[str],
    output_dir: Path,
    image_format: str,
) -> list[Path]:
    written: list[Path] = []
    for task in LEVEL_TASKS:
        for direction in directions:
            active_dimensions: list[str] = []
            human_values: list[float] = []
            model_values: list[float] = []
            for dimension in dimensions:
                summary = intensity_summary(results, dimension, task, direction)
                if not summary or int(summary.get("items") or 0) == 0:
                    continue
                human_delta = summary.get("mean_human_delta")
                model_delta = summary.get("mean_model_delta")
                if human_delta is None or model_delta is None:
                    continue
                active_dimensions.append(dimension)
                human_values.append(float(human_delta))
                model_values.append(float(model_delta))
            if not active_dimensions:
                continue
            x_positions = list(range(len(active_dimensions)))
            width = 0.38
            fig_width = max(7.5, 1.2 * len(active_dimensions) + 2.0)
            fig, ax = plt.subplots(figsize=(fig_width, 4.5))
            ax.bar(
                [x - width / 2 for x in x_positions],
                human_values,
                width,
                label="human",
            )
            ax.bar(
                [x + width / 2 for x in x_positions],
                model_values,
                width,
                label="model",
            )
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(active_dimensions, rotation=30, ha="right")
            ax.set_ylabel("Mean counterfactual - original intensity")
            ax.set_title(f"Intensity delta: {task}, {direction}")
            ax.legend()
            fig.tight_layout()
            filename = sanitize_filename(f"intensity_delta_{task}_{direction}.{image_format}")
            path = output_dir / filename
            fig.savefig(path, dpi=200)
            plt.close(fig)
            written.append(path)
    return written


def plot_top_transitions(
    plt: Any,
    results: dict[str, Any],
    dimensions: list[str],
    directions: list[str],
    output_dir: Path,
    image_format: str,
    top_k: int,
) -> list[Path]:
    written: list[Path] = []
    for dimension in dimensions:
        for task in LABEL_TASKS:
            for actor in ACTORS:
                for direction in directions:
                    summary = label_summary(results, dimension, task, actor, direction)
                    if not summary:
                        continue
                    transitions = summary.get("top_transitions", [])
                    if not isinstance(transitions, list):
                        continue
                    transitions = [
                        item
                        for item in transitions[:top_k]
                        if isinstance(item, dict) and item.get("count")
                    ]
                    if not transitions:
                        continue
                    labels = [
                        f"{item.get('from')} -> {item.get('to')}"
                        for item in transitions
                    ]
                    counts = [int(item.get("count") or 0) for item in transitions]
                    fig_height = max(3.2, 0.42 * len(labels) + 1.2)
                    fig, ax = plt.subplots(figsize=(9.0, fig_height))
                    positions = list(range(len(labels)))
                    ax.barh(positions, counts)
                    ax.set_yticks(positions)
                    ax.set_yticklabels(labels)
                    ax.invert_yaxis()
                    ax.set_xlabel("Count")
                    ax.set_title(f"Top label transitions: {dimension}, {task}, {actor}, {direction}")
                    fig.tight_layout()
                    filename = sanitize_filename(
                        f"top_transitions_{dimension}_{task}_{actor}_{direction}.{image_format}"
                    )
                    path = output_dir / filename
                    fig.savefig(path, dpi=200)
                    plt.close(fig)
                    written.append(path)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot counterfactual emotion label/intensity transitions"
    )
    parser.add_argument("--results_file", type=Path, required=True)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Defaults to <results_file stem>_transition_plots next to results_file.",
    )
    parser.add_argument(
        "--directions",
        type=str,
        default=",".join(DEFAULT_DIRECTIONS),
        help="Comma-separated appraisal directions for detailed CSVs/plots.",
    )
    parser.add_argument(
        "--simple_directions",
        type=str,
        default="all",
        help="Comma-separated appraisal directions for the origin-vs-counterfactual heatmap.",
    )
    parser.add_argument(
        "--simple_actor",
        choices=["model", "human", "both"],
        default="model",
        help="Whose labels to draw in the origin-vs-counterfactual heatmap.",
    )
    parser.add_argument(
        "--split_simple_by_dimension",
        action="store_true",
        help="Write one origin-vs-counterfactual heatmap per appraisal dimension and direction.",
    )
    parser.add_argument("--format", choices=["png", "pdf", "svg"], default="png")
    parser.add_argument("--top_k", type=int, default=12)
    parser.add_argument(
        "--only_simple",
        action="store_true",
        help="Only write the origin-vs-counterfactual heatmap, plus CSV tables.",
    )
    parser.add_argument(
        "--plot_top_transitions",
        action="store_true",
        help="Also write one top-transition bar chart per dimension/task/actor/direction.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.top_k <= 0:
        raise ValueError("--top_k must be positive")

    results = load_json(args.results_file)
    dimensions = iter_dimensions(results)
    if not dimensions:
        raise ValueError("No dimensions found in results file")
    directions = parse_directions(args.directions)
    output_dir = args.output_dir or (
        args.results_file.parent / f"{args.results_file.stem}_transition_plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    simple_directions = parse_directions(args.simple_directions)
    simple_actors = parse_simple_actors(args.simple_actor)

    label_rows = build_label_rows(results, dimensions, directions)
    prevalence_rows = build_label_prevalence_rows(results, dimensions, directions)
    intensity_rows = build_intensity_rows(results, dimensions, directions)
    transition_rows = build_transition_rows(results, dimensions, directions)

    write_csv(
        output_dir / "label_net_changes.csv",
        label_rows,
        [
            "dimension",
            "task",
            "actor",
            "appraisal_direction",
            "label",
            "items",
            "added",
            "removed",
            "kept_present",
            "kept_absent",
            "added_rate",
            "removed_rate",
            "net_change",
            "net_change_rate",
        ],
    )
    write_csv(
        output_dir / "label_prevalence.csv",
        prevalence_rows,
        [
            "dimension",
            "task",
            "actor",
            "appraisal_direction",
            "label",
            "items",
            "origin_count",
            "counterfactual_count",
            "origin_rate",
            "counterfactual_rate",
        ],
    )
    write_csv(
        output_dir / "intensity_changes.csv",
        intensity_rows,
        [
            "dimension",
            "task",
            "appraisal_direction",
            "items",
            "mean_human_delta",
            "mean_model_delta",
            "same_direction_rate",
            "mean_original_human",
            "mean_counterfactual_human",
            "mean_original_model",
            "mean_counterfactual_model",
        ],
    )
    write_csv(
        output_dir / "top_label_transitions.csv",
        transition_rows,
        [
            "dimension",
            "task",
            "actor",
            "appraisal_direction",
            "items",
            "from",
            "to",
            "count",
            "rate",
        ],
    )

    written_plots: list[Path] = []
    plt = import_matplotlib()
    if plt is None:
        print("[warning] matplotlib is not installed; wrote CSV tables only.")
    else:
        written_plots.extend(
            plot_origin_counterfactual_label_prevalence(
                plt,
                results,
                dimensions,
                simple_directions,
                simple_actors,
                output_dir,
                args.format,
                split_by_dimension=args.split_simple_by_dimension,
            )
        )
        if not args.only_simple:
            written_plots.extend(
                plot_label_heatmaps(
                    plt,
                    results,
                    dimensions,
                    directions,
                    output_dir,
                    args.format,
                )
            )
            written_plots.extend(
                plot_intensity_bars(
                    plt,
                    results,
                    dimensions,
                    directions,
                    output_dir,
                    args.format,
                )
            )
            if args.plot_top_transitions:
                written_plots.extend(
                    plot_top_transitions(
                        plt,
                        results,
                        dimensions,
                        directions,
                        output_dir,
                        args.format,
                        args.top_k,
                    )
                )

    if not label_rows:
        print(
            "[warning] no label_transitions found; rerun evaluate_counterfactual_emotion.py "
            "with the updated code before plotting label prevalence."
        )
    print(f"[done] wrote CSV tables to: {output_dir}")
    if written_plots:
        print(f"[done] wrote {len(written_plots)} plot(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
