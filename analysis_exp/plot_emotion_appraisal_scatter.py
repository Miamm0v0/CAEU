#!/usr/bin/env python3
"""Plot sample-level emotion quality against appraisal reasoning quality.

By default, the script discovers ``*/samples.csv`` files under
``output_analysis/emotion_correct`` and draws one panel per run. Explicit
series can be supplied as ``--series LABEL=PATH``.

Examples:
  python analysis_exp/plot_emotion_appraisal_scatter.py

  python analysis_exp/plot_emotion_appraisal_scatter.py \
    --series Base=output_analysis/emotion_correct/base/samples.csv \
    --series Ours=output_analysis/emotion_correct/grpo_from_warmup/samples.csv \
    --layout panels \
    --output_file output_analysis/emotion_correct/emotion_appraisal_scatter.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS_ROOT = REPO_ROOT / "output_analysis" / "emotion_correct"
DEFAULT_EMOTION_FIELD = "emotion_label_f1"
DEFAULT_APPRAISAL_FIELD = "appraisal_reasoning_rouge_l"
FIELD_LABELS = {
    "emotion_label_f1": "Emotion label F1",
    "appraisal_reasoning_bleu": "Appraisal reasoning BLEU-4",
    "appraisal_reasoning_rouge_1": "Appraisal reasoning ROUGE-1 F1",
    "appraisal_reasoning_rouge_2": "Appraisal reasoning ROUGE-2 F1",
    "appraisal_reasoning_rouge_l": "Appraisal reasoning ROUGE-L F1",
    "appraisal_reasoning_bertscore": "Appraisal reasoning BERTScore F1",
    "appraisal_accuracy": "Appraisal rating accuracy",
    "appraisal_normalized_mae": "Appraisal rating normalized MAE",
    "appraisal_normalized_rmse": "Appraisal rating normalized RMSE",
    "appraisal_within_one_accuracy": "Appraisal rating within-one accuracy",
}
APPRAISAL_FIELD_BOUNDS: dict[str, tuple[float, float]] = {
    "appraisal_reasoning_bleu": (0.0, 1.0),
    "appraisal_reasoning_rouge_1": (0.0, 1.0),
    "appraisal_reasoning_rouge_2": (0.0, 1.0),
    "appraisal_reasoning_rouge_l": (0.0, 1.0),
    "appraisal_reasoning_bertscore": (-1.0, 1.0),
    "appraisal_accuracy": (0.0, 1.0),
    "appraisal_mae": (0.0, 4.0),
    "appraisal_normalized_mae": (0.0, 1.0),
    "appraisal_normalized_rmse": (0.0, 1.0),
    "appraisal_within_one_accuracy": (0.0, 1.0),
    "appraisal_exact_count": (0.0, 22.0),
    "appraisal_opposite_side_errors": (0.0, 22.0),
}
DEFAULT_THRESHOLD = 0.6
COLORS = (
    "#4C6A92",
    "#C5543D",
    "#2F7D63",
    "#8C6D31",
    "#76528B",
    "#3B7C8C",
)
MARKERS = ("o", "s", "^", "D", "P", "X")


@dataclass(frozen=True)
class Point:
    sample_id: str
    emotion: float
    appraisal: float


@dataclass(frozen=True)
class SeriesData:
    label: str
    path: Path
    points: tuple[Point, ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot emotion metric versus appraisal reasoning metric from "
            "emotion_correct analysis samples.csv files."
        )
    )
    parser.add_argument(
        "--analysis_root",
        type=Path,
        default=DEFAULT_ANALYSIS_ROOT,
        help=(
            "Directory containing one samples.csv per run subdirectory. "
            "Default: output_analysis/emotion_correct."
        ),
    )
    parser.add_argument(
        "--series",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help=(
            "Explicit series label and samples.csv path. Repeat for multiple "
            "runs. When omitted, runs are discovered under --analysis_root."
        ),
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Optional run subdirectories to select during auto-discovery.",
    )
    parser.add_argument(
        "--emotion_field",
        default=DEFAULT_EMOTION_FIELD,
        help=f"CSV field for the x-axis. Default: {DEFAULT_EMOTION_FIELD}.",
    )
    parser.add_argument(
        "--appraisal_field",
        default=DEFAULT_APPRAISAL_FIELD,
        help=f"CSV field for the y-axis. Default: {DEFAULT_APPRAISAL_FIELD}.",
    )
    parser.add_argument(
        "--emotion_threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Draw a vertical emotion-selection threshold. Default: 0.6.",
    )
    parser.add_argument(
        "--no_threshold",
        action="store_true",
        help="Do not draw the emotion-selection threshold.",
    )
    parser.add_argument(
        "--layout",
        choices=("panels", "overlay"),
        default="panels",
        help="Draw one panel per run or overlay all runs. Default: panels.",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=None,
        help=(
            "PNG/PDF/SVG output path. Defaults to "
            "<analysis_root>/emotion_appraisal_scatter.png."
        ),
    )
    parser.add_argument(
        "--summary_file",
        type=Path,
        default=None,
        help="Correlation summary CSV path. Defaults next to --output_file.",
    )
    parser.add_argument("--title", default="Emotion-Appraisal Relationship")
    parser.add_argument("--dpi", type=int, default=220)
    return parser


def display_label(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    if normalized == "base":
        return "Base"
    if normalized in {"ours", "grpo", "grpo_from_warmup"}:
        return "Ours (GRPO from warmup)"
    return name.replace("_", " ").strip().title()


def field_label(field: str) -> str:
    return FIELD_LABELS.get(field, field.replace("_", " ").strip().title())


def parse_series_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return display_label(path.parent.name or path.stem), path
    label, raw_path = value.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise ValueError(f"Invalid --series value: {value!r}; expected LABEL=PATH")
    return label.strip(), Path(raw_path.strip())


def discover_series(
    analysis_root: Path, requested_runs: Sequence[str] | None
) -> list[tuple[str, Path]]:
    if not analysis_root.is_dir():
        raise FileNotFoundError(f"analysis_root not found: {analysis_root}")

    if requested_runs:
        candidates = [analysis_root / run / "samples.csv" for run in requested_runs]
    else:
        candidates = sorted(analysis_root.glob("*/samples.csv"))
        direct = analysis_root / "samples.csv"
        if direct.is_file():
            candidates.insert(0, direct)

    missing = [path for path in candidates if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing samples.csv file(s): " + ", ".join(str(path) for path in missing)
        )
    if not candidates:
        raise FileNotFoundError(
            f"No samples.csv files found under analysis_root: {analysis_root}"
        )

    discovered: list[tuple[str, Path]] = []
    for path in candidates:
        run_name = path.parent.name if path.parent != analysis_root else analysis_root.name
        discovered.append((display_label(run_name), path))
    discovered.sort(key=lambda item: (item[0] != "Base", item[0].lower()))
    return discovered


def parse_bool(value: Any) -> bool:
    if value is None or value == "":
        return True
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_series(
    label: str,
    path: Path,
    emotion_field: str,
    appraisal_field: str,
) -> SeriesData:
    if not path.is_file():
        raise FileNotFoundError(f"samples.csv not found: {path}")

    points: list[Point] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required = {emotion_field, appraisal_field}
        if not required.issubset(fieldnames):
            raise ValueError(
                f"{path} is missing required field(s): "
                + ", ".join(sorted(required - fieldnames))
            )
        for index, row in enumerate(reader):
            if "included" in fieldnames and not parse_bool(row.get("included")):
                continue
            emotion = finite_float(row.get(emotion_field))
            appraisal = finite_float(row.get(appraisal_field))
            if emotion is None or appraisal is None:
                continue
            points.append(
                Point(
                    sample_id=(row.get("sample_id") or f"row_{index}").strip(),
                    emotion=emotion,
                    appraisal=appraisal,
                )
            )
    if not points:
        raise ValueError(f"No valid included points found in: {path}")
    return SeriesData(label=label, path=path, points=tuple(points))


def average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def pearson(x_values: Sequence[float], y_values: Sequence[float]) -> float | None:
    if len(x_values) < 2 or len(x_values) != len(y_values):
        return None
    mean_x = statistics.fmean(x_values)
    mean_y = statistics.fmean(y_values)
    centered_x = [value - mean_x for value in x_values]
    centered_y = [value - mean_y for value in y_values]
    denominator = math.sqrt(
        sum(value * value for value in centered_x)
        * sum(value * value for value in centered_y)
    )
    if denominator == 0:
        return None
    return sum(a * b for a, b in zip(centered_x, centered_y)) / denominator


def linear_fit(
    x_values: Sequence[float], y_values: Sequence[float]
) -> tuple[float, float] | None:
    if len(x_values) < 2 or len(x_values) != len(y_values):
        return None
    mean_x = statistics.fmean(x_values)
    mean_y = statistics.fmean(y_values)
    denominator = sum((value - mean_x) ** 2 for value in x_values)
    if denominator == 0:
        return None
    slope = sum(
        (x_value - mean_x) * (y_value - mean_y)
        for x_value, y_value in zip(x_values, y_values)
    ) / denominator
    return slope, mean_y - slope * mean_x


def summarize(series: SeriesData, threshold: float) -> dict[str, Any]:
    x_values = [point.emotion for point in series.points]
    y_values = [point.appraisal for point in series.points]
    selected = [
        point.appraisal for point in series.points if point.emotion >= threshold
    ]
    not_selected = [
        point.appraisal for point in series.points if point.emotion < threshold
    ]
    fit = linear_fit(x_values, y_values)
    return {
        "label": series.label,
        "samples_csv": str(series.path),
        "samples": len(series.points),
        "emotion_mean": statistics.fmean(x_values),
        "appraisal_mean": statistics.fmean(y_values),
        "pearson": pearson(x_values, y_values),
        "spearman": pearson(average_ranks(x_values), average_ranks(y_values)),
        "regression_slope": fit[0] if fit else None,
        "regression_intercept": fit[1] if fit else None,
        "emotion_threshold": threshold,
        "above_threshold_samples": len(selected),
        "appraisal_mean_above_threshold": (
            statistics.fmean(selected) if selected else None
        ),
        "below_threshold_samples": len(not_selected),
        "appraisal_mean_below_threshold": (
            statistics.fmean(not_selected) if not_selected else None
        ),
    }


def write_summary(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_stat(value: float | None) -> str:
    return "NA" if value is None else f"{value:.3f}"


def padded_data_limits(
    values: Sequence[float],
    bounds: tuple[float, float] | None,
) -> tuple[float, float]:
    data_low = min(values)
    data_high = max(values)
    span = data_high - data_low
    if bounds is None:
        padding = max(0.02, span * 0.08)
        if span == 0.0:
            padding = max(0.02, abs(data_low) * 0.05)
        return data_low - padding, data_high + padding

    bound_low, bound_high = bounds
    if data_low < bound_low or data_high > bound_high:
        return padded_data_limits(values, None)
    padding = max((bound_high - bound_low) * 0.02, span * 0.08)
    low = max(bound_low, data_low - padding)
    high = min(bound_high, data_high + padding)
    if low == high:
        low = max(bound_low, low - padding)
        high = min(bound_high, high + padding)
    return low, high


def axis_limits(
    all_series: Sequence[SeriesData], appraisal_field: str
) -> tuple[tuple[float, float], tuple[float, float]]:
    x_values = [point.emotion for series in all_series for point in series.points]
    y_values = [point.appraisal for series in all_series for point in series.points]
    x_low = min(0.0, min(x_values))
    x_high = max(1.0, max(x_values))
    y_limits = padded_data_limits(
        y_values, APPRAISAL_FIELD_BOUNDS.get(appraisal_field)
    )
    return (x_low, x_high), y_limits


def draw_series(
    ax: Any,
    series: SeriesData,
    summary: dict[str, Any],
    *,
    color: str,
    marker: str,
    threshold: float | None,
    show_label: bool,
    show_annotation: bool,
) -> None:
    x_values = [point.emotion for point in series.points]
    y_values = [point.appraisal for point in series.points]
    ax.scatter(
        x_values,
        y_values,
        s=42,
        alpha=0.7,
        color=color,
        edgecolors="white",
        linewidths=0.45,
        marker=marker,
        label=series.label if show_label else None,
        zorder=3,
    )
    if threshold is not None:
        ax.axvline(
            threshold,
            color="#555555",
            linestyle="--",
            linewidth=1.2,
            alpha=0.75,
            zorder=2,
        )

    if show_annotation:
        annotation = (
            f"n = {summary['samples']}\n"
            f"Pearson r = {format_stat(summary['pearson'])}\n"
            f"Spearman r_s = {format_stat(summary['spearman'])}"
        )
        ax.text(
            0.03,
            0.97,
            annotation,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="#202020",
            bbox={
                "boxstyle": "square,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#D0D0D0",
                "alpha": 0.9,
            },
            zorder=5,
        )


def create_figure(
    all_series: Sequence[SeriesData],
    summaries: Sequence[dict[str, Any]],
    *,
    layout: str,
    threshold: float | None,
    title: str,
    emotion_field: str,
    appraisal_field: str,
) -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_limits, y_limits = axis_limits(all_series, appraisal_field)
    if layout == "overlay":
        figure, axis = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
        axes = [axis]
        for index, (series, summary) in enumerate(zip(all_series, summaries)):
            draw_series(
                axis,
                series,
                summary,
                color=COLORS[index % len(COLORS)],
                marker=MARKERS[index % len(MARKERS)],
                threshold=threshold if index == 0 else None,
                show_label=True,
                show_annotation=False,
            )
        axis.legend(frameon=False, loc="lower right")
    else:
        columns = min(2, len(all_series))
        rows = math.ceil(len(all_series) / columns)
        figure, raw_axes = plt.subplots(
            rows,
            columns,
            figsize=(6.2 * columns, 4.8 * rows),
            squeeze=False,
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        axes = [axis for row in raw_axes for axis in row]
        for index, (series, summary) in enumerate(zip(all_series, summaries)):
            axis = axes[index]
            draw_series(
                axis,
                series,
                summary,
                color=COLORS[index % len(COLORS)],
                marker=MARKERS[index % len(MARKERS)],
                threshold=threshold,
                show_label=False,
                show_annotation=True,
            )
            axis.set_title(series.label, fontsize=12, fontweight="semibold")
        for axis in axes[len(all_series) :]:
            axis.set_visible(False)

    for axis in axes[: len(all_series)] if layout == "panels" else axes:
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.set_xlabel(field_label(emotion_field))
        axis.set_ylabel(field_label(appraisal_field))
        axis.grid(True, color="#E5E5E5", linewidth=0.8, zorder=1)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.suptitle(title, fontsize=14, fontweight="semibold")
    return figure


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    if not 0.0 <= args.emotion_threshold <= 1.0:
        raise ValueError("--emotion_threshold must be between 0 and 1")

    specs = (
        [parse_series_spec(value) for value in args.series]
        if args.series
        else discover_series(args.analysis_root, args.runs)
    )
    all_series = [
        load_series(label, path, args.emotion_field, args.appraisal_field)
        for label, path in specs
    ]
    summaries = [
        summarize(series, args.emotion_threshold) for series in all_series
    ]

    output_file = args.output_file or (
        args.analysis_root / "emotion_appraisal_scatter.png"
    )
    if output_file.suffix.lower() not in {".png", ".pdf", ".svg"}:
        raise ValueError("--output_file must end in .png, .pdf, or .svg")
    summary_file = args.summary_file or output_file.with_name(
        f"{output_file.stem}_summary.csv"
    )

    threshold = None if args.no_threshold else args.emotion_threshold
    try:
        figure = create_figure(
            all_series,
            summaries,
            layout=args.layout,
            threshold=threshold,
            title=args.title,
            emotion_field=args.emotion_field,
            appraisal_field=args.appraisal_field,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        raise SystemExit(
            "matplotlib is required to render the scatter plot; "
            "install it with `pip install matplotlib`."
        ) from exc
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=args.dpi, bbox_inches="tight")
    write_summary(summary_file, summaries)

    print(f"[done] series={len(all_series)} figure={output_file}")
    print(f"[done] summary={summary_file}")
    for row in summaries:
        print(
            "[summary] {} n={} pearson={} spearman={}".format(
                row["label"],
                row["samples"],
                format_stat(row["pearson"]),
                format_stat(row["spearman"]),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
