#!/usr/bin/env python3
"""Create paper-ready appraisal comparison panels for Base and Ours.

The script combines the outputs of the two analysis experiments:

* ``emotion_correct/*/samples.csv`` and ``dimensions.csv`` provide panels a/b;
* ``appraisal_third_person/*/samples.csv`` provides panel c.

Panel c defines the per-situation perspective margin as

``M_i = ROUGE-L(first-person) - ROUGE-L(third-person mean reference)``.

Positive values therefore indicate greater similarity to first-person gold.
The Base/Ours difference is paired by ``sample_id`` and tested with a
two-sided paired random-sign permutation test.

Example:
  python analysis_exp/plot_appraisal_comparison_panels.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMOTION_ROOT = REPO_ROOT / "output_analysis" / "emotion_correct"
DEFAULT_PERSPECTIVE_ROOT = (
    REPO_ROOT / "output_analysis" / "appraisal_third_person"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output_analysis" / "figures"

BASE_COLOR = "#4C78A8"
OURS_COLOR = "#F28E2B"
LIGHT_GRAY = "#C9CDD2"
TEXT_COLOR = "#222222"
GRID_COLOR = "#E3E5E8"
DIMENSIONS = (
    "relevance",
    "congruence",
    "accountability",
    "control",
    "certainty",
)
DIMENSION_LABELS = {
    "relevance": "Relevance",
    "congruence": "Congruence",
    "accountability": "Accountability",
    "control": "Control",
    "certainty": "Certainty",
}
DIMENSION_FIELDS = {
    "all": "all_complete_rouge_l_f1",
    "emotion-correct": "emotion_correct_rouge_l_f1",
    "emotion-incorrect": "emotion_incorrect_rouge_l_f1",
}


@dataclass(frozen=True)
class PerspectivePair:
    sample_id: str
    base: float
    ours: float

    @property
    def delta(self) -> float:
        return self.ours - self.base


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot emotion-stratified appraisal ROUGE-L, dimension-level "
            "ROUGE-L, and paired first/third-person perspective margins."
        )
    )
    parser.add_argument("--emotion_root", type=Path, default=DEFAULT_EMOTION_ROOT)
    parser.add_argument(
        "--perspective_root", type=Path, default=DEFAULT_PERSPECTIVE_ROOT
    )
    parser.add_argument("--base_run", default="base")
    parser.add_argument("--ours_run", default="grpo_from_warmup")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--dimension_subset",
        choices=tuple(DIMENSION_FIELDS),
        default="all",
        help="Sample subset used for panel b. Default: all.",
    )
    parser.add_argument(
        "--perspective_subset",
        choices=("all", "selected-low"),
        default="all",
        help="Use all paired situations or only rows selected as low quality.",
    )
    parser.add_argument(
        "--bootstrap_samples",
        type=int,
        default=10000,
        help="Bootstrap resamples used for mean confidence intervals.",
    )
    parser.add_argument(
        "--permutation_samples",
        type=int,
        default=50000,
        help="Random-sign permutations used for the paired p-value.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=("png", "pdf"),
    )
    return parser


def finite_float(value: Any, *, field: str, path: Path) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: invalid {field} value {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{path}: non-finite {field} value {value!r}")
    return number


def parse_bool(value: Any, *, field: str, path: Path) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"{path}: invalid {field} value {value!r}")


def read_csv(path: Path) -> tuple[list[dict[str, str]], set[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        rows = list(reader)
    return rows, fields


def require_fields(path: Path, fields: set[str], required: set[str]) -> None:
    missing = sorted(required - fields)
    if missing:
        raise ValueError(f"{path}: missing field(s): {', '.join(missing)}")


def load_emotion_groups(
    path: Path,
    value_field: str = "appraisal_reasoning_rouge_l",
) -> dict[bool, list[float]]:
    rows, fields = read_csv(path)
    require_fields(path, fields, {"emotion_correct", value_field})
    groups: dict[bool, list[float]] = {False: [], True: []}
    for row in rows:
        if "included" in fields and not parse_bool(
            row.get("included"), field="included", path=path
        ):
            continue
        correct = parse_bool(
            row.get("emotion_correct"), field="emotion_correct", path=path
        )
        groups[correct].append(
            finite_float(row.get(value_field), field=value_field, path=path)
        )
    for correct, values in groups.items():
        if not values:
            label = "correct" if correct else "incorrect"
            raise ValueError(f"{path}: no included emotion-{label} rows")
    return groups


def load_dimension_scores(path: Path, field: str) -> dict[str, float]:
    rows, fields = read_csv(path)
    require_fields(path, fields, {"dimension", field})
    scores: dict[str, float] = {}
    for row in rows:
        dimension = str(row.get("dimension", "")).strip().lower()
        if dimension not in DIMENSIONS:
            continue
        if dimension in scores:
            raise ValueError(f"{path}: duplicate dimension {dimension!r}")
        scores[dimension] = finite_float(
            row.get(field), field=field, path=path
        )
    missing = [dimension for dimension in DIMENSIONS if dimension not in scores]
    if missing:
        raise ValueError(f"{path}: missing dimension(s): {', '.join(missing)}")
    return scores


def load_perspective_margins(
    path: Path,
    subset: str,
) -> dict[str, float]:
    first_field = "rouge-l_first_person"
    third_field = "rouge-l_third_person_mean_reference"
    rows, fields = read_csv(path)
    required = {"sample_id", first_field, third_field}
    if subset == "selected-low":
        required.add("selected_low")
    require_fields(path, fields, required)

    margins: dict[str, float] = {}
    for row in rows:
        if subset == "selected-low" and not parse_bool(
            row.get("selected_low"), field="selected_low", path=path
        ):
            continue
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError(f"{path}: empty sample_id")
        if sample_id in margins:
            raise ValueError(f"{path}: duplicate sample_id {sample_id!r}")
        first_score = finite_float(
            row.get(first_field), field=first_field, path=path
        )
        third_score = finite_float(
            row.get(third_field), field=third_field, path=path
        )
        margins[sample_id] = first_score - third_score
    if not margins:
        raise ValueError(f"{path}: no rows available for subset {subset!r}")
    return margins


def pair_perspective_margins(
    base: Mapping[str, float], ours: Mapping[str, float]
) -> list[PerspectivePair]:
    shared = sorted(set(base) & set(ours))
    if not shared:
        raise ValueError("Base and Ours have no shared perspective sample_id values")
    return [PerspectivePair(sample_id, base[sample_id], ours[sample_id]) for sample_id in shared]


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_means(
    values: Sequence[float], resamples: int, rng: random.Random
) -> list[float]:
    if not values:
        raise ValueError("bootstrap requires at least one value")
    if resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    count = len(values)
    return [
        statistics.fmean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(resamples)
    ]


def summarize_values(
    values: Sequence[float], resamples: int, rng: random.Random
) -> dict[str, float | int]:
    bootstrapped = bootstrap_means(values, resamples, rng)
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "ci_low": percentile(bootstrapped, 0.025),
        "ci_high": percentile(bootstrapped, 0.975),
        "median": statistics.median(values),
        "q1": percentile(values, 0.25),
        "q3": percentile(values, 0.75),
    }


def paired_permutation_pvalue(
    differences: Sequence[float], resamples: int, rng: random.Random
) -> float:
    if not differences:
        raise ValueError("paired permutation test requires paired differences")
    if resamples <= 0:
        raise ValueError("permutation resamples must be positive")
    observed = abs(statistics.fmean(differences))
    if observed == 0.0:
        return 1.0
    extreme = 0
    for _ in range(resamples):
        permuted = statistics.fmean(
            value if rng.random() < 0.5 else -value for value in differences
        )
        if abs(permuted) >= observed - 1.0e-15:
            extreme += 1
    return (extreme + 1.0) / (resamples + 1.0)


def padded_limits(values: Sequence[float], include_zero: bool = False) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    if include_zero:
        low = min(low, 0.0)
        high = max(high, 0.0)
    span = high - low
    padding = max(0.015, span * 0.12)
    return low - padding, high + padding


def draw_panel_a(
    ax: Any,
    groups: Mapping[str, Mapping[bool, Sequence[float]]],
    summaries: Mapping[str, Mapping[bool, Mapping[str, float | int]]],
    seed: int,
) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    method_colors = {"Base": BASE_COLOR, "Ours": OURS_COLOR}
    offsets = {False: -0.21, True: 0.21}
    rng = random.Random(seed)
    all_values: list[float] = []

    for method_index, method in enumerate(("Base", "Ours")):
        for correct in (False, True):
            values = list(groups[method][correct])
            all_values.extend(values)
            position = method_index + offsets[correct]
            color = method_colors[method]
            box = ax.boxplot(
                [values],
                positions=[position],
                widths=0.32,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
                medianprops={"color": TEXT_COLOR, "linewidth": 1.6},
                whiskerprops={"color": color, "linewidth": 1.1},
                capprops={"color": color, "linewidth": 1.1},
            )
            patch = box["boxes"][0]
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
            patch.set_alpha(0.35 if not correct else 0.68)
            if not correct:
                patch.set_hatch("///")

            jitter = [position + rng.uniform(-0.075, 0.075) for _ in values]
            ax.scatter(
                jitter,
                values,
                s=17,
                color=color,
                alpha=0.48,
                edgecolors="white",
                linewidths=0.3,
                zorder=3,
            )
            summary = summaries[method][correct]
            mean = float(summary["mean"])
            ax.errorbar(
                [position],
                [mean],
                yerr=[
                    [mean - float(summary["ci_low"])],
                    [float(summary["ci_high"]) - mean],
                ],
                fmt="D",
                markersize=5.8,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.4,
                ecolor=color,
                elinewidth=1.5,
                capsize=3,
                zorder=5,
            )

    ax.set_xticks((0, 1), ("Base", "Ours"))
    ax.set_ylabel("Appraisal reasoning ROUGE-L")
    ax.set_ylim(*padded_limits(all_values))
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(
        handles=(
            Patch(
                facecolor="#B8BCC2",
                edgecolor="#777777",
                hatch="///",
                alpha=0.45,
                label="Emotion incorrect",
            ),
            Patch(
                facecolor="#777777",
                edgecolor="#777777",
                alpha=0.68,
                label="Emotion correct",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                color="#555555",
                markerfacecolor="white",
                linestyle="none",
                label="Mean and 95% bootstrap CI",
            ),
        ),
        frameon=False,
        fontsize=8.5,
        loc="best",
    )


def draw_panel_b(
    ax: Any, dimension_scores: Mapping[str, Mapping[str, float]], subset: str
) -> None:
    all_values: list[float] = []
    for index, dimension in enumerate(DIMENSIONS):
        base = dimension_scores["Base"][dimension]
        ours = dimension_scores["Ours"][dimension]
        all_values.extend((base, ours))
        ax.plot((base, ours), (index, index), color=LIGHT_GRAY, linewidth=1.2, zorder=1)
        ax.scatter(base, index, s=48, color=BASE_COLOR, zorder=3)
        ax.scatter(ours, index, s=48, color=OURS_COLOR, zorder=3)

    low = min(all_values)
    high = max(all_values)
    span = max(high - low, 0.03)
    left = low - 0.12 * span
    annotation_x = high + 0.18 * span
    right = high + 0.85 * span
    for index, dimension in enumerate(DIMENSIONS):
        delta = dimension_scores["Ours"][dimension] - dimension_scores["Base"][dimension]
        ax.text(
            annotation_x,
            index,
            f"{delta:+.3f}",
            ha="left",
            va="center",
            fontsize=9,
            color=TEXT_COLOR,
        )

    ax.set_xlim(left, right)
    ax.set_yticks(range(len(DIMENSIONS)), [DIMENSION_LABELS[d] for d in DIMENSIONS])
    ax.invert_yaxis()
    ax.set_xlabel("ROUGE-L")
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.scatter([], [], color=BASE_COLOR, label="Base")
    ax.scatter([], [], color=OURS_COLOR, label="Ours")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.text(
        annotation_x,
        -0.72,
        "Ours - Base",
        ha="left",
        va="center",
        fontsize=8.5,
        color="#555555",
    )


def format_pvalue(value: float) -> str:
    return "< 0.001" if value < 0.001 else f"= {value:.3f}"


def paired_line_alphas(
    differences: Sequence[float], minimum: float = 0.12, maximum: float = 0.72
) -> list[float]:
    """Map paired differences to opacity, with more negative values fainter."""
    if not differences:
        return []
    low = min(differences)
    high = max(differences)
    if low == high:
        return [(minimum + maximum) / 2.0 for _ in differences]
    scale = maximum - minimum
    return [minimum + scale * (value - low) / (high - low) for value in differences]


def draw_panel_c(
    raw_ax: Any,
    effect_ax: Any,
    pairs: Sequence[PerspectivePair],
    summaries: Mapping[str, Mapping[str, float | int]],
    delta_summary: Mapping[str, float | int],
    delta_bootstrap: Sequence[float],
    p_value: float,
) -> None:
    line_alphas = paired_line_alphas([pair.delta for pair in pairs])
    for pair, alpha in zip(pairs, line_alphas):
        raw_ax.plot(
            (0, 1),
            (pair.base, pair.ours),
            color=LIGHT_GRAY,
            linewidth=0.7,
            alpha=alpha,
            zorder=1,
        )
    raw_ax.scatter(
        [0] * len(pairs),
        [pair.base for pair in pairs],
        s=24,
        color=BASE_COLOR,
        alpha=0.72,
        edgecolors="white",
        linewidths=0.35,
        zorder=3,
    )
    raw_ax.scatter(
        [1] * len(pairs),
        [pair.ours for pair in pairs],
        s=24,
        color=OURS_COLOR,
        alpha=0.72,
        edgecolors="white",
        linewidths=0.35,
        zorder=3,
    )
    for position, method, color in ((0, "Base", BASE_COLOR), (1, "Ours", OURS_COLOR)):
        summary = summaries[method]
        mean = float(summary["mean"])
        raw_ax.errorbar(
            [position],
            [mean],
            yerr=[
                [mean - float(summary["ci_low"])],
                [float(summary["ci_high"]) - mean],
            ],
            fmt="D",
            markersize=7,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.7,
            ecolor=color,
            elinewidth=2.0,
            capsize=4,
            zorder=5,
        )
    raw_ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0)
    raw_ax.set_xticks((0, 1), ("Base", "Ours"))
    raw_ax.set_ylabel(r"Perspective margin $M_i$")
    raw_ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        raw_ax.spines[side].set_visible(False)

    violin = effect_ax.violinplot(
        [list(delta_bootstrap)],
        positions=[0],
        widths=0.6,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in violin["bodies"]:
        body.set_facecolor(OURS_COLOR)
        body.set_edgecolor(OURS_COLOR)
        body.set_alpha(0.28)
    delta_mean = float(delta_summary["mean"])
    effect_ax.errorbar(
        [0],
        [delta_mean],
        yerr=[
            [delta_mean - float(delta_summary["ci_low"])],
            [float(delta_summary["ci_high"]) - delta_mean],
        ],
        fmt="D",
        markersize=7,
        markerfacecolor="white",
        markeredgecolor=OURS_COLOR,
        markeredgewidth=1.7,
        ecolor=OURS_COLOR,
        elinewidth=2.0,
        capsize=4,
        zorder=5,
    )
    effect_ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0)
    effect_ax.set_xticks((0,), ("Ours - Base",))
    effect_ax.set_ylabel(r"Effect size, $\Delta M$")
    effect_ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        effect_ax.spines[side].set_visible(False)
    effect_ax.text(
        0.5,
        0.98,
        (
            f"Delta M = {delta_mean:+.3f}\n"
            f"95% CI [{float(delta_summary['ci_low']):+.3f}, "
            f"{float(delta_summary['ci_high']):+.3f}]\n"
            f"paired permutation p {format_pvalue(p_value)}"
        ),
        transform=effect_ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.7,
        color=TEXT_COLOR,
    )


def save_figure(
    figure: Any, output_dir: Path, stem: str, formats: Sequence[str], dpi: int
) -> list[Path]:
    paths: list[Path] = []
    for extension in dict.fromkeys(formats):
        path = output_dir / f"{stem}.{extension}"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    return paths


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap_samples must be positive")
    if args.permutation_samples <= 0:
        raise ValueError("--permutation_samples must be positive")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")

    emotion_paths = {
        "Base": args.emotion_root / args.base_run / "samples.csv",
        "Ours": args.emotion_root / args.ours_run / "samples.csv",
    }
    dimension_paths = {
        "Base": args.emotion_root / args.base_run / "dimensions.csv",
        "Ours": args.emotion_root / args.ours_run / "dimensions.csv",
    }
    perspective_paths = {
        "Base": args.perspective_root / args.base_run / "samples.csv",
        "Ours": args.perspective_root / args.ours_run / "samples.csv",
    }

    groups = {
        method: load_emotion_groups(path) for method, path in emotion_paths.items()
    }
    rng = random.Random(args.seed)
    group_summaries = {
        method: {
            correct: summarize_values(values, args.bootstrap_samples, rng)
            for correct, values in method_groups.items()
        }
        for method, method_groups in groups.items()
    }

    dimension_field = DIMENSION_FIELDS[args.dimension_subset]
    dimension_scores = {
        method: load_dimension_scores(path, dimension_field)
        for method, path in dimension_paths.items()
    }

    margins = {
        method: load_perspective_margins(path, args.perspective_subset)
        for method, path in perspective_paths.items()
    }
    pairs = pair_perspective_margins(margins["Base"], margins["Ours"])
    base_values = [pair.base for pair in pairs]
    ours_values = [pair.ours for pair in pairs]
    differences = [pair.delta for pair in pairs]
    perspective_summaries = {
        "Base": summarize_values(base_values, args.bootstrap_samples, rng),
        "Ours": summarize_values(ours_values, args.bootstrap_samples, rng),
    }
    delta_bootstrap = bootstrap_means(differences, args.bootstrap_samples, rng)
    delta_summary = {
        "n": len(differences),
        "mean": statistics.fmean(differences),
        "ci_low": percentile(delta_bootstrap, 0.025),
        "ci_high": percentile(delta_bootstrap, 0.975),
        "median": statistics.median(differences),
        "q1": percentile(differences, 0.25),
        "q3": percentile(differences, 0.75),
    }
    p_value = paired_permutation_pvalue(
        differences, args.permutation_samples, random.Random(args.seed + 1)
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        raise SystemExit(
            "matplotlib is required; install it with `pip install matplotlib`."
        ) from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    figure_a, axis_a = plt.subplots(figsize=(7.0, 5.2), constrained_layout=True)
    draw_panel_a(axis_a, groups, group_summaries, args.seed)
    saved.extend(
        save_figure(
            figure_a,
            args.output_dir,
            "a_emotion_correctness_rouge_l",
            args.formats,
            args.dpi,
        )
    )
    plt.close(figure_a)

    figure_b, axis_b = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    draw_panel_b(axis_b, dimension_scores, args.dimension_subset)
    saved.extend(
        save_figure(
            figure_b,
            args.output_dir,
            "b_dimension_rouge_l",
            args.formats,
            args.dpi,
        )
    )
    plt.close(figure_b)

    figure_c = plt.figure(figsize=(10.2, 5.2), constrained_layout=True)
    grid_c = figure_c.add_gridspec(1, 2, width_ratios=(3.1, 1.35))
    raw_c = figure_c.add_subplot(grid_c[0, 0])
    effect_c = figure_c.add_subplot(grid_c[0, 1])
    draw_panel_c(
        raw_c,
        effect_c,
        pairs,
        perspective_summaries,
        delta_summary,
        delta_bootstrap,
        p_value,
    )
    saved.extend(
        save_figure(
            figure_c,
            args.output_dir,
            "c_perspective_margin_estimation",
            args.formats,
            args.dpi,
        )
    )
    plt.close(figure_c)

    combined = plt.figure(figsize=(14.0, 10.2), constrained_layout=True)
    grid = combined.add_gridspec(2, 2, height_ratios=(1.0, 1.05))
    combined_a = combined.add_subplot(grid[0, 0])
    combined_b = combined.add_subplot(grid[0, 1])
    lower = grid[1, :].subgridspec(1, 2, width_ratios=(3.1, 1.35))
    combined_c = combined.add_subplot(lower[0, 0])
    combined_effect = combined.add_subplot(lower[0, 1])
    draw_panel_a(combined_a, groups, group_summaries, args.seed)
    draw_panel_b(combined_b, dimension_scores, args.dimension_subset)
    draw_panel_c(
        combined_c,
        combined_effect,
        pairs,
        perspective_summaries,
        delta_summary,
        delta_bootstrap,
        p_value,
    )
    saved.extend(
        save_figure(
            combined,
            args.output_dir,
            "appraisal_comparison_abc",
            args.formats,
            args.dpi,
        )
    )
    plt.close(combined)

    statistics_path = args.output_dir / "appraisal_comparison_statistics.json"
    write_json(
        statistics_path,
        {
            "meta": {
                "emotion_root": str(args.emotion_root),
                "perspective_root": str(args.perspective_root),
                "base_run": args.base_run,
                "ours_run": args.ours_run,
                "dimension_subset": args.dimension_subset,
                "dimension_field": dimension_field,
                "perspective_subset": args.perspective_subset,
                "perspective_margin_definition": (
                    "rouge-l_first_person - "
                    "rouge-l_third_person_mean_reference"
                ),
                "bootstrap_samples": args.bootstrap_samples,
                "permutation_samples": args.permutation_samples,
                "seed": args.seed,
            },
            "panel_a": {
                method: {
                    "emotion_incorrect": method_summaries[False],
                    "emotion_correct": method_summaries[True],
                }
                for method, method_summaries in group_summaries.items()
            },
            "panel_b": {
                dimension: {
                    "base": dimension_scores["Base"][dimension],
                    "ours": dimension_scores["Ours"][dimension],
                    "ours_minus_base": (
                        dimension_scores["Ours"][dimension]
                        - dimension_scores["Base"][dimension]
                    ),
                }
                for dimension in DIMENSIONS
            },
            "panel_c": {
                "paired_samples": len(pairs),
                "base": perspective_summaries["Base"],
                "ours": perspective_summaries["Ours"],
                "ours_minus_base": delta_summary,
                "paired_permutation_p_two_sided": p_value,
            },
        },
    )

    print(f"[done] paired perspective samples={len(pairs)}")
    print(
        "[effect] Delta M={:+.4f} 95% CI [{:+.4f}, {:+.4f}] p{}".format(
            float(delta_summary["mean"]),
            float(delta_summary["ci_low"]),
            float(delta_summary["ci_high"]),
            format_pvalue(p_value),
        )
    )
    for path in saved:
        print(f"[done] figure={path}")
    print(f"[done] statistics={statistics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
