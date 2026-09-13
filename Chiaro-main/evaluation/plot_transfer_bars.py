"""Regenerate the transfer-accuracy bar chart (paper Figure 2).

Grouped bars for the three RoBERTa-large checkpoints (CHIARO-only, GoEm-only,
Combined) on the CHIARO held-out test split and ten external emotion benchmarks.
The best checkpoint per dataset is bold-labelled.

Usage:
  python plot_transfer_bars.py                       # paper numbers, writes transfer_bars.{png,pdf}
  python plot_transfer_bars.py --results my.json     # {"dataset": [chiaro_only, goem_only, combined], ...}
  python plot_transfer_bars.py --errors  my_std.json # optional per-cell std for error bars (same shape)
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# dataset -> [CHIARO-only, GoEm-only, Combined]  (top-1 accuracy, %)
PAPER = {
    "CHIARO": [69.5, 15.5, 73.5],
    "GoEmotions":         [24.8, 82.9, 82.0],
    "ISEAR":              [47.0, 55.2, 59.2],
    "CARER":              [42.5, 49.7, 46.1],
    "TweetEval":          [37.4, 55.2, 56.9],
    "SemEval-2018":       [47.5, 59.5, 62.6],
    "XED":                [40.6, 39.4, 43.5],
    "DailyDialog":        [28.5, 50.4, 54.6],
    "EmotionX-2019":      [53.9, 37.9, 34.2],
    "MELD":               [38.9, 31.6, 31.7],
    "EmoBench EU":        [19.2, 17.3, 21.2],
}
SERIES = ["CHIARO-only", "GoEm-only", "Combined"]
FILL = ["#aec7e8", "#ffbb78", "#f7b6d2"]
EDGE = ["#1f77b4", "#ff7f0e", "#e377c2"]


def plot(results, errors=None, out="transfer_bars"):
    names = list(results.keys())
    vals = np.array([results[n] for n in names], dtype=float)  # (n_datasets, 3)
    errs = np.array([errors[n] for n in names], dtype=float) if errors else None

    n = len(names)
    width = 0.25
    # leave a visual gap after the first (CHIARO) group
    x = np.arange(n, dtype=float)
    x[1:] += 0.6

    fig, ax = plt.subplots(figsize=(16.9, 5.6), dpi=200)
    for j, (label, fc, ec) in enumerate(zip(SERIES, FILL, EDGE)):
        xs = x + (j - 1) * width
        ax.bar(xs, vals[:, j], width, label=label, color=fc, edgecolor=ec, linewidth=1.3,
               yerr=None if errs is None else errs[:, j],
               error_kw=dict(ecolor="#000", capsize=2, lw=1))
        for i, xi in enumerate(xs):
            best = vals[i].argmax() == j
            ax.text(xi, vals[i, j] + 1.5, f"{vals[i, j]:.1f}", ha="center", va="bottom",
                    rotation=90, fontsize=12, fontweight="bold" if best else "normal",
                    color="black" if best else "#000")

    # section divider
    div = (x[0] + x[1]) / 2
    ax.axvline(div, color="#000", linestyle=":", linewidth=1)

    ax.set_ylabel("Accuracy (%)", fontsize=15)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="y", labelsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=14)
    ax.set_xlim(x[0] - 0.7, x[-1] + 0.7)
    ax.yaxis.grid(True, linestyle="--", color="#ddd", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):  # black x and y axes
        ax.spines[s].set_color("black")
        ax.spines[s].set_linewidth(1.2)
    ax.tick_params(axis="both", colors="black")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.13), ncol=3, frameon=False, fontsize=14)

    fig.tight_layout()
    import os
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    print(f"wrote {os.path.abspath(out)}.png / .pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", help="JSON: {dataset: [chiaro_only, goem_only, combined]}")
    ap.add_argument("--errors", help="JSON with the same shape, values = std (draws error bars)")
    ap.add_argument("--out", default="transfer_bars")
    a = ap.parse_args()
    results = json.load(open(a.results)) if a.results else PAPER
    errors = json.load(open(a.errors)) if a.errors else None
    plot(results, errors, a.out)


if __name__ == "__main__":
    main()
