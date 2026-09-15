from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from analysis_exp.plot_emotion_appraisal_scatter import (
    Point,
    SeriesData,
    average_ranks,
    axis_limits,
    discover_series,
    field_label,
    load_series,
    parse_series_spec,
    pearson,
    summarize,
)


class PlotEmotionAppraisalScatterTest(unittest.TestCase):
    def write_samples(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "sample_id",
                    "included",
                    "emotion_label_f1",
                    "appraisal_reasoning_rouge_l",
                ),
            )
            writer.writeheader()
            writer.writerows(
                (
                    {
                        "sample_id": "a",
                        "included": "True",
                        "emotion_label_f1": "0.2",
                        "appraisal_reasoning_rouge_l": "0.1",
                    },
                    {
                        "sample_id": "b",
                        "included": "True",
                        "emotion_label_f1": "0.8",
                        "appraisal_reasoning_rouge_l": "0.3",
                    },
                    {
                        "sample_id": "excluded",
                        "included": "False",
                        "emotion_label_f1": "1.0",
                        "appraisal_reasoning_rouge_l": "1.0",
                    },
                )
            )

    def test_load_and_summarize(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "base" / "samples.csv"
            self.write_samples(path)
            series = load_series(
                "Base",
                path,
                "emotion_label_f1",
                "appraisal_reasoning_rouge_l",
            )
            summary = summarize(series, 0.6)

        self.assertEqual(len(series.points), 2)
        self.assertAlmostEqual(summary["emotion_mean"], 0.5)
        self.assertAlmostEqual(summary["appraisal_mean"], 0.2)
        self.assertAlmostEqual(summary["pearson"], 1.0)
        self.assertEqual(summary["above_threshold_samples"], 1)

    def test_discovery_orders_base_first(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_samples(root / "grpo_from_warmup" / "samples.csv")
            self.write_samples(root / "base" / "samples.csv")
            discovered = discover_series(root, None)

        self.assertEqual(discovered[0][0], "Base")
        self.assertEqual(discovered[1][0], "Ours (GRPO from warmup)")

    def test_series_spec_and_tied_ranks(self) -> None:
        label, path = parse_series_spec("Ours=some/path.csv")
        self.assertEqual(label, "Ours")
        self.assertEqual(path, Path("some/path.csv"))
        self.assertEqual(average_ranks([1.0, 1.0, 3.0]), [1.5, 1.5, 3.0])
        self.assertAlmostEqual(pearson([1.0, 2.0], [3.0, 5.0]), 1.0)
        self.assertEqual(
            field_label("appraisal_reasoning_bertscore"),
            "Appraisal reasoning BERTScore F1",
        )

    def test_y_limits_follow_the_selected_appraisal_field(self) -> None:
        series = SeriesData(
            label="Run",
            path=Path("samples.csv"),
            points=(
                Point("a", 0.2, 1.5),
                Point("b", 0.8, 2.0),
            ),
        )
        _, mae_limits = axis_limits([series], "appraisal_mae")
        self.assertGreater(mae_limits[1], 1.0)

        accuracy_series = SeriesData(
            label="Run",
            path=Path("samples.csv"),
            points=(
                Point("a", 0.2, 0.2),
                Point("b", 0.8, 0.8),
            ),
        )
        _, accuracy_limits = axis_limits(
            [accuracy_series], "appraisal_accuracy"
        )
        self.assertGreaterEqual(accuracy_limits[0], 0.0)
        self.assertLessEqual(accuracy_limits[1], 1.0)

        bertscore_series = SeriesData(
            label="Run",
            path=Path("samples.csv"),
            points=(
                Point("a", 0.2, 0.82),
                Point("b", 0.8, 0.88),
            ),
        )
        _, bertscore_limits = axis_limits(
            [bertscore_series], "appraisal_reasoning_bertscore"
        )
        self.assertGreater(bertscore_limits[0], 0.0)
        self.assertLess(bertscore_limits[1], 1.0)


if __name__ == "__main__":
    unittest.main()
