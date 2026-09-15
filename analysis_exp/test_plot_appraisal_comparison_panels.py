from __future__ import annotations

import csv
import random
import tempfile
import unittest
from pathlib import Path

from analysis_exp.plot_appraisal_comparison_panels import (
    bootstrap_means,
    load_dimension_scores,
    load_emotion_groups,
    load_perspective_margins,
    pair_perspective_margins,
    paired_permutation_pvalue,
    paired_line_alphas,
    percentile,
    summarize_values,
)


class PlotAppraisalComparisonPanelsTest(unittest.TestCase):
    def write_csv(
        self, path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, str]]
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_emotion_groups_and_dimension_scores(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = root / "samples.csv"
            self.write_csv(
                samples,
                (
                    "sample_id",
                    "included",
                    "emotion_correct",
                    "appraisal_reasoning_rouge_l",
                ),
                [
                    {
                        "sample_id": "a",
                        "included": "True",
                        "emotion_correct": "False",
                        "appraisal_reasoning_rouge_l": "0.1",
                    },
                    {
                        "sample_id": "b",
                        "included": "True",
                        "emotion_correct": "True",
                        "appraisal_reasoning_rouge_l": "0.3",
                    },
                    {
                        "sample_id": "excluded",
                        "included": "False",
                        "emotion_correct": "True",
                        "appraisal_reasoning_rouge_l": "1.0",
                    },
                ],
            )
            groups = load_emotion_groups(samples)

            dimensions = root / "dimensions.csv"
            self.write_csv(
                dimensions,
                ("dimension", "all_complete_rouge_l_f1"),
                [
                    {
                        "dimension": dimension,
                        "all_complete_rouge_l_f1": str(index / 10),
                    }
                    for index, dimension in enumerate(
                        (
                            "relevance",
                            "congruence",
                            "accountability",
                            "control",
                            "certainty",
                        ),
                        start=1,
                    )
                ],
            )
            scores = load_dimension_scores(
                dimensions, "all_complete_rouge_l_f1"
            )

        self.assertEqual(groups[False], [0.1])
        self.assertEqual(groups[True], [0.3])
        self.assertAlmostEqual(scores["control"], 0.4)

    def test_perspective_margin_direction_and_pairing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_path = root / "base.csv"
            ours_path = root / "ours.csv"
            fields = (
                "sample_id",
                "selected_low",
                "rouge-l_first_person",
                "rouge-l_third_person_mean_reference",
            )
            self.write_csv(
                base_path,
                fields,
                [
                    {
                        "sample_id": "a",
                        "selected_low": "True",
                        "rouge-l_first_person": "0.4",
                        "rouge-l_third_person_mean_reference": "0.1",
                    },
                    {
                        "sample_id": "base_only",
                        "selected_low": "False",
                        "rouge-l_first_person": "0.2",
                        "rouge-l_third_person_mean_reference": "0.3",
                    },
                ],
            )
            self.write_csv(
                ours_path,
                fields,
                [
                    {
                        "sample_id": "a",
                        "selected_low": "True",
                        "rouge-l_first_person": "0.6",
                        "rouge-l_third_person_mean_reference": "0.2",
                    }
                ],
            )
            base = load_perspective_margins(base_path, "all")
            ours = load_perspective_margins(ours_path, "all")
            pairs = pair_perspective_margins(base, ours)

        self.assertAlmostEqual(base["a"], 0.3)
        self.assertAlmostEqual(ours["a"], 0.4)
        self.assertEqual(len(pairs), 1)
        self.assertAlmostEqual(pairs[0].delta, 0.1)

    def test_bootstrap_summary_and_permutation_are_bounded(self) -> None:
        values = [0.1, 0.2, 0.3, 0.4]
        bootstrap = bootstrap_means(values, 200, random.Random(4))
        summary = summarize_values(values, 200, random.Random(4))
        p_value = paired_permutation_pvalue(
            [0.1, 0.2, 0.1, 0.2], 500, random.Random(5)
        )

        self.assertEqual(len(bootstrap), 200)
        self.assertAlmostEqual(percentile(values, 0.5), 0.25)
        self.assertAlmostEqual(summary["mean"], 0.25)
        self.assertLessEqual(summary["ci_low"], summary["mean"])
        self.assertGreaterEqual(summary["ci_high"], summary["mean"])
        self.assertGreater(p_value, 0.0)
        self.assertLessEqual(p_value, 1.0)

    def test_paired_line_opacity_increases_with_delta(self) -> None:
        alphas = paired_line_alphas([-0.2, -0.1, 0.0, 0.3])

        self.assertEqual(alphas, sorted(alphas))
        self.assertAlmostEqual(alphas[0], 0.12)
        self.assertAlmostEqual(alphas[-1], 0.72)


if __name__ == "__main__":
    unittest.main()
