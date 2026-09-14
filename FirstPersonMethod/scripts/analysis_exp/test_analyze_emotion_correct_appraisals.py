from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import analyze_emotion_correct_appraisals as analysis


class EmotionCorrectAppraisalAnalysisTests(unittest.TestCase):
    def test_conditional_analysis_finds_emotion_correct_appraisal_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prediction_root = root / "predictions"
            output_dir = root / "analysis"
            for task in analysis.REQUIRED_TASKS:
                (prediction_root / task).mkdir(parents=True)
            (prediction_root / analysis.REASONING_TASK).mkdir()

            prompt = analysis.load_prompt(analysis.DEFAULT_PROMPT)
            dimension_to_statement = prompt["appraisals"]["dimension_to_statement"]
            dimensions = list(dimension_to_statement)
            label_map = prompt["label_maps"]["appraisals"]
            score_to_label = {int(score): label for label, score in label_map.items()}
            positive_label = prompt["label_options"]["positive-labels"]["values"][2]
            second_positive_label = prompt["label_options"]["positive-labels"][
                "values"
            ][3]

            gold: dict[str, dict] = {}
            for sample_id in ("emotion_correct", "emotion_incorrect"):
                gold[sample_id] = {
                    "appraisal_ratings": {
                        statement: score_to_label[5]
                        for statement in dimension_to_statement.values()
                    },
                    "emotion_labels": {
                        "positive_emotion_labels": [
                            positive_label,
                            second_positive_label,
                        ],
                        "negative_emotion_labels": [],
                    },
                    "cognitive_questions": {
                        "summary_answers": {
                            dimension: f"Reference appraisal for {dimension}."
                            for dimension in analysis.CORE_APPRAISAL_DIMENSIONS
                        }
                    },
                }

            gold_path = root / "first_person.json"
            gold_path.write_text(json.dumps(gold), encoding="utf-8")

            correct_appraisals = {}
            for index, dimension in enumerate(dimensions):
                score = 5 if index < 10 else 1
                correct_appraisals[dimension] = {
                    "score": score,
                    "label": score_to_label[score],
                }
            exact_appraisals = {
                dimension: {"score": 5, "label": score_to_label[5]}
                for dimension in dimensions
            }
            self._write_json(
                prediction_root / "appraisals" / "emotion_correct.json",
                correct_appraisals,
            )
            self._write_json(
                prediction_root / "appraisals" / "emotion_incorrect.json",
                exact_appraisals,
            )
            self._write_json(
                prediction_root / "positive-labels" / "emotion_correct.json",
                {"labels": [positive_label]},
            )
            self._write_json(
                prediction_root / "negative-labels" / "emotion_correct.json",
                {"labels": []},
            )
            self._write_json(
                prediction_root / "positive-labels" / "emotion_incorrect.json",
                {"labels": []},
            )
            self._write_json(
                prediction_root / "negative-labels" / "emotion_incorrect.json",
                {"labels": []},
            )

            exit_code = analysis.main(
                [
                    "--gold",
                    str(gold_path),
                    "--pred_root",
                    str(prediction_root),
                    "--output_dir",
                    str(output_dir),
                    "--emotion_f1_threshold",
                    "0.6",
                    "--bootstrap_samples",
                    "100",
                    "--permutation_samples",
                    "100",
                    "--seed",
                    "7",
                ]
            )
            self.assertEqual(exit_code, 0)

            summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            expected_accuracy = 10.0 / len(dimensions)
            self.assertEqual(summary["coverage"]["complete_samples"], 2)
            self.assertEqual(summary["headline"]["emotion_correct_samples"], 1)
            self.assertEqual(
                summary["meta"]["emotion_selection_metric"],
                "sample-level joint emotion-label F1",
            )
            self.assertEqual(summary["meta"]["emotion_f1_threshold"], 0.6)
            self.assertAlmostEqual(
                summary["headline"][
                    "appraisal_accuracy_given_emotion_correct"
                ],
                expected_accuracy,
            )
            self.assertEqual(
                summary["headline"][
                    "emotion_correct_but_low_appraisal_accuracy_samples"
                ],
                1,
            )
            self.assertEqual(
                summary["headline"][
                    "emotion_correct_with_opposite_side_appraisal_error_samples"
                ],
                1,
            )
            self.assertAlmostEqual(
                summary["correct_vs_incorrect"]["correct_minus_incorrect"],
                expected_accuracy - 1.0,
            )
            self.assertTrue((output_dir / "samples.csv").is_file())
            self.assertTrue((output_dir / "dimensions.csv").is_file())
            self.assertTrue((output_dir / "core_dimensions.csv").is_file())
            self.assertTrue((output_dir / "headline.csv").is_file())
            self.assertTrue(
                (output_dir / "emotion_correct_appraisal_errors.csv").is_file()
            )

            reasoning_output_dir = root / "reasoning_analysis"
            correct_reasoning = {
                dimension: {
                    "answer": (
                        f"Reference appraisal for {dimension}."
                        if index < 2
                        else "Completely unrelated response."
                    )
                }
                for index, dimension in enumerate(
                    analysis.CORE_APPRAISAL_DIMENSIONS
                )
            }
            exact_reasoning = {
                dimension: {
                    "answer": f"Reference appraisal for {dimension}."
                }
                for dimension in analysis.CORE_APPRAISAL_DIMENSIONS
            }
            self._write_json(
                prediction_root
                / analysis.REASONING_TASK
                / "emotion_correct.json",
                correct_reasoning,
            )
            self._write_json(
                prediction_root
                / analysis.REASONING_TASK
                / "emotion_incorrect.json",
                exact_reasoning,
            )
            self.assertEqual(
                analysis.main(
                    [
                        "--gold",
                        str(gold_path),
                        "--pred_root",
                        str(prediction_root),
                        "--output_dir",
                        str(reasoning_output_dir),
                        "--appraisal_target",
                        "appraisal",
                        "--emotion_f1_threshold",
                        "0.6",
                        "--bootstrap_samples",
                        "100",
                        "--permutation_samples",
                        "100",
                    ]
                ),
                0,
            )
            reasoning_summary = json.loads(
                (reasoning_output_dir / "summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                reasoning_summary["meta"]["appraisal_target"], "appraisal"
            )
            self.assertEqual(
                reasoning_summary["headline"]["emotion_correct_samples"], 1
            )
            self.assertAlmostEqual(
                reasoning_summary["headline"][
                    "appraisal_reasoning_rouge_l_given_emotion_correct"
                ],
                2.0 / len(analysis.CORE_APPRAISAL_DIMENSIONS),
            )
            self.assertEqual(
                reasoning_summary["headline"][
                    "emotion_correct_but_low_appraisal_score_samples"
                ],
                1,
            )
            self.assertTrue(
                (
                    reasoning_output_dir
                    / "emotion_correct_appraisal_reasoning.csv"
                ).is_file()
            )

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
