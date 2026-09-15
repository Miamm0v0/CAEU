from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import analyze_low_reasoning_third_person_gold as analysis


class ThirdPersonReasoningGoldAnalysisTests(unittest.TestCase):
    def test_low_first_person_sample_matches_third_person_references(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prediction_root = root / "predictions"
            prediction_dir = prediction_root / "core-appraisals"
            prediction_dir.mkdir(parents=True)
            output_dir = root / "analysis"

            first_person = {
                "low": self._gold_item("Alpha beta gamma delta epsilon."),
                "high": self._gold_item("Shared appraisal reference text."),
            }
            third_person = {
                "low": [
                    self._gold_item(
                        "Third annotator interpretation matches exactly.",
                        participant_id="third_1",
                    ),
                    self._gold_item(
                        "Third annotator interpretation matches exactly.",
                        participant_id="third_2",
                    ),
                ],
                "high": [
                    self._gold_item(
                        "Shared appraisal reference text.",
                        participant_id="third_3",
                    )
                ],
            }
            predictions = {
                "low": self._prediction_item(
                    "Third annotator interpretation matches exactly."
                ),
                "high": self._prediction_item(
                    "Shared appraisal reference text."
                ),
            }

            first_path = root / "first_person.json"
            third_path = root / "third_person.json"
            first_path.write_text(json.dumps(first_person), encoding="utf-8")
            third_path.write_text(json.dumps(third_person), encoding="utf-8")
            for sample_id, payload in predictions.items():
                (prediction_dir / f"{sample_id}.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )

            self.assertEqual(
                analysis.main(
                    [
                        "--first_person_gold",
                        str(first_path),
                        "--third_person_gold",
                        str(third_path),
                        "--pred_root",
                        str(prediction_root),
                        "--output_dir",
                        str(output_dir),
                        "--selection_metric",
                        "rouge-l",
                        "--selection_mode",
                        "threshold",
                        "--low_threshold",
                        "0.2",
                        "--bootstrap_samples",
                        "100",
                        "--permutation_samples",
                        "100",
                        "--seed",
                        "9",
                    ]
                ),
                0,
            )

            summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["coverage"]["matched_complete_samples"], 2)
            self.assertEqual(summary["coverage"]["low_first_person_samples"], 1)
            self.assertEqual(
                summary["headline"]["low_subset_first_person_mean"], 0.0
            )
            self.assertEqual(
                summary["headline"][
                    "low_subset_third_person_mean_reference"
                ],
                1.0,
            )
            self.assertEqual(
                summary["headline"]["low_subset_mean_reference_uplift"], 1.0
            )
            self.assertEqual(
                summary["headline"][
                    "low_subset_third_mean_better_sample_rate"
                ],
                1.0,
            )
            self.assertTrue((output_dir / "samples.csv").is_file())
            self.assertTrue((output_dir / "dimensions.csv").is_file())
            self.assertTrue((output_dir / "low_examples.csv").is_file())
            self.assertTrue(
                (output_dir / "third_person_reference_details.csv").is_file()
            )

    def test_chain_emotion_policy_dimension_mapping(self) -> None:
        payload = {
            "appraisal_reasoning": {
                "relevance": "Relevant.",
                "goal_congruence": "Incongruent.",
                "agency_accountability": "Someone else caused it.",
                "control_coping_potential": "No control.",
                "epistemic": "Unexpected.",
            }
        }
        self.assertEqual(
            analysis.core_prediction(payload, "chain-emotion", "certainty"),
            "Unexpected.",
        )

    @staticmethod
    def _gold_item(text: str, participant_id: str | None = None) -> dict:
        payload = {
            "story_collection": {"final_scenario": "A situation."},
            "cognitive_questions": {
                "summary_answers": {
                    dimension: text for dimension in analysis.CORE_DIMENSIONS
                }
            },
        }
        if participant_id is not None:
            payload["participant_id"] = participant_id
        return payload

    @staticmethod
    def _prediction_item(text: str) -> dict:
        return {
            dimension: {"answer": text}
            for dimension in analysis.CORE_DIMENSIONS
        }


if __name__ == "__main__":
    unittest.main()
