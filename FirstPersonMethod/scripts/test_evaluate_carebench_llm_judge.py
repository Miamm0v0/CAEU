from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import evaluate_carebench_llm_judge as evaluator


class CareBenchReferenceFreeJudgeTest(unittest.TestCase):
    def candidate(self) -> dict:
        return {
            "appraisal_reasoning": {
                "relevance": "This matters to me because it affects my goal.",
                "epistemic": "I did not expect this outcome.",
                "goal_congruence": "The outcome blocks what I wanted.",
                "agency_accountability": "I see the other person as responsible.",
                "control_coping_potential": "I have little control but can seek help.",
            },
            "emotion": {
                "positive_intensity": 0,
                "negative_intensity": 4,
                "positive_labels": [],
                "negative_labels": ["sad", "angry"],
            },
        }

    def judgment(self) -> dict:
        return {
            **{
                criterion: {"score": 3, "rationale": "Supported by the event."}
                for criterion in evaluator.CRITERIA
            },
            "overall_feedback": "Mostly strong and grounded.",
        }

    def test_judge_envelope_has_no_gold_or_reference(self) -> None:
        judge = object.__new__(evaluator.ReferenceFreeJudge)
        judge.system_prompt = evaluator.judge_system_prompt()
        messages = judge.request_messages(
            "I received disappointing news.", self.candidate()
        )
        envelope_text = messages[1]["content"]
        self.assertIn('"situation"', envelope_text)
        self.assertIn('"candidate"', envelope_text)
        self.assertNotIn('"gold"', envelope_text)
        self.assertNotIn('"reference"', envelope_text)

    def test_strict_judgment_parser_and_local_overall(self) -> None:
        parsed = evaluator.parse_judge_output(json.dumps(self.judgment()))
        scores = evaluator.judgment_scores(parsed)
        self.assertEqual(scores["overall_raw_0_4"], 3.0)
        self.assertEqual(scores["overall_normalized_0_1"], 0.75)

        invalid = self.judgment()
        invalid["extra"] = "not allowed"
        with self.assertRaisesRegex(ValueError, "keys mismatch"):
            evaluator.parse_judge_output(json.dumps(invalid))

    def test_load_situations_ignores_annotation_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample-1.json"
            path.write_text(
                json.dumps(
                    {
                        "story_collection": {"final_scenario": "I got the job."},
                        "cognitive_questions": {"secret": "must not be used"},
                        "emotion": {"secret": "must not be used"},
                    }
                ),
                encoding="utf-8",
            )
            rows = evaluator.load_situations(Path(directory))
        self.assertEqual(
            rows,
            [{"sample_id": "sample-1", "situation": "I got the job."}],
        )

    def test_candidate_accepts_policy_and_legacy_appraisal_keys(self) -> None:
        candidate = evaluator.normalize_policy_candidate(self.candidate())
        self.assertEqual(
            set(candidate["appraisal_reasoning"]),
            set(evaluator.POLICY_DIMENSIONS),
        )

        legacy = self.candidate()
        legacy["appraisal_reasoning"] = {
            "relevance": "This matters to me.",
            "certainty": "I did not expect it.",
            "congruence": "It blocks my goal.",
            "accountability": "The other person caused it.",
            "control": "I have little control.",
        }
        normalized = evaluator.normalize_policy_candidate(legacy)
        self.assertIn("epistemic", normalized["appraisal_reasoning"])

    def test_aggregation_excludes_invalid_instead_of_scoring_zero(self) -> None:
        samples = [
            {"sample_id": "valid", "situation": "A"},
            {"sample_id": "invalid", "situation": "B"},
        ]
        successful = {
            "sample_id": "valid",
            "status": "success",
            "candidate_source": "chain-emotion",
            **evaluator.judgment_scores(self.judgment()),
        }
        rows = {
            "valid": successful,
            "invalid": {
                "sample_id": "invalid",
                "status": "invalid_prediction",
                "candidate_source": "chain-emotion",
            },
        }
        summary = evaluator.aggregate_results(
            samples,
            rows,
            "chain-emotion",
            {
                "provider": "glm",
                "model": "judge",
                "base_url": "",
                "response_format": "json_object",
            },
        )
        self.assertEqual(summary["coverage"]["judge_success"], 1)
        self.assertEqual(summary["coverage"]["invalid_predictions"], 1)
        self.assertEqual(summary["scores"]["overall_mean_raw_0_4"], 3.0)


if __name__ == "__main__":
    unittest.main()
