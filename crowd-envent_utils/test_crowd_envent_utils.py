from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


UTILS_DIR = Path(__file__).resolve().parent
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from crowd_envent_metrics import build_metrics_report
from crowd_envent_schema import (
    APPRAISAL_FIELDS,
    METHOD_APPRAISAL_DIMENSIONS,
    carebench_aligned_output_schema,
    chain_target_output_schema,
    output_schema,
    parse_carebench_aligned_response,
    parse_chain_target_response,
    parse_policy_response,
)
import evaluate_policy_crowd_envent as evaluator
from evaluate_policy_crowd_envent import normalize_prediction_record
from judge_crowd_envent_reasoning import (
    aggregate_judgments,
    judge_output_example,
    parse_judgment,
)
from prepare_crowd_envent_eval import (
    CONFIRMED_REAL,
    first_person_situation,
    mask_emotion_cues,
    select_generation_rows,
)


class CrowdEnventSchemaTests(unittest.TestCase):
    def test_chain_policy_schema_round_trip(self) -> None:
        payload = output_schema(include_reasoning=True)
        parsed = parse_policy_response(json.dumps(payload), include_reasoning=True)
        self.assertEqual(set(parsed["appraisal_reasoning"]), set(METHOD_APPRAISAL_DIMENSIONS))
        self.assertEqual(set(parsed["appraisal_ratings"]), set(APPRAISAL_FIELDS))
        self.assertEqual(parsed["emotion"]["label"], "joy")

    def test_chain_all_target_schemas_and_saved_record(self) -> None:
        appraisal_branch = parse_chain_target_response(
            json.dumps(chain_target_output_schema("appraisals")), "appraisals"
        )
        emotion_branch = parse_chain_target_response(
            json.dumps(chain_target_output_schema("emotion")), "emotion"
        )
        record = normalize_prediction_record(
            {
                "sample_id": "sample-chain-all",
                "generation_schema": "chain-all",
                "appraisal_reasoning": {
                    "appraisals": appraisal_branch["appraisal_reasoning"],
                    "emotion": emotion_branch["appraisal_reasoning"],
                },
                "appraisal_ratings": appraisal_branch["appraisal_ratings"],
                "emotion": emotion_branch["emotion"],
            }
        )
        self.assertEqual(
            set(record["appraisal_reasoning"]), {"appraisals", "emotion"}
        )
        self.assertEqual(record["generation_schema"], "chain-all")

    def test_carebench_ratings_and_native_emotion_project_to_fields(self) -> None:
        appraisal_payload = carebench_aligned_output_schema("appraisals")
        emotion_payload = carebench_aligned_output_schema("emotion")
        emotion_payload["emotion"] = {"label": "sadness", "intensity": 4}
        parsed_appraisals = parse_carebench_aligned_response(
            json.dumps(appraisal_payload), "appraisals"
        )
        parsed_emotion = parse_carebench_aligned_response(
            json.dumps(emotion_payload), "emotion"
        )

        self.assertEqual(
            set(parsed_appraisals["appraisal_reasoning"]),
            set(METHOD_APPRAISAL_DIMENSIONS),
        )
        self.assertEqual(
            set(parsed_appraisals["appraisal_ratings"]),
            set(APPRAISAL_FIELDS),
        )
        self.assertEqual(parsed_emotion["emotion"], {"label": "sadness", "intensity": 4})
        self.assertIn("carebench_appraisal_ratings", parsed_appraisals)
        self.assertNotIn("carebench_emotion", parsed_emotion)

    def test_emotion_mask_is_uniform_and_first_person(self) -> None:
        masked = mask_emotion_cues("I felt joy, pride and ... after the event.")
        self.assertNotIn("joy", masked.lower())
        self.assertNotIn("pride", masked.lower())
        self.assertIn("[EMOTION]", masked)
        situation = first_person_situation(masked, "I experienced this:", False)
        self.assertTrue(situation.startswith("I experienced this:"))

    def test_strict_validated_selection(self) -> None:
        rows = [
            {"text_id": "1", "did_you_lie?": CONFIRMED_REAL},
            {"text_id": "2", "did_you_lie?": CONFIRMED_REAL},
            {"text_id": "3", "did_you_lie?": "--"},
        ]
        selected = select_generation_rows(rows, {"1": [{}]}, "strict-validated")
        self.assertEqual([row["text_id"] for row in selected], ["1"])


class CrowdEnventMetricTests(unittest.TestCase):
    def test_perfect_prediction_scores_perfectly(self) -> None:
        ratings = {field: (index % 5) + 1 for index, field in enumerate(APPRAISAL_FIELDS)}
        reasoning = {dimension: f"My {dimension} reasoning." for dimension in METHOD_APPRAISAL_DIMENSIONS}
        sample = {
            "id": "sample-1",
            "reference": {
                "appraisal_ratings": ratings,
                "emotion": {"label": "joy", "intensity": 4},
            },
            "validation_reference": {
                "emotion_labels": ["joy"] * 5,
                "majority_emotion_labels": ["joy"],
                "mean_appraisal_ratings": {field: float(value) for field, value in ratings.items()},
            },
        }
        prediction = {
            "sample_id": "sample-1",
            "appraisal_reasoning": reasoning,
            "appraisal_ratings": ratings,
            "emotion": {"label": "joy", "intensity": 4},
        }
        report = build_metrics_report(
            [sample], {"sample-1": prediction}, allow_incomplete=False
        )
        author = report["author_self_report"]
        self.assertEqual(author["appraisals"]["overall_21_ratings"]["normalized_rmse"], 0.0)
        self.assertEqual(author["emotion_classification"]["accuracy"], 1.0)
        self.assertEqual(author["emotion_candidate_hit"]["hit_rate"], 1.0)
        self.assertEqual(
            author["emotion_hit_at_k"]["by_k"]["1"]["hit_rate"], 1.0
        )
        self.assertEqual(author["emotion_intensity"]["normalized_rmse"], 0.0)
        self.assertEqual(report["coverage"]["coverage"], 1.0)

    def test_candidate_hit_uses_complete_original_label_set(self) -> None:
        ratings = {field: 3 for field in APPRAISAL_FIELDS}
        sample = {
            "id": "candidate-hit",
            "reference": {
                "appraisal_ratings": ratings,
                "emotion": {"label": "trust", "intensity": 4},
            },
            "validation_reference": {
                "emotion_labels": [],
                "majority_emotion_labels": [],
                "mean_appraisal_ratings": {},
            },
        }
        prediction = {
            "sample_id": "candidate-hit",
            "generation_schema": "direct",
            "appraisal_ratings": ratings,
            "emotion": {"label": "joy", "intensity": 4},
            "emotion_candidate_labels": ["joy", "trust", "fear"],
            "multilabel_emotion": {
                "labels": ["joy", "trust", "fear"],
                "intensity": 4,
            },
        }

        report = build_metrics_report(
            [sample], {"candidate-hit": prediction}, allow_incomplete=False
        )
        author = report["author_self_report"]
        self.assertEqual(author["emotion_classification"]["accuracy"], 0.0)
        self.assertEqual(author["emotion_candidate_hit"]["hit_rate"], 1.0)
        self.assertEqual(
            author["emotion_candidate_hit"]["mean_candidate_count"], 3.0
        )
        self.assertEqual(
            author["emotion_hit_at_k"]["by_k"]["1"]["hit_rate"], 0.0
        )
        self.assertEqual(
            author["emotion_hit_at_k"]["by_k"]["2"]["hit_rate"], 1.0
        )
        self.assertEqual(
            author["emotion_hit_at_k"]["ranking_sources"][
                "direct_multilabel_labels"
            ],
            1,
        )
        self.assertEqual(
            list(author["emotion_hit_at_k"]["by_k"]), ["1", "2", "3"]
        )

    def test_chain_hit_at_k_uses_evaluation_only_ranking(self) -> None:
        ratings = {field: 3 for field in APPRAISAL_FIELDS}
        sample = {
            "id": "chain-ranking",
            "reference": {
                "appraisal_ratings": ratings,
                "emotion": {"label": "fear", "intensity": 4},
            },
            "validation_reference": {
                "emotion_labels": [],
                "majority_emotion_labels": [],
                "mean_appraisal_ratings": {},
            },
        }
        prediction = {
            "sample_id": "chain-ranking",
            "generation_schema": "carebench",
            "appraisal_ratings": ratings,
            "emotion": {"label": "joy", "intensity": 4},
            "emotion_candidate_labels": ["fear", "joy", "trust"],
            "evaluation_only_emotion_ranking": ["fear", "trust", "joy"],
            "carebench_emotion": {
                "positive": {"label": "joy", "intensity": 4},
                "negative": {"label": "fear", "intensity": 2},
            },
        }

        report = build_metrics_report(
            [sample], {"chain-ranking": prediction}, allow_incomplete=False
        )
        author = report["author_self_report"]
        self.assertEqual(author["emotion_classification"]["accuracy"], 1.0)
        self.assertEqual(author["emotion_intensity"]["mae"], 2.0)
        self.assertEqual(
            author["emotion_hit_at_k"]["by_k"]["1"]["hit_rate"], 1.0
        )
        self.assertEqual(
            author["emotion_hit_at_k"]["ranking_sources"][
                "chain_evaluation_only_ranking"
            ],
            1,
        )

    def test_missing_prediction_can_be_penalized(self) -> None:
        ratings = {field: 5 for field in APPRAISAL_FIELDS}
        sample = {
            "id": "missing-sample",
            "reference": {
                "appraisal_ratings": ratings,
                "emotion": {"label": "joy", "intensity": 5},
            },
            "validation_reference": {
                "emotion_labels": ["joy"],
                "majority_emotion_labels": ["joy"],
                "mean_appraisal_ratings": {
                    field: float(value) for field, value in ratings.items()
                },
            },
        }

        report = build_metrics_report(
            [sample],
            {},
            allow_incomplete=False,
            missing_prediction_policy="penalize",
        )

        coverage = report["coverage"]
        self.assertEqual(coverage["valid_predictions"], 0)
        self.assertEqual(coverage["evaluated_samples"], 1)
        self.assertEqual(coverage["penalized_missing_predictions"], 1)
        self.assertEqual(coverage["coverage"], 0.0)
        self.assertEqual(coverage["metric_sample_coverage"], 1.0)
        author = report["author_self_report"]
        self.assertEqual(author["emotion_classification"]["accuracy"], 0.0)
        self.assertEqual(author["emotion_candidate_hit"]["hit_rate"], 0.0)
        self.assertEqual(
            author["emotion_hit_at_k"]["by_k"]["3"]["hit_rate"], 0.0
        )
        self.assertEqual(author["emotion_intensity"]["normalized_rmse"], 1.0)
        self.assertEqual(
            author["appraisals"]["overall_21_ratings"]["normalized_rmse"],
            0.5,
        )

    def test_chain_all_reports_two_reasoning_traces(self) -> None:
        ratings = {field: 3 for field in APPRAISAL_FIELDS}
        reasoning = {
            dimension: f"My {dimension} reasoning."
            for dimension in METHOD_APPRAISAL_DIMENSIONS
        }
        sample = {
            "id": "sample-2",
            "reference": {
                "appraisal_ratings": ratings,
                "emotion": {"label": "joy", "intensity": 3},
            },
            "validation_reference": {
                "emotion_labels": [],
                "majority_emotion_labels": [],
                "mean_appraisal_ratings": {},
            },
        }
        prediction = {
            "sample_id": "sample-2",
            "appraisal_reasoning": {
                "appraisals": reasoning,
                "emotion": reasoning,
            },
            "appraisal_ratings": ratings,
            "emotion": {"label": "joy", "intensity": 3},
        }
        report = build_metrics_report(
            [sample], {"sample-2": prediction}, allow_incomplete=False
        )
        traces = report["appraisal_reasoning"]["by_trace"]
        self.assertEqual(set(traces), {"appraisals", "emotion"})


class CrowdEnventGenerationTests(unittest.TestCase):
    def test_metrics_entrypoint_defaults_to_penalize_missing_predictions(self) -> None:
        args = evaluator.build_parser(fixed_mode="metrics").parse_args([])
        self.assertEqual(args.mode, "metrics")
        self.assertEqual(args.missing_prediction_policy, "penalize")

    def test_chain_all_makes_two_calls_and_merges_outputs(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, system_prompt: str, user_prompt: str) -> SimpleNamespace:
                self.calls += 1
                target = "appraisals" if "21 crowd-enVent appraisal ratings" in system_prompt else "emotion"
                return SimpleNamespace(
                    content=json.dumps(chain_target_output_schema(target)),
                    raw={"target": target},
                )

        client = FakeClient()
        args = SimpleNamespace(
            max_parse_retries=0,
            overwrite_predictions=False,
            generation_schema="chain-all",
        )
        sample = {
            "id": "sample-generation",
            "source_id": "1",
            "situation": "I experienced a good event.",
        }
        with tempfile.TemporaryDirectory() as temporary:
            predictions_path = Path(temporary) / "predictions.jsonl"
            invalid_path = Path(temporary) / "invalid.jsonl"
            with patch.object(evaluator, "create_policy_client", return_value=client):
                predictions = evaluator.generate_predictions(
                    args, [sample], predictions_path, invalid_path
                )
            self.assertEqual(client.calls, 2)
            record = predictions["sample-generation"]
            self.assertEqual(record["generation_schema"], "chain-all")
            self.assertEqual(
                set(record["appraisal_reasoning"]), {"appraisals", "emotion"}
            )
            self.assertFalse(invalid_path.exists())

    def test_carebench_generation_uses_two_calls_and_saves_projected_fields(self) -> None:
        appraisal_payload = carebench_aligned_output_schema("appraisals")
        emotion_payload = carebench_aligned_output_schema("emotion")
        emotion_payload["emotion"] = {"label": "joy", "intensity": 5}

        class FakeClient:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, system_prompt: str, user_prompt: str) -> SimpleNamespace:
                self.calls += 1
                self.last_system = system_prompt
                self.last_user = user_prompt
                payload = appraisal_payload if self.calls == 1 else emotion_payload
                return SimpleNamespace(content=json.dumps(payload), raw={"call": self.calls})

        client = FakeClient()
        args = SimpleNamespace(
            max_parse_retries=0,
            overwrite_predictions=False,
            generation_schema="carebench",
        )
        sample = {
            "id": "sample-carebench",
            "source_id": "1",
            "situation": "I experienced a good event.",
        }
        with tempfile.TemporaryDirectory() as temporary:
            predictions_path = Path(temporary) / "predictions.jsonl"
            invalid_path = Path(temporary) / "invalid.jsonl"
            with patch.object(evaluator, "create_policy_client", return_value=client):
                predictions = evaluator.generate_predictions(
                    args, [sample], predictions_path, invalid_path
                )
            self.assertEqual(client.calls, 2)
            record = predictions["sample-carebench"]
            self.assertEqual(record["generation_schema"], "carebench")
            self.assertEqual(
                set(record["appraisal_reasoning"]),
                {"appraisals", "emotion"},
            )
            self.assertEqual(record["emotion"], {"label": "joy", "intensity": 5})
            self.assertIn("carebench_appraisal_ratings", record)
            self.assertNotIn("carebench_emotion", record)
            self.assertEqual(
                record["projection"]["emotion"],
                "native_crowd_envent_single_label",
            )
            self.assertFalse(invalid_path.exists())


class CrowdEnventJudgeTests(unittest.TestCase):
    def test_judge_schema_round_trip(self) -> None:
        parsed = parse_judgment(json.dumps(judge_output_example()))
        self.assertEqual(set(parsed["appraisals"]), set(METHOD_APPRAISAL_DIMENSIONS))
        self.assertEqual(parsed["chain"]["appraisal_emotion_linkage"], 3)

    def test_missing_judgment_can_be_penalized(self) -> None:
        sample = {"id": "sample-missing"}
        report = aggregate_judgments(
            [sample],
            {},
            allow_incomplete=False,
            missing_judgment_policy="penalize",
        )
        coverage = report["coverage"]
        self.assertEqual(coverage["valid_judgments"], 0)
        self.assertEqual(coverage["evaluated_judgments"], 1)
        self.assertEqual(coverage["penalized_missing_judgments"], 1)
        self.assertEqual(coverage["coverage"], 0.0)
        self.assertEqual(coverage["metric_sample_coverage"], 1.0)
        self.assertEqual(report["mean_appraisal_score"], 1.0)
        self.assertEqual(report["chain"]["appraisal_emotion_linkage"], 1.0)


if __name__ == "__main__":
    unittest.main()
