from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from .evaluate_policy_checkpoints import (
    BertScoreBackend,
    CheckpointSpec,
    aggregate_judge_results,
    discover_checkpoint_specs,
    evaluate_automatic_metrics,
    format_coverage,
)


DIMENSIONS = [
    "relevance",
    "epistemic",
    "goal_congruence",
    "agency_accountability",
    "control_coping_potential",
]


class EvaluatePolicyCheckpointsTests(unittest.TestCase):
    def test_discovers_ordered_adapter_checkpoints_and_final(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("checkpoint-100", "checkpoint-25"):
                path = root / name
                path.mkdir()
                (path / "adapter_config.json").write_text(
                    json.dumps({"base_model_name_or_path": "/models/base"}),
                    encoding="utf-8",
                )
            (root / "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": "/models/base"}),
                encoding="utf-8",
            )
            specs, base = discover_checkpoint_specs(
                str(root),
                [],
                include_final=True,
                include_base_model=True,
                base_model_name_or_path="",
            )
            self.assertEqual(
                [spec.label for spec in specs],
                ["base", "checkpoint-25", "checkpoint-100", "final"],
            )
            self.assertEqual(base, "/models/base")
            self.assertTrue(all(isinstance(spec, CheckpointSpec) for spec in specs))

    def test_automatic_metrics_use_valid_referenced_outputs_only(self) -> None:
        reasoning = {
            dimension: "I know this event matters to my current goal."
            for dimension in DIMENSIONS
        }
        emotion = {
            "positive_intensity": 4,
            "negative_intensity": 1,
            "positive_labels": ["hopeful"],
            "negative_labels": [],
        }
        reference = {
            "legacy_human_appraisal_reasoning": reasoning,
            "gold_emotion": emotion,
        }
        examples = {
            "valid": {
                "sample_id": "valid",
                "reference_json": json.dumps(reference),
            },
            "invalid": {
                "sample_id": "invalid",
                "reference_json": json.dumps(reference),
            },
        }
        rows = [
            {
                "sample_id": "valid",
                "parsed_output": {
                    "appraisal_reasoning": reasoning,
                    "emotion": emotion,
                },
                "generation_error": "",
                "parse_error": "",
                "prompt_tokens": 100,
                "completion_tokens": 50,
            },
            {
                "sample_id": "invalid",
                "parsed_output": None,
                "generation_error": "",
                "parse_error": "invalid JSON",
                "prompt_tokens": 100,
                "completion_tokens": 5,
            },
        ]
        backend = BertScoreBackend(
            enabled=False,
            device="cpu",
            model_type="",
            batch_size=2,
            rescale_with_baseline=False,
        )
        metrics = evaluate_automatic_metrics(
            rows,
            examples,
            DIMENSIONS,
            backend,
        )
        self.assertEqual(
            metrics["coverage"]["reasoning_reference_pairs"],
            len(DIMENSIONS),
        )
        self.assertEqual(
            metrics["appraisal_reasoning"]["overall"]["bleu"],
            1.0,
        )
        self.assertEqual(
            metrics["positive_emotion"]["intensity_normalized_rmse"],
            0.0,
        )
        self.assertEqual(metrics["positive_emotion"]["example_f1"], 1.0)
        self.assertEqual(metrics["negative_emotion"]["example_f1"], 1.0)
        coverage = format_coverage(rows)
        self.assertEqual(coverage["generation_success_rate"], 1.0)
        self.assertEqual(coverage["format_valid_rate"], 0.5)

    def test_judge_api_failures_are_excluded_from_reward_mean(self) -> None:
        valid = {
            "reward_components": {
                "appraisal_reward": 0.75,
                "coherence_reward": 1.0,
                "transition_reward": 0.75,
                "outcome_label_reward": 0.8,
                "outcome_intensity_reward": 0.9,
                "outcome_reward": 0.6375,
            },
            "outcome_diagnostics": {
                "positive_label_score": 0.8,
                "negative_label_score": 0.8,
                "positive_intensity_score": 0.9,
                "negative_intensity_score": 0.9,
                "process_gate": 0.75,
            },
            "raw_transition_reward": 0.75,
            "dimension_scores": {dimension: 0.75 for dimension in DIMENSIONS},
            "appraisal_criterion_scores": {
                "dimension_specific_validity": 0.75,
                "situation_grounding": 0.75,
                "experiencer_fidelity": 0.75,
            },
            "cache_hit": False,
            "valid": True,
            "failed": False,
        }
        invalid = {
            "reward_components": {
                "appraisal_reward": -1.0,
                "coherence_reward": -1.0,
                "transition_reward": -1.0,
                "outcome_label_reward": -1.0,
                "outcome_intensity_reward": -1.0,
                "outcome_reward": -1.0,
            },
            "outcome_diagnostics": {},
            "valid": False,
            "failed": False,
        }
        api_failure = {
            "reward_components": None,
            "outcome_diagnostics": {},
            "valid": True,
            "failed": True,
        }
        summary = aggregate_judge_results(
            [valid, invalid, api_failure],
            DIMENSIONS,
        )
        self.assertAlmostEqual(
            summary["reward_component_mean_with_hard_gate"][
                "appraisal_reward"
            ],
            -0.125,
        )
        self.assertEqual(
            summary["reward_component_mean_valid"]["outcome_reward"],
            0.6375,
        )
        self.assertEqual(summary["judge_api_or_parse_failures"], 1)


if __name__ == "__main__":
    unittest.main()
