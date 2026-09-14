"""Tests for deterministic outcome rewards and component-wise GRPO scaling."""

from __future__ import annotations

import unittest

from .reward_components import (
    REWARD_COMPONENT_FIELDS,
    build_reward_components,
    compute_process_gate,
    group_normalize_components,
    invalid_reward_components,
)


class RewardComponentTests(unittest.TestCase):
    def test_gold_outcome_scores_and_process_gate(self) -> None:
        candidate = {
            "emotion": {
                "positive_intensity": 4,
                "negative_intensity": 1,
                "positive_labels": ["hopeful", "proud"],
                "negative_labels": [],
            }
        }
        reference = {
            "gold_emotion": {
                "positive_intensity": 6,
                "negative_intensity": 0,
                "positive_labels": ["hopeful", "excited"],
                "negative_labels": [],
            }
        }
        components, diagnostics = build_reward_components(
            {
                "appraisal_reward": 0.8,
                "coherence_reward": 0.75,
                "transition_reward": 0.5,
            },
            candidate,
            reference,
            outcome_label_weight=0.5,
            outcome_intensity_weight=0.5,
            use_process_gate=True,
            process_gate_mode="min",
        )

        self.assertAlmostEqual(diagnostics["positive_label_score"], 0.5)
        self.assertAlmostEqual(diagnostics["negative_label_score"], 1.0)
        self.assertAlmostEqual(diagnostics["positive_intensity_score"], 2 / 3)
        self.assertAlmostEqual(diagnostics["negative_intensity_score"], 5 / 6)
        self.assertAlmostEqual(components["outcome_label_reward"], 0.75)
        self.assertAlmostEqual(components["outcome_intensity_reward"], 0.75)
        self.assertAlmostEqual(diagnostics["process_gate"], 0.5)
        self.assertAlmostEqual(components["outcome_reward"], 0.375)
        self.assertEqual(tuple(components), REWARD_COMPONENT_FIELDS)
        self.assertNotIn("total_reward", components)

    def test_process_gate_can_be_disabled(self) -> None:
        candidate = {
            "emotion": {
                "positive_intensity": 6,
                "negative_intensity": 0,
                "positive_labels": ["hopeful"],
                "negative_labels": [],
            }
        }
        reference = {"gold_emotion": dict(candidate["emotion"])}
        components, diagnostics = build_reward_components(
            {
                "appraisal_reward": 0.2,
                "coherence_reward": 0.5,
                "transition_reward": 0.25,
            },
            candidate,
            reference,
            outcome_label_weight=0.5,
            outcome_intensity_weight=0.5,
            use_process_gate=False,
            process_gate_mode="min",
        )
        self.assertEqual(diagnostics["process_gate"], 1.0)
        self.assertEqual(diagnostics["process_gate_enabled"], 0.0)
        self.assertEqual(components["outcome_reward"], 1.0)

    def test_process_gate_modes(self) -> None:
        self.assertAlmostEqual(compute_process_gate(0.8, 0.5, "min"), 0.5)
        self.assertAlmostEqual(compute_process_gate(0.8, 0.5, "product"), 0.4)
        self.assertAlmostEqual(
            compute_process_gate(0.8, 0.5, "geometric_mean"),
            0.4**0.5,
        )

    def test_every_component_is_normalized_within_group(self) -> None:
        rows = [
            {field: float(index) for field in REWARD_COMPONENT_FIELDS}
            for index in range(4)
        ]
        normalized, signals = group_normalize_components(
            rows,
            group_size=4,
            optimization_weights={
                "appraisal_reward": 0.7,
                "coherence_reward": 0.1,
                "transition_reward": 0.1,
                "outcome_reward": 0.1,
            },
        )
        for field in REWARD_COMPONENT_FIELDS:
            values = [float(row[field]) for row in normalized if row]
            self.assertAlmostEqual(sum(values), 0.0, places=10)
        for normalized_row, signal in zip(normalized, signals):
            assert normalized_row is not None
            self.assertAlmostEqual(
                float(signal),
                normalized_row["appraisal_reward"],
            )

    def test_api_failure_row_stays_excluded(self) -> None:
        valid = {field: 1.0 for field in REWARD_COMPONENT_FIELDS}
        normalized, signals = group_normalize_components(
            [valid, None, valid, valid],
            group_size=4,
            optimization_weights={
                "appraisal_reward": 0.7,
                "coherence_reward": 0.1,
                "transition_reward": 0.1,
                "outcome_reward": 0.1,
            },
        )
        self.assertIsNone(normalized[1])
        self.assertIsNone(signals[1])
        self.assertEqual(signals[0], 0.0)

    def test_process_gate_is_applied_after_outcome_normalization(self) -> None:
        rows = []
        for index in range(4):
            row = {field: 1.0 for field in REWARD_COMPONENT_FIELDS}
            row["outcome_label_reward"] = index / 3
            row["outcome_intensity_reward"] = index / 3
            row["outcome_reward"] = 0.2 * index / 3
            rows.append(row)
        normalized, _ = group_normalize_components(
            rows,
            group_size=4,
            optimization_weights={
                "appraisal_reward": 0.7,
                "coherence_reward": 0.1,
                "transition_reward": 0.1,
                "outcome_reward": 0.1,
            },
            process_gates=[0.2] * 4,
        )
        for row in normalized:
            assert row is not None
            self.assertAlmostEqual(
                row["outcome_reward"],
                0.2 * row["outcome_label_reward"],
            )

    def test_invalid_format_has_component_hard_gate(self) -> None:
        components = invalid_reward_components(-1.0)
        self.assertEqual(set(components), set(REWARD_COMPONENT_FIELDS))
        self.assertTrue(all(value == -1.0 for value in components.values()))


if __name__ == "__main__":
    unittest.main()
