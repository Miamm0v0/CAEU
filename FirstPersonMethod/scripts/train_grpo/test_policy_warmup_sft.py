"""Tests for Policy warm-up target construction without ML dependencies."""

from __future__ import annotations

import json
import unittest

from .parsing import parse_policy_output
from .spec import (
    APPRAISAL_DIMENSIONS,
    POLICY_SYSTEM_PROMPT,
    build_policy_system_prompt,
)
from .train_policy_warmup_sft import (
    DEFAULT_MISSING_NORM_VALUE_TARGET,
    build_warmup_example,
    build_warmup_parser,
    normalize_warmup_reasoning,
)

CAREBENCH_DIMENSIONS = APPRAISAL_DIMENSIONS[:5]


def sample_record() -> dict[str, object]:
    return {
        "id": "participant_sample",
        "perspective": "first_person",
        "situation": "My friend helped me finish an important task.",
        "appraisal_reasoning": {
            "relevance": "This mattered because the task was important to me.",
            "certainty": "I knew the task was now likely to be completed.",
            "congruence": "The help moved me closer to my goal.",
            "accountability": "My friend intentionally helped me.",
            "control": "I could continue working with my friend's support.",
        },
        "emotion": {
            "positive_intensity": 3,
            "negative_intensity": 2,
            "positive_labels": ["grateful", "sorrow"],
            "negative_labels": ["worried"],
        },
    }


class PolicyWarmupSFTTests(unittest.TestCase):
    def test_target_uses_current_policy_schema(self) -> None:
        example = build_warmup_example(sample_record())
        self.assertEqual(example["prompt"][0]["content"], POLICY_SYSTEM_PROMPT)
        self.assertEqual(
            example["chat_template_kwargs"],
            {"enable_thinking": False},
        )

        target = parse_policy_output(example["completion"][0]["content"])
        self.assertEqual(
            set(target["appraisal_reasoning"]),
            set(APPRAISAL_DIMENSIONS),
        )
        self.assertEqual(
            target["appraisal_reasoning"]["norm_value_compatibility"],
            DEFAULT_MISSING_NORM_VALUE_TARGET,
        )

    def test_misbucketed_labels_are_canonicalized_by_valence(self) -> None:
        example = build_warmup_example(sample_record())
        target = json.loads(example["completion"][0]["content"])

        self.assertIn(
            "grateful",
            target["emotion"]["positive_labels"],
        )
        self.assertNotIn(
            "despair",
            target["emotion"]["positive_labels"],
        )
        self.assertIn(
            "despair",
            target["emotion"]["negative_labels"],
        )

    def test_curated_norm_value_can_be_required(self) -> None:
        record = sample_record()
        with self.assertRaisesRegex(ValueError, "norm_value_compatibility"):
            normalize_warmup_reasoning(record, "error")

        record["appraisal_reasoning"]["norm_value_compatibility"] = (
            "I see helping a friend as consistent with my values."
        )
        reasoning = normalize_warmup_reasoning(record, "error")
        self.assertIn("consistent with my values", reasoning[
            "norm_value_compatibility"
        ])

    def test_carebench_target_uses_only_first_five_dimensions(self) -> None:
        example = build_warmup_example(
            sample_record(),
            appraisal_dimensions=CAREBENCH_DIMENSIONS,
        )
        self.assertEqual(
            example["prompt"][0]["content"],
            build_policy_system_prompt(CAREBENCH_DIMENSIONS),
        )
        self.assertNotIn(
            "norm_value_compatibility",
            example["completion"][0]["content"],
        )
        target = parse_policy_output(
            example["completion"][0]["content"],
            CAREBENCH_DIMENSIONS,
        )
        self.assertEqual(
            list(target["appraisal_reasoning"]),
            CAREBENCH_DIMENSIONS,
        )
        reasoning = normalize_warmup_reasoning(
            sample_record(),
            "error",
            CAREBENCH_DIMENSIONS,
        )
        self.assertEqual(list(reasoning), CAREBENCH_DIMENSIONS)

    def test_pilot_and_thinking_arguments_are_available(self) -> None:
        args = build_warmup_parser().parse_args(
            [
                "--max_train_samples",
                "16",
                "--max_eval_samples",
                "4",
                "--enable_thinking",
                "--appraisal_dimensions",
                "control_coping_potential,relevance,epistemic",
            ]
        )
        self.assertEqual(args.max_train_samples, 16)
        self.assertEqual(args.max_eval_samples, 4)
        self.assertTrue(args.enable_thinking)
        self.assertEqual(args.eval_ratio, 0.0)
        self.assertEqual(
            args.appraisal_dimensions,
            [
                "relevance",
                "epistemic",
                "control_coping_potential",
            ],
        )


if __name__ == "__main__":
    unittest.main()
