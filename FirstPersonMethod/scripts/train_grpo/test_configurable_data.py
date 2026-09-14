"""Tests for dimension-aware GRPO data preparation."""

from __future__ import annotations

import json
import unittest

from .data import build_hidden_reference, normalize_policy_record
from .parsing import parse_judge_output
from .spec import (
    APPRAISAL_DIMENSIONS,
    build_judge_system_prompt,
    judge_output_example,
)


CAREBENCH_DIMENSIONS = APPRAISAL_DIMENSIONS[:5]


def carebench_record() -> dict[str, object]:
    return {
        "id": "participant_sample",
        "perspective": "first_person",
        "situation": "My friend helped me finish an important task.",
        "appraisal_reasoning": {
            "relevance": "The task mattered to me.",
            "certainty": "I knew I was likely to finish.",
            "congruence": "The help advanced my goal.",
            "accountability": "My friend intentionally helped me.",
            "control": "I could keep working with support.",
        },
        "emotion": {
            "positive_intensity": 3,
            "negative_intensity": 0,
            "positive_labels": ["grateful"],
            "negative_labels": [],
        },
    }


class ConfigurableDataTests(unittest.TestCase):
    def test_judge_prompt_contains_complete_strict_output_template(self) -> None:
        prompt = build_judge_system_prompt(CAREBENCH_DIMENSIONS)
        self.assertIn("REQUIRED OUTPUT TEMPLATE:", prompt)
        self.assertIn('"appraisals": {', prompt)
        self.assertIn('"dimension_specific_validity": {', prompt)
        self.assertIn('"score": 0', prompt)
        self.assertIn('"rationale":', prompt)
        self.assertIn('"coherence": {', prompt)
        self.assertIn('"transition": {', prompt)
        self.assertNotIn("emotion_reference_alignment", prompt)
        self.assertIn('"overall_feedback":', prompt)
        self.assertNotIn('"norm_value_compatibility": {', prompt)
        self.assertIn("Never shorten a criterion to a bare number", prompt)

        example = judge_output_example(CAREBENCH_DIMENSIONS)
        parsed = parse_judge_output(
            json.dumps(example),
            CAREBENCH_DIMENSIONS,
        )
        self.assertEqual(
            list(parsed["appraisals"]),
            CAREBENCH_DIMENSIONS,
        )

    def test_hidden_reference_contains_only_selected_dimensions(self) -> None:
        reference = build_hidden_reference(
            carebench_record(),
            CAREBENCH_DIMENSIONS,
        )
        reasoning = reference["legacy_human_appraisal_reasoning"]
        self.assertEqual(list(reasoning), CAREBENCH_DIMENSIONS)
        self.assertNotIn("norm_value_compatibility", reasoning)
        self.assertEqual(
            reference["gold_emotion"]["positive_labels"],
            ["grateful"],
        )

    def test_reference_mode_none_still_keeps_local_gold_outcome(self) -> None:
        example = normalize_policy_record(
            carebench_record(),
            index=0,
            reference_mode="none",
            appraisal_dimensions=CAREBENCH_DIMENSIONS,
        )
        reference = json.loads(example["reference_json"])
        self.assertIn("gold_emotion", reference)
        self.assertNotIn("legacy_human_appraisal_reasoning", reference)

    def test_carebench_reference_can_be_required_for_five_dimensions(
        self,
    ) -> None:
        example = normalize_policy_record(
            carebench_record(),
            index=0,
            reference_mode="required",
            appraisal_dimensions=CAREBENCH_DIMENSIONS,
        )
        reference = json.loads(example["reference_json"])
        self.assertEqual(
            list(reference["legacy_human_appraisal_reasoning"]),
            CAREBENCH_DIMENSIONS,
        )
        prompt_text = "\n".join(
            message["content"] for message in example["prompt"]
        )
        self.assertNotIn("norm_value_compatibility", prompt_text)

        with self.assertRaisesRegex(ValueError, "every selected"):
            normalize_policy_record(
                carebench_record(),
                index=0,
                reference_mode="required",
                appraisal_dimensions=APPRAISAL_DIMENSIONS,
            )


if __name__ == "__main__":
    unittest.main()
