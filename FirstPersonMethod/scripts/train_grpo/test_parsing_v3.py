"""Regression tests for configurable rubric parsing and gated reward."""

from __future__ import annotations

import json
import unittest

from .parsing import (
    aggregate_judgment,
    parse_judge_output,
    parse_policy_output,
)
from .spec import (
    APPRAISAL_CRITERIA,
    APPRAISAL_DIMENSIONS,
    policy_output_example,
    resolve_appraisal_dimensions,
)

CAREBENCH_DIMENSIONS = APPRAISAL_DIMENSIONS[:5]


def score_item(score: int) -> dict[str, object]:
    return {"score": score, "rationale": "Test rationale."}


def judgment_with_scores(
    appraisal_scores: dict[str, int],
    coherence_score: int,
    transition_score: int,
    appraisal_dimensions: list[str] | None = None,
) -> dict[str, object]:
    selected = appraisal_dimensions or APPRAISAL_DIMENSIONS
    return {
        "appraisals": {
            dimension: {
                criterion: score_item(appraisal_scores[criterion])
                for criterion in APPRAISAL_CRITERIA
            }
            for dimension in selected
        },
        "coherence": score_item(coherence_score),
        "transition": score_item(transition_score),
        "overall_feedback": "Complete.",
    }


class ConfigurableRubricParsingTests(unittest.TestCase):
    def test_separate_process_schema_is_accepted(self) -> None:
        judgment = judgment_with_scores(
            {criterion: 4 for criterion in APPRAISAL_CRITERIA},
            coherence_score=4,
            transition_score=4,
        )
        parsed = parse_judge_output(json.dumps(judgment))
        self.assertEqual(parsed["coherence"]["score"], 4)
        self.assertEqual(parsed["transition"]["score"], 4)

    def test_transition_is_gated_by_validity_and_grounding(self) -> None:
        judgment = judgment_with_scores(
            {
                "dimension_specific_validity": 2,
                "situation_grounding": 1,
                "experiencer_fidelity": 4,
            },
            coherence_score=4,
            transition_score=4,
        )
        result = aggregate_judgment(judgment)

        self.assertAlmostEqual(
            result["appraisal_criterion_scores"][
                "dimension_specific_validity"
            ],
            0.5,
        )
        self.assertAlmostEqual(
            result["appraisal_criterion_scores"]["situation_grounding"],
            0.25,
        )
        self.assertAlmostEqual(
            result["raw_transition_reward"],
            1.0,
        )
        self.assertAlmostEqual(
            result["transition_gate_components"][
                "effective_transition_reward"
            ],
            0.25,
        )
        self.assertAlmostEqual(result["appraisal_reward"], 7.0 / 12.0)
        self.assertAlmostEqual(result["coherence_reward"], 1.0)
        self.assertAlmostEqual(result["transition_reward"], 0.25)

    def test_old_chain_schema_is_rejected(self) -> None:
        judgment = judgment_with_scores(
            {criterion: 4 for criterion in APPRAISAL_CRITERIA},
            coherence_score=4,
            transition_score=4,
        )
        judgment["chain"] = {
            "cross_dimension_coherence": score_item(4),
            "emotion_reference_alignment": score_item(4),
            "appraisal_emotion_linkage": score_item(4),
        }
        with self.assertRaisesRegex(ValueError, "judge.root keys mismatch"):
            parse_judge_output(json.dumps(judgment))

    def test_carebench_five_dimension_policy_and_judgment_are_accepted(
        self,
    ) -> None:
        policy = policy_output_example(CAREBENCH_DIMENSIONS)
        parsed_policy = parse_policy_output(
            json.dumps(policy),
            CAREBENCH_DIMENSIONS,
        )
        self.assertEqual(
            list(parsed_policy["appraisal_reasoning"]),
            CAREBENCH_DIMENSIONS,
        )

        judgment = judgment_with_scores(
            {criterion: 4 for criterion in APPRAISAL_CRITERIA},
            coherence_score=4,
            transition_score=4,
            appraisal_dimensions=CAREBENCH_DIMENSIONS,
        )
        parsed_judgment = parse_judge_output(
            json.dumps(judgment),
            CAREBENCH_DIMENSIONS,
        )
        result = aggregate_judgment(
            parsed_judgment,
            appraisal_dimensions=CAREBENCH_DIMENSIONS,
        )
        self.assertEqual(
            list(result["dimension_scores"]),
            CAREBENCH_DIMENSIONS,
        )
        self.assertEqual(result["appraisal_reward"], 1.0)
        self.assertEqual(result["coherence_reward"], 1.0)
        self.assertEqual(result["transition_reward"], 1.0)

    def test_selected_dimension_schema_rejects_missing_and_extra_keys(
        self,
    ) -> None:
        five_dimension_policy = policy_output_example(
            CAREBENCH_DIMENSIONS
        )
        with self.assertRaisesRegex(
            ValueError,
            "appraisal_reasoning keys mismatch",
        ):
            parse_policy_output(json.dumps(five_dimension_policy))

        six_dimension_policy = policy_output_example()
        with self.assertRaisesRegex(
            ValueError,
            "appraisal_reasoning keys mismatch",
        ):
            parse_policy_output(
                json.dumps(six_dimension_policy),
                CAREBENCH_DIMENSIONS,
            )

    def test_emotion_synonyms_are_normalized_to_short_ids(self) -> None:
        policy = policy_output_example(CAREBENCH_DIMENSIONS)
        policy["emotion"] = {
            "positive_intensity": 4,
            "negative_intensity": 2,
            "positive_labels": ["Excited", "Hopeful", "Proud"],
            "negative_labels": ["Fearful", "Sorrow"],
        }
        parsed = parse_policy_output(
            json.dumps(policy),
            CAREBENCH_DIMENSIONS,
        )
        self.assertEqual(
            parsed["emotion"]["positive_labels"],
            ["hopeful", "proud", "excited"],
        )
        self.assertEqual(
            parsed["emotion"]["negative_labels"],
            ["worried", "despair"],
        )

    def test_multiple_synonyms_for_one_group_are_merged(self) -> None:
        policy = policy_output_example(CAREBENCH_DIMENSIONS)
        policy["emotion"]["positive_intensity"] = 3
        policy["emotion"]["positive_labels"] = ["Hopeful", "optimistic"]
        parsed = parse_policy_output(
            json.dumps(policy),
            CAREBENCH_DIMENSIONS,
        )
        self.assertEqual(
            parsed["emotion"]["positive_labels"],
            ["hopeful"],
        )

    def test_dimension_selection_is_validated_and_canonicalized(self) -> None:
        selected = resolve_appraisal_dimensions(
            "control_coping_potential,relevance,epistemic"
        )
        self.assertEqual(
            selected,
            [
                "relevance",
                "epistemic",
                "control_coping_potential",
            ],
        )
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            resolve_appraisal_dimensions("relevance,relevance")
        with self.assertRaisesRegex(ValueError, "Unknown"):
            resolve_appraisal_dimensions("relevance,not_a_dimension")


if __name__ == "__main__":
    unittest.main()
