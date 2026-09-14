from __future__ import annotations

import json
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_appraisal_counterfactual_data as builder  # noqa: E402
import counterfactual_prefix_transformers as module  # noqa: E402
import evaluate_counterfactual as ratings_evaluator  # noqa: E402
import evaluate_counterfactual_emotion as emotion_evaluator  # noqa: E402
from baseline_transformers_backend import TransformersClient  # noqa: E402


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def continue_chat(self, system_prompt, user_prompt, assistant_prefix):
        self.calls.append((system_prompt, user_prompt, assistant_prefix))
        if '"appraisals"' in user_prompt:
            appraisals = {
                key: {"score": 3, "label": "Neither agree nor disagree"}
                for _, key in module.chain.sft_common.APPRAISAL_KEY_MAP
            }
            suffix = '"appraisals": ' + json.dumps(appraisals) + "\n}"
        else:
            emotion = {
                "positive_intensity": 4,
                "negative_intensity": 1,
                "positive_labels": ["hopeful"],
                "negative_labels": ["worried"],
            }
            suffix = '"emotion": ' + json.dumps(emotion) + "\n}"
        return module.baseline.ChatResponse(
            content=suffix,
            raw={"backend": "fake", "assistant_prefill": True},
        )


class PrefixUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reasoning = {
            dimension: f"Original text for {dimension}."
            for dimension in module.POLICY_DIMENSIONS
        }

    def test_prefix_stops_before_target_and_complete_profiles_are_normalized(self):
        original_prefix = module.build_assistant_prefix(self.reasoning)
        self.assertTrue(original_prefix.endswith(",\n  "))
        self.assertNotIn('"emotion"', original_prefix)
        self.assertNotIn('"appraisals"', original_prefix)

        carebench_profile = {
            dimension: f"Gold text for {dimension}."
            for dimension in module.CAREBENCH_TO_POLICY
        }
        normalized = module.extract_complete_profile(
            {"third_person_appraisal_reasoning": carebench_profile},
            "third-person",
            "test item",
        )
        self.assertEqual(set(normalized), set(module.POLICY_DIMENSIONS))
        self.assertEqual(
            module.changed_dimensions(self.reasoning, normalized),
            list(module.POLICY_DIMENSIONS),
        )

    def test_selected_dimensions_replace_only_requested_appraisals(self):
        source = {
            dimension: f"Human text for {dimension}."
            for dimension in module.POLICY_DIMENSIONS
        }
        selected = module.normalize_replacement_dimensions(
            ["relevance", "certainty"]
        )
        replaced = module.replace_appraisal_dimensions(
            self.reasoning, source, selected
        )

        self.assertEqual(selected, ("relevance", "epistemic"))
        self.assertEqual(replaced["relevance"], source["relevance"])
        self.assertEqual(replaced["epistemic"], source["epistemic"])
        self.assertEqual(
            replaced["goal_congruence"], self.reasoning["goal_congruence"]
        )
        with self.assertRaises(ValueError):
            module.normalize_replacement_dimensions(["all", "relevance"])

    def test_builder_keeps_native_situation_separate_from_appraisals(self):
        record = builder.NormalizedRecord(
            sample_id="sample",
            participant_id="person",
            situation="I received unexpected good news.",
            appraisal_ratings={},
            appraisal_scores={},
            appraisal_reasoning={
                dimension: f"Original {dimension}."
                for dimension in builder.CORE_DIMENSION_ORDER
            },
            emotion_labels=None,
            raw={},
        )
        payload = builder.first_person_payload(record, "situation-only")
        self.assertEqual(
            payload["story_collection"]["final_scenario"], record.situation
        )
        self.assertEqual(
            payload["cognitive_questions"]["final_scenario"], record.situation
        )
        self.assertEqual(payload["appraisal_reasoning"], record.appraisal_reasoning)
        self.assertEqual(
            payload["first_person_appraisal_reasoning"],
            record.appraisal_reasoning,
        )
        self.assertNotIn("Appraisal profile:", payload["story_collection"]["final_scenario"])

    def test_builder_preserves_complete_first_and_third_person_profiles(self):
        first_reasoning = {
            dimension: f"First {dimension}."
            for dimension in builder.CORE_DIMENSION_ORDER
        }
        third_reasoning = {
            dimension: f"Third {dimension}."
            for dimension in builder.CORE_DIMENSION_ORDER
        }
        first = builder.NormalizedRecord(
            sample_id="sample",
            participant_id="first",
            situation="A shared situation.",
            appraisal_ratings={},
            appraisal_scores={},
            appraisal_reasoning=first_reasoning,
            emotion_labels=None,
            raw={},
        )
        third = builder.ThirdPersonRecord(
            source_sample_id="sample",
            participant_id="third",
            appraisal_ratings={},
            appraisal_scores={},
            appraisal_reasoning=third_reasoning,
            emotion_labels=None,
            raw={},
        )
        payload = builder.third_person_core_counterfactual_payload(
            first,
            third,
            "relevance",
            {
                "opposite_count": 1,
                "comparable_count": 1,
                "opposite_fraction": 1.0,
                "neutral_or_missing_count": 0,
                "opposite_items": [],
            },
        )

        self.assertEqual(
            payload["first_person_appraisal_reasoning"], first_reasoning
        )
        self.assertEqual(
            payload["third_person_appraisal_reasoning"], third_reasoning
        )
        self.assertNotEqual(
            payload["appraisal_reasoning"],
            payload["third_person_appraisal_reasoning"],
        )

    def test_transformers_backend_keeps_assistant_message_open(self):
        class FakeInputs:
            shape = (1, 4)

            def to(self, _device):
                return self

        class FakeGenerated:
            shape = (2,)

        class FakeOutput:
            def __getitem__(self, key):
                self.key = key
                return FakeGenerated()

        class FakeTokenizer:
            pad_token_id = 0

            def __init__(self):
                self.messages = None
                self.kwargs = None

            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                self.kwargs = kwargs
                return {"input_ids": FakeInputs(), "attention_mask": FakeInputs()}

            def decode(self, _ids, **_kwargs):
                return '"emotion": {}\n}'

        class FakeModel:
            def generate(self, **_kwargs):
                return FakeOutput()

        class InferenceMode:
            def __enter__(self):
                return None

            def __exit__(self, *_args):
                return False

        tokenizer = FakeTokenizer()
        client = TransformersClient.__new__(TransformersClient)
        client.tokenizer = tokenizer
        client.model = FakeModel()
        client.torch = types.SimpleNamespace(
            Tensor=type("UnusedTensor", (), {}),
            manual_seed=lambda _seed: None,
            cuda=types.SimpleNamespace(
                is_available=lambda: False,
                manual_seed_all=lambda _seed: None,
            ),
            inference_mode=lambda: InferenceMode(),
        )
        client.input_device = "cpu"
        client.generation_lock = threading.Lock()
        client.max_tokens = 32
        client.do_sample = False
        client.temperature = 0.2
        client.top_p = 1.0
        client.enable_thinking = False
        client.seed = 42
        client.request_count = 0
        client.model_name_or_path = "fake"
        client.is_adapter = False
        client.base_model_name_or_path = ""

        response = client.continue_chat("system", "user", "{prefix,")
        self.assertEqual(response.content, '"emotion": {}\n}')
        self.assertEqual(
            tokenizer.messages[-1],
            {"role": "assistant", "content": "{prefix,"},
        )
        self.assertTrue(tokenizer.kwargs["continue_final_message"])
        self.assertFalse(tokenizer.kwargs["add_generation_prompt"])
        self.assertTrue(response.raw["assistant_prefill"])


class PrefixRunnerTests(unittest.TestCase):
    def test_paired_runner_uses_native_prompt_and_writes_both_outcomes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_root = root / "origin_data"
            source_root = root / "counterfactual_data"
            baseline_root = root / "baseline"
            target_root = root / "prefix_run"
            first_root.mkdir()
            (source_root / "relevance").mkdir(parents=True)
            (baseline_root / "chain-emotion").mkdir(parents=True)

            situation = "I received the award I had worked toward."
            first_person_ratings = {
                statement: "Strongly agree"
                for statement in builder.SPEC_BY_STATEMENT
            }
            first_payload = {
                "id": "sample",
                "participant_id": "first-person-1",
                "situation": situation,
                "story_collection": {"final_scenario": situation},
                "appraisal_reasoning": {
                    dimension: f"First-person gold appraisal for {dimension}."
                    for dimension in builder.CORE_DIMENSION_ORDER
                },
                "appraisal_ratings": first_person_ratings,
                "emotion_labels": {
                    "positive_level": "5 - Very positive",
                    "negative_level": "0 - Not at all negative",
                    "positive_emotion_labels": [
                        "Proud, confident, determined"
                    ],
                    "negative_emotion_labels": [],
                },
            }
            (first_root / "sample.json").write_text(
                json.dumps(first_payload), encoding="utf-8"
            )

            original_reasoning = {
                dimension: f"Original model appraisal for {dimension}."
                for dimension in module.POLICY_DIMENSIONS
            }
            baseline_payload = {
                "appraisal_reasoning": original_reasoning,
                "emotion": {
                    "positive_intensity": 5,
                    "negative_intensity": 0,
                    "positive_labels": ["proud"],
                    "negative_labels": [],
                },
            }
            (baseline_root / "chain-emotion" / "sample.json").write_text(
                json.dumps(baseline_payload), encoding="utf-8"
            )

            replacement = "This outcome does not matter to me at all."
            third_person_reasoning = {
                dimension: f"Third-person gold appraisal for {dimension}."
                for dimension in builder.CORE_DIMENSION_ORDER
            }
            third_person_reasoning["relevance"] = replacement
            counterfactual_item = {
                "participant_id": "third-person-1",
                "situation": situation,
                "appraisal_reasoning": {"relevance": replacement},
                "third_person_appraisal_reasoning": third_person_reasoning,
                "appraisal_ratings": {
                    statement: "Strongly disagree"
                    for statement in builder.SPEC_BY_STATEMENT
                },
                "metadata": {
                    "source_sample_id": "sample",
                    "counterfactual_dimension": "relevance",
                },
                "emotion_labels": {
                    "positive_level": "2 - Slightly positive",
                    "negative_level": "3 - Moderately negative",
                    "positive_emotion_labels": [],
                    "negative_emotion_labels": [
                        "Worried, nervous, fearful"
                    ],
                },
            }
            (source_root / "relevance" / "sample.json").write_text(
                json.dumps([counterfactual_item]), encoding="utf-8"
            )

            fake_client = FakeClient()
            prompt_path = (
                module.REPO_ROOT
                / "FirstPersonMethod"
                / "scripts"
                / "prompts"
                / "baseline_prompt.toml"
            )
            argv = [
                "counterfactual_prefix_transformers.py",
                "--model",
                "dummy",
                "--first_person_root",
                str(first_root),
                "--source_root",
                str(source_root),
                "--baseline_root",
                str(baseline_root),
                "--target_root",
                str(target_root),
                "--prompt_path",
                str(prompt_path),
                "--outcome",
                "all",
                "--verbose",
                "false",
            ]
            with patch.object(module, "build_transformers_client", return_value=fake_client):
                with patch.object(sys, "argv", argv):
                    self.assertEqual(module.main(), 0)

            self.assertEqual(len(fake_client.calls), 6)
            for _, user_prompt, _ in fake_client.calls:
                self.assertIn(situation, user_prompt)
                self.assertNotIn(replacement, user_prompt)

            prefixes = [call[2] for call in fake_client.calls]
            self.assertEqual(sum(replacement in prefix for prefix in prefixes), 2)
            self.assertEqual(
                sum("First-person gold appraisal" in prefix for prefix in prefixes),
                2,
            )
            self.assertEqual(
                sum("Original model appraisal" in prefix for prefix in prefixes),
                2,
            )

            origin_emotion = json.loads(
                (target_root / "origin" / "positive-level" / "sample.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(origin_emotion["score"], 4)

            cf_emotion = json.loads(
                (
                    target_root
                    / "third-person-appraisal"
                    / "positive-labels"
                    / "relevance"
                    / "sample.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(cf_emotion[0]["labels"], [
                "Hopeful, optimistic, encouraged"
            ])

            cf_ratings = json.loads(
                (
                    target_root
                    / "third-person-appraisal"
                    / "appraisals"
                    / "relevance"
                    / "sample.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(len(cf_ratings[0]["target_appraisals"]), 22)

            first_person_emotion = json.loads(
                (
                    target_root
                    / "first-person-appraisal"
                    / "positive-level"
                    / "sample.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(first_person_emotion["score"], 4)

            audit = json.loads(
                (
                    target_root
                    / "trajectories"
                    / "third-person-appraisal"
                    / "emotion"
                    / "relevance"
                    / "sample.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                audit[0]["replaced_dimensions"],
                list(module.POLICY_DIMENSIONS),
            )
            self.assertEqual(
                audit[0]["third_person_appraisal_reasoning"]["relevance"],
                replacement,
            )
            manifest = json.loads(
                (target_root / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["replacement_dimensions"],
                list(module.POLICY_DIMENSIONS),
            )

            evaluation_file = root / "emotion_results.json"
            evaluation_argv = [
                "evaluate_counterfactual_emotion.py",
                "--first_person_root",
                str(first_root),
                "--counterfactual_gold_root",
                str(source_root),
                "--baseline_root",
                str(target_root / "origin"),
                "--counterfactual_pred_root",
                str(target_root / "third-person-appraisal"),
                "--prompt_path",
                str(prompt_path),
                "--output_file",
                str(evaluation_file),
            ]
            with patch.object(sys, "argv", evaluation_argv):
                self.assertEqual(emotion_evaluator.main(), 0)
            evaluation = json.loads(evaluation_file.read_text(encoding="utf-8"))
            self.assertEqual(
                evaluation["dimension"]["relevance"]["positive-level"]["stats"][
                    "evaluated_samples"
                ],
                1,
            )

            ratings_evaluation_file = root / "ratings_results.json"
            ratings_evaluation_argv = [
                "evaluate_counterfactual.py",
                "--first_person_root",
                str(first_root),
                "--counterfactual_gold_root",
                str(source_root),
                "--baseline_root",
                str(target_root / "origin"),
                "--counterfactual_pred_root",
                str(target_root / "third-person-appraisal"),
                "--prompt_path",
                str(prompt_path),
                "--output_file",
                str(ratings_evaluation_file),
            ]
            with patch.object(sys, "argv", ratings_evaluation_argv):
                self.assertEqual(ratings_evaluator.main(), 0)
            ratings_evaluation = json.loads(
                ratings_evaluation_file.read_text(encoding="utf-8")
            )
            self.assertEqual(
                ratings_evaluation["dimension"]["relevance.general"]["stats"][
                    "evaluated_samples"
                ],
                1,
            )
            self.assertEqual(
                len(
                    ratings_evaluation["intervention_dimension"]["relevance"][
                        "ratings"
                    ]
                ),
                6,
            )


if __name__ == "__main__":
    unittest.main()
