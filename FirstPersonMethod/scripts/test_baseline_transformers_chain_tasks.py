from __future__ import annotations

import json
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# The project runtime installs tomli on Python < 3.11.  Keep these parser-only
# tests runnable in a minimal environment where that optional backport is not
# installed; the tests construct their prompt config directly below.
try:
    import tomllib  # noqa: F401
except ImportError:
    try:
        import tomli  # noqa: F401
    except ImportError:
        tomli_stub = types.ModuleType("tomli")
        tomli_stub.load = lambda *_args, **_kwargs: {}
        sys.modules["tomli"] = tomli_stub

import baseline_transformers_chain as module  # noqa: E402
import baseline_transformers as runner  # noqa: E402
from train_grpo.spec import (  # noqa: E402
    build_policy_system_prompt,
    build_policy_user_prompt,
    policy_output_example,
)


class FakeClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def chat(self, system_prompt, user_prompt):
        del system_prompt, user_prompt
        self.calls += 1
        return module.baseline.ChatResponse(
            content=json.dumps(self.payload), raw={"backend": "fake"}
        )


class ChainTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        chain_template = {
            "user": (
                "{scenario}\n{core_appraisal_questions}\n{appraisal_statements}\n"
                "{positive_labels}\n{negative_labels}\n{output_schema}"
            )
        }
        cls.prompt_cfg = {
            "templates": {
                task: dict(chain_template) for task in module.CHAIN_TASKS
            },
            "core-appraisals": {
                "questions": {
                    dimension: f"Question about {dimension}?"
                    for dimension in module.CORE_APPRAISAL_DIMENSIONS
                }
            },
            "appraisals": {
                "dimension_to_statement": {
                    key: f"Statement about {key}."
                    for _, key in module.sft_common.APPRAISAL_KEY_MAP
                }
            },
            "label_maps": {
                "appraisals": {
                    "Strongly disagree": 1,
                    "Somewhat disagree": 2,
                    "Neither agree nor disagree": 3,
                    "Somewhat agree": 4,
                    "Strongly agree": 5,
                },
                "positive-level": {
                    "Not at all positive": 0,
                    "Very slightly positive": 1,
                    "Slightly positive": 2,
                    "Moderately positive": 3,
                    "Quite a bit positive": 4,
                    "Very positive": 5,
                    "Extremely positive": 6,
                },
                "negative-level": {
                    "Not at all negative": 0,
                    "Very slightly negative": 1,
                    "Slightly negative": 2,
                    "Moderately negative": 3,
                    "Quite a bit negative": 4,
                    "Very negative": 5,
                    "Extremely negative": 6,
                },
            },
            "label_options": {
                "positive-labels": {
                    "values": list(
                        module.sft_common.POSITIVE_LABEL_DESCRIPTIONS.values()
                    )
                },
                "negative-labels": {
                    "values": list(
                        module.sft_common.NEGATIVE_LABEL_DESCRIPTIONS.values()
                    )
                },
            },
        }
        cls.prompt_cfg["templates"][module.CHAIN_EMOTION_TASK] = {
            "system": "Dimensions: {appraisal_dimension_names}",
            "user": (
                "{scenario}\n{appraisal_definition_lines}\n{positive_labels}\n"
                "{negative_labels}\n{output_schema}"
            ),
        }
        cls.prompt_cfg["templates"][module.FULL_CHAIN_TASK] = dict(
            chain_template
        )
        cls.core = {
            dimension: f"I give a grounded answer about {dimension}."
            for dimension in module.CORE_APPRAISAL_DIMENSIONS
        }
        cls.policy_reasoning = {
            dimension: f"I give a grounded answer about {dimension}."
            for dimension in module.POLICY_APPRAISAL_DIMENSIONS
        }

    def payload_for(self, task):
        if task in module.GRPO_SCHEMA_CHAIN_TASKS:
            payload = {"appraisal_reasoning": dict(self.policy_reasoning)}
        else:
            payload = {"core_appraisals": dict(self.core)}
        if task == "chain-appraisals":
            payload["appraisals"] = {
                key: {"score": 3, "label": "Neither agree nor disagree"}
                for _, key in module.sft_common.APPRAISAL_KEY_MAP
            }
        elif task == "chain-positive-level":
            payload["positive_level"] = {
                "score": 4,
                "label": "Quite a bit positive",
            }
        elif task == "chain-negative-level":
            payload["negative_level"] = {
                "score": 2,
                "label": "Slightly negative",
            }
        elif task == "chain-positive-labels":
            payload["positive_labels"] = ["Hopeful", "excited"]
        elif task == "chain-negative-labels":
            payload["negative_labels"] = ["sad", "Surprised"]
        return payload

    def test_chain_all_expands_to_six_chain_tasks(self):
        self.assertEqual(
            module.resolve_selected_tasks("chain-all"), list(module.CHAIN_TASKS)
        )
        self.assertEqual(
            module.resolve_selected_tasks("chain-all", "separate"),
            list(module.CHAIN_TASKS),
        )
        self.assertEqual(len(module.CHAIN_TASKS), 6)
        self.assertNotIn(module.CHAIN_EMOTION_TASK, module.CHAIN_TASKS)
        self.assertEqual(
            module.resolve_selected_tasks("chain-all", "chain-emotion"),
            list(module.CHAIN_ALL_WITH_JOINT_EMOTION_TASKS),
        )
        self.assertEqual(
            list(module.CHAIN_ALL_WITH_JOINT_EMOTION_TASKS),
            [
                "chain-appraisals",
                "chain-emotion",
                "chain-core-appraisals",
            ],
        )
        self.assertEqual(
            module.resolve_selected_tasks("chain-emotion"),
            [module.CHAIN_EMOTION_TASK],
        )
        self.assertEqual(
            module.resolve_selected_tasks(
                "chain-all", "chain-emotion", "chain-emotion"
            ),
            ["chain-appraisals", "chain-emotion"],
        )
        with self.assertRaisesRegex(ValueError, "requires"):
            module.resolve_selected_tasks(
                "chain-all", "separate", "chain-emotion"
            )

    def test_chain_all_emotion_mode_cli(self):
        parser = runner.build_argument_parser()
        default_args = parser.parse_args(["--model", "dummy", "--task", "chain-all"])
        self.assertEqual(default_args.chain_all_emotion_mode, "separate")
        self.assertEqual(
            default_args.chain_core_appraisals_source, "separate"
        )
        joint_args = parser.parse_args(
            [
                "--model",
                "dummy",
                "--task",
                "chain-all",
                "--chain_all_emotion_mode",
                "chain-emotion",
                "--chain_core_appraisals_source",
                "chain-emotion",
            ]
        )
        self.assertEqual(joint_args.chain_all_emotion_mode, "chain-emotion")
        self.assertEqual(
            joint_args.chain_core_appraisals_source, "chain-emotion"
        )

    def test_every_chain_prompt_builds_and_every_output_parses(self):
        for task in module.CHAIN_TASKS:
            with self.subTest(task=task):
                _, prompt = module.build_chain_task_prompts(
                    self.prompt_cfg, "I experienced an event.", task
                )
                reasoning_field = (
                    "appraisal_reasoning"
                    if task in module.GRPO_SCHEMA_CHAIN_TASKS
                    else "core_appraisals"
                )
                self.assertIn(reasoning_field, prompt)
                trace, derived = module.parse_chain_task_output(
                    json.dumps(self.payload_for(task)), self.prompt_cfg, task
                )
                self.assertEqual(list(trace)[0], reasoning_field)
                self.assertIsInstance(derived, dict)

    def test_each_chain_task_uses_one_model_call(self):
        for task in module.CHAIN_TASKS:
            with self.subTest(task=task):
                client = FakeClient(self.payload_for(task))
                trace, derived, issues = module.process_chain_task(
                    client, self.prompt_cfg, "I experienced an event.", task
                )
                self.assertEqual(client.calls, 1)
                self.assertIsNotNone(trace)
                self.assertIsNotNone(derived)
                self.assertEqual(issues, [])

    def test_label_outputs_are_converted_to_benchmark_groups(self):
        _, positive = module.parse_chain_task_output(
            json.dumps(self.payload_for("chain-positive-labels")),
            self.prompt_cfg,
            "chain-positive-labels",
        )
        self.assertEqual(
            positive,
            {
                "labels": [
                    "Hopeful, optimistic, encouraged",
                    "Excited, enthusiastic, elated",
                ]
            },
        )

    def test_missing_core_dimension_is_invalid(self):
        payload = self.payload_for("chain-positive-level")
        del payload["core_appraisals"][module.CORE_APPRAISAL_DIMENSIONS[-1]]
        with self.assertRaisesRegex(ValueError, "core_appraisals keys mismatch"):
            module.parse_chain_task_output(
                json.dumps(payload), self.prompt_cfg, "chain-positive-level"
            )

    def test_selected_chain_tasks_use_policy_keys_and_map_at_boundary(self):
        appraisal_payload = self.payload_for("chain-appraisals")
        appraisal_trace, _ = module.parse_chain_task_output(
            json.dumps(appraisal_payload),
            self.prompt_cfg,
            "chain-appraisals",
        )
        self.assertEqual(
            set(appraisal_trace["appraisal_reasoning"]),
            set(module.POLICY_APPRAISAL_DIMENSIONS),
        )

        core_payload = self.payload_for("chain-core-appraisals")
        core_trace, carebench_output = module.parse_chain_task_output(
            json.dumps(core_payload),
            self.prompt_cfg,
            "chain-core-appraisals",
        )
        self.assertEqual(
            set(core_trace["appraisal_reasoning"]),
            set(module.POLICY_APPRAISAL_DIMENSIONS),
        )
        self.assertEqual(
            set(carebench_output), set(module.CORE_APPRAISAL_DIMENSIONS)
        )
        for policy_dimension, carebench_dimension in (
            module.POLICY_TO_CAREBENCH_CORE_DIMENSION.items()
        ):
            self.assertEqual(
                carebench_output[carebench_dimension]["answer"],
                self.policy_reasoning[policy_dimension],
            )

    def test_full_chain_uses_policy_keys_and_maps_at_boundary(self):
        payload = {
            "appraisal_reasoning": dict(self.policy_reasoning),
            "appraisal_ratings": {
                key: 3 for _, key in module.sft_common.APPRAISAL_KEY_MAP
            },
            "emotion": {
                "positive_intensity": 4,
                "negative_intensity": 2,
                "positive_labels": ["hopeful"],
                "negative_labels": ["sad"],
            },
        }
        _, prompt = module.build_full_chain_prompts(
            self.prompt_cfg, "I experienced an event."
        )
        for dimension in module.POLICY_APPRAISAL_DIMENSIONS:
            self.assertIn(f'"{dimension}"', prompt)

        full_chain, task_outputs = module.parse_full_chain_output(
            json.dumps(payload), self.prompt_cfg
        )
        self.assertEqual(
            set(full_chain["appraisal_reasoning"]),
            set(module.POLICY_APPRAISAL_DIMENSIONS),
        )
        self.assertEqual(
            set(task_outputs["core-appraisals"]),
            set(module.CORE_APPRAISAL_DIMENSIONS),
        )
        for policy_dimension, carebench_dimension in (
            module.POLICY_TO_CAREBENCH_CORE_DIMENSION.items()
        ):
            self.assertEqual(
                task_outputs["core-appraisals"][carebench_dimension]["answer"],
                self.policy_reasoning[policy_dimension],
            )

    def test_chain_emotion_uses_one_call_and_derives_four_tasks(self):
        payload = policy_output_example(module.POLICY_APPRAISAL_DIMENSIONS)
        payload["emotion"] = {
            "positive_intensity": 4,
            "negative_intensity": 2,
            "positive_labels": ["Hopeful", "excited"],
            "negative_labels": ["sad"],
        }
        client = FakeClient(payload)
        trace, derived, issues = module.process_chain_emotion(
            client, self.prompt_cfg, "I experienced an event."
        )
        self.assertEqual(client.calls, 1)
        self.assertEqual(issues, [])
        self.assertEqual(trace["emotion"]["positive_labels"], ["hopeful", "excited"])
        self.assertEqual(set(derived), set(module.CHAIN_EMOTION_OUTPUT_TASKS))
        self.assertEqual(derived["positive-level"]["score"], 4)
        self.assertEqual(
            derived["positive-labels"]["labels"],
            [
                "Hopeful, optimistic, encouraged",
                "Excited, enthusiastic, elated",
            ],
        )

    def test_chain_emotion_can_supply_carebench_core_appraisals(self):
        payload = policy_output_example(module.POLICY_APPRAISAL_DIMENSIONS)
        client = FakeClient(payload)
        trace, derived, issues = module.process_chain_emotion(
            client,
            self.prompt_cfg,
            "I experienced an event.",
            include_core_appraisals=True,
        )
        self.assertEqual(client.calls, 1)
        self.assertEqual(issues, [])
        self.assertIsNotNone(trace)
        self.assertIsNotNone(derived)
        self.assertEqual(
            set(derived["core-appraisals"]),
            set(module.CORE_APPRAISAL_DIMENSIONS),
        )
        for policy_dimension, carebench_dimension in (
            module.POLICY_TO_CAREBENCH_CORE_DIMENSION.items()
        ):
            self.assertEqual(
                derived["core-appraisals"][carebench_dimension]["answer"],
                trace["appraisal_reasoning"][policy_dimension],
            )

        with tempfile.TemporaryDirectory() as directory:
            error = runner._write_task_outputs(
                Path(directory),
                "sample-1",
                module.CHAIN_EMOTION_TASK,
                trace,
                derived,
                None,
                "chain-emotion",
            )
            self.assertIsNone(error)
            saved_path = Path(directory) / "core-appraisals" / "sample-1.json"
            saved = json.loads(saved_path.read_text(encoding="utf-8"))
            self.assertEqual(
                set(saved), set(module.CORE_APPRAISAL_DIMENSIONS)
            )

            saved_path.write_text('{"stale": true}', encoding="utf-8")
            runner._refresh_core_appraisals_from_trajectory(
                Path(directory)
                / module.CHAIN_EMOTION_TASK
                / "sample-1.json",
                Path(directory),
                "sample-1",
                module.CHAIN_EMOTION_TASK,
                self.prompt_cfg,
            )
            refreshed = json.loads(saved_path.read_text(encoding="utf-8"))
            self.assertEqual(
                set(refreshed), set(module.CORE_APPRAISAL_DIMENSIONS)
            )

    def test_toml_chain_emotion_prompt_matches_grpo_training_prompt(self):
        prompt_text = (
            SCRIPTS_DIR / "prompts" / "baseline_prompt.toml"
        ).read_text(encoding="utf-8")
        match = re.search(
            r'\[templates\.chain-emotion\]\s*'
            r'system\s*=\s*"""(.*?)"""\s*'
            r'user\s*=\s*"""(.*?)"""',
            prompt_text,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(match)
        prompt_cfg = dict(self.prompt_cfg)
        prompt_cfg["templates"] = dict(self.prompt_cfg["templates"])
        prompt_cfg["templates"][module.CHAIN_EMOTION_TASK] = {
            "system": match.group(1),
            "user": match.group(2),
        }
        scenario = "I experienced an event."
        actual_system, actual_user = module.build_chain_emotion_prompts(
            prompt_cfg, scenario
        )
        self.assertEqual(
            actual_system,
            build_policy_system_prompt(module.POLICY_APPRAISAL_DIMENSIONS),
        )
        self.assertEqual(
            actual_user,
            build_policy_user_prompt(scenario, module.POLICY_APPRAISAL_DIMENSIONS),
        )


if __name__ == "__main__":
    unittest.main()
