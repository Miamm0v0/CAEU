"""Provider-level tests that never make a real Judge API request."""

from __future__ import annotations

import asyncio
import io
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from .api_judge import APIJudge, GLM_DEFAULT_BASE_URL, resolve_judge_settings
from .config import build_parser
from .spec import (
    APPRAISAL_CRITERIA,
    APPRAISAL_DIMENSIONS,
    policy_output_example,
)


CAREBENCH_DIMENSIONS = APPRAISAL_DIMENSIONS[:5]


def gold_reference() -> str:
    return json.dumps(
        {
            "gold_emotion": {
                "positive_intensity": 0,
                "negative_intensity": 0,
                "positive_labels": [],
                "negative_labels": [],
            }
        }
    )


def valid_judgment_json(
    appraisal_dimensions: list[str] | None = None,
) -> str:
    selected = appraisal_dimensions or APPRAISAL_DIMENSIONS

    def score_item() -> dict[str, object]:
        return {"score": 4, "rationale": "Supported."}

    judgment = {
        "appraisals": {
            dimension: {
                criterion: score_item()
                for criterion in APPRAISAL_CRITERIA
            }
            for dimension in selected
        },
        "coherence": score_item(),
        "transition": score_item(),
        "overall_feedback": "The chain is valid.",
    }
    return json.dumps(judgment)


class FakeCompletions:
    def __init__(self, owner: "FakeAsyncOpenAI") -> None:
        self.owner = owner

    async def create(self, **kwargs: object) -> SimpleNamespace:
        self.owner.request_kwargs = kwargs
        return SimpleNamespace(
            request_id="glm-request-id",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=valid_judgment_json(
                            self.owner.response_dimensions
                        ),
                        refusal=None,
                    )
                )
            ],
        )


class FakeAsyncOpenAI:
    last_instance: "FakeAsyncOpenAI | None" = None
    response_dimensions: list[str] = list(APPRAISAL_DIMENSIONS)

    def __init__(self, **kwargs: object) -> None:
        self.client_kwargs = kwargs
        self.request_kwargs: dict[str, object] = {}
        self.chat = SimpleNamespace(completions=FakeCompletions(self))
        FakeAsyncOpenAI.last_instance = self


class JudgeProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeAsyncOpenAI.last_instance = None
        FakeAsyncOpenAI.response_dimensions = list(APPRAISAL_DIMENSIONS)

    def test_glm_defaults_and_request_fields(self) -> None:
        args = build_parser().parse_args(
            ["--judge_provider", "glm", "--judge_model", "glm-test"]
        )
        with patch.dict(os.environ, {"ZAI_API_KEY": "test-key"}, clear=True):
            settings = resolve_judge_settings(args)

        self.assertEqual(settings["provider"], "glm")
        self.assertEqual(settings["provider_section"], "glm")
        self.assertEqual(settings["api_key_env"], "ZAI_API_KEY")
        self.assertEqual(settings["base_url"], GLM_DEFAULT_BASE_URL)
        self.assertEqual(settings["response_format"], "json_object")
        self.assertEqual(settings["token_parameter"], "max_tokens")

        judge = APIJudge(args, settings, FakeAsyncOpenAI, cache=None)
        _, request_id = asyncio.run(
            judge.request_judgment("A situation.", {}, {})
        )
        instance = FakeAsyncOpenAI.last_instance
        self.assertIsNotNone(instance)
        assert instance is not None
        self.assertEqual(
            instance.client_kwargs["base_url"],
            GLM_DEFAULT_BASE_URL,
        )
        self.assertEqual(instance.request_kwargs["max_tokens"], 2500)
        self.assertNotIn(
            "max_completion_tokens",
            instance.request_kwargs,
        )
        self.assertEqual(
            instance.request_kwargs["response_format"],
            {"type": "json_object"},
        )
        self.assertEqual(
            instance.request_kwargs["extra_body"],
            {"do_sample": False},
        )
        self.assertNotIn("temperature", instance.request_kwargs)
        self.assertEqual(request_id, "glm-request-id")

    def test_glm_normalizes_openai_only_options(self) -> None:
        args = build_parser().parse_args(
            [
                "--judge_provider",
                "glm",
                "--judge_model",
                "glm-test",
                "--judge_response_format",
                "json_schema",
                "--judge_token_parameter",
                "max_completion_tokens",
            ]
        )
        with patch.dict(
            os.environ,
            {"ZHIPUAI_API_KEY": "legacy-test-key"},
            clear=True,
        ):
            settings = resolve_judge_settings(args)

        self.assertEqual(settings["api_key_env"], "ZHIPUAI_API_KEY")
        self.assertEqual(settings["requested_response_format"], "json_schema")
        self.assertEqual(settings["response_format"], "json_object")
        self.assertEqual(
            settings["requested_token_parameter"],
            "max_completion_tokens",
        )
        self.assertEqual(settings["token_parameter"], "max_tokens")

    def test_glm_thinking_can_be_disabled_explicitly(self) -> None:
        args = build_parser().parse_args(
            [
                "--judge_provider",
                "glm",
                "--judge_model",
                "glm-test",
                "--judge_thinking",
                "disabled",
            ]
        )
        with patch.dict(os.environ, {"ZAI_API_KEY": "test-key"}, clear=True):
            settings = resolve_judge_settings(args)

        judge = APIJudge(args, settings, FakeAsyncOpenAI, cache=None)
        asyncio.run(judge.request_judgment("A situation.", {}, {}))
        instance = FakeAsyncOpenAI.last_instance
        self.assertIsNotNone(instance)
        assert instance is not None
        self.assertEqual(
            instance.request_kwargs["extra_body"],
            {
                "do_sample": False,
                "thinking": {"type": "disabled"},
            },
        )

    def test_glm_thinking_is_sent_when_temperature_is_omitted(self) -> None:
        args = build_parser().parse_args(
            [
                "--judge_provider",
                "glm",
                "--judge_model",
                "glm-test",
                "--judge_omit_temperature",
                "--judge_thinking",
                "disabled",
            ]
        )
        with patch.dict(os.environ, {"ZAI_API_KEY": "test-key"}, clear=True):
            settings = resolve_judge_settings(args)

        judge = APIJudge(args, settings, FakeAsyncOpenAI, cache=None)
        asyncio.run(judge.request_judgment("A situation.", {}, {}))
        instance = FakeAsyncOpenAI.last_instance
        self.assertIsNotNone(instance)
        assert instance is not None
        self.assertEqual(
            instance.request_kwargs["extra_body"],
            {"thinking": {"type": "disabled"}},
        )

    def test_openai_auto_keeps_existing_request_behavior(self) -> None:
        args = build_parser().parse_args(["--judge_model", "openai-test"])
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test-key"},
            clear=True,
        ):
            settings = resolve_judge_settings(args)

        self.assertEqual(settings["provider"], "openai")
        self.assertEqual(settings["base_url"], "")
        self.assertEqual(settings["response_format"], "json_schema")
        self.assertEqual(
            settings["token_parameter"],
            "max_completion_tokens",
        )

        judge = APIJudge(args, settings, FakeAsyncOpenAI, cache=None)
        asyncio.run(judge.request_judgment("A situation.", {}, {}))
        instance = FakeAsyncOpenAI.last_instance
        self.assertIsNotNone(instance)
        assert instance is not None
        self.assertEqual(
            instance.request_kwargs["max_completion_tokens"],
            2500,
        )
        self.assertNotIn("max_tokens", instance.request_kwargs)
        self.assertEqual(
            instance.request_kwargs["response_format"]["type"],
            "json_schema",
        )
        self.assertEqual(instance.request_kwargs["temperature"], 0.0)
        self.assertNotIn("extra_body", instance.request_kwargs)

    def test_raw_judge_output_can_be_logged_for_diagnostics(self) -> None:
        args = build_parser().parse_args(
            [
                "--judge_provider",
                "glm",
                "--judge_model",
                "glm-test",
                "--log_judge_raw_outputs",
            ]
        )
        with patch.dict(os.environ, {"ZAI_API_KEY": "test-key"}, clear=True):
            settings = resolve_judge_settings(args)
        judge = APIJudge(args, settings, FakeAsyncOpenAI, cache=None)

        stream = io.StringIO()
        with patch("sys.stdout", stream):
            asyncio.run(
                judge.request_judgment(
                    "A situation.",
                    {},
                    {},
                    sample_id="sample-raw",
                )
            )
        logged = stream.getvalue().strip()
        self.assertTrue(logged.startswith("[judge-raw] "))
        payload = json.loads(logged[len("[judge-raw] "):])
        self.assertEqual(payload["sample_id"], "sample-raw")
        self.assertRegex(payload["candidate_sha256"], r"^[0-9a-f]{12}$")
        self.assertEqual(payload["attempt"], 1)
        self.assertEqual(payload["request_id"], "glm-request-id")
        self.assertIn('"appraisals"', payload["content"])

    def test_gold_emotion_is_not_sent_to_process_judge(self) -> None:
        args = build_parser().parse_args(
            ["--judge_provider", "glm", "--judge_model", "glm-test"]
        )
        with patch.dict(os.environ, {"ZAI_API_KEY": "test-key"}, clear=True):
            settings = resolve_judge_settings(args)
        judge = APIJudge(args, settings, FakeAsyncOpenAI, cache=None)
        messages = judge.request_messages(
            "A situation.",
            policy_output_example(),
            {
                **json.loads(gold_reference()),
                "legacy_human_appraisal_reasoning": {
                    "relevance": "It mattered to me."
                },
            },
        )
        user_message = messages[1]["content"]
        self.assertNotIn("gold_emotion", user_message)
        self.assertNotIn("positive_intensity", user_message.split('"candidate"')[0])
        self.assertIn("optional_human_appraisal_reference", user_message)

    def test_scores_and_logging_flow_through_api_judge(self) -> None:
        args = build_parser().parse_args(
            ["--judge_provider", "glm", "--judge_model", "glm-test"]
        )
        with patch.dict(os.environ, {"ZAI_API_KEY": "test-key"}, clear=True):
            settings = resolve_judge_settings(args)
        judge = APIJudge(args, settings, FakeAsyncOpenAI, cache=None)
        extras: dict[str, list[object]] = {}
        metrics: dict[str, float] = {}

        rewards = asyncio.run(
            judge.score_batch(
                completions=[json.dumps(policy_output_example())] * 4,
                situations=["A situation."] * 4,
                sample_ids=[f"sample-{index}" for index in range(4)],
                references=[gold_reference()] * 4,
                log_extra=lambda name, values: extras.update(
                    {name: values}
                ),
                log_metric=lambda name, value: metrics.update(
                    {name: value}
                ),
            )
        )

        self.assertEqual(rewards, [0.0] * 4)
        self.assertIn("reward_components", extras)
        self.assertIn("normalized_reward_components", extras)
        self.assertIn("judge_appraisal_criterion_scores", extras)
        self.assertIn("judge_transition_gate_components", extras)
        self.assertEqual(metrics["appraisal_reward"], 1.0)
        self.assertEqual(metrics["coherence_reward"], 1.0)
        self.assertEqual(metrics["transition_reward"], 1.0)
        self.assertEqual(metrics["outcome_reward"], 1.0)
        self.assertNotIn("judge_emotion_reference_alignment", metrics)

    def test_carebench_five_dimensions_flow_through_api_judge(self) -> None:
        args = build_parser().parse_args(
            [
                "--judge_provider",
                "glm",
                "--judge_model",
                "glm-test",
                "--appraisal_dimensions",
                ",".join(CAREBENCH_DIMENSIONS),
            ]
        )
        FakeAsyncOpenAI.response_dimensions = list(CAREBENCH_DIMENSIONS)
        with patch.dict(os.environ, {"ZAI_API_KEY": "test-key"}, clear=True):
            settings = resolve_judge_settings(args)
        judge = APIJudge(args, settings, FakeAsyncOpenAI, cache=None)

        result = asyncio.run(
            judge.score_one(
                "sample-1",
                "A situation.",
                json.dumps(policy_output_example(CAREBENCH_DIMENSIONS)),
                gold_reference(),
            )
        )

        instance = FakeAsyncOpenAI.last_instance
        self.assertIsNotNone(instance)
        assert instance is not None
        self.assertEqual(result["reward_components"]["appraisal_reward"], 1.0)
        self.assertEqual(result["reward_components"]["outcome_reward"], 1.0)
        self.assertNotIn(
            "norm_value_compatibility",
            instance.request_kwargs["messages"][0]["content"],
        )
        self.assertEqual(judge.appraisal_dimensions, CAREBENCH_DIMENSIONS)

    def test_invalid_policy_output_is_a_hard_gate(self) -> None:
        args = build_parser().parse_args(
            ["--judge_provider", "glm", "--judge_model", "glm-test"]
        )
        with patch.dict(os.environ, {"ZAI_API_KEY": "test-key"}, clear=True):
            settings = resolve_judge_settings(args)
        judge = APIJudge(args, settings, FakeAsyncOpenAI, cache=None)

        result = asyncio.run(
            judge.score_one(
                "sample-1",
                "A situation.",
                "not valid JSON",
                gold_reference(),
            )
        )

        instance = FakeAsyncOpenAI.last_instance
        self.assertIsNotNone(instance)
        assert instance is not None
        self.assertTrue(
            all(
                value == -1.0
                for value in result["reward_components"].values()
            )
        )
        self.assertEqual(instance.request_kwargs, {})


if __name__ == "__main__":
    unittest.main()
