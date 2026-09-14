#!/usr/bin/env python3
"""Reference-free LLM-as-Judge evaluation for CAREBench Policy outputs.

The Judge sees exactly two semantic inputs: the first-person ``situation`` and
the candidate Policy's ``appraisal_reasoning`` plus ``emotion``.  Human
CAREBench appraisal annotations, ratings, and emotion labels/intensities are
never accessed for scoring or sent to the Judge.

The five compact criteria retain the main concepts of the training rubric:

1. appraisal validity;
2. situation grounding plus first-person fidelity;
3. cross-appraisal coherence;
4. appraisal-to-emotion transition quality; and
5. reference-free emotion plausibility and calibration.

Scores use integer anchors from 0 to 4.  Missing/invalid predictions and
Judge/API failures are excluded rather than converted to zero.

Example:

    python FirstPersonMethod/scripts/evaluate_carebench_llm_judge.py \
      --source_folder data/first_person_split/raw/test \
      --pred_folder output/test_results/policy/chain-all \
      --prediction_task chain-emotion \
      --output_dir output/test_results/policy/carebench_llm_judge \
      --judge_provider glm --judge_model glm-5.1 \
      --judge_thinking disabled --judge_max_concurrency 8
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from sft_common import (
    NEGATIVE_LABEL_DESCRIPTIONS,
    POSITIVE_LABEL_DESCRIPTIONS,
    canonicalize_emotion_labels,
)
from train_grpo.api_judge import JudgeCache, resolve_judge_settings
from train_grpo.parsing import object_without_duplicate_keys, parse_policy_output
from train_grpo.spec import (
    APPRAISAL_DEFINITIONS,
    CAREBENCH_REASONING_KEYS,
    resolve_appraisal_dimensions,
)


RUBRIC_VERSION = "carebench-reference-free-chain-judge-v1"
POLICY_DIMENSIONS = tuple(resolve_appraisal_dimensions(None)[:5])
PREDICTION_TASKS = ("auto", "chain-emotion", "full-chain", "derived")
SCORE_MINIMUM = 0
SCORE_MAXIMUM = 4

CRITERIA = (
    "appraisal_validity",
    "grounding_and_first_person",
    "cross_appraisal_coherence",
    "appraisal_emotion_transition",
    "emotion_plausibility_calibration",
)

CRITERION_TITLES = {
    "appraisal_validity": "Appraisal validity",
    "grounding_and_first_person": "Grounding and first-person fidelity",
    "cross_appraisal_coherence": "Cross-appraisal coherence",
    "appraisal_emotion_transition": "Appraisal-to-emotion transition",
    "emotion_plausibility_calibration": "Emotion plausibility and calibration",
}

CRITERION_GUIDANCE = {
    "appraisal_validity": (
        "Assess whether the five appraisal dimensions are used correctly and "
        "form a psychologically plausible interpretation of the situation. "
        "Penalize category errors, materially wrong inferences, failure to "
        "capture salient appraisals, and filling a dimension with invented "
        "content merely because the schema requires it."
    ),
    "grounding_and_first_person": (
        "Assess whether appraisal claims are supported by the situation and "
        "faithfully represent the event author's first-person perspective. "
        "Penalize invented facts, goals, beliefs, intentions, relationships, "
        "consequences, causes, coping resources, and outside-observer framing."
    ),
    "cross_appraisal_coherence": (
        "Assess whether the five appraisals are mutually compatible, "
        "sufficiently differentiated, and together describe one coherent "
        "interpretation of the event. Penalize contradictions and reasoning "
        "that merely repeats the event without connecting the dimensions."
    ),
    "appraisal_emotion_transition": (
        "Assess whether the predicted emotion labels, valence, and intensities "
        "follow from the candidate appraisals. Self-consistency alone cannot "
        "rescue appraisals that are unsupported by the situation; take the "
        "quality of the preceding appraisal process into account."
    ),
    "emotion_plausibility_calibration": (
        "Without using any hidden gold answer, assess whether the predicted "
        "emotions are plausible for the first-person situation and whether "
        "labels and 0-to-6 intensities are appropriately calibrated. Penalize "
        "unsupported emotions, implausible intensity, internal valence/label "
        "mismatches, and omission of an unmistakably salient emotional "
        "reaction. Do not penalize a reasonable alternative merely because "
        "another emotion could also be plausible."
    ),
}

TRAINING_RUBRIC_COMPRESSION = {
    "appraisal_validity": [
        "per-dimension dimension_specific_validity",
    ],
    "grounding_and_first_person": [
        "per-dimension situation_grounding",
        "per-dimension experiencer_fidelity",
    ],
    "cross_appraisal_coherence": [
        "cross_dimension_coherence",
    ],
    "appraisal_emotion_transition": [
        "appraisal_emotion_linkage",
    ],
    "emotion_plausibility_calibration": [
        "reference-free test-time analogue of outcome quality; not the "
        "gold-based training outcome reward",
    ],
}

GENERAL_SCORE_ANCHORS = {
    0: "Unusable: absent, fundamentally wrong, contradictory, or unsupported.",
    1: "Poor: major errors dominate; only a small amount is defensible.",
    2: "Mixed: partially reasonable but has important omissions or errors.",
    3: "Good: mostly correct and well supported, with only minor weaknesses.",
    4: "Excellent: correct, specific, coherent, grounded, and well calibrated.",
}


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reference-free LLM-as-Judge evaluation of CAREBench appraisal "
            "reasoning and emotion predictions"
        )
    )
    data = parser.add_argument_group("CAREBench situations and predictions")
    data.add_argument(
        "--source_folder",
        type=Path,
        required=True,
        help=(
            "Folder of CAREBench sample JSON files. Only "
            "story_collection.final_scenario is read; annotations are ignored"
        ),
    )
    data.add_argument(
        "--pred_folder",
        type=Path,
        required=True,
        help="baseline_transformers.py target folder containing task subfolders",
    )
    data.add_argument(
        "--prediction_task",
        choices=list(PREDICTION_TASKS),
        default="chain-emotion",
        help=(
            "Candidate source. auto prefers chain-emotion, then full-chain, "
            "then reconstructed derived task files"
        ),
    )
    data.add_argument("--output_dir", type=Path, required=True)
    data.add_argument("--max_samples", type=int, default=None)
    data.add_argument("--seed", type=int, default=42)
    data.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore an existing judge_results.jsonl and evaluate afresh",
    )

    judge = parser.add_argument_group("API LLM Judge")
    judge.add_argument(
        "--judge_provider",
        choices=["openai", "glm"],
        default="openai",
    )
    judge.add_argument("--judge_model", type=str, default="")
    judge.add_argument("--judge_config", type=str, default="")
    judge.add_argument("--judge_provider_section", type=str, default="")
    judge.add_argument("--judge_api_key_env", type=str, default="")
    judge.add_argument("--judge_base_url", type=str, default="")
    judge.add_argument(
        "--judge_response_format",
        choices=["auto", "json_schema", "json_object", "text"],
        default="auto",
    )
    judge.add_argument(
        "--judge_token_parameter",
        choices=["auto", "max_tokens", "max_completion_tokens"],
        default="auto",
    )
    judge.add_argument("--judge_max_tokens", type=int, default=2500)
    judge.add_argument("--judge_temperature", type=float, default=0.0)
    judge.add_argument("--judge_omit_temperature", action="store_true")
    judge.add_argument(
        "--judge_thinking",
        choices=["auto", "enabled", "disabled"],
        default="disabled",
    )
    judge.add_argument("--judge_timeout_seconds", type=float, default=180.0)
    judge.add_argument("--judge_max_retries", type=int, default=3)
    judge.add_argument("--judge_parse_retries", type=int, default=1)
    judge.add_argument("--judge_max_concurrency", type=int, default=8)
    judge.add_argument(
        "--judge_failure_policy",
        choices=["skip", "error"],
        default="skip",
    )
    judge.add_argument("--judge_cache_path", type=Path, default=None)
    judge.add_argument("--no_judge_cache", action="store_true")
    judge.add_argument("--log_judge_raw_outputs", action="store_true")
    judge.add_argument(
        "--save_raw_outputs",
        action="store_true",
        help="Retain all successful raw Judge responses in judge_results.jsonl",
    )
    judge.add_argument(
        "--preview_only",
        action="store_true",
        help="Validate inputs and print one Judge envelope without calling the API",
    )
    return parser


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: top-level JSON must be an object")
    return payload


def load_situations(source_folder: Path) -> list[dict[str, str]]:
    """Load only CAREBench's model input, never its human target fields."""
    if not source_folder.is_dir():
        raise FileNotFoundError(f"Source folder not found: {source_folder}")
    rows: list[dict[str, str]] = []
    for path in sorted(source_folder.glob("*.json")):
        payload = read_json_object(path)
        story_collection = payload.get("story_collection")
        situation = (
            story_collection.get("final_scenario")
            if isinstance(story_collection, dict)
            else None
        )
        if not isinstance(situation, str) or not situation.strip():
            raise ValueError(
                f"{path}: story_collection.final_scenario must be non-empty"
            )
        rows.append(
            {
                "sample_id": path.stem,
                "situation": situation.strip(),
            }
        )
    if not rows:
        raise ValueError(f"No JSON samples found in {source_folder}")
    return rows


def stable_limit(
    rows: list[dict[str, str]],
    maximum: int | None,
    seed: int,
) -> list[dict[str, str]]:
    if maximum is None:
        return rows
    if maximum <= 0:
        raise ValueError("--max_samples must be positive")
    if maximum >= len(rows):
        return rows

    def stable_key(row: Mapping[str, str]) -> str:
        identity = f"carebench-judge:{seed}:{row['sample_id']}"
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    return sorted(rows, key=stable_key)[:maximum]


def normalize_reasoning(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise ValueError("appraisal_reasoning must be an object")
    normalized: dict[str, str] = {}
    for policy_dimension in POLICY_DIMENSIONS:
        carebench_key = CAREBENCH_REASONING_KEYS[policy_dimension]
        value = raw.get(policy_dimension, raw.get(carebench_key))
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"appraisal_reasoning.{policy_dimension} must be non-empty"
            )
        normalized[policy_dimension] = value.strip()
    allowed = set(POLICY_DIMENSIONS) | {
        CAREBENCH_REASONING_KEYS[dimension] for dimension in POLICY_DIMENSIONS
    }
    unsupported = sorted(set(raw) - allowed)
    if unsupported:
        raise ValueError(
            f"appraisal_reasoning contains unsupported dimensions: {unsupported}"
        )
    return normalized


def normalize_policy_candidate(payload: Mapping[str, Any]) -> dict[str, Any]:
    reasoning = normalize_reasoning(payload.get("appraisal_reasoning"))
    minimal = {
        "appraisal_reasoning": reasoning,
        "emotion": payload.get("emotion"),
    }
    return parse_policy_output(
        json.dumps(minimal, ensure_ascii=False),
        POLICY_DIMENSIONS,
    )


def normalize_derived_label_list(raw: Any, valence: str) -> list[str]:
    if not isinstance(raw, list):
        raise ValueError(f"{valence}-labels.labels must be a list")
    descriptions = (
        POSITIVE_LABEL_DESCRIPTIONS
        if valence == "positive"
        else NEGATIVE_LABEL_DESCRIPTIONS
    )
    exact_descriptions = {
        description.casefold(): canonical
        for canonical, description in descriptions.items()
    }
    aliases: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{valence}-labels.labels contains invalid value")
        cleaned = item.strip()
        aliases.append(exact_descriptions.get(cleaned.casefold(), cleaned))
    return canonicalize_emotion_labels(
        aliases,
        valence,
        field=f"emotion.{valence}_labels",
        reject_duplicates=False,
    )


def load_derived_candidate(
    pred_folder: Path,
    sample_id: str,
) -> tuple[dict[str, Any], list[str]]:
    paths = {
        task: pred_folder / task / f"{sample_id}.json"
        for task in (
            "core-appraisals",
            "positive-level",
            "negative-level",
            "positive-labels",
            "negative-labels",
        )
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "derived candidate is missing task files: " + ", ".join(missing)
        )

    core = read_json_object(paths["core-appraisals"])
    raw_reasoning: dict[str, Any] = {}
    for carebench_dimension, raw_value in core.items():
        if isinstance(raw_value, dict):
            raw_reasoning[carebench_dimension] = raw_value.get("answer")
        else:
            raw_reasoning[carebench_dimension] = raw_value

    positive_level = read_json_object(paths["positive-level"])
    negative_level = read_json_object(paths["negative-level"])
    positive_labels = read_json_object(paths["positive-labels"])
    negative_labels = read_json_object(paths["negative-labels"])
    candidate = normalize_policy_candidate(
        {
            "appraisal_reasoning": raw_reasoning,
            "emotion": {
                "positive_intensity": positive_level.get("score"),
                "negative_intensity": negative_level.get("score"),
                "positive_labels": normalize_derived_label_list(
                    positive_labels.get("labels"), "positive"
                ),
                "negative_labels": normalize_derived_label_list(
                    negative_labels.get("labels"), "negative"
                ),
            },
        }
    )
    return candidate, [str(path) for path in paths.values()]


def candidate_task_order(prediction_task: str) -> tuple[str, ...]:
    if prediction_task == "auto":
        return ("chain-emotion", "full-chain", "derived")
    return (prediction_task,)


def load_candidate(
    pred_folder: Path,
    sample_id: str,
    prediction_task: str,
) -> tuple[dict[str, Any], str, list[str]]:
    errors: list[str] = []
    found_any = False
    for task in candidate_task_order(prediction_task):
        try:
            if task == "derived":
                candidate, paths = load_derived_candidate(pred_folder, sample_id)
                return candidate, task, paths
            path = pred_folder / task / f"{sample_id}.json"
            if not path.is_file():
                errors.append(f"{task}: missing {path}")
                continue
            found_any = True
            candidate = normalize_policy_candidate(read_json_object(path))
            return candidate, task, [str(path)]
        except FileNotFoundError as exc:
            errors.append(f"{task}: {exc}")
            continue
        except (TypeError, ValueError) as exc:
            errors.append(f"{task}: {exc}")
            found_any = True
    if not found_any:
        raise FileNotFoundError("; ".join(errors))
    raise ValueError("; ".join(errors))


def score_item_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["score", "rationale"],
        "properties": {
            "score": {
                "type": "integer",
                "minimum": SCORE_MINIMUM,
                "maximum": SCORE_MAXIMUM,
            },
            "rationale": {"type": "string", "minLength": 1},
        },
    }


def judge_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [*CRITERIA, "overall_feedback"],
        "properties": {
            **{criterion: score_item_schema() for criterion in CRITERIA},
            "overall_feedback": {"type": "string", "minLength": 1},
        },
    }


def judge_output_example() -> dict[str, Any]:
    return {
        **{
            criterion: {
                "score": 0,
                "rationale": f"Evidence-based rationale for {criterion}.",
            }
            for criterion in CRITERIA
        },
        "overall_feedback": "Concise overall assessment.",
    }


def judge_system_prompt() -> str:
    dimension_lines = "\n".join(
        f"- {dimension}: {APPRAISAL_DEFINITIONS[dimension]}"
        for dimension in POLICY_DIMENSIONS
    )
    criteria_lines = "\n".join(
        f"- {criterion} ({CRITERION_TITLES[criterion]}): "
        f"{CRITERION_GUIDANCE[criterion]}"
        for criterion in CRITERIA
    )
    anchor_lines = "\n".join(
        f"- {score}: {description}"
        for score, description in GENERAL_SCORE_ANCHORS.items()
    )
    output = json.dumps(judge_output_example(), ensure_ascii=False, indent=2)
    return f"""You are a strict, reference-free evaluator of first-person
cognitive appraisal and emotion reasoning. You receive only a situation and a
candidate response. You do not have access to human CAREBench appraisal or
emotion annotations. Never invent or assume a hidden gold answer. Judge every
claim only against the situation and the internal quality of the candidate.

Multiple emotional reactions can be reasonable for the same event. Do not
penalize a plausible candidate merely because another reaction is also
possible. Prefer concise, supported inference over elaborate speculation. Do
not reward surface fluency when the underlying appraisal is wrong or invented.

Appraisal dimension meanings:
{dimension_lines}

Score all five compact criteria independently:
{criteria_lines}

Use the same integer anchors for every criterion:
{anchor_lines}

Return exactly one JSON object with every required key. Each rationale must
cite concrete evidence or a concrete defect from the supplied situation and
candidate. Do not return markdown, commentary, a total score, or extra keys.

Required output structure:
{output}""".strip()


def strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def require_exact_keys(
    value: Any,
    expected: set[str],
    location: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be an object")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{location} keys mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def parse_judge_output(text: str) -> dict[str, Any]:
    cleaned = strip_code_fence(text)
    if not cleaned:
        raise ValueError("empty Judge response")
    try:
        payload = json.loads(
            cleaned,
            object_pairs_hook=object_without_duplicate_keys,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid Judge JSON: {exc.msg}") from exc
    root = require_exact_keys(
        payload,
        {*CRITERIA, "overall_feedback"},
        "judge.root",
    )
    for criterion in CRITERIA:
        item = require_exact_keys(
            root[criterion], {"score", "rationale"}, f"judge.{criterion}"
        )
        score = item["score"]
        if (
            isinstance(score, bool)
            or not isinstance(score, int)
            or not SCORE_MINIMUM <= score <= SCORE_MAXIMUM
        ):
            raise ValueError(
                f"judge.{criterion}.score must be an integer in [0, 4]"
            )
        rationale = item["rationale"]
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError(
                f"judge.{criterion}.rationale must be a non-empty string"
            )
        item["rationale"] = rationale.strip()
    feedback = root["overall_feedback"]
    if not isinstance(feedback, str) or not feedback.strip():
        raise ValueError("judge.overall_feedback must be a non-empty string")
    root["overall_feedback"] = feedback.strip()
    return root


def judgment_scores(judgment: Mapping[str, Any]) -> dict[str, Any]:
    raw = {
        criterion: int(judgment[criterion]["score"])
        for criterion in CRITERIA
    }
    normalized = {
        criterion: score / SCORE_MAXIMUM
        for criterion, score in raw.items()
    }
    overall_raw = sum(raw.values()) / len(raw)
    return {
        "scores_raw_0_4": raw,
        "scores_normalized_0_1": normalized,
        "overall_raw_0_4": overall_raw,
        "overall_normalized_0_1": overall_raw / SCORE_MAXIMUM,
    }


def response_format(settings: Mapping[str, str]) -> dict[str, Any] | None:
    mode = settings["response_format"]
    if mode == "text":
        return {"type": "text"} if settings["provider"] == "glm" else None
    if mode == "json_object":
        return {"type": "json_object"}
    if mode == "json_schema":
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "carebench_reference_free_chain_judgment",
                "strict": True,
                "schema": judge_json_schema(),
            },
        }
    raise ValueError(f"Unsupported effective response format: {mode}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class JudgeRequestError(RuntimeError):
    """A failed Judge request together with every parse/request attempt."""

    def __init__(self, message: str, attempts: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.attempts = attempts


class ReferenceFreeJudge:
    def __init__(
        self,
        args: argparse.Namespace,
        settings: dict[str, str],
        async_openai_class: Any,
        cache: JudgeCache | None,
    ) -> None:
        self.args = args
        self.settings = settings
        self.cache = cache
        client_kwargs: dict[str, Any] = {
            "api_key": settings["api_key"],
            "timeout": args.judge_timeout_seconds,
            "max_retries": args.judge_max_retries,
        }
        if settings["base_url"]:
            client_kwargs["base_url"] = settings["base_url"]
        self.client = async_openai_class(**client_kwargs)
        self._semaphore: asyncio.Semaphore | None = None
        self._semaphore_loop: asyncio.AbstractEventLoop | None = None
        self.system_prompt = judge_system_prompt()

    def request_semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self._semaphore is None or self._semaphore_loop is not loop:
            self._semaphore = asyncio.Semaphore(self.args.judge_max_concurrency)
            self._semaphore_loop = loop
        return self._semaphore

    def input_identity(
        self,
        situation: str,
        candidate: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "rubric_version": RUBRIC_VERSION,
            "judge_provider": self.settings["provider"],
            "judge_model": self.settings["model"],
            "judge_base_url": self.settings["base_url"],
            "response_format": self.settings["response_format"],
            "token_parameter": self.settings["token_parameter"],
            "judge_thinking": self.args.judge_thinking,
            "situation": situation,
            "candidate": candidate,
        }

    def input_sha256(
        self,
        situation: str,
        candidate: Mapping[str, Any],
    ) -> str:
        return sha256_json(self.input_identity(situation, candidate))

    def request_messages(
        self,
        situation: str,
        candidate: Mapping[str, Any],
        validation_error: str = "",
    ) -> list[dict[str, str]]:
        # Deliberately no reference/gold field in this envelope.
        envelope = {
            "situation": situation,
            "candidate": candidate,
        }
        content = (
            "Score this situation and candidate with all five required "
            "criteria:\n"
            + json.dumps(envelope, ensure_ascii=False, indent=2)
        )
        if validation_error:
            content += (
                "\n\nYour previous response failed validation. Return a fresh, "
                "complete JSON judgment. Validation error: "
                + validation_error[:500]
            )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content},
        ]

    async def request_judgment(
        self,
        sample_id: str,
        situation: str,
        candidate: Mapping[str, Any],
    ) -> tuple[dict[str, Any], str | None, list[dict[str, Any]]]:
        format_payload = response_format(self.settings)
        candidate_digest = sha256_json(candidate)[:12]
        attempts: list[dict[str, Any]] = []
        last_error = ""
        request_id: str | None = None
        for attempt_index in range(self.args.judge_parse_retries + 1):
            request_kwargs: dict[str, Any] = {
                "model": self.settings["model"],
                "messages": self.request_messages(
                    situation,
                    candidate,
                    validation_error=last_error,
                ),
            }
            request_kwargs[self.settings["token_parameter"]] = (
                self.args.judge_max_tokens
            )
            extra_body: dict[str, Any] = {}
            if not self.args.judge_omit_temperature:
                if (
                    self.settings["provider"] == "glm"
                    and self.args.judge_temperature == 0
                ):
                    extra_body["do_sample"] = False
                else:
                    request_kwargs["temperature"] = self.args.judge_temperature
            if (
                self.settings["provider"] == "glm"
                and self.args.judge_thinking != "auto"
            ):
                extra_body["thinking"] = {"type": self.args.judge_thinking}
            if extra_body:
                request_kwargs["extra_body"] = extra_body
            if format_payload is not None:
                request_kwargs["response_format"] = format_payload

            try:
                async with self.request_semaphore():
                    response = await self.client.chat.completions.create(
                        **request_kwargs
                    )
            except Exception as exc:
                last_error = f"API request failed: {exc}"
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "valid": False,
                        "error": last_error[:1000],
                    }
                )
                continue

            request_id = (
                getattr(response, "_request_id", None)
                or getattr(response, "request_id", None)
                or getattr(response, "id", None)
            )
            if not response.choices:
                last_error = "Judge response contains no choices"
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "valid": False,
                        "request_id": request_id,
                        "error": last_error,
                    }
                )
                continue
            message = response.choices[0].message
            refusal = getattr(message, "refusal", None)
            content = getattr(message, "content", None)
            if refusal:
                last_error = f"Judge refused: {refusal}"
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "valid": False,
                        "request_id": request_id,
                        "error": last_error[:1000],
                    }
                )
                continue
            if not isinstance(content, str) or not content.strip():
                last_error = "Judge returned empty content"
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "valid": False,
                        "request_id": request_id,
                        "error": last_error,
                    }
                )
                continue
            if self.args.log_judge_raw_outputs:
                print(
                    "[judge-raw] "
                    + json.dumps(
                        {
                            "sample_id": sample_id,
                            "candidate_sha256": candidate_digest,
                            "attempt": attempt_index + 1,
                            "request_id": request_id,
                            "provider": self.settings["provider"],
                            "model": self.settings["model"],
                            "content": content,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
            try:
                judgment = parse_judge_output(content)
            except (TypeError, ValueError) as exc:
                last_error = str(exc)
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "valid": False,
                        "request_id": request_id,
                        "error": last_error[:1000],
                        "raw_output": content,
                    }
                )
                continue
            attempts.append(
                {
                    "attempt": attempt_index + 1,
                    "valid": True,
                    "request_id": request_id,
                    "raw_output": content,
                }
            )
            return judgment, request_id, attempts
        raise JudgeRequestError(
            "Judge did not return a valid complete five-criterion rubric after "
            f"{self.args.judge_parse_retries + 1} request(s): {last_error}",
            attempts,
        )

    async def score(
        self,
        sample_id: str,
        situation: str,
        candidate: Mapping[str, Any],
    ) -> dict[str, Any]:
        input_sha = self.input_sha256(situation, candidate)
        if self.cache is not None:
            cached = self.cache.get(input_sha)
            if cached is not None:
                return {
                    **cached,
                    "cache_hit": True,
                    "input_sha256": input_sha,
                }
        try:
            judgment, request_id, attempts = await self.request_judgment(
                sample_id,
                situation,
                candidate,
            )
            result = {
                "status": "success",
                "cache_hit": False,
                "request_id": request_id,
                "input_sha256": input_sha,
                "judgment": judgment,
                **judgment_scores(judgment),
            }
            if self.args.save_raw_outputs:
                result["judge_attempts"] = attempts
            if self.cache is not None:
                self.cache.put(input_sha, self.settings["model"], result)
            return result
        except Exception as exc:
            if self.args.judge_failure_policy == "error":
                raise
            failure = {
                "status": "judge_failed",
                "cache_hit": False,
                "request_id": None,
                "input_sha256": input_sha,
                "error": str(exc)[:2000],
            }
            if isinstance(exc, JudgeRequestError):
                # Failed raw responses are always retained for diagnosis;
                # --save_raw_outputs controls only successful responses.
                failure["judge_attempts"] = exc.attempts
            return failure


def load_jsonl_index(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSONL: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"{path}:{line_number}: missing sample_id")
            rows[sample_id] = row
    return rows


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")
        handle.flush()


def write_jsonl_atomic(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                )
                handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def population_std(values: Sequence[float]) -> float | None:
    if not values:
        return None
    average = sum(values) / len(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / len(values))


def aggregate_results(
    selected_samples: Sequence[Mapping[str, str]],
    rows: Mapping[str, Mapping[str, Any]],
    prediction_task: str,
    settings: Mapping[str, str],
) -> dict[str, Any]:
    selected_ids = [sample["sample_id"] for sample in selected_samples]
    selected_rows = [rows[sample_id] for sample_id in selected_ids]
    statuses = Counter(str(row.get("status", "unknown")) for row in selected_rows)
    successes = [row for row in selected_rows if row.get("status") == "success"]
    criterion_raw = {
        criterion: [
            float(row["scores_raw_0_4"][criterion]) for row in successes
        ]
        for criterion in CRITERIA
    }
    overall_raw = [float(row["overall_raw_0_4"]) for row in successes]
    distributions = {
        criterion: {
            str(score): sum(
                int(value == score) for value in criterion_raw[criterion]
            )
            for score in range(SCORE_MINIMUM, SCORE_MAXIMUM + 1)
        }
        for criterion in CRITERIA
    }
    source_counts = Counter(
        str(row.get("candidate_source", "unknown")) for row in selected_rows
    )
    valid_predictions = sum(
        statuses.get(status, 0) for status in ("success", "judge_failed")
    )
    judge_success = len(successes)
    selected_count = len(selected_rows)
    return {
        "rubric_version": RUBRIC_VERSION,
        "evaluation_type": "reference_free_llm_as_judge",
        "human_gold_used": False,
        "human_gold_sent_to_judge": False,
        "prediction_task": prediction_task,
        "judge": {
            "provider": settings["provider"],
            "model": settings["model"],
            "base_url": settings["base_url"] or None,
            "response_format": settings["response_format"],
        },
        "coverage": {
            "selected_samples": selected_count,
            "valid_predictions": valid_predictions,
            "judge_success": judge_success,
            "judge_failures": statuses.get("judge_failed", 0),
            "missing_predictions": statuses.get("missing_prediction", 0),
            "invalid_predictions": statuses.get("invalid_prediction", 0),
            "judge_coverage_total": (
                judge_success / selected_count if selected_count else 0.0
            ),
            "judge_success_rate_among_valid_predictions": (
                judge_success / valid_predictions if valid_predictions else 0.0
            ),
            "status_counts": dict(sorted(statuses.items())),
            "candidate_source_counts": dict(sorted(source_counts.items())),
        },
        "scores": {
            "scale": {"minimum": 0, "maximum": 4},
            "criterion_mean_raw_0_4": {
                criterion: mean(values)
                for criterion, values in criterion_raw.items()
            },
            "criterion_mean_normalized_0_1": {
                criterion: (
                    None if not values else mean(values) / SCORE_MAXIMUM
                )
                for criterion, values in criterion_raw.items()
            },
            "criterion_population_std_raw_0_4": {
                criterion: population_std(values)
                for criterion, values in criterion_raw.items()
            },
            "criterion_score_distribution": distributions,
            "overall_mean_raw_0_4": mean(overall_raw),
            "overall_mean_normalized_0_1": (
                None if not overall_raw else mean(overall_raw) / SCORE_MAXIMUM
            ),
            "overall_population_std_raw_0_4": population_std(overall_raw),
        },
        "rubric_compression": TRAINING_RUBRIC_COMPRESSION,
    }


def write_summary_csv(path: Path, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    coverage = summary["coverage"]
    scores = summary["scores"]
    row: dict[str, Any] = {
        "rubric_version": summary["rubric_version"],
        "evaluation_type": summary["evaluation_type"],
        "human_gold_used": summary["human_gold_used"],
        "human_gold_sent_to_judge": summary[
            "human_gold_sent_to_judge"
        ],
        "prediction_task": summary["prediction_task"],
        "judge_provider": summary["judge"]["provider"],
        "judge_model": summary["judge"]["model"],
        **coverage,
        "overall_mean_raw_0_4": scores["overall_mean_raw_0_4"],
        "overall_mean_normalized_0_1": scores[
            "overall_mean_normalized_0_1"
        ],
    }
    row.pop("status_counts", None)
    row.pop("candidate_source_counts", None)
    for criterion in CRITERIA:
        row[f"{criterion}_mean_raw_0_4"] = scores[
            "criterion_mean_raw_0_4"
        ][criterion]
        row[f"{criterion}_mean_normalized_0_1"] = scores[
            "criterion_mean_normalized_0_1"
        ][criterion]
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


async def run_pending_judgments(
    judge: ReferenceFreeJudge,
    pending: Sequence[Mapping[str, Any]],
    results_path: Path,
) -> list[dict[str, Any]]:
    async def evaluate_one(item: Mapping[str, Any]) -> dict[str, Any]:
        scored = await judge.score(
            str(item["sample_id"]),
            str(item["situation"]),
            item["candidate"],
        )
        return {
            "sample_id": item["sample_id"],
            "candidate_source": item["candidate_source"],
            "candidate_paths": item["candidate_paths"],
            "candidate_sha256": sha256_json(item["candidate"]),
            "candidate": item["candidate"],
            **scored,
        }

    tasks = [asyncio.create_task(evaluate_one(item)) for item in pending]
    completed: list[dict[str, Any]] = []
    for index, future in enumerate(asyncio.as_completed(tasks), start=1):
        row = await future
        append_jsonl(results_path, row)
        completed.append(row)
        if index == 1 or index % 10 == 0 or index == len(tasks):
            print(
                f"[judge] completed={index}/{len(tasks)} "
                f"sample={row['sample_id']} status={row['status']}",
                flush=True,
            )
    return completed


def sanitized_settings(settings: Mapping[str, str]) -> dict[str, Any]:
    return {
        key: value
        for key, value in settings.items()
        if key != "api_key"
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.judge_max_tokens <= 0:
        raise ValueError("--judge_max_tokens must be positive")
    if args.judge_timeout_seconds <= 0:
        raise ValueError("--judge_timeout_seconds must be positive")
    if args.judge_max_retries < 0 or args.judge_parse_retries < 0:
        raise ValueError("Judge retry counts must be non-negative")
    if args.judge_max_concurrency <= 0:
        raise ValueError("--judge_max_concurrency must be positive")
    if not math.isfinite(args.judge_temperature):
        raise ValueError("--judge_temperature must be finite")
    if not args.pred_folder.is_dir():
        raise FileNotFoundError(f"Prediction folder not found: {args.pred_folder}")


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    samples = stable_limit(
        load_situations(args.source_folder),
        args.max_samples,
        args.seed,
    )
    settings = resolve_judge_settings(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "judge_results.jsonl"
    summary_path = args.output_dir / "summary.json"
    summary_csv_path = args.output_dir / "summary.csv"
    rubric_path = args.output_dir / "rubric.json"
    if args.overwrite and results_path.exists():
        results_path.unlink()
    existing = {} if args.overwrite else load_jsonl_index(results_path)

    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "LLM-as-Judge evaluation requires the openai package"
        ) from exc

    cache: JudgeCache | None = None
    if not args.no_judge_cache:
        cache_path = args.judge_cache_path or args.output_dir / "judge_cache.sqlite3"
        cache = JudgeCache(cache_path)
    judge = ReferenceFreeJudge(args, settings, AsyncOpenAI, cache)

    all_rows: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = sample["sample_id"]
        try:
            candidate, source, candidate_paths = load_candidate(
                args.pred_folder,
                sample_id,
                args.prediction_task,
            )
        except FileNotFoundError as exc:
            row = {
                "sample_id": sample_id,
                "status": "missing_prediction",
                "candidate_source": args.prediction_task,
                "error": str(exc)[:2000],
            }
            all_rows[sample_id] = row
            continue
        except (TypeError, ValueError) as exc:
            row = {
                "sample_id": sample_id,
                "status": "invalid_prediction",
                "candidate_source": args.prediction_task,
                "error": str(exc)[:2000],
            }
            all_rows[sample_id] = row
            continue

        input_sha = judge.input_sha256(sample["situation"], candidate)
        previous = existing.get(sample_id)
        if (
            isinstance(previous, dict)
            and previous.get("status") == "success"
            and previous.get("input_sha256") == input_sha
        ):
            all_rows[sample_id] = previous
            continue
        pending.append(
            {
                **sample,
                "candidate": candidate,
                "candidate_source": source,
                "candidate_paths": candidate_paths,
            }
        )

    print(
        f"[prepare] selected={len(samples)} reused_success="
        f"{sum(row.get('status') == 'success' for row in all_rows.values())} "
        f"pending_judge={len(pending)} pre_judge_skipped="
        f"{sum(row.get('status') != 'success' for row in all_rows.values())}",
        flush=True,
    )
    if args.preview_only:
        if pending:
            preview = pending[0]
            print(
                json.dumps(
                    {
                        "sample_id": preview["sample_id"],
                        "messages": judge.request_messages(
                            preview["situation"], preview["candidate"]
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print("[preview] no valid pending candidate")
        return 0

    if pending:
        completed = asyncio.run(
            run_pending_judgments(judge, pending, results_path)
        )
        for row in completed:
            all_rows[str(row["sample_id"])] = row

    # Missing/invalid rows are persisted only after candidate validation; API
    # rows are appended immediately for crash-safe progress, then compacted.
    ordered_rows = [all_rows[sample["sample_id"]] for sample in samples]
    write_jsonl_atomic(results_path, ordered_rows)
    summary = aggregate_results(
        samples,
        all_rows,
        args.prediction_task,
        settings,
    )
    write_json_atomic(summary_path, summary)
    write_summary_csv(summary_csv_path, summary)
    write_json_atomic(
        rubric_path,
        {
            "rubric_version": RUBRIC_VERSION,
            "human_gold_used": False,
            "human_gold_sent_to_judge": False,
            "criteria": {
                criterion: {
                    "title": CRITERION_TITLES[criterion],
                    "guidance": CRITERION_GUIDANCE[criterion],
                    "training_rubric_sources": TRAINING_RUBRIC_COMPRESSION[
                        criterion
                    ],
                }
                for criterion in CRITERIA
            },
            "score_anchors": GENERAL_SCORE_ANCHORS,
            "judge_settings": sanitized_settings(settings),
        },
    )
    coverage = summary["coverage"]
    print(
        f"[done] judge_success={coverage['judge_success']}/"
        f"{coverage['selected_samples']} "
        f"coverage={coverage['judge_coverage_total']:.4f} "
        f"overall_0_4={summary['scores']['overall_mean_raw_0_4']} "
        f"results={results_path} summary={summary_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
