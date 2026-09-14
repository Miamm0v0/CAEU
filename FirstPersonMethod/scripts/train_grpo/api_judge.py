"""Provider-aware asynchronous Judge, retries, cache, and GRPO reward."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .parsing import (
    aggregate_judgment,
    completion_text,
    parse_judge_output,
    parse_policy_output,
)
from .reward_components import (
    REWARD_COMPONENT_FIELDS,
    build_reward_components,
    group_normalize_components,
    invalid_reward_components,
)
from .spec import (
    RUBRIC_VERSION,
    build_judge_system_prompt,
    judge_output_schema,
)


GLM_DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
DEFAULT_API_KEY_ENVS = {
    "openai": ("OPENAI_API_KEY",),
    "glm": ("ZAI_API_KEY", "ZHIPUAI_API_KEY"),
}


class JudgeCache:
    """SQLite cache shared safely by async tasks and distributed processes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS judgments (
                    cache_key TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    model TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=60.0)
        connection.execute("PRAGMA busy_timeout=60000")
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def get(self, key: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM judgments WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        try:
            value = json.loads(row[0])
        except (TypeError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def put(self, key: str, model: str, payload: dict[str, Any]) -> None:
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO judgments
                    (cache_key, created_at, model, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (key, created_at, model, serialized),
            )


def load_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    with path.open("rb") as handle:
        value = tomllib.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a TOML object")
    return value


def effective_response_format(provider_name: str, requested: str) -> str:
    """Resolve a portable format to one supported by the selected API."""
    if requested == "auto":
        return "json_object" if provider_name == "glm" else "json_schema"
    if provider_name == "glm" and requested == "json_schema":
        # GLM supports text and json_object, but not OpenAI strict schemas.
        return "json_object"
    return requested


def effective_token_parameter(provider_name: str, requested: str) -> str:
    """Resolve the provider-specific output-token request field."""
    if provider_name == "glm":
        return "max_tokens"
    if requested == "auto":
        return "max_completion_tokens"
    return requested


def resolve_judge_settings(args: argparse.Namespace) -> dict[str, str]:
    provider_name = str(
        getattr(args, "judge_provider", "openai")
    ).strip().lower()
    if provider_name not in DEFAULT_API_KEY_ENVS:
        raise ValueError(f"Unsupported judge provider: {provider_name}")

    provider_section = str(
        getattr(args, "judge_provider_section", "")
    ).strip() or provider_name
    provider: dict[str, Any] = {}
    if args.judge_config:
        path = Path(args.judge_config)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Judge config not found: {path}")
        config = load_toml(path)
        providers = config.get("providers")
        if not isinstance(providers, dict):
            raise ValueError(
                f"{path} must contain a [providers] TOML table"
            )
        selected = providers.get(provider_section)
        if not isinstance(selected, dict):
            raise ValueError(
                f"{path} has no [providers.{provider_section}] table"
            )
        provider = selected

    configured_key_env = (
        str(getattr(args, "judge_api_key_env", "")).strip()
        or str(provider.get("api_key_env", "")).strip()
    )
    key_env_candidates = (
        (configured_key_env,)
        if configured_key_env
        else DEFAULT_API_KEY_ENVS[provider_name]
    )
    api_key = ""
    resolved_key_env = configured_key_env
    for environment_name in key_env_candidates:
        environment_key = os.environ.get(environment_name, "").strip()
        if environment_key:
            api_key = environment_key
            resolved_key_env = environment_name
            break
    if not api_key:
        api_key = str(provider.get("api_key", "")).strip()
        if api_key:
            resolved_key_env = ""

    model = args.judge_model.strip() or str(provider.get("model", "")).strip()
    base_url = (
        args.judge_base_url.strip()
        or str(provider.get("base_url", "")).strip()
    )
    if not base_url and provider_name == "glm":
        base_url = GLM_DEFAULT_BASE_URL

    requested_response_format = str(
        getattr(args, "judge_response_format", "auto")
    )
    requested_token_parameter = str(
        getattr(args, "judge_token_parameter", "auto")
    )
    response_format = effective_response_format(
        provider_name,
        requested_response_format,
    )
    token_parameter = effective_token_parameter(
        provider_name,
        requested_token_parameter,
    )

    missing: list[str] = []
    if not api_key:
        env_description = " or ".join(
            f"${name}" for name in key_env_candidates
        )
        missing.append(
            f"API key in {env_description} or --judge_config"
        )
    if not model:
        missing.append("--judge_model or judge config model")
    if missing:
        raise ValueError("Missing judge configuration: " + ", ".join(missing))
    return {
        "provider": provider_name,
        "provider_section": provider_section,
        "api_key": api_key,
        "api_key_env": resolved_key_env,
        "model": model,
        "base_url": base_url,
        "requested_response_format": requested_response_format,
        "response_format": response_format,
        "requested_token_parameter": requested_token_parameter,
        "token_parameter": token_parameter,
    }


def judge_response_format(
    mode: str,
    provider_name: str = "openai",
    appraisal_dimensions: list[str] | None = None,
) -> dict[str, Any] | None:
    mode = effective_response_format(provider_name, mode)
    if mode == "text":
        return {"type": "text"} if provider_name == "glm" else None
    if mode == "json_object":
        return {"type": "json_object"}
    if mode == "json_schema":
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "first_person_appraisal_chain_judgment",
                "strict": True,
                "schema": judge_output_schema(appraisal_dimensions),
            },
        }
    raise ValueError(f"Unknown judge response format: {mode}")


class APIJudge:
    """API process Judge plus deterministic, component-wise GRPO rewards."""

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
        self.appraisal_dimensions = list(args.appraisal_dimensions)
        self.judge_system_prompt = build_judge_system_prompt(
            self.appraisal_dimensions
        )
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

    def request_semaphore(self) -> asyncio.Semaphore:
        """Create the limiter inside the event loop that performs requests."""
        loop = asyncio.get_running_loop()
        if self._semaphore is None or self._semaphore_loop is not loop:
            self._semaphore = asyncio.Semaphore(
                self.args.judge_max_concurrency
            )
            self._semaphore_loop = loop
        return self._semaphore

    def cache_key(
        self,
        situation: str,
        candidate: dict[str, Any],
        reference: dict[str, Any],
    ) -> str:
        identity = {
            "rubric_version": RUBRIC_VERSION,
            "judge_provider": self.settings["provider"],
            "judge_model": self.settings["model"],
            "judge_base_url": self.settings["base_url"],
            "appraisal_dimensions": self.appraisal_dimensions,
            "response_format": self.settings["response_format"],
            "token_parameter": self.settings["token_parameter"],
            "judge_thinking": getattr(self.args, "judge_thinking", "auto"),
            "situation": situation,
            "candidate": candidate,
            "reference": reference,
        }
        serialized = json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def request_messages(
        self,
        situation: str,
        candidate: dict[str, Any],
        reference: dict[str, Any],
        validation_error: str = "",
    ) -> list[dict[str, str]]:
        appraisal_reference = reference.get(
            "legacy_human_appraisal_reasoning"
        )
        envelope = {
            "situation": situation,
            "optional_human_appraisal_reference": (
                appraisal_reference
                if isinstance(appraisal_reference, dict)
                else None
            ),
            "candidate": candidate,
        }
        user_content = (
            "Score this data with every required rubric field:\n"
            + json.dumps(envelope, ensure_ascii=False, indent=2)
        )
        if validation_error:
            user_content += (
                "\n\nA prior response failed schema validation. Return a fresh, "
                f"complete judgment. Validation error: {validation_error[:500]}"
            )
        return [
            {"role": "system", "content": self.judge_system_prompt},
            {"role": "user", "content": user_content},
        ]

    async def request_judgment(
        self,
        situation: str,
        candidate: dict[str, Any],
        reference: dict[str, Any],
        sample_id: str = "<unknown>",
    ) -> tuple[dict[str, Any], str | None]:
        response_format = judge_response_format(
            self.settings["response_format"],
            self.settings["provider"],
            self.appraisal_dimensions,
        )
        candidate_digest = hashlib.sha256(
            json.dumps(
                candidate,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:12]
        last_error = ""
        request_id: str | None = None
        for attempt_index in range(self.args.judge_parse_retries + 1):
            request_kwargs: dict[str, Any] = {
                "model": self.settings["model"],
                "messages": self.request_messages(
                    situation,
                    candidate,
                    reference,
                    validation_error=last_error,
                ),
            }
            request_kwargs[self.settings["token_parameter"]] = (
                self.args.judge_max_tokens
            )
            glm_extra_body: dict[str, Any] = {}
            if not self.args.judge_omit_temperature:
                if (
                    self.settings["provider"] == "glm"
                    and self.args.judge_temperature == 0
                ):
                    # GLM exposes deterministic decoding as do_sample=false.
                    # The OpenAI transport forwards this provider extension.
                    glm_extra_body["do_sample"] = False
                else:
                    request_kwargs["temperature"] = (
                        self.args.judge_temperature
                    )
            judge_thinking = getattr(self.args, "judge_thinking", "auto")
            if (
                self.settings["provider"] == "glm"
                and judge_thinking != "auto"
            ):
                # GLM's OpenAI-compatible API accepts this provider extension
                # in the request body. Keeping it separate from
                # --enable_thinking avoids changing policy generation.
                glm_extra_body["thinking"] = {"type": judge_thinking}
            if glm_extra_body:
                request_kwargs["extra_body"] = glm_extra_body
            if response_format is not None:
                request_kwargs["response_format"] = response_format
            async with self.request_semaphore():
                response = await self.client.chat.completions.create(
                    **request_kwargs
                )
            request_id = (
                getattr(response, "_request_id", None)
                or getattr(response, "request_id", None)
                or getattr(response, "id", None)
            )
            if not response.choices:
                last_error = "judge response contains no choices"
                continue
            message = response.choices[0].message
            refusal = getattr(message, "refusal", None)
            content = getattr(message, "content", None)
            if refusal:
                last_error = f"judge refused: {refusal}"
                continue
            if not isinstance(content, str) or not content.strip():
                last_error = "judge returned empty content"
                continue
            if getattr(self.args, "log_judge_raw_outputs", False):
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
                return parse_judge_output(
                    content,
                    self.appraisal_dimensions,
                ), request_id
            except (TypeError, ValueError) as exc:
                last_error = str(exc)
        raise RuntimeError(
            "Judge did not return a valid complete rubric after "
            f"{self.args.judge_parse_retries + 1} request(s): {last_error}"
        )

    async def score_valid_candidate(
        self,
        sample_id: str,
        situation: str,
        candidate: dict[str, Any],
        reference: dict[str, Any],
    ) -> dict[str, Any]:
        key = self.cache_key(situation, candidate, reference)
        if self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                cached = dict(cached)
                cached["cache_hit"] = True
                return cached
        try:
            judgment, request_id = await self.request_judgment(
                situation,
                candidate,
                reference,
                sample_id=sample_id,
            )
        except Exception as exc:
            digest = hashlib.sha256(
                json.dumps(
                    candidate,
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()[:12]
            raise RuntimeError(
                f"API judge failed for sample={sample_id} "
                f"candidate_sha256={digest}: {exc}"
            ) from exc
        scores = aggregate_judgment(
            judgment,
            self.appraisal_dimensions,
        )
        result = {
            **scores,
            "judgment": judgment,
            "request_id": request_id,
            "cache_hit": False,
        }
        if self.cache is not None:
            self.cache.put(key, self.settings["model"], result)
        return result

    async def score_one(
        self,
        sample_id: str,
        situation: str,
        completion: Any,
        reference_json: str,
    ) -> dict[str, Any]:
        try:
            candidate = parse_policy_output(
                completion_text(completion),
                self.appraisal_dimensions,
            )
        except (TypeError, ValueError) as exc:
            reward_components = invalid_reward_components(
                self.args.invalid_component_reward
            )
            return {
                "reward_components": reward_components,
                "outcome_diagnostics": {},
                "normalized_reward_components": None,
                "dimension_scores": {},
                "appraisal_criterion_scores": {},
                "transition_gate_components": {},
                "cache_hit": False,
                "valid": False,
                "failed": False,
                "skipped": False,
                "error": f"invalid policy output: {exc}",
                "feedback": "",
            }
        try:
            reference = json.loads(reference_json) if reference_json else {}
            if not isinstance(reference, dict):
                raise ValueError("reference_json must decode to an object")
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"{sample_id}: internal reference_json is invalid: {exc}"
            ) from exc
        try:
            result = await self.score_valid_candidate(
                sample_id,
                situation,
                candidate,
                reference,
            )
            judgment = result.get("judgment", {})
            reward_components, outcome_diagnostics = build_reward_components(
                result,
                candidate,
                reference,
                self.args.outcome_label_weight,
                self.args.outcome_intensity_weight,
                self.args.use_process_gate,
                self.args.process_gate_mode,
            )
            return {
                **result,
                "reward_components": reward_components,
                "outcome_diagnostics": outcome_diagnostics,
                "normalized_reward_components": None,
                "valid": True,
                "failed": False,
                "skipped": False,
                "error": "",
                "feedback": str(judgment.get("overall_feedback", "")),
            }
        except Exception as exc:
            if self.args.judge_failure_policy == "error":
                raise
            return {
                "reward_components": None,
                "outcome_diagnostics": {},
                "normalized_reward_components": None,
                "dimension_scores": {},
                "appraisal_criterion_scores": {},
                "transition_gate_components": {},
                "cache_hit": False,
                "valid": True,
                "failed": True,
                "skipped": True,
                "error": str(exc)[:500],
                "feedback": "",
            }

    async def score_batch(
        self,
        completions: list[Any],
        situations: list[str],
        sample_ids: list[str],
        references: list[str],
        log_extra: Callable[..., Any] | None,
        log_metric: Callable[..., Any] | None,
    ) -> list[float | None]:
        count = len(completions)
        if not (
            len(situations) == count
            and len(sample_ids) == count
            and len(references) == count
        ):
            raise ValueError(
                "GRPO did not align completions with situation/sample/reference "
                "dataset columns"
            )
        results = await asyncio.gather(
            *[
                self.score_one(sample_id, situation, completion, reference)
                for sample_id, situation, completion, reference in zip(
                    sample_ids,
                    situations,
                    completions,
                    references,
                )
            ]
        )
        optimization_weights = {
            "appraisal_reward": self.args.appraisal_reward_weight,
            "coherence_reward": self.args.coherence_reward_weight,
            "transition_reward": self.args.transition_reward_weight,
            "outcome_reward": self.args.outcome_reward_weight,
        }
        normalized_rows, training_signals = group_normalize_components(
            [item["reward_components"] for item in results],
            group_size=self.args.num_generations,
            optimization_weights=optimization_weights,
            process_gates=[
                (
                    None
                    if item["reward_components"] is None
                    else (
                        1.0
                        if not item["valid"]
                        else float(item["outcome_diagnostics"]["process_gate"])
                    )
                )
                for item in results
            ],
            outcome_label_weight=self.args.outcome_label_weight,
            outcome_intensity_weight=self.args.outcome_intensity_weight,
            epsilon=self.args.reward_normalization_epsilon,
        )
        for item, normalized in zip(results, normalized_rows):
            item["normalized_reward_components"] = normalized
        self._log_results(
            results,
            training_signals,
            log_extra,
            log_metric,
        )
        return training_signals

    @staticmethod
    def _log_results(
        results: list[dict[str, Any]],
        training_signals: list[float | None],
        log_extra: Callable[..., Any] | None,
        log_metric: Callable[..., Any] | None,
    ) -> None:
        if log_extra is not None:
            extra_values = {
                "reward_components": [
                    json.dumps(
                        item["reward_components"],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    for item in results
                ],
                "normalized_reward_components": [
                    json.dumps(
                        item["normalized_reward_components"],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    for item in results
                ],
                "outcome_diagnostics": [
                    json.dumps(
                        item["outcome_diagnostics"],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    for item in results
                ],
                "judge_dimension_scores": [
                    json.dumps(
                        item["dimension_scores"],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    for item in results
                ],
                "judge_appraisal_criterion_scores": [
                    json.dumps(
                        item["appraisal_criterion_scores"],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    for item in results
                ],
                "judge_transition_gate_components": [
                    json.dumps(
                        item["transition_gate_components"],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    for item in results
                ],
                "judge_cache_hit": [
                    bool(item["cache_hit"]) for item in results
                ],
                "judge_skipped": [bool(item["skipped"]) for item in results],
                "judge_error": [
                    str(item["error"])[:500] for item in results
                ],
                "judge_feedback": [
                    str(item["feedback"])[:1000] for item in results
                ],
            }
            for name, values in extra_values.items():
                log_extra(name, values)
        if log_metric is None or not results:
            return

        judged = [
            item for item in results if item["valid"] and not item["failed"]
        ]

        def judged_mean(field: str) -> float:
            if not judged:
                return 0.0
            return sum(float(item[field]) for item in judged) / len(judged)

        def judged_mapping_mean(field: str, key: str) -> float:
            if not judged:
                return 0.0
            return sum(
                float(item[field][key]) for item in judged
            ) / len(judged)

        def component_mean(field: str, normalized: bool = False) -> float:
            mapping_name = (
                "normalized_reward_components"
                if normalized
                else "reward_components"
            )
            values = [
                float(item[mapping_name][field])
                for item in results
                if isinstance(item.get(mapping_name), dict)
            ]
            return sum(values) / len(values) if values else 0.0

        def component_std(field: str) -> float:
            values = [
                float(item["normalized_reward_components"][field])
                for item in results
                if isinstance(item.get("normalized_reward_components"), dict)
            ]
            if len(values) <= 1:
                return 0.0
            mean = sum(values) / len(values)
            return (
                sum((value - mean) ** 2 for value in values)
                / (len(values) - 1)
            ) ** 0.5

        def outcome_mean(field: str) -> float:
            values = [
                float(item["outcome_diagnostics"][field])
                for item in judged
                if field in item["outcome_diagnostics"]
            ]
            return sum(values) / len(values) if values else 0.0

        metrics = {
            "strict_format_rate": (
                sum(bool(item["valid"]) for item in results) / len(results)
            ),
            "judge_api_failure_rate": (
                sum(bool(item["failed"]) for item in results) / len(results)
            ),
            "judge_skipped_rate": (
                sum(bool(item["skipped"]) for item in results) / len(results)
            ),
            "judge_cache_hit_rate": (
                sum(bool(item["cache_hit"]) for item in results) / len(results)
            ),
            "judge_dimension_specific_validity": judged_mapping_mean(
                "appraisal_criterion_scores",
                "dimension_specific_validity",
            ),
            "judge_situation_grounding": judged_mapping_mean(
                "appraisal_criterion_scores",
                "situation_grounding",
            ),
            "judge_experiencer_fidelity": judged_mapping_mean(
                "appraisal_criterion_scores",
                "experiencer_fidelity",
            ),
            "judge_raw_appraisal_emotion_linkage": judged_mean(
                "raw_transition_reward"
            ),
            "process_gate": outcome_mean("process_gate"),
            "process_gate_enabled": outcome_mean("process_gate_enabled"),
            "positive_label_score": outcome_mean("positive_label_score"),
            "negative_label_score": outcome_mean("negative_label_score"),
            "positive_intensity_score": outcome_mean(
                "positive_intensity_score"
            ),
            "negative_intensity_score": outcome_mean(
                "negative_intensity_score"
            ),
        }
        for field in REWARD_COMPONENT_FIELDS:
            metrics[field] = component_mean(field)
            metrics[f"normalized_{field}_mean"] = component_mean(
                field,
                normalized=True,
            )
            metrics[f"normalized_{field}_std"] = component_std(field)
        available_signals = [
            float(value) for value in training_signals if value is not None
        ]
        metrics["componentwise_training_signal_mean"] = (
            sum(available_signals) / len(available_signals)
            if available_signals
            else 0.0
        )
        for name, value in metrics.items():
            log_metric(name, value)

    def reward_function(self) -> Callable[..., Any]:
        async def api_judge_reward(
            completions: list[Any],
            situation: list[str],
            sample_id: list[str] | None = None,
            reference_json: list[str] | None = None,
            log_extra: Callable[..., Any] | None = None,
            log_metric: Callable[..., Any] | None = None,
            **_: Any,
        ) -> list[float | None]:
            count = len(completions)
            sample_ids = sample_id or [
                f"<unknown-{index}>" for index in range(count)
            ]
            references = reference_json or ["{}"] * count
            return await self.score_batch(
                completions,
                situation,
                sample_ids,
                references,
                log_extra,
                log_metric,
            )

        api_judge_reward.__name__ = "componentwise_group_normalized_reward"
        return api_judge_reward
