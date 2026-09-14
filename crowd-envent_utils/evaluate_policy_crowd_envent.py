#!/usr/bin/env python3
"""Compatibility entry point for crowd-enVent Policy generation and metrics.

The default ``chain`` mode makes one end-to-end call per situation and emits
the GRPO-aligned five-dimensional appraisal reasoning, all 21 native ratings,
and one native crowd-enVent emotion plus intensity.  ``chain-all`` makes two
independent calls: reasoning -> ratings and reasoning -> emotion.  ``direct``
omits natural-language reasoning while retaining the same scored outcomes.

Generation/parse failures are written to an invalid JSONL file. Metrics
penalize missing predictions by default so invalid generations remain in the
denominator; ``--allow_incomplete`` explicitly scores valid coverage only for
debugging.

For the recommended split workflow, use ``generate_policy_crowd_envent.py``
first and then ``compute_crowd_envent_metrics.py`` in a fresh process.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


UTILS_DIR = Path(__file__).resolve().parent
REPO_ROOT = UTILS_DIR.parent
FIRST_PERSON_SCRIPTS = REPO_ROOT / "FirstPersonMethod/scripts"
COVIDET_UTILS = REPO_ROOT / "covidet_utils"
for directory in (UTILS_DIR, FIRST_PERSON_SCRIPTS, COVIDET_UTILS):
    value = str(directory)
    if value not in sys.path:
        sys.path.insert(0, value)

from crowd_envent_metrics import build_metrics_report
from crowd_envent_metrics import MISSING_PREDICTION_POLICIES
from crowd_envent_schema import (
    CAREBENCH_APPRAISALS_SYSTEM_PROMPT,
    CAREBENCH_EMOTION_SYSTEM_PROMPT,
    CHAIN_APPRAISALS_SYSTEM_PROMPT,
    CHAIN_EMOTION_SYSTEM_PROMPT,
    CHAIN_SYSTEM_PROMPT,
    CHAIN_TARGETS,
    DIRECT_SYSTEM_PROMPT,
    EMOTION_LABELS,
    build_carebench_aligned_prompt,
    build_chain_target_prompt,
    normalize_emotion,
    normalize_ratings,
    normalize_reasoning,
    parse_carebench_aligned_response,
    parse_chain_target_response,
    parse_policy_response,
    build_policy_prompt,
)
from carebench_aligned_projection import (
    normalize_carebench_appraisal_ratings,
    normalize_crowd_envent_valenced_emotion,
)
from covidet_api_client import (
    PolicyAPIClient,
    PolicyAPIError,
    resolve_api_settings,
)


DEFAULT_EVAL_FILE = REPO_ROOT / "FirstPersonMethod/data/crowd_envent/test.json"
GENERATION_SCHEMAS = ("chain", "chain-all", "direct", "carebench")
EVALUATION_ONLY_EMOTION_RANKING_FIELD = "evaluation_only_emotion_ranking"


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got {value!r}")


def build_parser(*, fixed_mode: str | None = None) -> argparse.ArgumentParser:
    if fixed_mode is not None and fixed_mode not in {"generate", "metrics"}:
        raise ValueError(
            "fixed_mode must be either 'generate', 'metrics', or None"
        )
    parser = argparse.ArgumentParser(
        description=(
            "Generate and score first-person Policy chains on crowd-enVent"
            if fixed_mode is None
            else (
                "Generate first-person Policy predictions on crowd-enVent"
                if fixed_mode == "generate"
                else "Compute crowd-enVent metrics from saved Policy predictions"
            )
        )
    )
    if fixed_mode is None:
        parser.add_argument(
            "--mode",
            choices=["run", "generate", "metrics"],
            default="run",
            help=(
                "run=generate+metrics; generate=inference only; "
                "metrics=rescore JSONL"
            ),
        )
    else:
        parser.set_defaults(mode=fixed_mode)
    if fixed_mode != "metrics":
        parser.add_argument(
            "--generation_schema",
            choices=list(GENERATION_SCHEMAS),
            default="chain",
            help=(
                "chain makes one joint reasoning/ratings/emotion call; chain-all "
                "makes separate reasoning->ratings and reasoning->emotion calls; "
                "direct predicts outcomes without natural-language reasoning; "
                "carebench makes separate calls with CAREBench appraisal "
                "ratings and a native crowd-enVent emotion output, then "
                "projects the ratings to native metrics fields"
            ),
        )
    else:
        # The saved predictions carry their schema; metrics-only users should
        # not need to repeat the generation setting.
        parser.set_defaults(generation_schema="chain")
    parser.add_argument("--eval_file", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument(
        "--predictions_file",
        type=Path,
        default=None,
        help=(
            "JSONL predictions; required by the metrics-only entry point"
            if fixed_mode == "metrics"
            else "JSONL destination; defaults to a model-specific output path"
        ),
    )
    if fixed_mode != "generate":
        parser.add_argument("--results_file", type=Path, default=None)
    else:
        parser.set_defaults(results_file=None)
    if fixed_mode != "metrics":
        parser.add_argument("--invalid_file", type=Path, default=None)
    else:
        parser.set_defaults(invalid_file=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--sample_seed", type=int, default=42)
    if fixed_mode != "generate":
        parser.add_argument(
            "--missing_prediction_policy",
            choices=list(MISSING_PREDICTION_POLICIES),
            default="penalize",
            help=(
                "How metrics handle samples absent from predictions.jsonl. "
                "penalize keeps them in the denominator with neutral ratings, "
                "no-emotion, and minimum intensity; error stops; skip scores "
                "valid coverage only."
            ),
        )
        parser.add_argument(
            "--allow_incomplete",
            action="store_true",
            help=(
                "Deprecated alias for --missing_prediction_policy skip, useful "
                "for debugging valid outputs only."
            ),
        )
    else:
        parser.set_defaults(
            allow_incomplete=False,
            missing_prediction_policy="penalize",
        )
    if fixed_mode != "metrics":
        parser.add_argument("--overwrite_predictions", action="store_true")
        parser.add_argument("--max_parse_retries", type=int, default=2)
    else:
        parser.set_defaults(overwrite_predictions=False, max_parse_retries=2)

    if fixed_mode != "metrics":
        model = parser.add_argument_group("Policy model")
        model.add_argument("--backend", choices=["transformers", "api"], default="transformers")
        model.add_argument("--model", type=str, default="")
        model.add_argument("--model_is_adapter", action="store_true")
        model.add_argument("--base_model", type=str, default="")
        model.add_argument("--tokenizer", type=str, default="")
        model.add_argument("--max_tokens", type=int, default=4096)
        model.add_argument("--temperature", type=float, default=0.2)
        model.add_argument("--top_p", type=float, default=1.0)
        model.add_argument("--do_sample", type=str2bool, default=False)
        model.add_argument("--enable_thinking", type=str2bool, default=False)
        model.add_argument(
            "--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto"
        )
        model.add_argument("--device", type=str, default="auto")
        model.add_argument(
            "--device_map",
            choices=["auto", "balanced", "balanced_low_0", "sequential"],
            default=None,
        )
        quantization = model.add_mutually_exclusive_group()
        quantization.add_argument("--load_in_4bit", action="store_true")
        quantization.add_argument("--load_in_8bit", action="store_true")
        model.add_argument(
            "--attn_implementation",
            choices=["eager", "sdpa", "flash_attention_2"],
            default=None,
        )
        model.add_argument("--trust_remote_code", action="store_true")
        model.add_argument("--local_files_only", action="store_true")
        model.add_argument("--revision", type=str, default="")
        model.add_argument("--seed", type=int, default=42)

        api = parser.add_argument_group("Policy API")
        api.add_argument("--api_provider", choices=["openai", "glm"], default="openai")
        api.add_argument("--api_model", type=str, default="")
        api.add_argument("--api_config", type=str, default="")
        api.add_argument("--api_provider_section", type=str, default="")
        api.add_argument("--api_key_env", type=str, default="")
        api.add_argument("--api_base_url", type=str, default="")
        api.add_argument(
            "--api_token_parameter",
            choices=["auto", "max_tokens", "max_completion_tokens"],
            default="auto",
        )
        api.add_argument(
            "--api_response_format",
            choices=["auto", "json_object", "text"],
            default="auto",
        )
        api.add_argument(
            "--api_thinking",
            choices=["auto", "enabled", "disabled"],
            default="auto",
        )
        api.add_argument("--api_timeout", type=float, default=180.0)
        api.add_argument("--api_max_retries", type=int, default=3)
    return parser


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


def load_evaluation_file(path: Path) -> dict[str, Any]:
    require_file(path, "crowd-enVent evaluation JSON")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{path} must contain a non-empty samples array")
    seen: set[str] = set()
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"samples[{index}] must be an object")
        sample_id = sample.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"samples[{index}] has no id")
        if sample_id in seen:
            raise ValueError(f"Duplicate sample id {sample_id}")
        seen.add(sample_id)
        if not isinstance(sample.get("situation"), str) or not sample["situation"].strip():
            raise ValueError(f"{sample_id}: situation must be non-empty")
        reference = sample.get("reference")
        if not isinstance(reference, dict):
            raise ValueError(f"{sample_id}: missing reference")
        # Reuse the strict Policy parser to validate the gold outcome fields.
        parse_policy_response(
            json.dumps(
                {
                    "appraisal_ratings": reference.get("appraisal_ratings"),
                    "emotion": reference.get("emotion"),
                }
            ),
            include_reasoning=False,
        )
    return payload


def select_samples(
    samples: list[dict[str, Any]], maximum: int | None, seed: int
) -> list[dict[str, Any]]:
    if maximum is None:
        return samples
    if maximum <= 0:
        raise ValueError("--max_samples must be positive")
    if maximum >= len(samples):
        return samples
    return random.Random(seed).sample(samples, maximum)


def create_policy_client(args: argparse.Namespace) -> Any:
    if args.backend == "api":
        settings = resolve_api_settings(args)
        return PolicyAPIClient(
            settings,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
            timeout=args.api_timeout,
            max_retries=args.api_max_retries,
        )
    if not args.model.strip():
        raise ValueError("--model is required for local Policy generation")
    try:
        from baseline_transformers import TransformersClient
    except ImportError as exc:
        raise RuntimeError(
            "Local Policy generation requires transformers/torch/peft dependencies"
        ) from exc
    return TransformersClient(
        model_name_or_path=args.model.strip(),
        tokenizer_name_or_path=args.tokenizer.strip(),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        enable_thinking=args.enable_thinking,
        dtype=args.dtype,
        device=args.device,
        device_map=args.device_map,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        revision=args.revision,
        seed=args.seed,
        base_model_name_or_path=args.base_model.strip(),
        model_is_adapter=args.model_is_adapter,
    )


def prediction_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.predictions_file is not None:
        predictions = args.predictions_file
    else:
        backend = getattr(args, "backend", "transformers")
        api_model = getattr(args, "api_model", "")
        model = getattr(args, "model", "")
        api_provider = getattr(args, "api_provider", "openai")
        identifier = (
            api_model.strip() or model.strip() or api_provider
            if backend == "api"
            else model.strip()
        )
        if not identifier:
            raise ValueError("--predictions_file is required when no model is specified")
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", identifier).strip("._") or "policy"
        predictions = (
            REPO_ROOT
            / "FirstPersonMethod/output/crowd_envent_policy_eval"
            / safe_name
            / "predictions.jsonl"
        )
    invalid = args.invalid_file or predictions.with_name(predictions.stem + ".invalid.jsonl")
    results = args.results_file or predictions.with_name("results.json")
    return predictions, invalid, results


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            records.append(value)
    return records


def _normalize_emotion_candidates(
    raw_candidates: Any,
    *,
    sample_id: str,
    field: str,
) -> list[str]:
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError(f"{sample_id}: {field} must be a non-empty JSON list")
    candidates: list[str] = []
    for index, raw_label in enumerate(raw_candidates):
        if not isinstance(raw_label, str):
            raise ValueError(f"{sample_id}: {field}[{index}] must be text")
        label = raw_label.strip().lower()
        if label not in EMOTION_LABELS:
            raise ValueError(
                f"{sample_id}: invalid emotion candidate {raw_label!r}; "
                f"allowed={list(EMOTION_LABELS)}"
            )
        if label not in candidates:
            candidates.append(label)
    if "no-emotion" in candidates and len(candidates) != 1:
        raise ValueError(
            f"{sample_id}: no-emotion cannot be combined with other candidates"
        )
    return candidates


def normalize_prediction_record(record: Mapping[str, Any]) -> dict[str, Any]:
    sample_id = record.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("Prediction record has no sample_id")
    schema = record.get("generation_schema", "chain")
    if schema not in GENERATION_SCHEMAS:
        raise ValueError(f"{sample_id}: invalid generation_schema {schema!r}")
    if schema == "chain-all":
        raw_reasoning = record.get("appraisal_reasoning")
        if not isinstance(raw_reasoning, dict) or set(raw_reasoning) != set(CHAIN_TARGETS):
            raise ValueError(
                f"{sample_id}: chain-all appraisal_reasoning must contain "
                f"exactly {list(CHAIN_TARGETS)}"
            )
        parsed = {
            "appraisal_reasoning": {
                target: normalize_reasoning(raw_reasoning[target])
                for target in CHAIN_TARGETS
            },
            "appraisal_ratings": normalize_ratings(record.get("appraisal_ratings")),
            "emotion": normalize_emotion(record.get("emotion")),
        }
    elif schema == "carebench":
        raw_reasoning = record.get("appraisal_reasoning")
        if (
            isinstance(raw_reasoning, dict)
            and set(raw_reasoning) == set(CHAIN_TARGETS)
        ):
            normalized_reasoning: dict[str, Any] = {
                target: normalize_reasoning(raw_reasoning[target])
                for target in CHAIN_TARGETS
            }
        else:
            normalized_reasoning = normalize_reasoning(raw_reasoning)
        parsed = {
            "appraisal_reasoning": normalized_reasoning,
            "carebench_appraisal_ratings": normalize_carebench_appraisal_ratings(
                record.get("carebench_appraisal_ratings")
            ),
            "appraisal_ratings": normalize_ratings(record.get("appraisal_ratings")),
            "emotion": normalize_emotion(record.get("emotion")),
        }
        raw_carebench_emotion = record.get("carebench_emotion")
        if raw_carebench_emotion is not None:
            parsed["carebench_emotion"] = normalize_crowd_envent_valenced_emotion(
                raw_carebench_emotion
            )
    else:
        candidate = {
            "appraisal_ratings": record.get("appraisal_ratings"),
            "emotion": record.get("emotion"),
        }
        if schema == "chain":
            candidate["appraisal_reasoning"] = record.get("appraisal_reasoning")
        parsed = parse_policy_response(
            json.dumps(candidate, ensure_ascii=False),
            include_reasoning=schema == "chain",
        )
    normalized = dict(record)
    normalized.update(parsed)
    raw_candidates = record.get("emotion_candidate_labels")
    candidates: list[str] | None = None
    if raw_candidates is not None:
        candidates = _normalize_emotion_candidates(
            raw_candidates,
            sample_id=sample_id,
            field="emotion_candidate_labels",
        )
        normalized["emotion_candidate_labels"] = [
            label for label in EMOTION_LABELS if label in set(candidates)
        ]
    raw_evaluation_ranking = record.get(EVALUATION_ONLY_EMOTION_RANKING_FIELD)
    if raw_evaluation_ranking is not None:
        if schema != "carebench":
            raise ValueError(
                f"{sample_id}: {EVALUATION_ONLY_EMOTION_RANKING_FIELD} is only "
                "valid for converted chain predictions"
            )
        chain_ranking = _normalize_emotion_candidates(
            raw_evaluation_ranking,
            sample_id=sample_id,
            field=EVALUATION_ONLY_EMOTION_RANKING_FIELD,
        )
        if candidates is not None and set(chain_ranking) != set(candidates):
            raise ValueError(
                f"{sample_id}: chain evaluation ranking and emotion candidate "
                "sets differ"
            )
        normalized[EVALUATION_ONLY_EMOTION_RANKING_FIELD] = chain_ranking

    raw_multilabel_emotion = record.get("multilabel_emotion")
    if raw_multilabel_emotion is not None:
        if schema != "direct" or not isinstance(raw_multilabel_emotion, dict):
            raise ValueError(
                f"{sample_id}: multilabel_emotion is only valid for converted "
                "direct predictions"
            )
        if set(raw_multilabel_emotion) != {"labels", "intensity"}:
            raise ValueError(
                f"{sample_id}: multilabel_emotion requires labels and intensity"
            )
        direct_ranking = _normalize_emotion_candidates(
            raw_multilabel_emotion.get("labels"),
            sample_id=sample_id,
            field="multilabel_emotion.labels",
        )
        intensity = raw_multilabel_emotion.get("intensity")
        if isinstance(intensity, bool) or not isinstance(intensity, int):
            raise ValueError(
                f"{sample_id}: multilabel_emotion.intensity must be an integer"
            )
        if not 1 <= intensity <= 5:
            raise ValueError(
                f"{sample_id}: multilabel_emotion.intensity must be in [1, 5]"
            )
        if direct_ranking[0] != normalized["emotion"]["label"]:
            raise ValueError(
                f"{sample_id}: first direct ranked label must equal the converted "
                "emotion label"
            )
        if candidates is not None and set(direct_ranking) != set(candidates):
            raise ValueError(
                f"{sample_id}: direct ranking and emotion candidate sets differ"
            )
        normalized["multilabel_emotion"] = {
            "labels": direct_ranking,
            "intensity": intensity,
        }
    normalized["generation_schema"] = schema
    return normalized


def prediction_index(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for raw_record in records:
        record = normalize_prediction_record(raw_record)
        sample_id = record["sample_id"]
        if sample_id in indexed:
            raise ValueError(f"Duplicate prediction for {sample_id}")
        indexed[sample_id] = record
    return indexed


def effective_missing_prediction_policy(args: argparse.Namespace) -> str:
    if getattr(args, "allow_incomplete", False):
        return "skip"
    policy = getattr(args, "missing_prediction_policy", "penalize")
    if policy not in MISSING_PREDICTION_POLICIES:
        raise ValueError(
            f"Invalid missing prediction policy {policy!r}; "
            f"allowed={list(MISSING_PREDICTION_POLICIES)}"
        )
    return policy


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False))
        handle.write("\n")
        handle.flush()


def generate_one(
    client: Any,
    args: argparse.Namespace,
    sample: Mapping[str, Any],
    chain_target: str | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any], list[dict[str, Any]]]:
    if args.generation_schema == "chain-all":
        if chain_target not in CHAIN_TARGETS:
            raise ValueError(
                f"chain_target must be one of {list(CHAIN_TARGETS)} for chain-all"
            )
        system_prompt = (
            CHAIN_APPRAISALS_SYSTEM_PROMPT
            if chain_target == "appraisals"
            else CHAIN_EMOTION_SYSTEM_PROMPT
        )
    elif args.generation_schema == "carebench":
        if chain_target not in CHAIN_TARGETS:
            raise ValueError(
                f"chain_target must be one of {list(CHAIN_TARGETS)} for carebench"
            )
        system_prompt = (
            CAREBENCH_APPRAISALS_SYSTEM_PROMPT
            if chain_target == "appraisals"
            else CAREBENCH_EMOTION_SYSTEM_PROMPT
        )
    else:
        if chain_target is not None:
            raise ValueError("chain_target is only valid for chain-style generation")
        include_reasoning = args.generation_schema == "chain"
        system_prompt = CHAIN_SYSTEM_PROMPT if include_reasoning else DIRECT_SYSTEM_PROMPT
    correction = ""
    attempts: list[dict[str, Any]] = []
    response_metadata: dict[str, Any] = {}
    for attempt in range(args.max_parse_retries + 1):
        try:
            if args.generation_schema == "chain-all":
                user_prompt = build_chain_target_prompt(
                    sample["situation"],
                    chain_target,
                    correction=correction,
                )
            elif args.generation_schema == "carebench":
                user_prompt = build_carebench_aligned_prompt(
                    sample["situation"],
                    chain_target,
                    correction=correction,
                )
            else:
                user_prompt = build_policy_prompt(
                    sample["situation"],
                    include_reasoning=include_reasoning,
                    correction=correction,
                )
            response = client.chat(
                system_prompt,
                user_prompt,
            )
        except PolicyAPIError as exc:
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "valid": False,
                    "error_type": "api_request",
                    "error": str(exc),
                }
            )
            correction = ""
            continue
        response_metadata = response.raw
        try:
            if args.generation_schema == "chain-all":
                parsed = parse_chain_target_response(response.content, chain_target)
            elif args.generation_schema == "carebench":
                parsed = parse_carebench_aligned_response(
                    response.content,
                    chain_target,
                )
            else:
                parsed = parse_policy_response(
                    response.content,
                    include_reasoning=include_reasoning,
                )
        except (TypeError, ValueError) as exc:
            correction = str(exc)
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "valid": False,
                    "error_type": "parse_or_schema",
                    "error": correction,
                    "raw_output": response.content,
                }
            )
            continue
        attempts.append(
            {"attempt": attempt + 1, "valid": True, "raw_output": response.content}
        )
        return parsed, response_metadata, attempts
    return None, response_metadata, attempts


def generate_predictions(
    args: argparse.Namespace,
    samples: list[dict[str, Any]],
    predictions_path: Path,
    invalid_path: Path,
) -> dict[str, dict[str, Any]]:
    if args.max_parse_retries < 0:
        raise ValueError("--max_parse_retries must be >= 0")
    if args.overwrite_predictions:
        for path in (predictions_path, invalid_path):
            if path.exists():
                path.unlink()
    existing = prediction_index(load_jsonl(predictions_path))
    incompatible = [
        sample_id
        for sample_id, record in existing.items()
        if record["generation_schema"] != args.generation_schema
    ]
    if incompatible:
        raise ValueError(
            "Prediction JSONL contains a different generation schema; use a "
            "new file or --overwrite_predictions"
        )
    pending = [sample for sample in samples if sample["id"] not in existing]
    print(
        f"[generate] selected={len(samples)} existing={len(samples)-len(pending)} "
        f"pending={len(pending)} schema={args.generation_schema} "
        f"calls_per_sample={2 if args.generation_schema in {'chain-all', 'carebench'} else 1}"
    )
    if not pending:
        return existing
    client = create_policy_client(args)
    try:
        from tqdm import tqdm
        iterator: Iterable[dict[str, Any]] = tqdm(pending, desc="crowd-enVent Policy")
    except ImportError:
        iterator = pending
    valid = invalid = 0
    for sample in iterator:
        if args.generation_schema in {"chain-all", "carebench"}:
            parsed_by_target: dict[str, dict[str, Any] | None] = {}
            metadata_by_target: dict[str, dict[str, Any]] = {}
            attempts_by_target: dict[str, list[dict[str, Any]]] = {}
            for target in CHAIN_TARGETS:
                parsed, metadata, attempts = generate_one(
                    client, args, sample, target
                )
                parsed_by_target[target] = parsed
                metadata_by_target[target] = metadata
                attempts_by_target[target] = attempts
            failed_targets = [
                target for target, parsed in parsed_by_target.items() if parsed is None
            ]
        else:
            parsed, metadata, attempts = generate_one(client, args, sample)
            parsed_by_target = {"joint": parsed}
            metadata_by_target = {"joint": metadata}
            attempts_by_target = {"joint": attempts}
            failed_targets = [] if parsed is not None else ["joint"]
        if failed_targets:
            append_jsonl(
                invalid_path,
                {
                    "sample_id": sample["id"],
                    "source_id": sample.get("source_id"),
                    "generation_schema": args.generation_schema,
                    "error": "all generation/parse attempts failed",
                    "failed_targets": failed_targets,
                    "attempts": attempts_by_target,
                },
            )
            invalid += 1
            continue
        if args.generation_schema == "chain-all":
            appraisal_branch = parsed_by_target["appraisals"]
            emotion_branch = parsed_by_target["emotion"]
            assert appraisal_branch is not None and emotion_branch is not None
            record = {
                "sample_id": sample["id"],
                "source_id": sample.get("source_id"),
                "generation_schema": "chain-all",
                "appraisal_reasoning": {
                    "appraisals": appraisal_branch["appraisal_reasoning"],
                    "emotion": emotion_branch["appraisal_reasoning"],
                },
                "appraisal_ratings": appraisal_branch["appraisal_ratings"],
                "emotion": emotion_branch["emotion"],
                "generation": metadata_by_target,
                "attempt_count": {
                    target: len(attempts_by_target[target])
                    for target in CHAIN_TARGETS
                },
                "raw_output": {
                    target: attempts_by_target[target][-1]["raw_output"]
                    for target in CHAIN_TARGETS
                },
            }
        elif args.generation_schema == "carebench":
            appraisal_branch = parsed_by_target["appraisals"]
            emotion_branch = parsed_by_target["emotion"]
            assert appraisal_branch is not None and emotion_branch is not None
            record = {
                "sample_id": sample["id"],
                "source_id": sample.get("source_id"),
                "generation_schema": "carebench",
                "appraisal_reasoning": {
                    "appraisals": appraisal_branch["appraisal_reasoning"],
                    "emotion": emotion_branch["appraisal_reasoning"],
                },
                "appraisal_ratings": appraisal_branch["appraisal_ratings"],
                "emotion": emotion_branch["emotion"],
                "carebench_appraisal_ratings": appraisal_branch[
                    "carebench_appraisal_ratings"
                ],
                "generation": metadata_by_target,
                "attempt_count": {
                    target: len(attempts_by_target[target])
                    for target in CHAIN_TARGETS
                },
                "raw_output": {
                    target: attempts_by_target[target][-1]["raw_output"]
                    for target in CHAIN_TARGETS
                },
                "projection": {
                    "appraisals": "carebench_statement_projection_v1",
                    "emotion": "native_crowd_envent_single_label",
                },
            }
        else:
            parsed = parsed_by_target["joint"]
            assert parsed is not None
            record = {
                "sample_id": sample["id"],
                "source_id": sample.get("source_id"),
                "generation_schema": args.generation_schema,
                "appraisal_ratings": parsed["appraisal_ratings"],
                "emotion": parsed["emotion"],
                "generation": metadata_by_target["joint"],
                "attempt_count": len(attempts_by_target["joint"]),
                "raw_output": attempts_by_target["joint"][-1]["raw_output"],
            }
            if args.generation_schema == "chain":
                record["appraisal_reasoning"] = parsed["appraisal_reasoning"]
        append_jsonl(predictions_path, record)
        existing[sample["id"]] = normalize_prediction_record(record)
        valid += 1
    print(
        f"[generate] valid_written={valid} invalid_written={invalid} "
        f"predictions={predictions_path}"
    )
    return existing


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def execute(args: argparse.Namespace) -> int:
    evaluation = load_evaluation_file(args.eval_file)
    samples = select_samples(
        evaluation["samples"], args.max_samples, args.sample_seed
    )
    predictions_path, invalid_path, results_path = prediction_paths(args)
    if args.mode in {"run", "generate"}:
        predictions = generate_predictions(
            args, samples, predictions_path, invalid_path
        )
    else:
        predictions = prediction_index(load_jsonl(predictions_path))
    if args.mode in {"run", "metrics"}:
        report = build_metrics_report(
            samples,
            predictions,
            allow_incomplete=args.allow_incomplete,
            missing_prediction_policy=effective_missing_prediction_policy(args),
        )
        active_schemas = sorted(
            {
                predictions[sample["id"]]["generation_schema"]
                for sample in samples
                if sample["id"] in predictions
            }
        )
        report["evaluation"] = {
            "eval_file": str(args.eval_file),
            "subset": evaluation.get("subset"),
            "generation_schema": (
                active_schemas[0] if len(active_schemas) == 1 else active_schemas
            ),
            "sample_seed": args.sample_seed,
            "predictions_file": str(predictions_path),
        }
        write_json_atomic(results_path, report)
        print(json.dumps(report["coverage"], ensure_ascii=False, indent=2))
        print(f"[metrics] {results_path}")
    return 0


def main() -> int:
    return execute(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
