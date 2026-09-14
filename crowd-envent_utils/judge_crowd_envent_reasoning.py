#!/usr/bin/env python3
"""API-rubric evaluation of crowd-enVent appraisal reasoning.

crowd-enVent has no human natural-language rationale.  This Judge therefore
uses the situation, the experiencer's 21 self-reported ratings, and their
self-reported emotion as a structured hidden reference.  API/parse failures are
penalized by default so invalid outputs remain in the evaluation denominator.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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

from crowd_envent_schema import (
    APPRAISAL_TO_METHOD_DIMENSION,
    CHAIN_TARGETS,
    METHOD_APPRAISAL_DEFINITIONS,
    METHOD_APPRAISAL_DIMENSIONS,
    extract_json_object,
    normalize_emotion,
    normalize_ratings,
    normalize_reasoning,
)
from covidet_api_client import message_content_text, response_to_dict
from train_grpo.api_judge import resolve_judge_settings


RUBRIC_FIELDS = (
    "dimension_specific_validity",
    "situation_grounding",
    "experiencer_fidelity",
)
CHAIN_FIELDS = (
    "cross_dimension_coherence",
    "appraisal_rating_alignment",
    "appraisal_emotion_linkage",
)
MISSING_JUDGMENT_POLICIES = ("error", "skip", "penalize")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use an API rubric Judge to evaluate crowd-enVent reasoning"
    )
    parser.add_argument(
        "--mode",
        choices=["run", "metrics"],
        default="run",
        help="run requests missing judgments; metrics only aggregates saved JSONL",
    )
    parser.add_argument(
        "--eval_file",
        type=Path,
        default=REPO_ROOT / "FirstPersonMethod/data/crowd_envent/test.json",
    )
    parser.add_argument("--predictions_file", type=Path, required=True)
    parser.add_argument("--judgments_file", type=Path, default=None)
    parser.add_argument("--invalid_file", type=Path, default=None)
    parser.add_argument("--results_file", type=Path, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument(
        "--missing_judgment_policy",
        choices=list(MISSING_JUDGMENT_POLICIES),
        default="penalize",
        help=(
            "How to handle missing Judge outputs: error, skip, or penalize "
            "with minimum rubric scores."
        ),
    )
    parser.add_argument(
        "--allow_incomplete",
        action="store_true",
        help="Deprecated alias for --missing_judgment_policy skip.",
    )
    parser.add_argument("--overwrite_judgments", action="store_true")
    parser.add_argument("--judge_provider", choices=["openai", "glm"], default="openai")
    parser.add_argument("--judge_model", type=str, default="")
    parser.add_argument("--judge_config", type=str, default="")
    parser.add_argument("--judge_provider_section", type=str, default="")
    parser.add_argument("--judge_api_key_env", type=str, default="")
    parser.add_argument("--judge_base_url", type=str, default="")
    parser.add_argument(
        "--judge_response_format",
        choices=["json_object", "text"],
        default="json_object",
    )
    parser.add_argument(
        "--judge_token_parameter",
        choices=["auto", "max_tokens", "max_completion_tokens"],
        default="auto",
    )
    parser.add_argument("--judge_max_tokens", type=int, default=3000)
    parser.add_argument("--judge_temperature", type=float, default=0.0)
    parser.add_argument(
        "--judge_thinking",
        choices=["auto", "enabled", "disabled"],
        default="disabled",
    )
    parser.add_argument("--judge_timeout_seconds", type=float, default=180.0)
    parser.add_argument("--judge_max_retries", type=int, default=3)
    parser.add_argument("--judge_parse_retries", type=int, default=1)
    parser.add_argument("--judge_max_concurrency", type=int, default=8)
    return parser


def judge_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    judgments = args.judgments_file or args.predictions_file.with_name(
        args.predictions_file.stem + ".judgments.jsonl"
    )
    invalid = args.invalid_file or judgments.with_name(
        judgments.stem + ".invalid.jsonl"
    )
    results = args.results_file or judgments.with_name("judge_results.json")
    return judgments, invalid, results


def effective_missing_judgment_policy(args: argparse.Namespace) -> str:
    if getattr(args, "allow_incomplete", False):
        return "skip"
    return getattr(args, "missing_judgment_policy", "penalize")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


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
                raise ValueError(f"{path}:{line_number}: expected object")
            records.append(value)
    return records


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False))
        handle.write("\n")
        handle.flush()


def record_index(records: Iterable[Mapping[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for raw in records:
        sample_id = raw.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{name} record has no sample_id")
        if sample_id in indexed:
            raise ValueError(f"Duplicate {name} record for {sample_id}")
        indexed[sample_id] = dict(raw)
    return indexed


def rating_groups() -> dict[str, list[str]]:
    result = {dimension: [] for dimension in METHOD_APPRAISAL_DIMENSIONS}
    for field, dimension in APPRAISAL_TO_METHOD_DIMENSION.items():
        if dimension in result:
            result[dimension].append(field)
    return result


def judge_output_example() -> dict[str, Any]:
    return {
        "appraisals": {
            dimension: {field: 3 for field in RUBRIC_FIELDS}
            for dimension in METHOD_APPRAISAL_DIMENSIONS
        },
        "chain": {field: 3 for field in CHAIN_FIELDS},
        "overall_feedback": "A concise explanation of the most important strengths and errors.",
    }


def penalized_judgment() -> dict[str, Any]:
    return {
        "appraisals": {
            dimension: {field: 1 for field in RUBRIC_FIELDS}
            for dimension in METHOD_APPRAISAL_DIMENSIONS
        },
        "chain": {field: 1 for field in CHAIN_FIELDS},
        "overall_feedback": "Missing or invalid Judge output; assigned minimum rubric scores.",
    }


def system_prompt() -> str:
    definitions = "\n".join(
        f"- {dimension}: {METHOD_APPRAISAL_DEFINITIONS[dimension]}"
        for dimension in METHOD_APPRAISAL_DIMENSIONS
    )
    return f"""You are a strict evaluator of first-person event -> cognitive
appraisal -> emotion reasoning. crowd-enVent provides no gold natural-language
rationale, so use the event author's 21 self-reported 1--5 appraisal ratings
and self-reported emotion as structured reference evidence. Do not demand
details that the situation or ratings do not support.

Appraisal dimensions:
{definitions}

Score every criterion with an integer from 1 to 4:
1 = materially wrong, contradicted, invented, or absent;
2 = major omissions or weak/partly unsupported reasoning;
3 = mostly correct with minor omissions or imprecision;
4 = correct, well grounded, and faithful.

The candidate may contain either one shared appraisal-reasoning trace or two
independent traces named appraisals and emotion. When two traces are present,
evaluate both: use the appraisals trace for rating alignment, the emotion trace
for emotion linkage, and penalize material conflicts between the traces.

For each appraisal dimension score both available traces jointly:
- dimension_specific_validity: theory-correct interpretation consistent with
  relevant self-report ratings;
- situation_grounding: claims are supported by the situation rather than invented;
- experiencer_fidelity: reasoning consistently adopts the first-person author.

Chain scores:
- cross_dimension_coherence: the five appraisals do not conflict within a trace
  or across the appraisals/emotion traces;
- appraisal_rating_alignment: reasoning supports the candidate's 21 ratings and
  those ratings are compatible with the author's structured reference;
- appraisal_emotion_linkage: reasoning supports the candidate emotion, while the
  author's emotion/intensity is the outcome reference. Do not reward a chain that
  is merely self-consistent with an incorrect outcome.

Return exactly one JSON object with this schema and no markdown:
{json.dumps(judge_output_example(), ensure_ascii=False, indent=2)}"""


def build_user_prompt(
    sample: Mapping[str, Any],
    prediction: Mapping[str, Any],
    correction: str = "",
) -> str:
    groups = rating_groups()
    gold_ratings = sample["reference"]["appraisal_ratings"]
    grouped_reference = {
        dimension: {field: gold_ratings[field] for field in fields}
        for dimension, fields in groups.items()
    }
    envelope = {
        "situation": sample["situation"],
        "structured_author_reference": {
            "appraisal_ratings_by_method_dimension": grouped_reference,
            "auxiliary_norm_value_ratings": {
                field: gold_ratings[field]
                for field, dimension in APPRAISAL_TO_METHOD_DIMENSION.items()
                if dimension == "auxiliary_norm_value"
            },
            "emotion": sample["reference"]["emotion"],
        },
        "candidate": {
            "appraisal_reasoning": prediction["appraisal_reasoning"],
            "appraisal_ratings": prediction["appraisal_ratings"],
            "emotion": prediction["emotion"],
        },
    }
    prompt = "Score every required field:\n" + json.dumps(
        envelope, ensure_ascii=False, indent=2
    )
    if correction:
        prompt += (
            "\n\nThe prior judgment was invalid. Return a fresh complete object. "
            f"Validation error: {correction[:1000]}"
        )
    return prompt


def normalize_reasoning_traces(value: Any) -> dict[str, dict[str, str]]:
    if not isinstance(value, dict):
        raise ValueError("prediction appraisal_reasoning must be an object")
    if set(value) == set(METHOD_APPRAISAL_DIMENSIONS):
        return {"joint": normalize_reasoning(value)}
    if set(value) == set(CHAIN_TARGETS):
        return {
            target: normalize_reasoning(value[target])
            for target in CHAIN_TARGETS
        }
    raise ValueError(
        "prediction appraisal_reasoning must be one five-dimensional trace or "
        f"exactly the two traces {list(CHAIN_TARGETS)}"
    )


def parse_score(value: Any, field: str) -> int:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 4:
        raise ValueError(f"{field} must be an integer in [1, 4]")
    return value


def parse_judgment(text: str) -> dict[str, Any]:
    payload = extract_json_object(text)
    expected_root = {"appraisals", "chain", "overall_feedback"}
    if set(payload) != expected_root:
        raise ValueError(
            f"Judge root keys mismatch: missing={sorted(expected_root-set(payload))}, "
            f"extra={sorted(set(payload)-expected_root)}"
        )
    raw_appraisals = payload["appraisals"]
    if not isinstance(raw_appraisals, dict) or set(raw_appraisals) != set(METHOD_APPRAISAL_DIMENSIONS):
        raise ValueError("Judge appraisals must contain exactly the five method dimensions")
    appraisals: dict[str, dict[str, int]] = {}
    for dimension in METHOD_APPRAISAL_DIMENSIONS:
        raw = raw_appraisals[dimension]
        if not isinstance(raw, dict) or set(raw) != set(RUBRIC_FIELDS):
            raise ValueError(f"Judge appraisals.{dimension} keys mismatch")
        appraisals[dimension] = {
            field: parse_score(raw[field], f"appraisals.{dimension}.{field}")
            for field in RUBRIC_FIELDS
        }
    raw_chain = payload["chain"]
    if not isinstance(raw_chain, dict) or set(raw_chain) != set(CHAIN_FIELDS):
        raise ValueError("Judge chain keys mismatch")
    feedback = payload["overall_feedback"]
    if not isinstance(feedback, str) or not feedback.strip():
        raise ValueError("overall_feedback must be non-empty text")
    return {
        "appraisals": appraisals,
        "chain": {
            field: parse_score(raw_chain[field], f"chain.{field}")
            for field in CHAIN_FIELDS
        },
        "overall_feedback": feedback.strip(),
    }


class JudgeClient:
    def __init__(self, args: argparse.Namespace, settings: Mapping[str, str]) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("API Judge requires the openai package") from exc
        kwargs: dict[str, Any] = {
            "api_key": settings["api_key"],
            "timeout": args.judge_timeout_seconds,
            "max_retries": args.judge_max_retries,
        }
        if settings["base_url"]:
            kwargs["base_url"] = settings["base_url"]
        self.client = OpenAI(**kwargs)
        self.args = args
        self.settings = settings

    def request(self, user_prompt: str) -> tuple[str, dict[str, Any]]:
        request: dict[str, Any] = {
            "model": self.settings["model"],
            "messages": [
                {"role": "system", "content": system_prompt()},
                {"role": "user", "content": user_prompt},
            ],
        }
        request[self.settings["token_parameter"]] = self.args.judge_max_tokens
        if self.settings["response_format"] == "json_object":
            request["response_format"] = {"type": "json_object"}
        extra_body: dict[str, Any] = {}
        if self.settings["provider"] == "glm" and self.args.judge_temperature == 0:
            extra_body["do_sample"] = False
        else:
            request["temperature"] = self.args.judge_temperature
        if self.settings["provider"] == "glm" and self.args.judge_thinking != "auto":
            extra_body["thinking"] = {"type": self.args.judge_thinking}
        if extra_body:
            request["extra_body"] = extra_body
        response = self.client.chat.completions.create(**request)
        content = message_content_text(response.choices[0].message.content)
        if not content:
            raise RuntimeError("Judge returned empty content")
        return content, response_to_dict(response)


def judge_one(
    client: JudgeClient,
    args: argparse.Namespace,
    sample: Mapping[str, Any],
    prediction: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    correction = ""
    attempts: list[dict[str, Any]] = []
    for attempt in range(args.judge_parse_retries + 1):
        try:
            content, raw_response = client.request(
                build_user_prompt(sample, prediction, correction)
            )
        except Exception as exc:
            attempts.append(
                {"attempt": attempt + 1, "valid": False, "error_type": "api", "error": str(exc)}
            )
            correction = ""
            continue
        try:
            judgment = parse_judgment(content)
        except (TypeError, ValueError) as exc:
            correction = str(exc)
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "valid": False,
                    "error_type": "parse_or_schema",
                    "error": correction,
                    "raw_output": content,
                    "raw_response": raw_response,
                }
            )
            continue
        attempts.append(
            {
                "attempt": attempt + 1,
                "valid": True,
                "raw_output": content,
                "raw_response": raw_response,
            }
        )
        return judgment, {"attempts": attempts}
    return None, {"attempts": attempts}


def aggregate_judgments(
    samples: list[Mapping[str, Any]],
    judgments: Mapping[str, Mapping[str, Any]],
    *,
    allow_incomplete: bool,
    missing_judgment_policy: str | None = None,
) -> dict[str, Any]:
    if missing_judgment_policy is None:
        policy = "skip" if allow_incomplete else "error"
    else:
        policy = missing_judgment_policy
    if policy not in MISSING_JUDGMENT_POLICIES:
        raise ValueError(
            "--missing_judgment_policy must be one of "
            + ", ".join(MISSING_JUDGMENT_POLICIES)
        )
    missing = [sample["id"] for sample in samples if sample["id"] not in judgments]
    if missing and policy == "error":
        raise ValueError(
            f"Missing {len(missing)} judgment(s). Use "
            "--missing_judgment_policy penalize to keep them in the denominator, "
            "or --allow_incomplete/--missing_judgment_policy skip for debugging."
        )
    evaluated: list[Mapping[str, Any]] = []
    penalized_missing: list[str] = []
    skipped_missing: list[str] = []
    for sample in samples:
        sample_id = sample["id"]
        record = judgments.get(sample_id)
        if record is not None:
            evaluated.append(record["judgment"])
        elif policy == "penalize":
            evaluated.append(penalized_judgment())
            penalized_missing.append(sample_id)
        else:
            skipped_missing.append(sample_id)
    valid_judgments = len(samples) - len(missing)
    if not evaluated:
        raise ValueError("No valid judgments are available")
    per_dimension: dict[str, dict[str, float]] = {}
    all_appraisal_scores: list[float] = []
    for dimension in METHOD_APPRAISAL_DIMENSIONS:
        per_dimension[dimension] = {}
        for field in RUBRIC_FIELDS:
            values = [float(item["appraisals"][dimension][field]) for item in evaluated]
            per_dimension[dimension][field] = sum(values) / len(values)
            all_appraisal_scores.extend(values)
    chain = {
        field: sum(float(item["chain"][field]) for item in evaluated) / len(evaluated)
        for field in CHAIN_FIELDS
    }
    return {
        "coverage": {
            "selected_samples": len(samples),
            "valid_judgments": valid_judgments,
            "evaluated_judgments": len(evaluated),
            "missing_judgments": len(missing),
            "penalized_missing_judgments": len(penalized_missing),
            "skipped_missing_judgments": len(skipped_missing),
            "coverage": valid_judgments / len(samples),
            "metric_sample_coverage": len(evaluated) / len(samples),
            "allow_incomplete": allow_incomplete,
            "missing_judgment_policy": policy,
            "missing_sample_ids": missing,
            "penalized_missing_sample_ids": penalized_missing,
            "penalty_defaults": (
                {"all_rubric_scores": 1}
                if policy == "penalize"
                else {}
            ),
        },
        "mean_appraisal_score": sum(all_appraisal_scores) / len(all_appraisal_scores),
        "appraisals": per_dimension,
        "chain": chain,
        "scale": {"minimum": 1, "maximum": 4, "higher_is_better": True},
    }


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def main() -> int:
    args = build_parser().parse_args()
    if args.judge_parse_retries < 0 or args.judge_max_concurrency <= 0:
        raise ValueError("Judge retries must be non-negative and concurrency positive")
    evaluation = load_json(args.eval_file)
    samples = evaluation.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Evaluation JSON has no samples")
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max_samples must be positive")
        if args.max_samples < len(samples):
            samples = random.Random(args.sample_seed).sample(
                samples, args.max_samples
            )
    predictions = record_index(load_jsonl(args.predictions_file), "prediction")
    for sample in samples:
        prediction = predictions.get(sample["id"])
        if prediction is None:
            continue
        prediction["appraisal_reasoning"] = normalize_reasoning_traces(
            prediction.get("appraisal_reasoning")
        )
        normalize_ratings(prediction.get("appraisal_ratings"))
        normalize_emotion(prediction.get("emotion"))

    judgments_path, invalid_path, results_path = judge_paths(args)
    if args.mode == "metrics" and args.overwrite_judgments:
        raise ValueError("--overwrite_judgments is not valid with --mode metrics")
    if args.overwrite_judgments:
        for path in (judgments_path, invalid_path):
            if path.exists():
                path.unlink()
    existing = record_index(load_jsonl(judgments_path), "judgment")
    pending = [
        sample
        for sample in samples
        if sample["id"] in predictions and sample["id"] not in existing
    ]
    settings: Mapping[str, str] | None = None
    if args.mode == "run" and pending:
        settings = resolve_judge_settings(args)
        incompatible = [
            sample_id
            for sample_id, record in existing.items()
            if record.get("judge")
            != {"provider": settings["provider"], "model": settings["model"]}
        ]
        if incompatible:
            raise ValueError(
                "Existing judgments were produced by another provider/model; "
                "use a separate file or --overwrite_judgments"
            )
        client = JudgeClient(args, settings)
        print(
            f"[judge] selected={len(samples)} existing={len(existing)} "
            f"pending={len(pending)} concurrency={args.judge_max_concurrency}"
        )
        with ThreadPoolExecutor(max_workers=args.judge_max_concurrency) as executor:
            future_to_sample = {
                executor.submit(
                    judge_one, client, args, sample, predictions[sample["id"]]
                ): sample
                for sample in pending
            }
            for future in as_completed(future_to_sample):
                sample = future_to_sample[future]
                try:
                    judgment, metadata = future.result()
                except Exception as exc:
                    judgment = None
                    metadata = {"fatal_error": str(exc)}
                if judgment is None:
                    append_jsonl(
                        invalid_path,
                        {
                            "sample_id": sample["id"],
                            "error": "all Judge API/parse attempts failed",
                            **metadata,
                        },
                    )
                    continue
                record = {
                    "sample_id": sample["id"],
                    "judgment": judgment,
                    "judge": {
                        "provider": settings["provider"],
                        "model": settings["model"],
                    },
                    **metadata,
                }
                append_jsonl(judgments_path, record)
                existing[sample["id"]] = record
    elif args.mode == "run":
        print(
            f"[judge] selected={len(samples)} existing={len(existing)} pending=0"
        )
    report = aggregate_judgments(
        samples,
        existing,
        allow_incomplete=args.allow_incomplete,
        missing_judgment_policy=effective_missing_judgment_policy(args),
    )
    report["files"] = {
        "eval_file": str(args.eval_file),
        "predictions_file": str(args.predictions_file),
        "judgments_file": str(judgments_path),
        "invalid_file": str(invalid_path),
    }
    judge_identities = sorted(
        {
            json.dumps(record.get("judge", {}), sort_keys=True)
            for record in existing.values()
            if record.get("judge")
        }
    )
    report["judge"] = (
        {
            "provider": settings["provider"],
            "model": settings["model"],
        }
        if settings is not None
        else {
            "saved_identities": [json.loads(value) for value in judge_identities]
        }
    )
    write_json_atomic(results_path, report)
    print(json.dumps(report["coverage"], ensure_ascii=False, indent=2))
    print(f"[judge-results] {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
