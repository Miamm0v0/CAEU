#!/usr/bin/env python3
"""Score blinded candidates with the exact GRPO training Judge (v7)."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train_grpo.api_judge import (  # noqa: E402
    APIJudge,
    JudgeCache,
    resolve_judge_settings,
)
from train_grpo.parsing import (  # noqa: E402
    aggregate_judgment,
    parse_judge_output,
    parse_policy_output,
)
from train_grpo.spec import (  # noqa: E402
    RUBRIC_VERSION,
    build_judge_system_prompt,
    resolve_appraisal_dimensions,
)


DEFAULT_DIMENSIONS = resolve_appraisal_dimensions(
    "relevance,epistemic,goal_congruence,agency_accountability,"
    "control_coping_potential"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the exact v7 GRPO API Judge on blinded candidates"
    )
    parser.add_argument("--master_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--appraisal_dimensions",
        type=resolve_appraisal_dimensions,
        default=list(DEFAULT_DIMENSIONS),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--retry_failures",
        action="store_true",
        help="On resume, retry prior error rows while retaining their audit trail",
    )
    parser.add_argument("--preview_only", action="store_true")

    parser.add_argument(
        "--judge_provider", choices=["openai", "glm"], default="openai"
    )
    parser.add_argument("--judge_model", type=str, default="")
    parser.add_argument("--judge_config", type=str, default="")
    parser.add_argument("--judge_provider_section", type=str, default="")
    parser.add_argument("--judge_api_key_env", type=str, default="")
    parser.add_argument("--judge_base_url", type=str, default="")
    parser.add_argument(
        "--judge_response_format",
        choices=["auto", "json_schema", "json_object", "text"],
        default="auto",
    )
    parser.add_argument(
        "--judge_token_parameter",
        choices=["auto", "max_completion_tokens", "max_tokens"],
        default="auto",
    )
    parser.add_argument("--judge_max_tokens", type=int, default=5000)
    parser.add_argument("--judge_temperature", type=float, default=0.0)
    parser.add_argument("--judge_omit_temperature", action="store_true")
    parser.add_argument(
        "--judge_thinking",
        choices=["auto", "enabled", "disabled"],
        default="disabled",
    )
    parser.add_argument("--judge_timeout_seconds", type=float, default=180.0)
    parser.add_argument("--judge_max_retries", type=int, default=3)
    parser.add_argument("--judge_parse_retries", type=int, default=1)
    parser.add_argument("--judge_max_concurrency", type=int, default=8)
    parser.add_argument("--judge_cache_path", type=Path, default=None)
    parser.add_argument("--no_judge_cache", action="store_true")
    parser.add_argument("--log_judge_raw_outputs", action="store_true")
    return parser


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(value)
    return rows


def append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def candidate_sha(candidate: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        candidate, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def same_number(left: Any, right: Any) -> bool:
    return isinstance(left, (int, float)) and isinstance(
        right, (int, float)
    ) and math.isclose(float(left), float(right), abs_tol=1e-12)


async def score_row(
    judge: APIJudge,
    row: Mapping[str, Any],
    dimensions: Sequence[str],
    prompt_sha: str,
) -> dict[str, Any]:
    candidate_id = str(row.get("candidate_id", ""))
    base = {
        "schema_version": "judge-human-train-judge-v1",
        "rubric_version": RUBRIC_VERSION,
        "judge_prompt_sha256": prompt_sha,
        "pair_id": row.get("pair_id"),
        "candidate_id": candidate_id,
        "candidate_slot": row.get("candidate_slot"),
        "source_sample_id": row.get("source_sample_id"),
    }
    try:
        situation = row.get("situation")
        if not isinstance(situation, str) or not situation.strip():
            raise ValueError("situation must be a non-empty string")
        candidate = parse_policy_output(
            json.dumps(row.get("candidate"), ensure_ascii=False), dimensions
        )
        result = await judge.score_valid_candidate(
            sample_id=candidate_id,
            situation=situation.strip(),
            candidate=candidate,
            # Deliberately no CAREBench gold/reference in this study.
            reference={},
        )
        # Re-run the exact public parser and aggregator so schema/reward drift
        # fails loudly rather than silently changing the study definition.
        judgment = parse_judge_output(
            json.dumps(result["judgment"], ensure_ascii=False), dimensions
        )
        aggregate = aggregate_judgment(judgment, dimensions)
        for name in (
            "appraisal_reward",
            "coherence_reward",
            "transition_reward",
            "raw_transition_reward",
        ):
            if not same_number(result.get(name), aggregate[name]):
                raise RuntimeError(f"APIJudge/{name} aggregation mismatch")
        return {
            **base,
            "status": "ok",
            "candidate_sha256": candidate_sha(candidate),
            "judgment": judgment,
            **aggregate,
            "request_id": result.get("request_id"),
            "cache_hit": bool(result.get("cache_hit", False)),
        }
    except Exception as exc:
        return {**base, "status": "error", "error": str(exc)[:2000]}


async def run(args: argparse.Namespace) -> int:
    dimensions = list(args.appraisal_dimensions)
    rows = read_jsonl(args.master_file)
    if not rows:
        raise ValueError("master file is empty")
    ids = [str(row.get("candidate_id", "")) for row in rows]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("master candidate_id values must be non-empty and unique")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "judge_results.jsonl"
    summary_path = args.output_dir / "judge_summary.json"
    if args.overwrite:
        if results_path.exists():
            results_path.unlink()
        if summary_path.exists():
            summary_path.unlink()

    prior_rows = read_jsonl(results_path) if results_path.is_file() else []
    if args.retry_failures:
        completed = {
            str(row.get("candidate_id"))
            for row in prior_rows
            if row.get("status") == "ok"
        }
    else:
        completed = {str(row.get("candidate_id")) for row in prior_rows}
    pending = [row for row in rows if str(row["candidate_id"]) not in completed]

    prompt = build_judge_system_prompt(dimensions)
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if args.preview_only:
        preview = {
            "rubric_version": RUBRIC_VERSION,
            "appraisal_dimensions": dimensions,
            "judge_prompt_sha256": prompt_sha,
            "judge_system_prompt": prompt,
            "candidate_count": len(rows),
            "pending_count": len(pending),
            "gold_reference_used": False,
        }
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return 0

    settings = resolve_judge_settings(args)
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError("Install the openai package to run the API Judge") from exc
    cache: JudgeCache | None = None
    if not args.no_judge_cache:
        cache_path = args.judge_cache_path or (
            args.output_dir / "train_judge_cache.sqlite3"
        )
        cache = JudgeCache(cache_path)
    judge = APIJudge(args, settings, AsyncOpenAI, cache)
    if judge.judge_system_prompt != prompt:
        raise RuntimeError("APIJudge prompt differs from build_judge_system_prompt")

    batch_size = max(1, args.judge_max_concurrency * 4)
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        results = await asyncio.gather(
            *[
                score_row(judge, row, dimensions, prompt_sha)
                for row in batch
            ]
        )
        append_jsonl(results_path, results)
        counts = Counter(row["status"] for row in results)
        done = min(start + len(batch), len(pending))
        print(
            f"Judge progress {done}/{len(pending)}: "
            f"ok={counts.get('ok', 0)} error={counts.get('error', 0)}",
            flush=True,
        )

    all_rows = read_jsonl(results_path) if results_path.is_file() else []
    latest: dict[str, dict[str, Any]] = {}
    for row in all_rows:
        latest[str(row.get("candidate_id"))] = row
    status_counts = Counter(row.get("status", "unknown") for row in latest.values())
    errors = [
        {
            "candidate_id": row.get("candidate_id"),
            "pair_id": row.get("pair_id"),
            "error": row.get("error"),
        }
        for row in latest.values()
        if row.get("status") != "ok"
    ]
    write_json(
        summary_path,
        {
            "rubric_version": RUBRIC_VERSION,
            "appraisal_dimensions": dimensions,
            "judge_prompt_sha256": prompt_sha,
            "judge_provider": settings["provider"],
            "judge_model": settings["model"],
            "candidate_count": len(rows),
            "status_counts": dict(status_counts),
            "gold_reference_used": False,
            "errors": errors,
        },
    )
    print(f"Saved Judge scores to {results_path}")
    return 0


def main() -> int:
    return asyncio.run(run(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
