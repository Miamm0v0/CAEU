#!/usr/bin/env python3
"""Compute crowd-enVent metrics from an existing Policy prediction JSONL.

No Policy client is created in this entry point. It only reads saved
predictions and writes the aggregate metrics JSON.
"""

from __future__ import annotations

from evaluate_policy_crowd_envent import build_parser as build_shared_parser
from evaluate_policy_crowd_envent import execute


def build_parser():
    return build_shared_parser(fixed_mode="metrics")


def main() -> int:
    return execute(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
