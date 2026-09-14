#!/usr/bin/env python3
"""Continue three selectable appraisal-profile branches to CAREBench ratings."""

from counterfactual_prefix_transformers import main


if __name__ == "__main__":
    raise SystemExit(main(default_outcome="ratings"))
