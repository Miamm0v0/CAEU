#!/usr/bin/env python3
"""Continue three selectable appraisal-profile branches to CAREBench emotion."""

from counterfactual_prefix_transformers import main


if __name__ == "__main__":
    raise SystemExit(main(default_outcome="emotion"))
