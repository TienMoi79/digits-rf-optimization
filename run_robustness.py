"""Five-seed development-only comparison and TPE startup ablation."""
import argparse
from pathlib import Path

from src.common import ROOT
from src.robustness import load_plan, run_robustness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/robustness.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/robustness")
    args = parser.parse_args()
    output = run_robustness(load_plan(args.config), args.output)
    from src.robustness_analysis import analyze_robustness
    from src.robustness_validation import validate_robustness
    analyze_robustness(output)
    validate_robustness(output)


if __name__ == "__main__":
    main()
