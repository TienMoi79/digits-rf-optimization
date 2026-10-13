"""Run the full experiment. Existing results are never overwritten."""
import argparse
from pathlib import Path

from src.common import ROOT, load_config
from src.experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    output = run_experiment(load_config(args.config), args.output)
    from src.analysis import analyze
    analyze(output)
    from src.validation import validate_results
    validate_results(output)


if __name__ == "__main__":
    main()
