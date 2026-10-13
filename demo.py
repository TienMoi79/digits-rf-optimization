"""Demo prediction using the saved model and a built-in Digits sample."""
import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_digits

from src.common import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=123, help="Dataset index, 0..1796")
    parser.add_argument("--pixels", type=Path, help="Optional CSV containing exactly 8x8 pixel values in [0,16]")
    parser.add_argument("--model", type=Path, default=ROOT / "results/models/selected.joblib")
    args = parser.parse_args()
    if not args.model.exists():
        parser.error("Model not found. Run run_experiment.py first or use --model.")
    if args.pixels:
        pixels = np.loadtxt(args.pixels, delimiter=",")
        if pixels.shape != (8,8) or not np.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 16:
            parser.error("Pixel CSV must be a finite 8x8 matrix with values from 0 to 16.")
        label = None
    else:
        digits = load_digits()
        if not 0 <= args.index < len(digits.target):
            parser.error("index must be from 0 to 1796.")
        pixels, label = digits.images[args.index], int(digits.target[args.index])
    model = joblib.load(args.model)
    x = pixels.reshape(1, -1)
    probs = model.predict_proba(x)[0]
    prediction = int(model.predict(x)[0])
    shades = " .:-=+*#%@"
    print("\nInput digit (8x8 pixels):")
    for row in pixels:
        print("".join(shades[int(round(value/16*(len(shades)-1)))]*2 for value in row))
    print(f"\nPrediction: {prediction}")
    if label is not None:
        print(f"Dataset label: {label}")
    print("Top class probabilities (RF scores, not calibrated confidence):")
    for i in np.argsort(probs)[-3:][::-1]:
        print(f"  digit {model.classes_[i]}: {probs[i]:.4f}")


if __name__ == "__main__":
    main()
