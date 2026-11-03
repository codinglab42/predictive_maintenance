#!/usr/bin/env python3
"""Estimate bearing failure risk from one FEMTO vibration acquisition."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import json
from pathlib import Path

import numpy as np

from src.femto_data import FEATURE_NAMES, read_signal_features
from src.ml_core_loader import load_ml_core
from train_predictive_maintenance import METADATA_PATH, MODEL_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--signal-csv",
        type=Path,
        required=True,
        help="path to one FEMTO acc_*.csv sample (native six-column or two-channel)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="override the alert threshold saved during training",
    )
    args = parser.parse_args()
    if args.threshold is not None and not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    return args


def main() -> None:
    args = parse_args()
    if not MODEL_PATH.is_file() or not METADATA_PATH.is_file():
        raise FileNotFoundError("model not found; run train_predictive_maintenance.py first")

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if metadata.get("format_version") != 2:
        raise ValueError("saved model is not a FEMTO model; retrain with the current pipeline")
    if metadata.get("feature_names") != FEATURE_NAMES:
        raise ValueError("saved feature schema does not match this version of the pipeline")

    means = np.asarray(metadata.get("means"), dtype=np.float64)
    scales = np.asarray(metadata.get("scales"), dtype=np.float64)
    if means.shape != (len(FEATURE_NAMES),) or scales.shape != means.shape:
        raise ValueError("saved preprocessing parameters are malformed; retrain the model")
    if not np.isfinite(means).all() or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("saved preprocessing parameters are invalid; retrain the model")

    threshold = metadata["threshold"] if args.threshold is None else args.threshold
    features = read_signal_features(args.signal_csv).reshape(1, -1)
    features = np.asfortranarray((features - means) / scales)
    model = load_ml_core().LogisticRegression()
    model.load(str(MODEL_PATH))
    probability = float(np.asarray(model.predict(features)).reshape(-1)[0])

    print(
        f"Estimated failure probability within {metadata['horizon_acquisitions']} "
        f"acquisitions: {probability:.1%}"
    )
    print("Alert: YES" if probability >= threshold else "Alert: NO")
    print(f"Threshold: {threshold:.1%}")


if __name__ == "__main__":
    main()
