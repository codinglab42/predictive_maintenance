#!/usr/bin/env python3
"""Estimate bearing failure risk from one FEMTO vibration acquisition."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.femto_data import read_signal_features
from src.model_artifacts import METADATA_PATH, MODEL_PATH
from src.predictor import load_prediction_config, predict_features


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
    config = load_prediction_config(MODEL_PATH, METADATA_PATH)
    if not config.checksum_verified:
        logging.warning(
            "model metadata has no checksum; artifact pairing cannot be verified. "
            "Retrain the model to record it."
        )
    features = read_signal_features(args.signal_csv)
    result = predict_features(features, config, args.threshold)

    print(
        f"Estimated failure probability within {result.horizon_acquisitions} "
        f"acquisitions: {result.probability:.1%}"
    )
    print("Alert: YES" if result.probability >= result.threshold else "Alert: NO")
    print(f"Threshold: {result.threshold:.1%}")
    print(f"Threshold source: {result.threshold_source}")
    print(f"Signal: {args.signal_csv.expanduser().resolve()}")
    print(f"Model: {config.model_path}")
    print(f"Model SHA-256: {result.model_sha256}")
    checksum_status = "verified" if result.checksum_verified else "not verifiable (legacy metadata)"
    print(f"Checksum status: {checksum_status}")


if __name__ == "__main__":
    main()
