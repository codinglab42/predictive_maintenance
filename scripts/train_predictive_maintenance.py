#!/usr/bin/env python3
"""Train and evaluate a bearing-failure baseline on NASA FEMTO/PRONOSTIA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import json
import os
import sys
from pathlib import Path

import numpy as np

from src.femto_data import (
    DEFAULT_DATA_DIR,
    FEATURE_NAMES,
    build_labeled_dataset,
    standardize,
)
from src.ml_core_loader import load_ml_core


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "predictive_maintenance_model.bin"
METADATA_PATH = MODEL_DIR / "preprocessing.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="directory containing the extracted FEMTO/PRONOSTIA dataset",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        help="directory for extracted features (default: processed/ beside --data-dir)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=20,
        help="positive label for samples within this many acquisitions of failure",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="probability threshold for raising an alert",
    )
    parser.add_argument(
        "--test-bearing",
        action="append",
        dest="test_bearings",
        help="bearing ID to hold out for evaluation; repeat to hold out multiple bearings",
    )
    args = parser.parse_args()
    if args.horizon < 1:
        parser.error("--horizon must be a positive number of acquisitions")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    return args


def save_processed_data(
    output_dir: Path,
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    sample_indices: np.ndarray,
    horizon: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "femto_features.npz"
    temporary_path = output_path.with_suffix(".npz.tmp")
    with temporary_path.open("wb") as output:
        np.savez_compressed(
            output,
            features=features,
            labels=labels,
            bearing_ids=groups,
            sample_indices=sample_indices,
        )
    os.replace(temporary_path, output_path)
    metadata_path = output_dir / "femto_features.json"
    metadata_temporary_path = metadata_path.with_suffix(".json.tmp")
    metadata_temporary_path.write_text(
        json.dumps(
            {
                "source": "NASA FEMTO/PRONOSTIA",
                "feature_names": FEATURE_NAMES,
                "label": f"positive in final {horizon} acquisitions including final sample",
                "samples": int(labels.size),
                "bearings": sorted(set(groups.tolist())),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(metadata_temporary_path, metadata_path)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    processed_dir = (
        args.processed_dir.expanduser().resolve()
        if args.processed_dir is not None
        else data_dir.parent.parent / "processed"
    )
    print("Fase 1/5: ricerca dei campioni ed estrazione delle feature", file=sys.stderr, flush=True)
    last_progress_bearing: str | None = None

    def report_feature_progress(processed: int, total: int, bearing_id: str) -> None:
        nonlocal last_progress_bearing
        if processed % 100 == 0 or processed == total or bearing_id != last_progress_bearing:
            percentage = processed / total * 100
            print(
                f"\r  Campioni elaborati: {processed:,}/{total:,} "
                f"({percentage:.1f}%) | {bearing_id}",
                end="\n" if processed == total else "",
                file=sys.stderr,
                flush=True,
            )
        last_progress_bearing = bearing_id

    features, labels, groups, sample_indices = build_labeled_dataset(
        data_dir,
        args.horizon,
        progress_callback=report_feature_progress,
    )
    print("Fase 2/5: suddivisione dei dati e standardizzazione", file=sys.stderr, flush=True)
    bearings = sorted(set(groups.tolist()))
    test_bearings = set(args.test_bearings or [bearings[-1]])
    unknown_bearings = test_bearings.difference(bearings)
    if unknown_bearings:
        raise ValueError(f"unknown test bearing(s): {', '.join(sorted(unknown_bearings))}")
    train_mask = ~np.isin(groups, list(test_bearings))
    test_mask = ~train_mask
    if not train_mask.any() or not test_mask.any():
        raise ValueError("the split must contain at least one training and one test bearing")
    if np.unique(labels[train_mask]).size != 2:
        raise ValueError(
            "training data must contain both failure classes; adjust --horizon or --test-bearing"
        )

    train_features, test_features, means, scales = standardize(
        features[train_mask],
        features[test_mask],
    )
    print("Fase 3/5: addestramento del modello", file=sys.stderr, flush=True)
    model = load_ml_core().LogisticRegression(0.05, 1500, 0.01, 1e-4, True)
    model.fit(train_features, labels[train_mask].astype(np.float64))

    print("Fase 4/5: valutazione sul cuscinetto di test", file=sys.stderr, flush=True)
    test_labels = labels[test_mask].astype(np.float64)
    probabilities = np.asarray(model.predict(test_features), dtype=np.float64).reshape(-1)
    predictions = (probabilities >= args.threshold).astype(np.int64)
    confusion = np.asarray(model.confusion_matrix(test_features, test_labels, args.threshold))
    precision, recall, f1 = map(
        float,
        model.precision_recall_f1(test_features, test_labels, args.threshold),
    )
    accuracy = float(np.mean(predictions == test_labels))

    print("Fase 5/5: salvataggio di feature, modello e preprocessing", file=sys.stderr, flush=True)
    save_processed_data(processed_dir, features, labels, groups, sample_indices, args.horizon)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_PATH))
    metadata = {
        "format_version": 2,
        "source": "NASA FEMTO/PRONOSTIA",
        "feature_names": FEATURE_NAMES,
        "means": means.tolist(),
        "scales": scales.tolist(),
        "threshold": args.threshold,
        "horizon_acquisitions": args.horizon,
        "training_bearings": sorted(set(groups[train_mask].tolist())),
        "test_bearings": sorted(test_bearings),
        "training_samples": int(train_mask.sum()),
        "test_samples": int(test_mask.sum()),
    }
    metadata_temporary_path = METADATA_PATH.with_suffix(".json.tmp")
    metadata_temporary_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(metadata_temporary_path, METADATA_PATH)

    print("NASA FEMTO/PRONOSTIA bearing-failure baseline")
    print(f"Bearings: {len(bearings)} | Failure horizon: {args.horizon} acquisitions")
    print(f"Training bearings: {', '.join(metadata['training_bearings'])}")
    print(f"Test bearings: {', '.join(metadata['test_bearings'])}")
    print(f"Train samples: {int(train_mask.sum())} | Test samples: {int(test_mask.sum())}")
    print(
        f"Accuracy: {accuracy:.1%} | Precision: {precision:.1%} | "
        f"Recall: {recall:.1%} | F1: {f1:.1%}"
    )
    print(
        "Confusion matrix (actual rows, predicted columns): "
        f"TN={int(confusion[0, 0])}, FP={int(confusion[0, 1])}, "
        f"FN={int(confusion[1, 0])}, TP={int(confusion[1, 1])}"
    )
    print(f"Saved processed features: {processed_dir / 'femto_features.npz'}")
    print(f"Saved model: {MODEL_PATH}")
    print("Experimental benchmark metrics are not evidence of industrial performance.")


if __name__ == "__main__":
    main()
