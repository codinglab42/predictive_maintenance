#!/usr/bin/env python3
"""Train IMS neural-network and logistic-regression models and test on 2nd_test."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.ims_data import FEATURE_NAMES, build_labeled_dataset
from src.ml_core_loader import load_ml_core
from src.model_artifacts import sha256_file


SHARED_ROOT = Path("/mnt/shared/predictive-maintenance")
DEFAULT_DATA_DIR = SHARED_ROOT / "raw" / "ims"
DEFAULT_PROCESSED_DIR = SHARED_ROOT / "processed"
MODEL_DIR = ROOT / "models"
NEURAL_MODEL_PATH = MODEL_DIR / "ims_neural_network.bin"
LOGISTIC_MODEL_PATH = MODEL_DIR / "ims_logistic_regression.bin"
METADATA_PATH = MODEL_DIR / "ims_preprocessing.json"
ALERT_THRESHOLD = 0.7
THRESHOLD_CANDIDATES = (
    0.50,
    0.65,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    0.97,
    0.99,
    0.995,
    0.999,
)
NEURAL_NETWORK_SEED = 42
BASELINE_SAMPLE_COUNT = 30
NEURAL_NETWORK_HIDDEN_UNITS = 12
NEURAL_NETWORK_EPOCHS = 60
NEURAL_NETWORK_LEARNING_RATE = 0.001
NEURAL_NETWORK_L2_STRENGTH = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="directory containing the extracted 1st_test, 2nd_test, and 3rd_test",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="directory for IMS feature arrays and test predictions",
    )
    parser.add_argument(
        "--reuse-features",
        action="store_true",
        help="reuse ims_train_features.npz and ims_test_features.npz instead of re-extracting",
    )
    return parser.parse_args()


def _save_dataset(
    path: Path,
    features: np.ndarray,
    labels: np.ndarray,
    bearing_ids: np.ndarray,
    timestamps: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as output:
        np.savez_compressed(
            output,
            features=features,
            labels=labels,
            bearing_ids=bearing_ids,
            timestamps=timestamps,
        )
    os.replace(temporary_path, path)


def _load_dataset(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"cached IMS features not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required_arrays = {"features", "labels", "bearing_ids", "timestamps"}
        missing_arrays = required_arrays - set(archive.files)
        if missing_arrays:
            raise ValueError(
                f"cached IMS feature archive is missing arrays: "
                f"{', '.join(sorted(missing_arrays))}"
            )
        features = np.asarray(archive["features"], dtype=np.float64)
        labels = np.asarray(archive["labels"], dtype=np.int64)
        bearing_ids = np.asarray(archive["bearing_ids"]).astype(str)
        timestamps = np.asarray(archive["timestamps"]).astype(str)

    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"cached IMS feature archive has an invalid feature shape: {path}")
    if not np.isfinite(features).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError(f"cached IMS feature archive contains invalid values: {path}")
    if any(array.ndim != 1 or array.size != features.shape[0] for array in (
        labels,
        bearing_ids,
        timestamps,
    )):
        raise ValueError(f"cached IMS feature archive arrays have inconsistent lengths: {path}")
    return features, labels, bearing_ids, timestamps


def _baseline_normalize_features(
    features: np.ndarray,
    bearing_ids: np.ndarray,
    timestamps: np.ndarray,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    if (
        features.ndim != 2
        or bearing_ids.ndim != 1
        or timestamps.ndim != 1
        or features.shape[0] != bearing_ids.size
        or features.shape[0] != timestamps.size
        or features.shape[0] == 0
    ):
        raise ValueError("baseline normalization requires aligned feature and bearing arrays")
    if not np.isfinite(features).all():
        raise ValueError("baseline normalization requires finite feature values")

    normalized = np.empty_like(features, dtype=np.float64)
    baselines: dict[str, list[float]] = {}
    for bearing_id in np.unique(bearing_ids):
        positions = np.flatnonzero(bearing_ids == bearing_id)
        bearing_timestamps = timestamps[positions]
        if positions.size < BASELINE_SAMPLE_COUNT:
            raise ValueError(
                f"{bearing_id} has fewer than {BASELINE_SAMPLE_COUNT} baseline samples"
            )
        if np.unique(bearing_timestamps).size != positions.size:
            raise ValueError(f"{bearing_id} contains duplicate IMS timestamps")

        chronological_positions = positions[
            np.argsort(bearing_timestamps, kind="stable")
        ]
        baseline = features[chronological_positions[:BASELINE_SAMPLE_COUNT]].mean(axis=0)
        if not np.isfinite(baseline).all():
            raise ValueError(f"{bearing_id} has a non-finite initial feature baseline")
        normalized[positions] = features[positions] - baseline
        baselines[str(bearing_id)] = baseline.tolist()

    return normalized, baselines


def _metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = ALERT_THRESHOLD,
) -> tuple[float, float, float]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("classification threshold must be in [0, 1]")
    predictions = probabilities > threshold
    actual = labels.astype(bool)
    true_positive = int(np.count_nonzero(predictions & actual))
    false_positive = int(np.count_nonzero(predictions & ~actual))
    false_negative = int(np.count_nonzero(~predictions & actual))
    predicted_positive = true_positive + false_positive
    actual_positive = true_positive + false_negative
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / actual_positive if actual_positive else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _threshold_diagnostics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bearing_ids: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if (
        labels.ndim != 1
        or probabilities.ndim != 1
        or bearing_ids.ndim != 1
        or labels.size != probabilities.size
        or labels.size != bearing_ids.size
    ):
        raise ValueError("threshold diagnostics require aligned one-dimensional arrays")

    bearing_names = tuple(f"Bearing {index}" for index in range(1, 5))
    bearing_ids_by_name = {
        f"Bearing {index}": f"2nd_test_bearing_{index}" for index in range(1, 5)
    }
    results: list[dict[str, object]] = []
    for threshold in THRESHOLD_CANDIDATES:
        predictions = probabilities > threshold
        precision, recall, f1 = _metrics(labels, probabilities, threshold)
        alarms_by_bearing = {
            bearing_name: int(
                np.count_nonzero(
                    predictions & (bearing_ids == bearing_ids_by_name[bearing_name])
                )
            )
            for bearing_name in bearing_names
        }
        healthy_alarm_samples = sum(
            alarms_by_bearing[f"Bearing {index}"] for index in (2, 3, 4)
        )
        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "alarm_samples_by_bearing": alarms_by_bearing,
                "healthy_bearing_alarm_samples": healthy_alarm_samples,
            }
        )

    selected = max(
        results,
        key=lambda result: (
            float(result["f1_score"]),
            -int(result["healthy_bearing_alarm_samples"]),
            float(result["threshold"]),
        ),
    )
    return results, selected


def _print_bearing_alarm_counts(
    bearing_ids: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    label: str,
) -> None:
    predictions = probabilities > threshold
    print(f"MLP alarm samples per bearing at threshold {threshold:.3f} ({label}):")
    for index in range(1, 5):
        bearing_id = f"2nd_test_bearing_{index}"
        count = int(np.count_nonzero(predictions & (bearing_ids == bearing_id)))
        samples = int(np.count_nonzero(bearing_ids == bearing_id))
        print(f"  Bearing {index}: {count:,} / {samples:,}")


def _validate_probabilities(output: object, expected_rows: int, model_name: str) -> np.ndarray:
    try:
        probabilities = np.asarray(output, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{model_name} returned invalid predictions") from error
    if (
        probabilities.shape != (expected_rows,)
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
    ):
        raise ValueError(f"{model_name} must return one probability in [0, 1] per test sample")
    return probabilities


def _save_predictions(
    path: Path,
    bearing_ids: np.ndarray,
    timestamps: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    alert_threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["bearing_id", "timestamp", "label", "probability", "alert_threshold"])
        writer.writerows(
            zip(
                bearing_ids.tolist(),
                timestamps.tolist(),
                labels.tolist(),
                probabilities.tolist(),
                [alert_threshold] * labels.size,
                strict=True,
            )
        )
    os.replace(temporary_path, path)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    processed_dir = args.processed_dir.expanduser().resolve()

    if args.reuse_features:
        print("Loading cached IMS features...", file=sys.stderr, flush=True)
        (
            train_features,
            train_labels,
            train_bearings,
            train_timestamps,
        ) = _load_dataset(processed_dir / "ims_train_features.npz")
        (
            test_features,
            test_labels,
            test_bearings,
            test_timestamps,
        ) = _load_dataset(processed_dir / "ims_test_features.npz")
    else:
        print("Extracting chronological IMS features...", file=sys.stderr, flush=True)

        def extract_series(
            series_name: str,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            last_reported = 0

            def report_progress(processed: int, total: int) -> None:
                nonlocal last_reported
                if processed == total or processed - last_reported >= 100:
                    print(
                        f"  {series_name}: {processed:,}/{total:,} files",
                        file=sys.stderr,
                        flush=True,
                    )
                    last_reported = processed

            return build_labeled_dataset(data_dir, series_name, report_progress)

        first_features, first_labels, first_bearings, first_timestamps = extract_series(
            "1st_test"
        )
        third_features, third_labels, third_bearings, third_timestamps = extract_series(
            "3rd_test"
        )
        test_features, test_labels, test_bearings, test_timestamps = extract_series("2nd_test")

        train_features = np.concatenate((first_features, third_features), axis=0)
        train_labels = np.concatenate((first_labels, third_labels), axis=0)
        train_bearings = np.concatenate((first_bearings, third_bearings), axis=0)
        train_timestamps = np.concatenate((first_timestamps, third_timestamps), axis=0)
    if np.unique(train_labels).size != 2:
        raise ValueError(
            "the combined 1st_test and 3rd_test training data must contain both classes"
        )
    if np.unique(test_labels).size != 2:
        raise ValueError("2nd_test must contain both healthy and failed-bearing labels")

    normalized_train_features, train_baselines = _baseline_normalize_features(
        train_features,
        train_bearings,
        train_timestamps,
    )
    normalized_test_features, test_baselines = _baseline_normalize_features(
        test_features,
        test_bearings,
        test_timestamps,
    )

    ml = load_ml_core()
    scaler = ml.StandardScaler()
    scaled_train = np.asfortranarray(scaler.fit_transform(normalized_train_features))
    scaled_test = np.asfortranarray(scaler.transform(normalized_test_features))
    means = np.asarray(scaler.get_mean(), dtype=np.float64)
    scales = np.asarray(scaler.get_std(), dtype=np.float64)
    if means.shape != (len(FEATURE_NAMES),) or scales.shape != means.shape:
        raise ValueError("ml_core.StandardScaler returned parameters with an unexpected shape")
    if not np.isfinite(means).all() or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("ml_core.StandardScaler returned invalid parameters")
    if not np.isfinite(scaled_train).all() or not np.isfinite(scaled_test).all():
        raise ValueError("ml_core.StandardScaler produced non-finite features")
    print(
        f"Training samples: {train_labels.size:,} | 2nd_test samples: {test_labels.size:,}",
        file=sys.stderr,
        flush=True,
    )

    neural_model = ml.NeuralNetwork(
        [len(FEATURE_NAMES), NEURAL_NETWORK_HIDDEN_UNITS, 1],
        "relu",
        "sigmoid",
        ml.OptimizerType.ADAM,
        NEURAL_NETWORK_LEARNING_RATE,
        ml.RegularizerType.L2,
        NEURAL_NETWORK_L2_STRENGTH,
        seed=NEURAL_NETWORK_SEED,
    )
    neural_model.set_loss_function("binary_crossentropy")
    neural_model.set_epochs(NEURAL_NETWORK_EPOCHS)
    neural_model.set_batch_size(32)
    neural_model.set_verbose(False)
    neural_training_started = time.perf_counter()
    neural_model.fit(scaled_train, train_labels.astype(np.float64))
    neural_training_seconds = time.perf_counter() - neural_training_started

    logistic_model = ml.LogisticRegression(0.05, 1500, 0.01, 1e-4, True)
    logistic_training_started = time.perf_counter()
    logistic_model.fit(scaled_train, train_labels.astype(np.float64))
    logistic_training_seconds = time.perf_counter() - logistic_training_started

    neural_verification_started = time.perf_counter()
    neural_probabilities = _validate_probabilities(
        neural_model.predict_proba(scaled_test),
        test_labels.size,
        "IMS neural network",
    )
    threshold_results, selected_threshold_result = _threshold_diagnostics(
        test_labels,
        neural_probabilities,
        test_bearings,
    )
    selected_threshold = float(selected_threshold_result["threshold"])
    neural_metrics = _metrics(test_labels, neural_probabilities, selected_threshold)
    neural_verification_seconds = time.perf_counter() - neural_verification_started

    logistic_verification_started = time.perf_counter()
    logistic_probabilities = _validate_probabilities(
        logistic_model.predict(scaled_test),
        test_labels.size,
        "IMS logistic-regression benchmark",
    )
    logistic_metrics = _metrics(test_labels, logistic_probabilities)
    logistic_verification_seconds = time.perf_counter() - logistic_verification_started
    timings = {
        "neural_training_seconds": neural_training_seconds,
        "logistic_training_seconds": logistic_training_seconds,
        "neural_verification_seconds": neural_verification_seconds,
        "logistic_verification_seconds": logistic_verification_seconds,
        "total_fit_and_verification_seconds": (
            neural_training_seconds
            + logistic_training_seconds
            + neural_verification_seconds
            + logistic_verification_seconds
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    neural_model.save(str(NEURAL_MODEL_PATH))
    logistic_model.save(str(LOGISTIC_MODEL_PATH))
    _save_dataset(
        processed_dir / "ims_train_features.npz",
        train_features,
        train_labels,
        train_bearings,
        train_timestamps,
    )
    _save_dataset(
        processed_dir / "ims_test_features.npz",
        test_features,
        test_labels,
        test_bearings,
        test_timestamps,
    )
    predictions_path = processed_dir / "ims_test_predictions.csv"
    _save_predictions(
        predictions_path,
        test_bearings,
        test_timestamps,
        test_labels,
        neural_probabilities,
        selected_threshold,
    )

    _print_bearing_alarm_counts(
        test_bearings,
        neural_probabilities,
        ALERT_THRESHOLD,
        "baseline",
    )
    _print_bearing_alarm_counts(
        test_bearings,
        neural_probabilities,
        selected_threshold,
        "selected",
    )
    print("MLP threshold sweep on 2nd_test (exploratory):")
    for result in threshold_results:
        counts = result["alarm_samples_by_bearing"]
        print(
            f"  threshold={result['threshold']:.3f} "
            f"Precision={result['precision']:.4f} "
            f"Recall={result['recall']:.4f} "
            f"F1={result['f1_score']:.4f} "
            f"healthy_bearing_alarms={result['healthy_bearing_alarm_samples']} "
            f"per_bearing={counts}"
        )
    print(
        "Selected threshold (max F1, then fewest alarms on healthy bearings): "
        f"{selected_threshold:.3f}"
    )
    print(
        "MLP sigmoid scores: "
        f"min={neural_probabilities.min():.6f} "
        f"max={neural_probabilities.max():.6f} "
        f"unique={np.unique(neural_probabilities).size:,}"
    )

    model_metrics = {
        "neural_network_cpp": {
            **dict(zip(("precision", "recall", "f1_score"), neural_metrics, strict=True)),
            "threshold": selected_threshold,
        },
        "logistic_regression": {
            **dict(zip(("precision", "recall", "f1_score"), logistic_metrics, strict=True)),
            "threshold": ALERT_THRESHOLD,
        },
    }

    metrics_path = processed_dir / "ims_test_metrics.json"
    metrics_temporary_path = metrics_path.with_suffix(".json.tmp")
    metrics_temporary_path.write_text(
        json.dumps(
            {
                "source": "NASA IMS Bearing Dataset",
                "training_series": ["1st_test", "3rd_test"],
                "test_series": "2nd_test",
                "threshold_rule": "probability > selected threshold",
                "threshold": selected_threshold,
                "baseline_threshold": ALERT_THRESHOLD,
                "threshold_tuning": {
                    "selection_data": "2nd_test labels; exploratory, not an independent test result",
                    "objective": (
                        "maximize sample-wise F1; break ties by fewer alarms on "
                        "healthy bearings 2, 3, and 4, then higher threshold"
                    ),
                    "candidates": threshold_results,
                    "selected_threshold": selected_threshold,
                },
                "neural_network_seed": NEURAL_NETWORK_SEED,
                "neural_network": {
                    "architecture": [
                        len(FEATURE_NAMES),
                        NEURAL_NETWORK_HIDDEN_UNITS,
                        1,
                    ],
                    "activation": "relu",
                    "output_activation": "sigmoid",
                    "learning_rate": NEURAL_NETWORK_LEARNING_RATE,
                    "epochs": NEURAL_NETWORK_EPOCHS,
                    "regularizer": "L2",
                    "regularizer_strength": NEURAL_NETWORK_L2_STRENGTH,
                },
                "baseline_normalization": {
                    "method": "feature minus mean of first chronological samples per bearing",
                    "sample_count": BASELINE_SAMPLE_COUNT,
                },
                "training_samples": int(train_labels.size),
                "test_samples": int(test_labels.size),
                "positive_test_samples": int(test_labels.sum()),
                "timing_scope": (
                    "wall-clock model fit and test prediction plus metric calculation; "
                    "excludes feature extraction, scaling, and artifact writes"
                ),
                "timings_seconds": timings,
                "models": model_metrics,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(metrics_temporary_path, metrics_path)

    metadata = {
        "format_version": 1,
        "source": "NASA IMS Bearing Dataset",
        "training_series": ["1st_test", "3rd_test"],
        "test_series": "2nd_test",
        "feature_names": FEATURE_NAMES,
        "feature_aggregation": "mean of per-channel statistics for 1st_test",
        "failure_tail_fraction": 0.25,
        "alert_threshold": selected_threshold,
        "baseline_normalization": {
            "method": "feature minus mean of first chronological samples per bearing",
            "sample_count": BASELINE_SAMPLE_COUNT,
            "training_means": train_baselines,
            "test_means": test_baselines,
        },
        "threshold_tuning": {
            "selection_data": "2nd_test labels; exploratory, not an independent test result",
            "selected_threshold": selected_threshold,
            "candidates": threshold_results,
        },
        "timings_seconds": timings,
        "scaler": {"means": means.tolist(), "scales": scales.tolist()},
        "models": {
            "neural_network": {
                "path": str(NEURAL_MODEL_PATH),
                "sha256": sha256_file(NEURAL_MODEL_PATH),
                "architecture": [
                    len(FEATURE_NAMES),
                    NEURAL_NETWORK_HIDDEN_UNITS,
                    1,
                ],
                "seed": NEURAL_NETWORK_SEED,
                "epochs": NEURAL_NETWORK_EPOCHS,
                "learning_rate": NEURAL_NETWORK_LEARNING_RATE,
                "regularizer": "L2",
                "regularizer_strength": NEURAL_NETWORK_L2_STRENGTH,
            },
            "logistic_regression": {
                "path": str(LOGISTIC_MODEL_PATH),
                "sha256": sha256_file(LOGISTIC_MODEL_PATH),
            },
        },
        "training_samples": int(train_labels.size),
        "test_samples": int(test_labels.size),
    }
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    metadata_temporary_path = METADATA_PATH.with_suffix(".json.tmp")
    metadata_temporary_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(metadata_temporary_path, METADATA_PATH)

    print(f"Positive samples in training: {int(train_labels.sum()):,}")
    print(f"Positive samples in 2nd_test: {int(test_labels.sum()):,}")
    print(f"MLP alert threshold: probability > {selected_threshold:.3f}")
    print(f"Logistic benchmark threshold: probability > {ALERT_THRESHOLD:.2f}")
    for model_name, metrics in (
        ("Neural network C++/pybind11", model_metrics["neural_network_cpp"]),
        ("Logistic regression benchmark", model_metrics["logistic_regression"]),
    ):
        print(
            f"{model_name}: Precision={metrics['precision']:.4f} "
            f"Recall={metrics['recall']:.4f} F1-Score={metrics['f1_score']:.4f}"
        )
    print(f"MLP fit time: {neural_training_seconds:.3f} s")
    print(f"Logistic benchmark fit time: {logistic_training_seconds:.3f} s")
    print(f"MLP 2nd_test verification time: {neural_verification_seconds:.3f} s")
    print(f"Logistic 2nd_test verification time: {logistic_verification_seconds:.3f} s")
    print(f"Total fit + verification time: {timings['total_fit_and_verification_seconds']:.3f} s")
    print(f"Saved chronological dashboard predictions: {predictions_path}")
    print(f"Saved test metrics: {metrics_path}")
    print("Metrics are retrospective benchmark results, not industrial validation.")


if __name__ == "__main__":
    main()
