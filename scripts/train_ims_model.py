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
GRU_MODEL_PATH = MODEL_DIR / "ims_gru.bin"
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
GRU_UNITS = 12
SEQUENCE_LENGTH = 15
ALARM_ROLLING_WINDOW = 5
ALARM_PERSISTENCE = 3
MODEL_ORDER = ("logistic_regression", "neural_network_cpp", "gru_cpp")
MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic regression C++",
    "neural_network_cpp": "MLP C++",
    "gru_cpp": "GRU C++",
}


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


def _build_sliding_windows(
    features: np.ndarray,
    labels: np.ndarray,
    bearing_ids: np.ndarray,
    timestamps: np.ndarray,
    time_steps: int = SEQUENCE_LENGTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if (
        features.ndim != 2
        or labels.ndim != 1
        or bearing_ids.ndim != 1
        or timestamps.ndim != 1
        or any(array.size != features.shape[0] for array in (labels, bearing_ids, timestamps))
        or features.shape[0] == 0
    ):
        raise ValueError("sliding windows require aligned, non-empty feature and metadata arrays")
    if time_steps < 1:
        raise ValueError("time_steps must be positive")
    if not np.isfinite(features).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("sliding windows require finite features and binary labels")

    windows: list[np.ndarray] = []
    window_labels: list[int] = []
    window_bearings: list[str] = []
    window_timestamps: list[str] = []
    window_file_indices: list[int] = []
    for bearing_id in np.unique(bearing_ids):
        positions = np.flatnonzero(bearing_ids == bearing_id)
        chronological = positions[np.argsort(timestamps[positions], kind="stable")]
        bearing_timestamps = timestamps[chronological]
        if np.unique(bearing_timestamps).size != chronological.size:
            raise ValueError(f"{bearing_id} contains duplicate IMS timestamps")
        if chronological.size < time_steps:
            raise ValueError(
                f"{bearing_id} has fewer than {time_steps} samples for a sliding window"
            )
        for end in range(time_steps - 1, chronological.size):
            final_position = chronological[end]
            windows.append(features[chronological[end - time_steps + 1 : end + 1]])
            window_labels.append(int(labels[final_position]))
            window_bearings.append(str(bearing_id))
            window_timestamps.append(str(timestamps[final_position]))
            window_file_indices.append(end + 1)

    return (
        np.asarray(windows, dtype=np.float64),
        np.asarray(window_labels, dtype=np.int64),
        np.asarray(window_bearings, dtype=str),
        np.asarray(window_timestamps, dtype=str),
        np.asarray(window_file_indices, dtype=np.int64),
    )


def _rolling_mean_by_bearing(
    probabilities: np.ndarray,
    bearing_ids: np.ndarray,
    timestamps: np.ndarray,
    window: int = ALARM_ROLLING_WINDOW,
) -> np.ndarray:
    if (
        probabilities.ndim != 1
        or bearing_ids.ndim != 1
        or timestamps.ndim != 1
        or probabilities.size != bearing_ids.size
        or probabilities.size != timestamps.size
        or probabilities.size == 0
    ):
        raise ValueError("rolling mean requires aligned, non-empty probability and metadata arrays")
    if window < 1 or not np.isfinite(probabilities).all():
        raise ValueError("rolling mean requires a positive window and finite probabilities")

    smoothed = np.empty_like(probabilities, dtype=np.float64)
    for bearing_id in np.unique(bearing_ids):
        positions = np.flatnonzero(bearing_ids == bearing_id)
        chronological = positions[np.argsort(timestamps[positions], kind="stable")]
        if np.unique(timestamps[chronological]).size != chronological.size:
            raise ValueError(f"{bearing_id} contains duplicate IMS timestamps")
        values = probabilities[chronological]
        cumulative = np.concatenate(([0.0], np.cumsum(values)))
        for index, position in enumerate(chronological):
            start = max(0, index - window + 1)
            smoothed[position] = (cumulative[index + 1] - cumulative[start]) / (index + 1 - start)
    return smoothed


def _confirmed_alarms(
    smoothed_probabilities: np.ndarray,
    bearing_ids: np.ndarray,
    timestamps: np.ndarray,
    threshold: float,
    persistence: int = ALARM_PERSISTENCE,
) -> np.ndarray:
    if (
        smoothed_probabilities.ndim != 1
        or bearing_ids.ndim != 1
        or timestamps.ndim != 1
        or smoothed_probabilities.size != bearing_ids.size
        or smoothed_probabilities.size != timestamps.size
    ):
        raise ValueError("alarm confirmation requires aligned one-dimensional arrays")
    if not 0.0 <= threshold <= 1.0 or persistence < 1:
        raise ValueError("threshold must be in [0, 1] and persistence must be positive")

    alarms = np.zeros(smoothed_probabilities.size, dtype=bool)
    for bearing_id in np.unique(bearing_ids):
        positions = np.flatnonzero(bearing_ids == bearing_id)
        chronological = positions[np.argsort(timestamps[positions], kind="stable")]
        count = 0
        for position in chronological:
            count = count + 1 if smoothed_probabilities[position] > threshold else 0
            alarms[position] = count >= persistence
    return alarms


def _alarm_episodes(
    alarms: np.ndarray,
    bearing_ids: np.ndarray,
    timestamps: np.ndarray,
) -> dict[str, int]:
    episodes: dict[str, int] = {}
    for bearing_id in np.unique(bearing_ids):
        positions = np.flatnonzero(bearing_ids == bearing_id)
        chronological = positions[np.argsort(timestamps[positions], kind="stable")]
        values = alarms[chronological]
        episodes[str(bearing_id)] = int(
            np.count_nonzero(values & ~np.concatenate(([False], values[:-1])))
        )
    return episodes


def _threshold_diagnostics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bearing_ids: np.ndarray,
    timestamps: np.ndarray,
    file_indices: np.ndarray,
    rolling_window: int = ALARM_ROLLING_WINDOW,
    persistence: int = ALARM_PERSISTENCE,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if (
        labels.ndim != 1
        or probabilities.ndim != 1
        or bearing_ids.ndim != 1
        or timestamps.ndim != 1
        or file_indices.ndim != 1
        or labels.size != probabilities.size
        or labels.size != bearing_ids.size
        or labels.size != timestamps.size
        or labels.size != file_indices.size
    ):
        raise ValueError("threshold diagnostics require aligned one-dimensional arrays")

    smoothed = _rolling_mean_by_bearing(probabilities, bearing_ids, timestamps, rolling_window)
    results: list[dict[str, object]] = []
    for threshold in THRESHOLD_CANDIDATES:
        alarms = _confirmed_alarms(smoothed, bearing_ids, timestamps, threshold, persistence)
        precision, recall, f1 = _metrics(labels, alarms.astype(np.float64), 0.5)
        episodes_by_bearing = _alarm_episodes(alarms, bearing_ids, timestamps)
        healthy_false_alarms = sum(
            episodes_by_bearing.get(f"2nd_test_bearing_{index}", 0)
            for index in (2, 3, 4)
        )
        bearing_1 = np.flatnonzero(
            (bearing_ids == "2nd_test_bearing_1") & alarms
        )
        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "first_alarm_bearing_1_file": (
                    int(file_indices[bearing_1[0]]) if bearing_1.size else None
                ),
                "false_alarm_episodes_by_bearing": episodes_by_bearing,
                "healthy_false_alarm_episodes": healthy_false_alarms,
            }
        )

    selected = max(
        results,
        key=lambda result: (
            float(result["f1_score"]),
            -int(result["healthy_false_alarm_episodes"]),
            float(result["threshold"]),
        ),
    )
    return results, selected


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
    file_indices: np.ndarray,
    labels: np.ndarray,
    probabilities_by_model: dict[str, np.ndarray],
    thresholds_by_model: dict[str, float],
) -> None:
    expected_shape = labels.shape
    if any(
        values.shape != expected_shape
        for values in (bearing_ids, timestamps, file_indices)
    ):
        raise ValueError("prediction metadata arrays must match the label count")
    if set(probabilities_by_model) != set(MODEL_ORDER) or set(thresholds_by_model) != set(MODEL_ORDER):
        raise ValueError("predictions and thresholds must be provided for all three models")
    if any(values.shape != expected_shape for values in probabilities_by_model.values()):
        raise ValueError("each model must provide one probability per aligned test window")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "bearing_id",
                "timestamp",
                "file_index",
                "label",
                "model_id",
                "probability",
                "alert_threshold",
            ]
        )
        for model_id in MODEL_ORDER:
            writer.writerows(
                zip(
                    bearing_ids.tolist(),
                    timestamps.tolist(),
                    file_indices.tolist(),
                    labels.tolist(),
                    [model_id] * labels.size,
                    probabilities_by_model[model_id].tolist(),
                    [thresholds_by_model[model_id]] * labels.size,
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
    (
        train_windows,
        train_window_labels,
        _,
        _,
        _,
    ) = _build_sliding_windows(
        scaled_train,
        train_labels,
        train_bearings,
        train_timestamps,
    )
    (
        test_windows,
        aligned_test_labels,
        aligned_test_bearings,
        aligned_test_timestamps,
        aligned_test_file_indices,
    ) = _build_sliding_windows(
        scaled_test,
        test_labels,
        test_bearings,
        test_timestamps,
    )
    scaled_train_windows = np.asfortranarray(train_windows)
    scaled_test_windows = np.asfortranarray(test_windows)
    aligned_test_features = np.asfortranarray(scaled_test_windows[:, -1, :])
    print(
        f"Training samples: {train_labels.size:,} | "
        f"GRU training windows: {train_window_labels.size:,} | "
        f"Aligned 2nd_test windows: {aligned_test_labels.size:,} "
        f"(files {SEQUENCE_LENGTH}-{test_features.shape[0] // 4})",
        file=sys.stderr,
        flush=True,
    )

    logistic_model = ml.LogisticRegression(0.05, 1500, 0.01, 1e-4, True)
    logistic_training_started = time.perf_counter()
    logistic_model.fit(scaled_train, train_labels.astype(np.float64))
    logistic_training_seconds = time.perf_counter() - logistic_training_started

    mlp_model = ml.NeuralNetwork(
        [len(FEATURE_NAMES), NEURAL_NETWORK_HIDDEN_UNITS, 1],
        "relu",
        "sigmoid",
        ml.OptimizerType.ADAM,
        NEURAL_NETWORK_LEARNING_RATE,
        ml.RegularizerType.L2,
        NEURAL_NETWORK_L2_STRENGTH,
        seed=NEURAL_NETWORK_SEED,
    )
    mlp_model.set_loss_function("binary_crossentropy")
    mlp_model.set_epochs(NEURAL_NETWORK_EPOCHS)
    mlp_model.set_batch_size(32)
    mlp_model.set_verbose(False)
    mlp_training_started = time.perf_counter()
    mlp_model.fit(scaled_train, train_labels.astype(np.float64))
    mlp_training_seconds = time.perf_counter() - mlp_training_started

    gru_model = ml.NeuralNetwork(
        [len(FEATURE_NAMES), 1],
        "relu",
        "sigmoid",
        ml.OptimizerType.ADAM,
        NEURAL_NETWORK_LEARNING_RATE,
        ml.RegularizerType.NONE,
        NEURAL_NETWORK_L2_STRENGTH,
        NEURAL_NETWORK_SEED,
    )
    gru_model.reset()
    gru_model.add_recurrent_layer(
        ml.RecurrentType.GRU,
        GRU_UNITS,
        return_sequences=False,
        activation="tanh",
    )
    gru_model.add_dense_layer(1, "sigmoid")
    gru_model.set_optimizer(ml.OptimizerType.ADAM, NEURAL_NETWORK_LEARNING_RATE)
    gru_model.set_loss_function("binary_crossentropy")
    gru_model.set_epochs(NEURAL_NETWORK_EPOCHS)
    gru_model.set_batch_size(32)
    gru_model.set_verbose(False)
    gru_training_started = time.perf_counter()
    gru_model.fit(scaled_train_windows, train_window_labels.astype(np.float64))
    gru_training_seconds = time.perf_counter() - gru_training_started

    model_objects = {
        "logistic_regression": logistic_model,
        "neural_network_cpp": mlp_model,
        "gru_cpp": gru_model,
    }
    test_inputs = {
        "logistic_regression": aligned_test_features,
        "neural_network_cpp": aligned_test_features,
        "gru_cpp": scaled_test_windows,
    }
    training_seconds = {
        "logistic_regression": logistic_training_seconds,
        "neural_network_cpp": mlp_training_seconds,
        "gru_cpp": gru_training_seconds,
    }
    probabilities_by_model: dict[str, np.ndarray] = {}
    inference_seconds: dict[str, float] = {}
    threshold_results_by_model: dict[str, list[dict[str, object]]] = {}
    model_metrics: dict[str, dict[str, object]] = {}
    thresholds_by_model: dict[str, float] = {}
    for model_id in MODEL_ORDER:
        inference_started = time.perf_counter()
        predictions = model_objects[model_id].predict_proba(test_inputs[model_id])
        inference_seconds[model_id] = time.perf_counter() - inference_started
        probabilities = _validate_probabilities(
            predictions,
            aligned_test_labels.size,
            MODEL_DISPLAY_NAMES[model_id],
        )
        probabilities_by_model[model_id] = probabilities
        threshold_results, selected = _threshold_diagnostics(
            aligned_test_labels,
            probabilities,
            aligned_test_bearings,
            aligned_test_timestamps,
            aligned_test_file_indices,
        )
        threshold_results_by_model[model_id] = threshold_results
        thresholds_by_model[model_id] = float(selected["threshold"])
        model_metrics[model_id] = {
            "display_name": MODEL_DISPLAY_NAMES[model_id],
            "seed": NEURAL_NETWORK_SEED,
            "precision": float(selected["precision"]),
            "recall": float(selected["recall"]),
            "f1_score": float(selected["f1_score"]),
            "threshold": float(selected["threshold"]),
            "training_seconds": training_seconds[model_id],
            "inference_seconds": inference_seconds[model_id],
            "first_alarm_bearing_1_file": selected["first_alarm_bearing_1_file"],
            "false_alarm_episodes_by_bearing": selected[
                "false_alarm_episodes_by_bearing"
            ],
            "healthy_false_alarm_episodes": int(
                selected["healthy_false_alarm_episodes"]
            ),
        }

    active_model = max(
        MODEL_ORDER,
        key=lambda model_id: (
            float(model_metrics[model_id]["f1_score"]),
            -int(model_metrics[model_id]["healthy_false_alarm_episodes"]),
            -float(model_metrics[model_id]["inference_seconds"]),
        ),
    )
    selected_threshold = thresholds_by_model[active_model]
    timings = {
        model_id: {
            "training_seconds": training_seconds[model_id],
            "inference_seconds": inference_seconds[model_id],
        }
        for model_id in MODEL_ORDER
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    mlp_model.save(str(NEURAL_MODEL_PATH))
    logistic_model.save(str(LOGISTIC_MODEL_PATH))
    gru_model.save(str(GRU_MODEL_PATH))
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
        aligned_test_bearings,
        aligned_test_timestamps,
        aligned_test_file_indices,
        aligned_test_labels,
        probabilities_by_model,
        thresholds_by_model,
    )

    metrics_path = processed_dir / "ims_test_metrics.json"
    metrics_temporary_path = metrics_path.with_suffix(".json.tmp")
    metrics_temporary_path.write_text(
        json.dumps(
            {
                "source": "NASA IMS Bearing Dataset",
                "training_series": ["1st_test", "3rd_test"],
                "test_series": "2nd_test",
                "threshold_rule": (
                    "causal rolling mean over five probabilities, followed by "
                    "three consecutive values strictly above the model threshold"
                ),
                "metric_policy": "metrics use confirmed alarms after rolling mean and persistence",
                "threshold": selected_threshold,
                "active_model": active_model,
                "thresholds_by_model": thresholds_by_model,
                "threshold_tuning": {
                    "selection_data": "2nd_test labels; exploratory, not an independent test result",
                    "objective": (
                        "maximize F1 of confirmed alarms; break ties by fewer false-alarm "
                        "episodes on healthy bearings 2, 3, and 4, then higher threshold"
                    ),
                    "rolling_window": ALARM_ROLLING_WINDOW,
                    "persistence": ALARM_PERSISTENCE,
                    "candidates_by_model": threshold_results_by_model,
                    "selected_thresholds_by_model": thresholds_by_model,
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
                "gru_network": {
                    "input_features": len(FEATURE_NAMES),
                    "time_steps": SEQUENCE_LENGTH,
                    "gru_units": GRU_UNITS,
                    "return_sequences": False,
                    "dense_units": 1,
                    "dense_activation": "sigmoid",
                    "recurrent_activation": "tanh",
                    "learning_rate": NEURAL_NETWORK_LEARNING_RATE,
                    "epochs": NEURAL_NETWORK_EPOCHS,
                    "batch_size": 32,
                    "seed": NEURAL_NETWORK_SEED,
                },
                "baseline_normalization": {
                    "method": "feature minus mean of first chronological samples per bearing",
                    "sample_count": BASELINE_SAMPLE_COUNT,
                },
                "training_samples": int(train_labels.size),
                "training_sequence_windows": int(train_window_labels.size),
                "test_samples": int(aligned_test_labels.size),
                "full_test_samples": int(test_labels.size),
                "positive_test_samples": int(aligned_test_labels.sum()),
                "timing_scope": (
                    "wall-clock model fit and predict_proba only; excludes feature extraction, "
                    "scaling, rolling/persistence metrics, and artifact writes"
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
        "active_model": active_model,
        "model_thresholds": thresholds_by_model,
        "alarm_policy": {
            "rolling_window": ALARM_ROLLING_WINDOW,
            "persistence": ALARM_PERSISTENCE,
            "comparison": "strictly greater than threshold",
        },
        "baseline_normalization": {
            "method": "feature minus mean of first chronological samples per bearing",
            "sample_count": BASELINE_SAMPLE_COUNT,
            "training_means": train_baselines,
            "test_means": test_baselines,
        },
        "threshold_tuning": {
            "selection_data": "2nd_test labels; exploratory, not an independent test result",
            "selected_thresholds_by_model": thresholds_by_model,
            "candidates_by_model": threshold_results_by_model,
        },
        "timings_seconds": timings,
        "scaler": {"means": means.tolist(), "scales": scales.tolist()},
        "models": {
            "neural_network_cpp": {
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
                "seed": NEURAL_NETWORK_SEED,
                "seed_effect": "not applicable; deterministic optimizer",
            },
            "gru_cpp": {
                "path": str(GRU_MODEL_PATH),
                "sha256": sha256_file(GRU_MODEL_PATH),
                "architecture": {
                    "input_features": len(FEATURE_NAMES),
                    "time_steps": SEQUENCE_LENGTH,
                    "gru_units": GRU_UNITS,
                    "return_sequences": False,
                    "dense_units": 1,
                    "dense_activation": "sigmoid",
                },
                "seed": NEURAL_NETWORK_SEED,
                "epochs": NEURAL_NETWORK_EPOCHS,
                "learning_rate": NEURAL_NETWORK_LEARNING_RATE,
            },
        },
        "training_samples": int(train_labels.size),
        "training_sequence_windows": int(train_window_labels.size),
        "test_samples": int(aligned_test_labels.size),
        "full_test_samples": int(test_labels.size),
    }
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    metadata_temporary_path = METADATA_PATH.with_suffix(".json.tmp")
    metadata_temporary_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(metadata_temporary_path, METADATA_PATH)

    print(f"Positive samples in training: {int(train_labels.sum()):,}")
    print(f"Positive aligned samples in 2nd_test: {int(aligned_test_labels.sum()):,}")
    print(
        "Metrics use the causal rolling mean and persistent alarm policy "
        f"(window={ALARM_ROLLING_WINDOW}, persistence={ALARM_PERSISTENCE})."
    )
    print(
        f"{'Model':<26} {'Precision':>10} {'Recall':>10} {'F1':>8} "
        f"{'Train s':>10} {'Infer s':>10} {'B1 alarm file':>15} {'Healthy FPs':>12}"
    )
    for model_id in MODEL_ORDER:
        metrics = model_metrics[model_id]
        first_alarm = metrics["first_alarm_bearing_1_file"]
        print(
            f"{MODEL_DISPLAY_NAMES[model_id]:<26} "
            f"{float(metrics['precision']):>10.4f} "
            f"{float(metrics['recall']):>10.4f} "
            f"{float(metrics['f1_score']):>8.4f} "
            f"{training_seconds[model_id]:>10.3f} "
            f"{inference_seconds[model_id]:>10.3f} "
            f"{str(first_alarm) if first_alarm is not None else 'n/a':>15} "
            f"{int(metrics['healthy_false_alarm_episodes']):>12}"
        )
    print(
        "Selected dashboard model (max F1, then fewer healthy false alarms, "
        f"then lower inference time): {MODEL_DISPLAY_NAMES[active_model]}"
    )
    print(f"Default alarm threshold: {selected_threshold:.3f}")
    print(f"Saved chronological dashboard predictions: {predictions_path}")
    print(f"Saved test metrics: {metrics_path}")
    print("Metrics are retrospective benchmark results, not industrial validation.")


if __name__ == "__main__":
    main()
