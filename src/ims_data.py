"""Chronological feature extraction for the NASA IMS bearing dataset."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np


FEATURE_NAMES = [
    "rms",
    "standard_deviation",
    "kurtosis",
    "skewness",
    "peak_to_peak",
    "crest_factor",
]
SERIES_BEARINGS = {
    "1st_test": 4,
    "2nd_test": 4,
    "3rd_test": 4,
}
FAILED_BEARINGS = {
    "1st_test": {3, 4},
    "2nd_test": {1},
    "3rd_test": {3},
}


def extract_channel_features(signal: np.ndarray) -> np.ndarray:
    """Return six finite population statistics for one vibration channel."""
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    if values.size < 2:
        raise ValueError("an IMS vibration channel must contain at least two samples")
    if not np.isfinite(values).all():
        raise ValueError("IMS vibration channel contains non-finite values")

    centered = values - np.mean(values)
    standard_deviation = float(np.sqrt(np.mean(np.square(centered))))
    rms = float(np.sqrt(np.mean(np.square(values))))
    if standard_deviation == 0.0:
        kurtosis = 0.0
        skewness = 0.0
    else:
        standardized = centered / standard_deviation
        skewness = float(np.mean(np.power(standardized, 3)))
        kurtosis = float(np.mean(np.power(standardized, 4)) - 3.0)
    return np.asarray(
        [
            rms,
            standard_deviation,
            kurtosis,
            skewness,
            float(np.ptp(values)),
            float(np.max(np.abs(values)) / rms) if rms > 0.0 else 0.0,
        ],
        dtype=np.float64,
    )


def _timestamp_from_path(path: Path) -> datetime:
    filename = path.name[:-4] if path.name.casefold().endswith(".txt") else path.name
    try:
        return datetime.strptime(filename, "%Y.%m.%d.%H.%M.%S")
    except ValueError as error:
        raise ValueError(f"IMS filename is not a timestamp: {path.name}") from error


def _series_files(data_dir: Path, series_name: str) -> list[tuple[datetime, Path]]:
    if series_name not in SERIES_BEARINGS:
        raise ValueError(f"unsupported IMS series: {series_name}")
    series_dir = data_dir / series_name
    if not series_dir.is_dir():
        raise FileNotFoundError(f"IMS series directory not found: {series_dir}")

    files = [
        (_timestamp_from_path(path), path)
        for path in series_dir.rglob("*")
        if path.is_file() and not path.name.startswith(".")
    ]
    if not files:
        raise ValueError(f"no timestamped IMS samples found under {series_dir}")
    files.sort(key=lambda item: item[0])
    timestamps = [timestamp for timestamp, _ in files]
    if len(timestamps) != len(set(timestamps)):
        raise ValueError(f"duplicate timestamps found in IMS series {series_name}")
    return files


def _read_sample(path: Path, series_name: str) -> np.ndarray:
    try:
        sample = np.loadtxt(path, dtype=np.float64, ndmin=2)
    except (OSError, ValueError) as error:
        raise ValueError(f"could not read IMS sample {path}: {error}") from error
    expected_columns = 8 if series_name == "1st_test" else 4
    if sample.ndim != 2 or sample.shape[1] != expected_columns or sample.shape[0] < 2:
        raise ValueError(
            f"IMS sample {path} must have at least two rows and "
            f"{expected_columns} columns; got {sample.shape}"
        )
    if not np.isfinite(sample).all():
        raise ValueError(f"IMS sample contains non-finite values: {path}")
    return sample


def _features_for_bearing(sample: np.ndarray, series_name: str, bearing_index: int) -> np.ndarray:
    if series_name == "1st_test":
        columns = sample[:, bearing_index * 2 : bearing_index * 2 + 2]
    else:
        columns = sample[:, bearing_index : bearing_index + 1]
    channel_features = [extract_channel_features(columns[:, column]) for column in range(columns.shape[1])]
    return np.mean(np.stack(channel_features), axis=0)


def build_labeled_dataset(
    data_dir: Path,
    series_name: str,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract six features and 75%-failure-horizon labels for one IMS series."""
    files = _series_files(data_dir, series_name)
    failed_bearings = FAILED_BEARINGS[series_name]
    features: list[np.ndarray] = []
    labels: list[int] = []
    bearing_ids: list[str] = []
    timestamps: list[str] = []
    samples_per_bearing = len(files)
    failure_start = int(samples_per_bearing * 0.75)

    for position, (timestamp, path) in enumerate(files):
        sample = _read_sample(path, series_name)
        for bearing_index in range(SERIES_BEARINGS[series_name]):
            features.append(_features_for_bearing(sample, series_name, bearing_index))
            bearing_ids.append(f"{series_name}_bearing_{bearing_index + 1}")
            labels.append(
                int(bearing_index + 1 in failed_bearings and position >= failure_start)
            )
            timestamps.append(timestamp.isoformat(sep=" "))
        if progress_callback is not None:
            progress_callback(position + 1, samples_per_bearing)

    feature_array = np.asarray(features, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    bearing_array = np.asarray(bearing_ids, dtype=str)
    timestamp_array = np.asarray(timestamps, dtype=str)
    if not np.isfinite(feature_array).all():
        raise ValueError(f"feature extraction produced non-finite values for {series_name}")
    return feature_array, label_array, bearing_array, timestamp_array


