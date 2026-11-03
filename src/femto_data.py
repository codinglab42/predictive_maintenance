"""Loading and feature extraction for the NASA FEMTO/PRONOSTIA bearing data."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "femto"
_SAMPLE_NAME = re.compile(r"^acc_(\d+)\.csv$", re.IGNORECASE)
_BEARING_NAME = re.compile(r"^Bearing\d+_\d+$", re.IGNORECASE)
_TRAINING_PARTS = {"learning_set", "train_set", "training_set", "train"}
_TEST_PARTS = {"test_set", "full_test_set", "testing_set", "test"}
_CHANNEL_FEATURES = ("rms", "std", "peak_to_peak", "max_abs", "crest_factor", "kurtosis")
FEATURE_NAMES = [
    f"{channel}_{feature}"
    for channel in ("horizontal", "vertical")
    for feature in _CHANNEL_FEATURES
]


def _bearing_sort_key(bearing_id: str) -> tuple[int, int]:
    match = re.fullmatch(r"Bearing(\d+)_(\d+)", bearing_id, re.IGNORECASE)
    if match is None:
        return (0, 0)
    return int(match.group(1)), int(match.group(2))


def infer_condition_id(bearing_id: str) -> int:
    """Return the operating-condition group for a FEMTO bearing ID."""
    match = re.fullmatch(r"Bearing(\d+)_(\d+)", bearing_id, re.IGNORECASE)
    if match is None:
        raise ValueError(f"unsupported bearing ID format: {bearing_id!r}")
    return int(match.group(1))


def group_bearings_by_condition(bearing_ids: Iterable[str]) -> dict[int, list[str]]:
    """Group bearings by FEMTO operating condition (the leading BearingN prefix)."""
    unique_bearings = list(dict.fromkeys(bearing_ids))
    grouped: dict[int, list[str]] = {}
    for bearing_id in unique_bearings:
        condition = infer_condition_id(bearing_id)
        grouped.setdefault(condition, []).append(bearing_id)
    for condition in grouped:
        grouped[condition] = sorted(grouped[condition], key=_bearing_sort_key)
    return dict(sorted(grouped.items()))


def discover_bearing_files(data_dir: Path) -> dict[str, list[tuple[int, Path]]]:
    """Find run-to-failure training sequences and sort each by acquisition ID."""
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"dataset directory not found: {data_dir}; run download_femto.py first"
        )

    candidates = [
        path
        for path in data_dir.rglob("*.csv")
        if _SAMPLE_NAME.fullmatch(path.name) and _BEARING_NAME.fullmatch(path.parent.name)
    ]
    training_files = [
        path
        for path in candidates
        if any(part.casefold() in _TRAINING_PARTS for part in path.relative_to(data_dir).parts[:-2])
    ]
    if training_files:
        shallowest_training_depth = min(
            len(path.relative_to(data_dir).parts[:-2]) for path in training_files
        )
        candidates = [
            path
            for path in training_files
            if len(path.relative_to(data_dir).parts[:-2]) == shallowest_training_depth
        ]
    else:
        candidates = [
            path
            for path in candidates
            if not any(part.casefold() in _TEST_PARTS for part in path.relative_to(data_dir).parts[:-2])
        ]

    bearings: dict[str, list[tuple[int, Path]]] = {}
    for path in candidates:
        match = _SAMPLE_NAME.fullmatch(path.name)
        if match is None:
            continue
        bearings.setdefault(path.parent.name, []).append((int(match.group(1)), path))

    for bearing_id, samples in bearings.items():
        samples.sort(key=lambda sample: sample[0])
        sample_ids = [sample_id for sample_id, _ in samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError(f"duplicate acquisition IDs found for {bearing_id}")

    if not bearings:
        raise ValueError(
            f"no FEMTO training samples found under {data_dir}; expected "
            "BearingN_M/acc_*.csv directories inside Learning_set or equivalent"
        )
    return dict(sorted(bearings.items(), key=lambda item: _bearing_sort_key(item[0])))


def extract_signal_features(signal: np.ndarray) -> np.ndarray:
    """Extract six finite time-domain features for each vibration channel."""
    if signal.ndim != 2 or signal.shape[1] != 2 or signal.shape[0] < 8:
        raise ValueError("a vibration sample must contain at least 8 rows and exactly 2 columns")
    if not np.isfinite(signal).all():
        raise ValueError("vibration sample contains non-finite values")

    features: list[float] = []
    for channel in signal.T:
        centered = channel - np.mean(channel)
        rms = float(np.sqrt(np.mean(np.square(channel))))
        std = float(np.sqrt(np.mean(np.square(centered))))
        max_abs = float(np.max(np.abs(channel)))
        variance = float(np.mean(np.square(centered)))
        kurtosis = (
            float(np.mean(np.power(centered, 4)) / (variance * variance) - 3.0)
            if variance > 0.0
            else 0.0
        )
        features.extend(
            [
                rms,
                std,
                float(np.ptp(channel)),
                max_abs,
                max_abs / rms if rms > 0.0 else 0.0,
                kurtosis,
            ]
        )
    result = np.asarray(features, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("feature extraction produced non-finite values")
    return result


def read_signal_features(path: Path) -> np.ndarray:
    """Read one headerless, two-channel FEMTO CSV sample and extract features."""
    try:
        signal = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
    except (OSError, ValueError) as error:
        raise ValueError(f"could not read vibration CSV {path}: {error}") from error
    if signal.shape[1] == 6:
        signal = signal[:, -2:]
    return extract_signal_features(signal)


def build_labeled_dataset(
    data_dir: Path,
    horizon: int,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build features and labels, optionally reporting progress after each sample."""
    if horizon < 1:
        raise ValueError("failure horizon must be a positive number of acquisitions")

    feature_rows: list[np.ndarray] = []
    labels: list[int] = []
    bearing_ids: list[str] = []
    sample_ids: list[int] = []

    bearings = discover_bearing_files(data_dir)
    total_samples = sum(len(samples) for samples in bearings.values())
    processed_samples = 0
    for bearing_id, samples in bearings.items():
        for sample_position, (sample_id, path) in enumerate(samples):
            feature_rows.append(read_signal_features(path))
            labels.append(int(len(samples) - sample_position <= horizon))
            bearing_ids.append(bearing_id)
            sample_ids.append(sample_id)
            processed_samples += 1
            if progress_callback is not None:
                progress_callback(processed_samples, total_samples, bearing_id)

    return (
        np.asarray(feature_rows, dtype=np.float64, order="F"),
        np.asarray(labels, dtype=np.int64),
        np.asarray(bearing_ids, dtype=str),
        np.asarray(sample_ids, dtype=np.int64),
    )


def standardize(
    train_features: np.ndarray,
    other_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scale features using only the training partition's statistics."""
    means = train_features.mean(axis=0)
    scales = train_features.std(axis=0)
    scales[scales == 0.0] = 1.0
    return (
        np.asfortranarray((train_features - means) / scales),
        np.asfortranarray((other_features - means) / scales),
        means,
        scales,
    )
