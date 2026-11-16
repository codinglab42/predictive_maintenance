"""Load and prepare the FEMTO benchmark replay used by the dashboard."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from src.femto_data import FEATURE_NAMES, infer_condition_id


Status = Literal["green", "yellow", "red"]


@dataclass(frozen=True)
class BearingSeries:
    bearing_id: str
    condition_id: int
    sample_indices: np.ndarray
    horizontal_rms: np.ndarray
    probabilities: np.ndarray
    thresholds: np.ndarray
    above_threshold: np.ndarray
    notifications: np.ndarray


_REQUIRED_COLUMNS = {
    "bearing_id",
    "condition_id",
    "sample_index",
    "probability",
    "balanced_f1_threshold",
    "balanced_f1_above_threshold",
    "balanced_f1_notification",
}


def load_dashboard_data(features_path: Path, predictions_path: Path) -> dict[str, BearingSeries]:
    """Join extracted FEMTO features with out-of-fold LOCO predictions."""
    with np.load(features_path, allow_pickle=False) as archive:
        missing_arrays = {"features", "bearing_ids", "sample_indices"} - set(archive.files)
        if missing_arrays:
            raise ValueError(f"feature archive is missing arrays: {', '.join(sorted(missing_arrays))}")
        features = np.asarray(archive["features"], dtype=np.float64)
        bearing_ids = np.asarray(archive["bearing_ids"]).astype(str)
        sample_indices = np.asarray(archive["sample_indices"], dtype=np.int64)

    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise ValueError("feature archive does not match the FEMTO feature schema")
    if bearing_ids.ndim != 1 or sample_indices.ndim != 1:
        raise ValueError("bearing IDs and sample indices must be one-dimensional")
    if features.shape[0] != bearing_ids.size or features.shape[0] != sample_indices.size:
        raise ValueError("feature, bearing ID, and sample index counts do not match")
    if not np.isfinite(features).all():
        raise ValueError("feature archive contains non-finite values")

    feature_rows: dict[tuple[str, int], float] = {}
    for bearing_id, sample_index, row in zip(bearing_ids, sample_indices, features, strict=True):
        key = (bearing_id, int(sample_index))
        if key in feature_rows:
            raise ValueError(f"duplicate feature sample found for {bearing_id} acquisition {sample_index}")
        feature_rows[key] = float(row[0])

    predictions: dict[tuple[str, int], tuple[int, float, float, bool, bool]] = {}
    try:
        with predictions_path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None:
                raise ValueError(f"prediction CSV is empty: {predictions_path}")
            missing_columns = _REQUIRED_COLUMNS - set(reader.fieldnames)
            if missing_columns:
                raise ValueError(
                    f"prediction CSV is missing columns: {', '.join(sorted(missing_columns))}"
                )
            for line_number, row in enumerate(reader, start=2):
                bearing_id = row["bearing_id"]
                sample_index = int(row["sample_index"])
                condition_id = int(row["condition_id"])
                probability = float(row["probability"])
                threshold = float(row["balanced_f1_threshold"])
                above_threshold = _parse_flag(row["balanced_f1_above_threshold"], line_number)
                notification = _parse_flag(row["balanced_f1_notification"], line_number)
                if condition_id != infer_condition_id(bearing_id):
                    raise ValueError(f"condition does not match {bearing_id} on CSV line {line_number}")
                if not 0.0 <= probability <= 1.0 or not 0.0 <= threshold <= 1.0:
                    raise ValueError(f"invalid probability or threshold on CSV line {line_number}")
                if above_threshold != (probability >= threshold):
                    raise ValueError(f"threshold flag does not match the score on CSV line {line_number}")

                key = (bearing_id, sample_index)
                if key in predictions:
                    raise ValueError(
                        f"duplicate prediction found for {bearing_id} acquisition {sample_index}"
                    )
                predictions[key] = (
                    condition_id,
                    probability,
                    threshold,
                    above_threshold,
                    notification,
                )
    except OSError as error:
        raise OSError(f"could not read prediction CSV {predictions_path}: {error}") from error
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith(
            ("prediction CSV", "condition does not", "invalid probability", "threshold flag", "duplicate prediction")
        ):
            raise
        raise ValueError(f"invalid prediction CSV {predictions_path}: {error}") from error

    if feature_rows.keys() != predictions.keys():
        missing_predictions = feature_rows.keys() - predictions.keys()
        missing_features = predictions.keys() - feature_rows.keys()
        details = []
        if missing_predictions:
            details.append(f"{len(missing_predictions)} feature samples lack predictions")
        if missing_features:
            details.append(f"{len(missing_features)} predictions lack features")
        raise ValueError("feature/prediction samples do not match: " + "; ".join(details))

    bearing_series: dict[str, BearingSeries] = {}
    bearing_order = sorted({bearing_id for bearing_id, _ in feature_rows})
    for bearing_id in bearing_order:
        keys = sorted(key for key in feature_rows if key[0] == bearing_id)
        rows = [predictions[key] for key in keys]
        bearing_series[bearing_id] = BearingSeries(
            bearing_id=bearing_id,
            condition_id=rows[0][0],
            sample_indices=np.asarray([key[1] for key in keys], dtype=np.int64),
            horizontal_rms=np.asarray([feature_rows[key] for key in keys], dtype=np.float64),
            probabilities=np.asarray([row[1] for row in rows], dtype=np.float64),
            thresholds=np.asarray([row[2] for row in rows], dtype=np.float64),
            above_threshold=np.asarray([row[3] for row in rows], dtype=bool),
            notifications=np.asarray([row[4] for row in rows], dtype=bool),
        )
        if not rows or any(row[0] != rows[0][0] for row in rows):
            raise ValueError(f"inconsistent operating condition for {bearing_id}")
    if not bearing_series:
        raise ValueError("no bearing predictions were loaded")
    return bearing_series


def status_at_position(series: BearingSeries, position: int) -> Status:
    """Return a latched replay status at a zero-based sequence position."""
    if not 0 <= position < series.sample_indices.size:
        raise IndexError("bearing sequence position is out of range")
    if np.any(series.notifications[: position + 1]):
        return "red"
    if series.above_threshold[position]:
        return "yellow"
    return "green"


def position_at_progress(series: BearingSeries, progress_percent: int) -> int:
    """Map shared replay progress to a sample position for one bearing."""
    if not 0 <= progress_percent <= 100:
        raise ValueError("replay progress must be between 0 and 100")
    return int((progress_percent * (series.sample_indices.size - 1) + 50) // 100)


def _parse_flag(value: str | None, line_number: int) -> bool:
    if value not in {"0", "1"}:
        raise ValueError(f"expected a 0/1 flag on CSV line {line_number}")
    return value == "1"
