"""Load and prepare the FEMTO benchmark replay used by the dashboard."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
from typing import Literal

import numpy as np

from src.femto_data import FEATURE_NAMES, infer_condition_id
from src.ims_data import FEATURE_NAMES as IMS_FEATURE_NAMES


Status = Literal["green", "yellow", "red"]
IMS_ALERT_THRESHOLD = 0.7


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


@dataclass(frozen=True)
class IMSBearingSeries:
    bearing_id: str
    timestamps: np.ndarray
    file_indices: np.ndarray
    probabilities: np.ndarray
    probabilities_by_model: dict[str, np.ndarray]
    labels: np.ndarray
    rms: np.ndarray
    kurtosis: np.ndarray
    above_threshold: np.ndarray
    alert_threshold: float
    alert_thresholds: dict[str, float]


_REQUIRED_COLUMNS = {
    "bearing_id",
    "condition_id",
    "sample_index",
    "probability",
    "balanced_f1_threshold",
    "balanced_f1_above_threshold",
    "balanced_f1_notification",
}
_IMS_LEGACY_COLUMNS = {"bearing_id", "timestamp", "label", "probability", "alert_threshold"}
_IMS_MULTI_MODEL_COLUMNS = {
    "bearing_id",
    "timestamp",
    "file_index",
    "label",
    "model_id",
    "probability",
    "alert_threshold",
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


def load_ims_dashboard_data(
    features_path: Path,
    predictions_path: Path,
    active_model: str | None = None,
) -> dict[str, IMSBearingSeries]:
    """Join chronological 2nd_test features to predictions from each model."""
    with np.load(features_path, allow_pickle=False) as archive:
        missing_arrays = {"features", "labels", "bearing_ids", "timestamps"} - set(archive.files)
        if missing_arrays:
            raise ValueError(f"IMS feature archive is missing arrays: {', '.join(sorted(missing_arrays))}")
        features = np.asarray(archive["features"], dtype=np.float64)
        labels = np.asarray(archive["labels"], dtype=np.int64)
        bearing_ids = np.asarray(archive["bearing_ids"]).astype(str)
        timestamps = np.asarray(archive["timestamps"]).astype(str)

    if features.ndim != 2 or features.shape[1] != len(IMS_FEATURE_NAMES):
        raise ValueError("IMS feature archive does not match the six-feature schema")
    if not np.isfinite(features).all():
        raise ValueError("IMS feature archive contains non-finite values")
    if any(array.ndim != 1 for array in (labels, bearing_ids, timestamps)):
        raise ValueError("IMS labels, bearing IDs, and timestamps must be one-dimensional")
    if any(array.size != features.shape[0] for array in (labels, bearing_ids, timestamps)):
        raise ValueError("IMS feature, label, bearing ID, and timestamp counts do not match")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("IMS labels must contain only 0 and 1")

    feature_rows: dict[tuple[str, str], tuple[float, float, int]] = {}
    for bearing_id, timestamp, label, row in zip(
        bearing_ids, timestamps, labels, features, strict=True
    ):
        try:
            parsed_timestamp = datetime.fromisoformat(timestamp)
        except ValueError as error:
            raise ValueError(f"invalid IMS timestamp for {bearing_id}: {timestamp}") from error
        key = (bearing_id, parsed_timestamp.isoformat(sep=" "))
        if key in feature_rows:
            raise ValueError(f"duplicate IMS feature sample found for {bearing_id} at {timestamp}")
        feature_rows[key] = (float(row[0]), float(row[2]), int(label))

    predictions: dict[tuple[str, str], dict[str, float]] = {}
    file_indices: dict[tuple[str, str], int] = {}
    thresholds_by_model: dict[str, float] = {}
    try:
        with predictions_path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None:
                raise ValueError(f"IMS prediction CSV is empty: {predictions_path}")
            columns = set(reader.fieldnames)
            required_columns = (
                _IMS_MULTI_MODEL_COLUMNS
                if _IMS_MULTI_MODEL_COLUMNS <= columns
                else _IMS_LEGACY_COLUMNS
            )
            missing_columns = required_columns - columns
            if missing_columns:
                raise ValueError(
                    f"IMS prediction CSV is missing columns: {', '.join(sorted(missing_columns))}"
                )
            for line_number, row in enumerate(reader, start=2):
                bearing_id = row["bearing_id"]
                try:
                    timestamp = datetime.fromisoformat(row["timestamp"]).isoformat(sep=" ")
                    label = int(row["label"])
                    probability = float(row["probability"])
                    threshold = float(row["alert_threshold"])
                    model_id = row["model_id"] if "model_id" in row else "neural_network_cpp"
                    file_index = int(row["file_index"]) if "file_index" in row else 0
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"invalid IMS prediction CSV values on line {line_number}"
                    ) from error
                if (
                    not model_id
                    or label not in {0, 1}
                    or not math.isfinite(probability)
                    or not 0 <= probability <= 1
                ):
                    raise ValueError(f"invalid IMS label or probability on CSV line {line_number}")
                if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                    raise ValueError(f"invalid IMS alert threshold on CSV line {line_number}")
                if "file_index" in row and file_index < 1:
                    raise ValueError(f"invalid IMS file index on CSV line {line_number}")
                if model_id in thresholds_by_model and not math.isclose(
                    threshold,
                    thresholds_by_model[model_id],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        f"IMS prediction CSV contains multiple thresholds for {model_id}"
                    )
                thresholds_by_model[model_id] = threshold
                key = (bearing_id, timestamp)
                model_predictions = predictions.setdefault(key, {})
                if model_id in model_predictions:
                    raise ValueError(
                        f"duplicate IMS prediction found for {bearing_id} at {timestamp}"
                    )
                if key not in feature_rows or feature_rows[key][2] != label:
                    raise ValueError(
                        f"IMS prediction does not match feature label on CSV line {line_number}"
                    )
                model_predictions[model_id] = probability
                if "file_index" in row:
                    if key in file_indices and file_indices[key] != file_index:
                        raise ValueError(
                            f"IMS models use different file indices on CSV line {line_number}"
                        )
                    file_indices[key] = file_index
    except OSError as error:
        raise OSError(f"could not read IMS prediction CSV {predictions_path}: {error}") from error

    model_ids = set(thresholds_by_model)
    if not model_ids:
        raise ValueError("IMS prediction CSV contains no model predictions")
    if any(set(values) != model_ids for values in predictions.values()):
        raise ValueError("IMS prediction CSV does not contain every model for every sample")
    missing_features = predictions.keys() - feature_rows.keys()
    if missing_features:
        raise ValueError(
            f"IMS prediction CSV contains {len(missing_features)} samples without features"
        )
    for bearing_id in sorted({key[0] for key in feature_rows}):
        feature_keys = sorted(
            (key for key in feature_rows if key[0] == bearing_id),
            key=lambda key: datetime.fromisoformat(key[1]),
        )
        prediction_keys = sorted(
            (key for key in predictions if key[0] == bearing_id),
            key=lambda key: datetime.fromisoformat(key[1]),
        )
        start = len(feature_keys) - len(prediction_keys)
        if start < 0 or feature_keys[start:] != prediction_keys:
            raise ValueError(
                f"IMS predictions for {bearing_id} must cover a contiguous chronological suffix"
            )
        if file_indices:
            expected_file_indices = list(range(start + 1, len(feature_keys) + 1))
            observed_file_indices = [file_indices[key] for key in prediction_keys]
            if observed_file_indices != expected_file_indices:
                raise ValueError(f"IMS file indices are not aligned for {bearing_id}")
    selected_model = active_model or sorted(model_ids)[0]
    if selected_model not in model_ids:
        raise ValueError(f"active IMS model {selected_model!r} is missing from predictions")

    series_by_bearing: dict[str, IMSBearingSeries] = {}
    for bearing_id in sorted({key[0] for key in feature_rows}):
        keys = sorted(
            (key for key in predictions if key[0] == bearing_id),
            key=lambda key: datetime.fromisoformat(key[1]),
        )
        probability_map = {
            model_id: np.asarray(
                [predictions[key][model_id] for key in keys], dtype=np.float64
            )
            for model_id in sorted(model_ids)
        }
        selected_threshold = thresholds_by_model[selected_model]
        series_by_bearing[bearing_id] = IMSBearingSeries(
            bearing_id=bearing_id,
            timestamps=np.asarray([key[1] for key in keys], dtype=str),
            file_indices=np.asarray(
                [
                    file_indices[key] if key in file_indices else index + 1
                    for index, key in enumerate(keys)
                ],
                dtype=np.int64,
            ),
            probabilities=probability_map[selected_model],
            probabilities_by_model=probability_map,
            labels=np.asarray([feature_rows[key][2] for key in keys], dtype=np.int64),
            rms=np.asarray([feature_rows[key][0] for key in keys], dtype=np.float64),
            kurtosis=np.asarray([feature_rows[key][1] for key in keys], dtype=np.float64),
            above_threshold=np.asarray(
                [predictions[key][selected_model] > selected_threshold for key in keys],
                dtype=bool,
            ),
            alert_threshold=selected_threshold,
            alert_thresholds=thresholds_by_model.copy(),
        )
    if not series_by_bearing:
        raise ValueError("no IMS 2nd_test predictions were loaded")
    return series_by_bearing


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
