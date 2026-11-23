"""Model artifact validation and prediction for a single FEMTO signal."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from src.femto_data import FEATURE_NAMES
from src.ml_core_loader import load_ml_core
from src.model_artifacts import METADATA_PATH, MODEL_PATH, sha256_file


@dataclass(frozen=True)
class PredictionConfig:
    model_path: Path
    means: np.ndarray
    scales: np.ndarray
    threshold: float
    horizon_acquisitions: int
    model_sha256: str
    checksum_verified: bool


@dataclass(frozen=True)
class PredictionResult:
    probability: float
    threshold: float
    threshold_source: Literal["saved model", "override"]
    horizon_acquisitions: int
    model_sha256: str
    checksum_verified: bool


def load_prediction_config(
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
) -> PredictionConfig:
    """Validate preprocessing metadata and its association with the model."""
    if not model_path.is_file():
        raise FileNotFoundError(
            f"model not found: {model_path}; run train_femto_predictive_maintenance.py first"
        )
    if not metadata_path.is_file():
        raise FileNotFoundError(f"preprocessing metadata not found: {metadata_path}")

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"could not read preprocessing metadata {metadata_path}: {error}"
        ) from error

    if not isinstance(metadata, dict):
        raise ValueError(f"preprocessing metadata must be a JSON object: {metadata_path}")
    if metadata.get("format_version") != 2:
        raise ValueError("saved model is not a FEMTO model; retrain with the current pipeline")
    if metadata.get("feature_names") != FEATURE_NAMES:
        raise ValueError("saved feature schema does not match this version of the pipeline")

    try:
        means = np.asarray(metadata.get("means"), dtype=np.float64)
        scales = np.asarray(metadata.get("scales"), dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("saved preprocessing parameters are malformed; retrain the model") from error
    if means.shape != (len(FEATURE_NAMES),) or scales.shape != means.shape:
        raise ValueError("saved preprocessing parameters are malformed; retrain the model")
    if not np.isfinite(means).all() or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("saved preprocessing parameters are invalid; retrain the model")

    threshold = metadata.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("saved alert threshold is invalid; retrain the model")
    try:
        threshold_value = float(threshold)
    except OverflowError as error:
        raise ValueError("saved alert threshold is invalid; retrain the model") from error
    if not math.isfinite(threshold_value) or not 0.0 <= threshold_value <= 1.0:
        raise ValueError("saved alert threshold is invalid; retrain the model")

    horizon = metadata.get("horizon_acquisitions")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("saved failure horizon is invalid; retrain the model")

    model_sha256 = sha256_file(model_path)
    expected_sha256 = metadata.get("model_sha256")
    if expected_sha256 is None and "model_sha256" not in metadata:
        checksum_verified = False
    elif (
        not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        raise ValueError("saved model checksum is malformed; retrain the model")
    elif expected_sha256 != model_sha256:
        raise ValueError(
            "model checksum does not match preprocessing metadata; "
            "restore the matching artifacts or retrain the model"
        )
    else:
        checksum_verified = True

    return PredictionConfig(
        model_path=model_path.resolve(),
        means=means,
        scales=scales,
        threshold=threshold_value,
        horizon_acquisitions=horizon,
        model_sha256=model_sha256,
        checksum_verified=checksum_verified,
    )


def predict_features(
    features: np.ndarray,
    config: PredictionConfig,
    threshold_override: float | None = None,
) -> PredictionResult:
    """Run one validated feature vector through the saved FEMTO model."""
    if threshold_override is None:
        threshold = config.threshold
        threshold_source = "saved model"
    else:
        if (
            isinstance(threshold_override, bool)
            or not isinstance(threshold_override, (int, float))
        ):
            raise ValueError("threshold override must be a finite value between 0 and 1")
        try:
            threshold = float(threshold_override)
        except OverflowError as error:
            raise ValueError(
                "threshold override must be a finite value between 0 and 1"
            ) from error
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold override must be a finite value between 0 and 1")
        threshold_source = "override"

    try:
        feature_vector = np.asarray(features, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("signal features are malformed") from error
    if feature_vector.shape != (len(FEATURE_NAMES),) or not np.isfinite(feature_vector).all():
        raise ValueError("signal features do not match the FEMTO feature schema")

    normalized_features = np.asfortranarray(
        ((feature_vector - config.means) / config.scales).reshape(1, -1)
    )
    if not np.isfinite(normalized_features).all():
        raise ValueError("standardized vibration features contain non-finite values")

    model = load_ml_core().LogisticRegression()
    model.load(str(config.model_path))
    probability = _validate_probability_output(model.predict(normalized_features))
    return PredictionResult(
        probability=probability,
        threshold=threshold,
        threshold_source=threshold_source,
        horizon_acquisitions=config.horizon_acquisitions,
        model_sha256=config.model_sha256,
        checksum_verified=config.checksum_verified,
    )


def _validate_probability_output(output: object) -> float:
    try:
        predictions = np.asarray(output, dtype=np.float64).reshape(-1)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("model returned an invalid probability") from error
    if predictions.shape != (1,):
        raise ValueError("model returned an unexpected number of predictions")
    probability = float(predictions[0])
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("model returned an invalid probability")
    return probability
