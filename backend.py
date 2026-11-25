"""Data access and report generation for the StoneCut PdM dashboard."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import io
import json
import math
from pathlib import Path
from typing import Literal

from src.dashboard_data import IMSBearingSeries, load_ims_dashboard_data


SHARED_ROOT = Path("/mnt/shared/predictive-maintenance/processed")
DEFAULT_FEATURES_PATH = SHARED_ROOT / "ims_test_features.npz"
DEFAULT_PREDICTIONS_PATH = SHARED_ROOT / "ims_test_predictions.csv"
DEFAULT_METRICS_PATH = SHARED_ROOT / "ims_test_metrics.json"

BEARING_ORDER = tuple(f"2nd_test_bearing_{index}" for index in range(1, 5))
COMPONENT_NAMES = {
    "2nd_test_bearing_1": "Main blade spindle (front)",
    "2nd_test_bearing_2": "Main blade spindle (rear)",
    "2nd_test_bearing_3": "Drive flywheel",
    "2nd_test_bearing_4": "Gearmotor transfer shaft",
}

MachineStatus = Literal["operational", "prealarm", "critical"]


@dataclass(frozen=True)
class DashboardData:
    series_by_bearing: dict[str, IMSBearingSeries]
    model_metrics: dict[str, dict[str, float]]
    selected_threshold: float


@dataclass(frozen=True)
class BearingReading:
    bearing_id: str
    timestamp: str
    probability: float
    health_index: float
    rms: float
    kurtosis: float
    label: int
    status: MachineStatus


def load_dashboard_data() -> DashboardData:
    return _load_dashboard_data(
        DEFAULT_FEATURES_PATH,
        DEFAULT_PREDICTIONS_PATH,
        DEFAULT_METRICS_PATH,
    )


def _load_dashboard_data(
    features_path: Path,
    predictions_path: Path,
    metrics_path: Path,
) -> DashboardData:
    series_by_bearing = load_ims_dashboard_data(
        features_path,
        predictions_path,
    )
    probability_values = {
        float(probability)
        for series in series_by_bearing.values()
        for probability in series.probabilities
    }
    if len(probability_values) <= 2:
        raise ValueError(
            "IMS predictions contain class-like scores; continuous sigmoid probabilities are required."
        )
    if set(series_by_bearing) != set(BEARING_ORDER):
        raise ValueError(
            "The IMS 2nd_test dashboard requires exactly four bearings "
            "named bearing_1 through bearing_4."
        )

    sample_counts = {series.timestamps.size for series in series_by_bearing.values()}
    if len(sample_counts) != 1:
        raise ValueError("IMS bearing sequences do not have matching sample counts.")
    reference_timestamps = series_by_bearing[BEARING_ORDER[0]].timestamps
    for bearing_id in BEARING_ORDER[1:]:
        if not (series_by_bearing[bearing_id].timestamps == reference_timestamps).all():
            raise ValueError("IMS bearing timestamps are not aligned across acquisition files.")

    try:
        metrics_document = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid model metrics JSON: {metrics_path}") from error
    models = metrics_document.get("models")
    if not isinstance(models, dict):
        raise ValueError(f"Model metrics are missing from {metrics_path}.")
    try:
        selected_threshold = float(metrics_document["threshold"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Selected alarm threshold is missing from {metrics_path}.") from error
    if not math.isfinite(selected_threshold) or not 0.0 <= selected_threshold <= 1.0:
        raise ValueError(f"Selected alarm threshold is invalid in {metrics_path}.")
    if any(
        not math.isclose(
            series.alert_threshold,
            selected_threshold,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for series in series_by_bearing.values()
    ):
        raise ValueError("Prediction CSV and metrics JSON use different alarm thresholds.")

    model_metrics: dict[str, dict[str, float]] = {}
    for model_key in ("neural_network_cpp", "logistic_regression"):
        model = models.get(model_key)
        if not isinstance(model, dict):
            raise ValueError(f"Metrics for {model_key} are missing from {metrics_path}.")
        values: dict[str, float] = {}
        for metric_name in ("precision", "recall", "f1_score"):
            try:
                value = float(model[metric_name])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid {metric_name} metric for {model_key}."
                ) from error
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"Invalid {metric_name} metric for {model_key}.")
            values[metric_name] = value
        model_metrics[model_key] = values

    return DashboardData(series_by_bearing, model_metrics, selected_threshold)


def classify_probability(probability: float, threshold: float) -> MachineStatus:
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("Failure probability must be finite and in [0, 1].")
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Alarm threshold must be finite and in [0, 1].")
    if probability > threshold:
        return "critical"
    if probability >= 0.45:
        return "prealarm"
    return "operational"


def readings_at_position(
    series_by_bearing: dict[str, IMSBearingSeries],
    position: int,
    threshold: float,
) -> tuple[BearingReading, ...]:
    readings = []
    for bearing_id in BEARING_ORDER:
        series = series_by_bearing[bearing_id]
        if not 0 <= position < series.timestamps.size:
            raise IndexError("Replay position is outside the IMS acquisition range.")
        probability = float(series.probabilities[position])
        readings.append(
            BearingReading(
                bearing_id=bearing_id,
                timestamp=str(series.timestamps[position]),
                probability=probability,
                health_index=(1.0 - probability) * 100.0,
                rms=float(series.rms[position]),
                kurtosis=float(series.kurtosis[position]),
                label=int(series.labels[position]),
                status=classify_probability(probability, threshold),
            )
        )
    return tuple(readings)


def build_maintenance_reports(
    readings: tuple[BearingReading, ...],
    threshold: float,
) -> tuple[str | None, str | None]:
    critical_readings = tuple(
        reading for reading in readings if reading.status == "critical"
    )
    if not critical_readings:
        return None, None

    csv_buffer = io.StringIO(newline="")
    columns = (
        "timestamp",
        "bearing_id",
        "component",
        "failure_probability",
        "health_index_percent",
        "rms",
        "kurtosis",
        "label",
        "alarm_threshold",
        "recommended_action",
    )
    writer = csv.DictWriter(csv_buffer, fieldnames=columns)
    writer.writeheader()
    report_lines = [
        "STONECUT PREDICTIVE MAINTENANCE — INSPECTION ORDER",
        f"Acquisition: {critical_readings[0].timestamp}",
        f"Failure alarm threshold: {threshold:.3f}",
        "",
    ]
    for reading in critical_readings:
        component = COMPONENT_NAMES[reading.bearing_id]
        writer.writerow(
            {
                "timestamp": reading.timestamp,
                "bearing_id": reading.bearing_id,
                "component": component,
                "failure_probability": f"{reading.probability:.6f}",
                "health_index_percent": f"{reading.health_index:.2f}",
                "rms": f"{reading.rms:.6f}",
                "kurtosis": f"{reading.kurtosis:.6f}",
                "label": reading.label,
                "alarm_threshold": f"{threshold:.3f}",
                "recommended_action": "Inspect; replacement requires human assessment",
            }
        )
        report_lines.extend(
            (
                f"Component: {component} ({reading.bearing_id})",
                f"Failure probability: {reading.probability:.1%}",
                f"Health index: {reading.health_index:.1f}%",
                f"RMS: {reading.rms:.5f} | Kurtosis: {reading.kurtosis:.5f}",
                "Action: inspect; replacement requires human assessment.",
                "",
            )
        )

    report_lines.append(
        "Benchmark scores are retrospective and are not calibrated for autonomous maintenance decisions."
    )
    return csv_buffer.getvalue(), "\n".join(report_lines)
