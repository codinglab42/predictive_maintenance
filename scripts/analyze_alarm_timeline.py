#!/usr/bin/env python3
"""Build comparable per-acquisition and alarm-episode timelines from nested OOF results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from src.femto_data import DEFAULT_DATA_DIR, FEATURE_NAMES
from validate_nested_policies import _notification_events


PROJECT_ROOT = Path(__file__).resolve().parent
HORIZONTAL_FEATURES = tuple(
    name for name in FEATURE_NAMES if name.startswith("horizontal_")
)
DISTANCE_BINS = (
    ("0-19", 0, 19),
    ("20-49", 20, 49),
    ("50-99", 50, 99),
    ("100-199", 100, 199),
    ("200+", 200, None),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        help=(
            "directory containing femto_features.npz and nested h20 results "
            "(default: processed/ beside the dataset)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="destination directory (default: --processed-dir)",
    )
    parser.add_argument(
        "--objective",
        choices=("balanced_f1", "sensitivity_first", "precision_at_80pct_recall"),
        default="balanced_f1",
        help="nested policy to compare (default: balanced_f1)",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, path)


def _episode_rows(
    bearing_id: str,
    sample_indices: np.ndarray,
    probabilities: np.ndarray,
    notifications: np.ndarray,
    features: np.ndarray,
    threshold: float,
    horizon: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    above = probabilities >= threshold
    assignments = np.zeros(probabilities.size, dtype=np.int64)
    episodes: list[dict[str, Any]] = []
    window_start = max(0, probabilities.size - horizon)
    start: int | None = None

    for position in range(probabilities.size + 1):
        is_above = position < probabilities.size and bool(above[position])
        if is_above:
            if start is None:
                start = position
            continue
        if start is None:
            continue

        end = position - 1
        episode_number = len(episodes) + 1
        assignments[start : end + 1] = episode_number
        notification_positions = np.flatnonzero(notifications[start : end + 1]) + start
        if notification_positions.size > 1:
            raise ValueError(
                f"{bearing_id} episode {episode_number} has multiple notifications"
            )
        notification_position = (
            int(notification_positions[0]) if notification_positions.size else None
        )

        if notification_position is None:
            category = "not_confirmed"
        elif notification_position < window_start:
            category = (
                "false_before_window" if end < window_start else "early_persistent"
            )
        else:
            category = "confirmed_in_window"

        episode_features = features[start : end + 1]
        row: dict[str, Any] = {
            "bearing_id": bearing_id,
            "episode": episode_number,
            "classification": category,
            "confirmed": int(notification_position is not None),
            "start_sample_index": int(sample_indices[start]),
            "end_sample_index": int(sample_indices[end]),
            "notification_sample_index": (
                int(sample_indices[notification_position])
                if notification_position is not None
                else ""
            ),
            "samples_in_episode": end - start + 1,
            "distance_to_end_at_start": probabilities.size - 1 - start,
            "distance_to_end_at_end": probabilities.size - 1 - end,
            "distance_to_end_at_notification": (
                probabilities.size - 1 - notification_position
                if notification_position is not None
                else ""
            ),
            "mean_probability": float(np.mean(probabilities[start : end + 1])),
            "max_probability": float(np.max(probabilities[start : end + 1])),
        }
        for column, feature_name in enumerate(HORIZONTAL_FEATURES):
            feature_values = episode_features[:, FEATURE_NAMES.index(feature_name)]
            row[f"mean_{feature_name}"] = float(np.mean(feature_values))
            row[f"max_{feature_name}"] = float(np.max(feature_values))
        episodes.append(row)
        start = None

    return episodes, assignments


def _distance_bin(distance: int) -> str:
    for label, lower, upper in DISTANCE_BINS:
        if distance >= lower and (upper is None or distance <= upper):
            return label
    raise ValueError(f"distance cannot be assigned to a bin: {distance}")


def build_timeline(
    processed_dir: Path,
    output_dir: Path,
    objective: str = "balanced_f1",
) -> tuple[Path, Path, Path]:
    summary_path = processed_dir / "femto_nested_h20_validation_results.json"
    predictions_path = processed_dir / "femto_nested_h20_oof_alarm_predictions.csv"
    features_path = processed_dir / "femto_features.npz"
    for path in (summary_path, predictions_path, features_path):
        if not path.is_file():
            raise FileNotFoundError(f"required validation input not found: {path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    horizon = int(summary["horizon_acquisitions"])
    if objective not in summary["objectives"]:
        raise ValueError(f"objective {objective!r} is missing from {summary_path}")

    with np.load(features_path, allow_pickle=False) as data:
        feature_matrix = data["features"]
        labels = data["labels"]
        bearing_ids = data["bearing_ids"]
        sample_indices = data["sample_indices"]
    if feature_matrix.ndim != 2 or feature_matrix.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"unexpected feature matrix shape: {feature_matrix.shape}")
    if not (
        len(labels) == len(bearing_ids) == len(sample_indices) == len(feature_matrix)
    ):
        raise ValueError("features, labels, bearing IDs, and sample indices are misaligned")

    prediction_rows = _read_csv(predictions_path)
    prediction_by_key: dict[tuple[str, int], dict[str, str]] = {}
    for row in prediction_rows:
        key = (row["bearing_id"], int(row["sample_index"]))
        if key in prediction_by_key:
            raise ValueError(f"duplicate nested prediction row: {key}")
        prediction_by_key[key] = row
    if len(prediction_by_key) != len(feature_matrix):
        raise ValueError("nested predictions do not match the saved feature sample count")

    feature_keys = {
        (str(bearing), int(sample_id))
        for bearing, sample_id in zip(bearing_ids, sample_indices)
    }
    if feature_keys != set(prediction_by_key):
        raise ValueError("nested predictions and feature samples have different keys")

    fold_by_bearing = {
        fold["outer_test_bearing"]: fold for fold in summary["fold_details"]
    }
    if set(fold_by_bearing) != set(bearing_ids.tolist()):
        raise ValueError("outer fold metadata does not cover the feature bearings")

    timeline_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    comparison: dict[str, dict[str, Any]] = {}
    counts = Counter()

    for bearing_id in sorted(fold_by_bearing):
        indices = np.flatnonzero(bearing_ids == bearing_id)
        indices = indices[np.argsort(sample_indices[indices])]
        bearing_samples = sample_indices[indices]
        saved_feature_labels = labels[indices]
        bearing_features = feature_matrix[indices]
        rows = [
            prediction_by_key[(bearing_id, int(sample_id))]
            for sample_id in bearing_samples
        ]
        probabilities = np.asarray([float(row["probability"]) for row in rows])
        saved_labels = np.asarray(
            [int(row["failure_within_horizon"]) for row in rows], dtype=np.int64
        )
        if int(saved_labels.sum()) != min(horizon, len(indices)):
            raise ValueError(f"target window does not contain {horizon} records for {bearing_id}")
        expected_labels = np.zeros(len(indices), dtype=np.int64)
        expected_labels[-horizon:] = 1
        if not np.array_equal(saved_labels, expected_labels):
            raise ValueError(f"nested target labels do not mark the final records for {bearing_id}")
        if not np.array_equal(saved_feature_labels[-horizon:], np.ones(min(horizon, len(indices)))):
            raise ValueError(
                f"feature labels disagree with the nested target in the final window for {bearing_id}"
            )
        bearing_labels = saved_labels

        policy = fold_by_bearing[bearing_id]["selected_policies"][objective]
        threshold = float(policy["threshold"])
        confirmation = int(policy["confirmation"])
        cooldown = int(policy["cooldown"])
        notifications, _ = _notification_events(
            probabilities, threshold, confirmation, cooldown
        )
        csv_notifications = np.asarray(
            [int(row[f"{objective}_notification"]) for row in rows],
            dtype=np.int64,
        )
        if not np.array_equal(notifications, csv_notifications):
            raise ValueError(f"saved notification flags do not match selected policy for {bearing_id}")

        episodes, episode_assignments = _episode_rows(
            bearing_id,
            bearing_samples,
            probabilities,
            notifications,
            bearing_features,
            threshold,
            horizon,
        )
        episode_rows.extend(episodes)
        window_start = max(0, len(indices) - horizon)
        pre_window_above = int(np.count_nonzero((np.arange(len(indices)) < window_start) & (probabilities >= threshold)))
        category_counts = Counter(row["classification"] for row in episodes)
        counts.update(category_counts)

        timeline_by_distance: dict[str, dict[str, Any]] = {}
        distances = np.arange(len(indices) - 1, -1, -1)
        for bin_label, lower, upper in DISTANCE_BINS:
            mask = distances >= lower
            if upper is not None:
                mask &= distances <= upper
            if not np.any(mask):
                continue
            feature_values = bearing_features[mask]
            timeline_by_distance[bin_label] = {
                "sample_count": int(mask.sum()),
                "positive_samples": int(bearing_labels[mask].sum()),
                "median_probability": float(np.median(probabilities[mask])),
                "mean_probability": float(np.mean(probabilities[mask])),
                "above_threshold_samples": int(np.count_nonzero(probabilities[mask] >= threshold)),
                "notifications": int(notifications[mask].sum()),
                "median_horizontal_features": {
                    name: float(np.median(feature_values[:, FEATURE_NAMES.index(name)]))
                    for name in HORIZONTAL_FEATURES
                },
            }

        comparison[bearing_id] = {
            "samples": int(len(indices)),
            "positive_samples": int(bearing_labels.sum()),
            "selected_policy": {
                "threshold": threshold,
                "confirmation": confirmation,
                "cooldown": cooldown,
                "inner_constraint_fallback": policy["inner_constraint_fallback"],
            },
            "pre_window_samples_above_threshold": pre_window_above,
            "episodes": dict(category_counts),
            "distance_to_failure_bins": timeline_by_distance,
        }

        for position, row in enumerate(rows):
            timeline_row: dict[str, Any] = {
                "bearing_id": bearing_id,
                "sample_index": int(bearing_samples[position]),
                "distance_to_end_acquisitions": int(distances[position]),
                "failure_within_horizon": int(bearing_labels[position]),
                "distance_bin": _distance_bin(int(distances[position])),
                "probability": float(probabilities[position]),
                "selected_threshold": threshold,
                "confirmation_samples": confirmation,
                "cooldown_samples": cooldown,
                "above_threshold": int(probabilities[position] >= threshold),
                "notification": int(notifications[position]),
                "episode": int(episode_assignments[position]) or "",
            }
            for name in HORIZONTAL_FEATURES:
                timeline_row[name] = float(
                    bearing_features[position, FEATURE_NAMES.index(name)]
                )
            timeline_rows.append(timeline_row)

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"h{horizon}_{objective}"
    timeline_path = output_dir / f"femto_alarm_timeline_{suffix}.csv"
    episodes_path = output_dir / f"femto_alarm_episodes_{suffix}.csv"
    comparison_path = output_dir / f"femto_alarm_timeline_{suffix}_summary.json"
    timeline_fields = [
        "bearing_id",
        "sample_index",
        "distance_to_end_acquisitions",
        "failure_within_horizon",
        "distance_bin",
        "probability",
        "selected_threshold",
        "confirmation_samples",
        "cooldown_samples",
        "above_threshold",
        "notification",
        "episode",
        *HORIZONTAL_FEATURES,
    ]
    episode_fields = [
        "bearing_id",
        "episode",
        "classification",
        "confirmed",
        "start_sample_index",
        "end_sample_index",
        "notification_sample_index",
        "samples_in_episode",
        "distance_to_end_at_start",
        "distance_to_end_at_end",
        "distance_to_end_at_notification",
        "mean_probability",
        "max_probability",
        *[
            field
            for name in HORIZONTAL_FEATURES
            for field in (f"mean_{name}", f"max_{name}")
        ],
    ]
    _write_csv(timeline_path, timeline_fields, timeline_rows)
    _write_csv(episodes_path, episode_fields, episode_rows)
    summary_out = {
        "source_validation": summary_path.name,
        "source_predictions": predictions_path.name,
        "source_features": features_path.name,
        "label_source": (
            "Nested h20 prediction CSV. The feature archive retains labels from the "
            "previous 21-acquisition target and is used only for feature values and sample keys."
        ),
        "objective": objective,
        "horizon_acquisitions": horizon,
        "distance_definition": (
            "Record positions remaining until the final record: zero is the final "
            "acquisition. Bins are indexed by this distance."
        ),
        "timeline_rows": len(timeline_rows),
        "episode_rows": len(episode_rows),
        "episode_classification_counts": dict(counts),
        "bearings": comparison,
    }
    temporary_path = comparison_path.with_suffix(comparison_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(summary_out, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, comparison_path)
    return timeline_path, episodes_path, comparison_path


def main() -> None:
    args = parse_args()
    processed_dir = (
        args.processed_dir.expanduser().resolve()
        if args.processed_dir
        else DEFAULT_DATA_DIR.parent.parent / "processed"
    )
    output_dir = (
        args.output_dir.expanduser().resolve() if args.output_dir else processed_dir
    )
    timeline, episodes, summary = build_timeline(
        processed_dir,
        output_dir,
        args.objective,
    )
    print(f"Timeline: {timeline}")
    print(f"Episodi: {episodes}")
    print(f"Confronto: {summary}")


if __name__ == "__main__":
    main()
