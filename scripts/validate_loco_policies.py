#!/usr/bin/env python3
"""Run leave-one-condition-out validation for FEMTO alarm policies."""

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
import sys
from pathlib import Path
from typing import Any

import numpy as np

from src.femto_data import (
    DEFAULT_DATA_DIR,
    build_labeled_dataset,
    group_bearings_by_condition,
    standardize,
)
from src.ml_core_loader import load_ml_core
from validate_nested_policies import (
    COOLDOWNS,
    CONFIRMATION_SAMPLES,
    MODEL_PARAMETERS,
    OBJECTIVES,
    THRESHOLDS,
    evaluate_policy,
    select_policies,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="directory containing the extracted FEMTO/PRONOSTIA dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="directory for validation CSV/JSON (default: processed/ beside the dataset)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=20,
        help="number of final acquisitions labeled positive and used as the event window",
    )
    parser.add_argument(
        "--weighted-loss",
        action="store_true",
        help=(
            "legacy flag name: oversample the minority class in LOCO training "
            "(this is not class-weighted loss)"
        ),
    )
    args = parser.parse_args()
    if args.horizon < 1:
        parser.error("--horizon must be a positive number of acquisitions")
    return args


def _rebalance_training_set(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = labels.astype(np.int64)
    classes, counts = np.unique(labels, return_counts=True)
    if classes.size != 2:
        return features, labels
    minority_class = int(classes[np.argmin(counts)])
    majority_class = int(classes[np.argmax(counts)])
    minority_mask = labels == minority_class
    minority_features = features[minority_mask]
    minority_labels = labels[minority_mask]
    majority_count = int(np.sum(labels == majority_class))
    repeat_count = max(1, int(np.ceil(majority_count / minority_features.shape[0])))
    if repeat_count > 1:
        minority_features = np.repeat(minority_features, repeat_count, axis=0)[:majority_count]
        minority_labels = np.repeat(minority_labels, repeat_count)[:majority_count]
    majority_features = features[labels == majority_class]
    majority_labels = labels[labels == majority_class]
    balanced_features = np.vstack([majority_features, minority_features])
    balanced_labels = np.concatenate([majority_labels, minority_labels])
    permutation = np.random.default_rng(0).permutation(balanced_features.shape[0])
    return balanced_features[permutation], balanced_labels[permutation]


def _fit_predict(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    train_bearings: set[str],
    test_bearing: str,
    *,
    weighted_loss: bool = False,
) -> np.ndarray:
    train_mask = np.isin(groups, sorted(train_bearings))
    test_mask = groups == test_bearing
    if not train_mask.any() or not test_mask.any():
        raise ValueError("each validation fold must contain training and test samples")
    if np.unique(labels[train_mask]).size != 2:
        raise ValueError(
            "a model-training fold must contain both failure classes; "
            "adjust --horizon or inspect the dataset"
        )

    train_features, test_features, _, _ = standardize(
        features[train_mask],
        features[test_mask],
    )
    model = load_ml_core().LogisticRegression(
        MODEL_PARAMETERS["learning_rate"],
        MODEL_PARAMETERS["iterations"],
        MODEL_PARAMETERS["regularization"],
        MODEL_PARAMETERS["tolerance"],
        MODEL_PARAMETERS["verbose"],
    )
    y_train = labels[train_mask].astype(np.float64)
    if weighted_loss:
        train_features, y_train = _rebalance_training_set(train_features, y_train)
    model.fit(train_features, y_train)
    probabilities = np.asarray(model.predict(test_features), dtype=np.float64).reshape(-1)
    if probabilities.shape[0] != int(test_mask.sum()):
        raise RuntimeError(f"model returned an unexpected prediction count for {test_bearing}")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise RuntimeError(f"model returned invalid probabilities for {test_bearing}")
    return probabilities


def _write_predictions(
    path: Path,
    bearing_ids: np.ndarray,
    sample_indices: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    selected_by_condition: dict[int, dict[str, dict[str, Any]]],
) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    objectives = tuple(OBJECTIVES)
    fieldnames = [
        "condition_id",
        "bearing_id",
        "sample_index",
        "failure_within_horizon",
        "probability",
    ]
    for objective in objectives:
        fieldnames.extend(
            [
                f"{objective}_threshold",
                f"{objective}_confirmation",
                f"{objective}_cooldown",
                f"{objective}_above_threshold",
                f"{objective}_notification",
            ]
        )

    with temporary_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for condition_id, condition_bearings in group_bearings_by_condition(bearing_ids.tolist()).items():
            for bearing_id in condition_bearings:
                indices = np.flatnonzero(bearing_ids == bearing_id)
                indices = indices[np.argsort(sample_indices[indices])]
                notification_flags = {
                    objective: _notification_events(
                        probabilities[indices],
                        float(selected_by_condition[condition_id][objective]["threshold"]),
                        int(selected_by_condition[condition_id][objective]["confirmation"]),
                        int(selected_by_condition[condition_id][objective]["cooldown"]),
                    )[0]
                    for objective in objectives
                }
                for position, index in enumerate(indices):
                    row: dict[str, Any] = {
                        "condition_id": condition_id,
                        "bearing_id": bearing_id,
                        "sample_index": int(sample_indices[index]),
                        "failure_within_horizon": int(labels[index]),
                        "probability": f"{probabilities[index]:.10f}",
                    }
                    for objective in objectives:
                        policy = selected_by_condition[condition_id][objective]
                        row[f"{objective}_threshold"] = f"{float(policy['threshold']):.2f}"
                        row[f"{objective}_confirmation"] = int(policy["confirmation"])
                        row[f"{objective}_cooldown"] = int(policy["cooldown"])
                        row[f"{objective}_above_threshold"] = int(probabilities[index] >= float(policy["threshold"]))
                        row[f"{objective}_notification"] = int(notification_flags[objective][position])
                    writer.writerow(row)
    os.replace(temporary_path, path)


def _notification_events(
    probabilities: np.ndarray,
    threshold: float,
    confirmation: int,
    cooldown: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    above = probabilities >= threshold
    notifications = np.zeros(probabilities.size, dtype=np.int64)
    events: list[tuple[int, int]] = []
    run_start: int | None = None
    last_notification: int | None = None

    for position in range(probabilities.size + 1):
        is_above = position < probabilities.size and bool(above[position])
        if is_above:
            if run_start is None:
                run_start = position
            continue
        if run_start is None:
            continue

        run_end = position - 1
        confirmation_position = run_start + confirmation - 1
        cooldown_elapsed = (
            last_notification is None
            or confirmation_position - last_notification >= cooldown
        )
        if confirmation_position <= run_end and cooldown_elapsed:
            notifications[confirmation_position] = 1
            events.append((confirmation_position, run_end))
            last_notification = confirmation_position
        run_start = None

    return notifications, events


def run_loco_validation(data_dir: Path, output_dir: Path, horizon: int, weighted_loss: bool = False) -> tuple[Path, Path]:
    print("Fase 1/4: estrazione feature e target", file=sys.stderr, flush=True)
    features, labels, bearing_ids, sample_indices = build_labeled_dataset(data_dir, horizon)
    unique_bearings = list(dict.fromkeys(bearing_ids.tolist()))
    condition_groups = group_bearings_by_condition(unique_bearings)
    if len(condition_groups) < 2:
        raise ValueError("Leave-one-condition-out needs at least two operating conditions")

    outer_probabilities = np.full(labels.size, np.nan, dtype=np.float64)
    selected_by_condition: dict[int, dict[str, dict[str, Any]]] = {}
    condition_details: list[dict[str, Any]] = []

    for condition_id in sorted(condition_groups):
        test_bearings = condition_groups[condition_id]
        training_bearings = [
            bearing
            for other_condition in condition_groups.values()
            for bearing in other_condition
            if bearing not in test_bearings
        ]
        if len(training_bearings) < 2:
            raise ValueError(f"training condition {condition_id} does not have enough bearings")

        print(
            f"Fase 2/4: piega esterna condizione {condition_id} "
            f"({', '.join(test_bearings)})",
            file=sys.stderr,
            flush=True,
        )

        inner_probabilities: dict[str, np.ndarray] = {}
        for inner_bearing in training_bearings:
            training_for_inner = [bearing for bearing in training_bearings if bearing != inner_bearing]
            inner_probabilities[inner_bearing] = _fit_predict(
                features,
                labels,
                bearing_ids,
                set(training_for_inner),
                inner_bearing,
                weighted_loss=weighted_loss,
            )

        selected = select_policies(inner_probabilities, horizon)
        selected_by_condition[condition_id] = selected

        outer_prediction_by_bearing: dict[str, np.ndarray] = {}
        for test_bearing in test_bearings:
            outer_prediction_by_bearing[test_bearing] = _fit_predict(
                features,
                labels,
                bearing_ids,
                set(training_bearings),
                test_bearing,
                weighted_loss=weighted_loss,
            )
            indices = np.flatnonzero(bearing_ids == test_bearing)
            indices = indices[np.argsort(sample_indices[indices])]
            if outer_prediction_by_bearing[test_bearing].size != indices.size:
                raise RuntimeError(f"outer predictions misaligned for {test_bearing}")
            outer_probabilities[indices] = outer_prediction_by_bearing[test_bearing]

        condition_detail = {
            "condition_id": condition_id,
            "test_bearings": test_bearings,
            "training_bearings": training_bearings,
            "selected_policies": selected,
        }
        for objective in OBJECTIVES:
            policy = selected[objective]
            condition_metrics = evaluate_policy(
                {bearing: outer_prediction_by_bearing[bearing] for bearing in test_bearings},
                float(policy["threshold"]),
                int(policy["confirmation"]),
                int(policy["cooldown"]),
                horizon,
            )
            condition_detail[objective] = {
                key: value
                for key, value in condition_metrics.items()
                if key != "per_bearing"
            }
        condition_details.append(condition_detail)

    if not np.isfinite(outer_probabilities).all():
        raise RuntimeError("LOCO validation did not produce one prediction per sample")

    print("Fase 3/4: aggregazione delle metriche evento", file=sys.stderr, flush=True)
    condition_results: dict[int, dict[str, Any]] = {}
    for condition_id in sorted(condition_groups):
        bearing_probs = {
            bearing: outer_probabilities[bearing_ids == bearing][
                np.argsort(sample_indices[bearing_ids == bearing])
            ]
            for bearing in condition_groups[condition_id]
        }
        per_objective: dict[str, dict[str, Any]] = {}
        for objective in OBJECTIVES:
            policy = selected_by_condition[condition_id][objective]
            metrics = evaluate_policy(
                bearing_probs,
                float(policy["threshold"]),
                int(policy["confirmation"]),
                int(policy["cooldown"]),
                horizon,
            )
            per_objective[objective] = {
                key: value
                for key, value in metrics.items()
                if key != "per_bearing"
            }
        condition_results[condition_id] = {
            "test_bearings": condition_groups[condition_id],
            "selected_policies": selected_by_condition[condition_id],
            "metrics_by_objective": per_objective,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"femto_loco_h{horizon}_alarm_predictions.csv"
    json_path = output_dir / f"femto_loco_h{horizon}_validation_results.json"
    _write_predictions(csv_path, bearing_ids, sample_indices, labels, outer_probabilities, selected_by_condition)

    print("Fase 4/4: salvataggio dei riepiloghi", file=sys.stderr, flush=True)
    summary = {
        "method": (
            "Leave-one-condition-out validation. All bearings in one operating condition are held out, "
            "then the policy is selected only using the remaining conditions."
        ),
        "data_dir": str(data_dir),
        "horizon_acquisitions": horizon,
        "candidate_thresholds": list(THRESHOLDS),
        "candidate_confirmation_samples": list(CONFIRMATION_SAMPLES),
        "candidate_cooldowns_samples": list(COOLDOWNS),
        "objectives": OBJECTIVES,
        "model": MODEL_PARAMETERS,
        "weighted_loss": weighted_loss,
        "training_sampling": "minority_class_oversampling" if weighted_loss else "none",
        "condition_to_bearings": {
            condition_id: bearings for condition_id, bearings in sorted(condition_groups.items())
        },
        "samples": int(labels.size),
        "positive_samples": int(labels.sum()),
        "condition_results": condition_results,
        "fold_details": condition_details,
        "predictions_csv": csv_path.name,
    }
    with json_path.open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, allow_nan=False)
        output.write("\n")

    print(f"Campioni: {labels.size:,} | positivi: {int(labels.sum()):,}")
    for condition_id, condition in sorted(condition_results.items()):
        objective = "balanced_f1"
        metrics = condition["metrics_by_objective"][objective]
        print(
            f"condizione {condition_id}: "
            f"bearings={len(condition['test_bearings'])}, "
            f"false episodes={metrics['false_episodes']}, "
            f"F1={metrics['event_f1']:.1%}, "
            f"recall={metrics['event_recall']:.1%}, "
            f"precision={metrics['event_precision']:.1%}"
        )
    if weighted_loss:
        print("modalita': oversampling della classe minoritaria attivo")
    print(f"Predizioni: {csv_path}")
    print(f"Riepilogo: {json_path}")
    return csv_path, json_path


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else data_dir.parent.parent / "processed"
    )
    run_loco_validation(data_dir, output_dir, args.horizon, weighted_loss=args.weighted_loss)


if __name__ == "__main__":
    main()
