#!/usr/bin/env python3
"""Run nested leave-one-bearing-out validation for FEMTO alarm policies."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import csv
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from src.femto_data import DEFAULT_DATA_DIR, build_labeled_dataset, standardize
from src.ml_core_loader import load_ml_core


MODEL_PARAMETERS = {
    "type": "LogisticRegression",
    "learning_rate": 0.05,
    "iterations": 1500,
    "regularization": 0.01,
    "tolerance": 1e-4,
    "verbose": False,
}
THRESHOLDS = (0.10, 0.20, 0.30, 0.50)
CONFIRMATION_SAMPLES = (1, 3, 5)
COOLDOWNS = (0, 10, 20, 50)
OBJECTIVES = {
    "sensitivity_first": (
        "Maximize inner held-out bearing recall; ties: fewer false episodes, "
        "higher event F1, greater upper-median lead."
    ),
    "balanced_f1": (
        "Maximize event F1; ties: higher recall, fewer false episodes, "
        "greater upper-median lead."
    ),
    "precision_at_80pct_recall": (
        "Among policies with at least 80% inner bearing recall, maximize event "
        "precision; ties: higher recall, fewer false episodes, greater "
        "upper-median lead. If none qualify, choose highest recall and mark fallback."
    ),
}


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
    args = parser.parse_args()
    if args.horizon < 1:
        parser.error("--horizon must be a positive number of acquisitions")
    return args


def _fit_predict(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    train_bearings: set[str],
    test_bearing: str,
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
    model.fit(train_features, labels[train_mask].astype(np.float64))
    probabilities = np.asarray(model.predict(test_features), dtype=np.float64).reshape(-1)
    if probabilities.shape[0] != int(test_mask.sum()):
        raise RuntimeError(f"model returned an unexpected prediction count for {test_bearing}")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0) or np.any(
        probabilities > 1.0
    ):
        raise RuntimeError(f"model returned invalid probabilities for {test_bearing}")
    return probabilities


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


def _upper_median(values: list[int]) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def evaluate_policy(
    probabilities_by_bearing: dict[str, np.ndarray],
    threshold: float,
    confirmation: int,
    cooldown: int,
    horizon: int,
) -> dict[str, Any]:
    per_bearing: list[dict[str, Any]] = []
    leads: list[int] = []
    false_episodes = 0
    early_persistent_episodes = 0
    in_window_episodes = 0

    for bearing_id, probabilities in probabilities_by_bearing.items():
        if probabilities.ndim != 1 or probabilities.size == 0:
            raise ValueError(f"{bearing_id} must have a non-empty 1D probability sequence")
        window_start = max(0, probabilities.size - horizon)
        _, events = _notification_events(probabilities, threshold, confirmation, cooldown)
        bearing_false = 0
        bearing_persistent = 0
        bearing_in_window = 0
        bearing_leads: list[int] = []
        for confirmation_position, run_end in events:
            if confirmation_position < window_start:
                if run_end < window_start:
                    bearing_false += 1
                else:
                    bearing_persistent += 1
            else:
                bearing_in_window += 1
                bearing_leads.append(probabilities.size - 1 - confirmation_position)

        false_episodes += bearing_false
        early_persistent_episodes += bearing_persistent
        in_window_episodes += bearing_in_window
        leads.extend(bearing_leads)
        per_bearing.append(
            {
                "bearing_id": bearing_id,
                "false_episodes": bearing_false,
                "early_persistent_episodes": bearing_persistent,
                "in_window_episodes": bearing_in_window,
                "detected_in_window": bearing_in_window > 0,
                "first_in_window_lead": max(bearing_leads) if bearing_leads else None,
                "in_window_leads": bearing_leads,
            }
        )

    bearing_count = len(probabilities_by_bearing)
    detected_bearings = sum(item["detected_in_window"] for item in per_bearing)
    recall = detected_bearings / bearing_count if bearing_count else 0.0
    precision_denominator = detected_bearings + false_episodes
    precision = (
        detected_bearings / precision_denominator if precision_denominator else 0.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "heldout_bearings": bearing_count,
        "detected_bearings": detected_bearings,
        "missed_bearings": bearing_count - detected_bearings,
        "false_episodes": false_episodes,
        "early_persistent_episodes": early_persistent_episodes,
        "in_window_episodes": in_window_episodes,
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
        "median_in_window_lead": _upper_median(leads),
        "per_bearing": per_bearing,
    }


def _selection_key(metrics: dict[str, Any], objective: str) -> tuple[float, ...]:
    precision = metrics["event_precision"]
    recall = metrics["event_recall"]
    f1 = metrics["event_f1"]
    fewer_false = -metrics["false_episodes"]
    lead = metrics["median_in_window_lead"]
    greater_lead = lead if lead is not None else -1.0
    if objective == "sensitivity_first":
        return (recall, fewer_false, f1, greater_lead)
    if objective == "balanced_f1":
        return (f1, recall, fewer_false, greater_lead)
    if objective == "precision_at_80pct_recall":
        return (precision, recall, fewer_false, greater_lead)
    raise ValueError(f"unknown policy-selection objective: {objective}")


def select_policies(
    probabilities_by_bearing: dict[str, np.ndarray],
    horizon: int,
    thresholds: tuple[float, ...] = THRESHOLDS,
) -> dict[str, dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for threshold, confirmation, cooldown in itertools.product(
        thresholds,
        CONFIRMATION_SAMPLES,
        COOLDOWNS,
    ):
        metrics = evaluate_policy(
            probabilities_by_bearing,
            threshold,
            confirmation,
            cooldown,
            horizon,
        )
        candidates.append(
            {
                "threshold": threshold,
                "confirmation": confirmation,
                "cooldown": cooldown,
                "metrics": metrics,
            }
        )

    selected: dict[str, dict[str, Any]] = {}
    for objective in OBJECTIVES:
        fallback = False
        eligible = candidates
        if objective == "precision_at_80pct_recall":
            eligible = [
                candidate
                for candidate in candidates
                if candidate["metrics"]["event_recall"] >= 0.8
            ]
            if not eligible:
                fallback = True
                eligible = candidates
                key = lambda candidate: (
                    candidate["metrics"]["event_recall"],
                    *_selection_key(candidate["metrics"], objective),
                )
            else:
                key = lambda candidate: _selection_key(candidate["metrics"], objective)
        else:
            key = lambda candidate: _selection_key(candidate["metrics"], objective)

        best = max(eligible, key=key)
        selected[objective] = {
            "threshold": best["threshold"],
            "confirmation": best["confirmation"],
            "cooldown": best["cooldown"],
            "inner_metrics": {
                key: value
                for key, value in best["metrics"].items()
                if key != "per_bearing"
            },
            "inner_constraint_fallback": fallback,
            "inner_per_bearing": best["metrics"]["per_bearing"],
        }
    return selected


def _atomic_write_text(path: Path, contents: str) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(contents, encoding="utf-8")
    os.replace(temporary_path, path)


def _write_predictions(
    path: Path,
    bearing_ids: np.ndarray,
    sample_indices: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    policy_by_bearing: dict[str, dict[str, dict[str, Any]]],
) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    objectives = tuple(OBJECTIVES)
    fieldnames = [
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
        for bearing_id in dict.fromkeys(bearing_ids.tolist()):
            indices = np.flatnonzero(bearing_ids == bearing_id)
            indices = indices[np.argsort(sample_indices[indices])]
            notification_flags = {
                objective: _notification_events(
                    probabilities[indices],
                    float(policy_by_bearing[bearing_id][objective]["threshold"]),
                    int(policy_by_bearing[bearing_id][objective]["confirmation"]),
                    int(policy_by_bearing[bearing_id][objective]["cooldown"]),
                )[0]
                for objective in objectives
            }
            for position, index in enumerate(indices):
                row: dict[str, Any] = {
                    "bearing_id": bearing_id,
                    "sample_index": int(sample_indices[index]),
                    "failure_within_horizon": int(labels[index]),
                    "probability": f"{probabilities[index]:.10f}",
                }
                for objective in objectives:
                    policy = policy_by_bearing[bearing_id][objective]
                    threshold = float(policy["threshold"])
                    row[f"{objective}_threshold"] = f"{threshold:.2f}"
                    row[f"{objective}_confirmation"] = int(policy["confirmation"])
                    row[f"{objective}_cooldown"] = int(policy["cooldown"])
                    row[f"{objective}_above_threshold"] = int(
                        probabilities[index] >= threshold
                    )
                    row[f"{objective}_notification"] = int(
                        notification_flags[objective][position]
                    )
                writer.writerow(row)

    os.replace(temporary_path, path)


def run_validation(
    data_dir: Path,
    output_dir: Path,
    horizon: int,
) -> tuple[Path, Path]:
    print("Fase 1/4: estrazione delle feature e target", file=sys.stderr, flush=True)
    last_progress_bearing: str | None = None

    def report_progress(processed: int, total: int, bearing_id: str) -> None:
        nonlocal last_progress_bearing
        if processed % 100 == 0 or processed == total or bearing_id != last_progress_bearing:
            print(
                f"\r  Campioni elaborati: {processed:,}/{total:,} "
                f"({processed / total:.1%}) | {bearing_id}",
                end="\n" if processed == total else "",
                file=sys.stderr,
                flush=True,
            )
        last_progress_bearing = bearing_id

    features, labels, bearing_ids, sample_indices = build_labeled_dataset(
        data_dir,
        horizon,
        progress_callback=report_progress,
    )
    bearings = list(dict.fromkeys(bearing_ids.tolist()))
    if len(bearings) < 3:
        raise ValueError("nested leave-one-bearing-out validation requires at least 3 bearings")

    outer_probabilities = np.full(labels.size, np.nan, dtype=np.float64)
    policies_by_bearing: dict[str, dict[str, dict[str, Any]]] = {}
    fold_details: list[dict[str, Any]] = []

    for fold_number, outer_bearing in enumerate(bearings, start=1):
        print(
            f"Fase 2/4: piega esterna {fold_number}/{len(bearings)} "
            f"({outer_bearing})",
            file=sys.stderr,
            flush=True,
        )
        inner_bearings = [bearing for bearing in bearings if bearing != outer_bearing]
        inner_probabilities: dict[str, np.ndarray] = {}
        for inner_bearing in inner_bearings:
            training_bearings = set(inner_bearings) - {inner_bearing}
            mask = bearing_ids == inner_bearing
            inner_probabilities[inner_bearing] = _fit_predict(
                features,
                labels,
                bearing_ids,
                training_bearings,
                inner_bearing,
            )
            if inner_probabilities[inner_bearing].size != int(mask.sum()):
                raise RuntimeError(f"inner predictions misaligned for {inner_bearing}")

        selected = select_policies(inner_probabilities, horizon)
        outer_prediction = _fit_predict(
            features,
            labels,
            bearing_ids,
            set(inner_bearings),
            outer_bearing,
        )
        outer_indices = np.flatnonzero(bearing_ids == outer_bearing)
        outer_indices = outer_indices[np.argsort(sample_indices[outer_indices])]
        if outer_prediction.size != outer_indices.size:
            raise RuntimeError(f"outer predictions misaligned for {outer_bearing}")
        outer_probabilities[outer_indices] = outer_prediction
        policies_by_bearing[outer_bearing] = selected
        fold_details.append(
            {
                "outer_test_bearing": outer_bearing,
                "inner_validation_bearings": inner_bearings,
                "selected_policies": selected,
            }
        )

    if not np.isfinite(outer_probabilities).all():
        raise RuntimeError("nested validation did not produce one prediction per sample")

    print("Fase 3/4: aggregazione delle metriche evento", file=sys.stderr, flush=True)
    outer_results: dict[str, dict[str, Any]] = {}
    probabilities_by_bearing = {
        bearing: outer_probabilities[bearing_ids == bearing][
            np.argsort(sample_indices[bearing_ids == bearing])
        ]
        for bearing in bearings
    }
    for objective in OBJECTIVES:
        # Each outer bearing uses only the policy selected without that bearing.
        per_bearing: list[dict[str, Any]] = []
        for bearing in bearings:
            policy = policies_by_bearing[bearing][objective]
            bearing_metrics = evaluate_policy(
                {bearing: probabilities_by_bearing[bearing]},
                policy["threshold"],
                policy["confirmation"],
                policy["cooldown"],
                horizon,
            )["per_bearing"][0]
            per_bearing.append(bearing_metrics)
        detected = sum(item["detected_in_window"] for item in per_bearing)
        false = sum(item["false_episodes"] for item in per_bearing)
        persistent = sum(item["early_persistent_episodes"] for item in per_bearing)
        in_window = sum(item["in_window_episodes"] for item in per_bearing)
        leads = [lead for item in per_bearing for lead in item["in_window_leads"]]
        recall = detected / len(bearings)
        denominator = detected + false
        precision = detected / denominator if denominator else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        outer_results[objective] = {
            "outer_folds": len(bearings),
            "detected_bearings": detected,
            "missed_bearings": len(bearings) - detected,
            "false_episodes": false,
            "early_persistent_episodes": persistent,
            "in_window_episodes": in_window,
            "event_precision": precision,
            "event_recall": recall,
            "event_f1": f1,
            "median_in_window_lead": _upper_median(leads),
            "per_bearing": per_bearing,
        }

    suffix = f"h{horizon}"
    csv_path = output_dir / f"femto_nested_{suffix}_oof_alarm_predictions.csv"
    json_path = output_dir / f"femto_nested_{suffix}_validation_results.json"

    _write_predictions(
        csv_path,
        bearing_ids,
        sample_indices,
        labels,
        outer_probabilities,
        policies_by_bearing,
    )
    print("Fase 4/4: salvataggio dei riepiloghi", file=sys.stderr, flush=True)
    summary = {
        "method": (
            "Nested leave-one-bearing-out validation. Each outer bearing is excluded "
            "from model fitting and policy selection; inner leave-one-bearing-out "
            "predictions select the alarm policy using only the other bearings."
        ),
        "scope": (
            "Model hyperparameters are fixed. The target and event window contain "
            "the final horizon records per bearing. No operational model or "
            "configuration is modified."
        ),
        "data_dir": str(data_dir),
        "horizon_acquisitions": horizon,
        "target_definition": (
            "The final horizon records in acquisition order, including the final "
            "record, are positive."
        ),
        "candidate_thresholds": list(THRESHOLDS),
        "candidate_confirmation_samples": list(CONFIRMATION_SAMPLES),
        "candidate_cooldowns_samples": list(COOLDOWNS),
        "objectives": OBJECTIVES,
        "model": MODEL_PARAMETERS,
        "samples": int(labels.size),
        "positive_samples": int(labels.sum()),
        "bearings": {
            bearing: {
                "samples": int(np.count_nonzero(bearing_ids == bearing)),
                "positive_samples": int(np.count_nonzero(labels[bearing_ids == bearing])),
            }
            for bearing in bearings
        },
        "episode_definitions": {
            "false_before_window": (
                "notification confirmed and episode ended before the final horizon records"
            ),
            "early_persistent": (
                "notification confirmed before the final horizon records and episode "
                "remained active into that window"
            ),
            "confirmed_in_window": (
                "notification confirmation occurred within the final horizon records"
            ),
            "cooldown": (
                "A later episode is suppressed when its confirmation occurs fewer than "
                "cooldown acquisitions after the previous notification."
            ),
            "lead": (
                "Number of record positions from notification confirmation to the "
                "final record; reported as the upper median for compatibility with "
                "the historical analysis."
            ),
        },
        "outer_results": outer_results,
        "fold_details": fold_details,
        "predictions_csv": csv_path.name,
    }
    _atomic_write_text(json_path, json.dumps(summary, indent=2, allow_nan=False) + "\n")

    print(f"Campioni: {labels.size:,} | positivi: {int(labels.sum()):,}")
    for objective, result in outer_results.items():
        print(
            f"{objective}: bearings={result['detected_bearings']}/{len(bearings)}, "
            f"false episodes={result['false_episodes']}, "
            f"event F1={result['event_f1']:.1%}, "
            f"upper-median lead={result['median_in_window_lead']}"
        )
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
    output_dir.mkdir(parents=True, exist_ok=True)
    run_validation(data_dir, output_dir, args.horizon)


if __name__ == "__main__":
    main()
