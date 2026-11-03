#!/usr/bin/env python3
"""Compare LOCO logistic regression with a horizontal-RMS alarm baseline."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.femto_data import (
    DEFAULT_DATA_DIR,
    FEATURE_NAMES,
    build_labeled_dataset,
    group_bearings_by_condition,
)
from validate_loco_policies import _notification_events
from validate_nested_policies import (
    COOLDOWNS,
    CONFIRMATION_SAMPLES,
    OBJECTIVES,
    THRESHOLDS,
    evaluate_policy,
    select_policies,
)


RMS_THRESHOLD_QUANTILES = (0.50, 0.75, 0.90, 0.95, 0.99)
RMS_FEATURE_INDEX = FEATURE_NAMES.index("horizontal_rms")
PRIMARY_OBJECTIVE = "balanced_f1"
LEAD_WINDOWS = {
    "h20": (0, 19),
    "last_50": (0, 49),
    "lead_10_to_50": (10, 50),
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
        help="directory for comparison CSV/JSON (default: processed/ beside the dataset)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=20,
        help="number of final acquisitions labeled positive and used as the primary event window",
    )
    args = parser.parse_args()
    if args.horizon < 1:
        parser.error("--horizon must be a positive number of acquisitions")
    return args


def _fit_rms_score(
    features: np.ndarray,
    groups: np.ndarray,
    train_bearings: set[str],
    test_bearing: str,
) -> np.ndarray:
    train_mask = np.isin(groups, sorted(train_bearings))
    test_mask = groups == test_bearing
    if not train_mask.any() or not test_mask.any():
        raise ValueError("each validation fold must contain training and test samples")

    train_rms = features[train_mask, RMS_FEATURE_INDEX]
    test_rms = features[test_mask, RMS_FEATURE_INDEX]
    mean = float(train_rms.mean())
    scale = float(train_rms.std())
    if scale == 0.0:
        scale = 1.0
    scores = (test_rms - mean) / scale
    if not np.isfinite(scores).all():
        raise RuntimeError(f"RMS baseline returned invalid scores for {test_bearing}")
    return scores


def _rms_threshold_candidates(scores_by_bearing: dict[str, np.ndarray]) -> tuple[float, ...]:
    scores = np.concatenate(list(scores_by_bearing.values()))
    candidates = np.unique(np.quantile(scores, RMS_THRESHOLD_QUANTILES))
    if candidates.size == 0 or not np.isfinite(candidates).all():
        raise ValueError("could not derive finite RMS thresholds from inner-fold scores")
    return tuple(float(value) for value in candidates)


def _load_logistic_reference(
    predictions_path: Path,
    summary_path: Path,
    bearing_ids: np.ndarray,
    sample_indices: np.ndarray,
    labels: np.ndarray,
    condition_groups: dict[int, list[str]],
    horizon: int,
) -> tuple[np.ndarray, dict[int, dict[str, dict[str, Any]]]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("horizon_acquisitions", -1)) != horizon:
        raise ValueError(f"logistic reference has a different horizon: {summary_path}")
    if "weighted_loss" not in summary:
        raise ValueError(
            f"logistic reference does not record its class-balancing mode; "
            f"regenerate standard LOCO outputs: {summary_path}"
        )
    if summary["weighted_loss"]:
        raise ValueError("the logistic reference must use the standard unweighted training set")
    saved_groups = {
        int(condition): bearings
        for condition, bearings in summary.get("condition_to_bearings", {}).items()
    }
    if saved_groups != condition_groups:
        raise ValueError("logistic reference condition groups do not match the current dataset")

    fold_by_condition = {
        int(fold["condition_id"]): fold for fold in summary.get("fold_details", [])
    }
    if set(fold_by_condition) != set(condition_groups):
        raise ValueError("logistic reference does not contain all LOCO folds")
    policies: dict[int, dict[str, dict[str, Any]]] = {}
    for condition_id, fold in fold_by_condition.items():
        expected_training = [
            bearing
            for other_condition, bearings in condition_groups.items()
            if other_condition != condition_id
            for bearing in bearings
        ]
        if (
            fold.get("test_bearings") != condition_groups[condition_id]
            or fold.get("training_bearings") != expected_training
        ):
            raise ValueError(f"logistic reference fold {condition_id} does not match LOCO split")
        policies[condition_id] = fold["selected_policies"]

    saved_predictions: dict[tuple[str, int], tuple[int, float]] = {}
    with predictions_path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            key = row["bearing_id"], int(row["sample_index"])
            if key in saved_predictions:
                raise ValueError(f"duplicate logistic reference prediction for {key}")
            saved_predictions[key] = (
                int(row["failure_within_horizon"]),
                float(row["probability"]),
            )

    logistic_scores = np.empty(labels.size, dtype=np.float64)
    expected_keys: set[tuple[str, int]] = set()
    for index, (bearing, sample_index, label) in enumerate(
        zip(bearing_ids.tolist(), sample_indices.tolist(), labels.tolist(), strict=True)
    ):
        key = bearing, int(sample_index)
        expected_keys.add(key)
        if key not in saved_predictions:
            raise ValueError(f"logistic reference is missing prediction for {key}")
        saved_label, probability = saved_predictions[key]
        if saved_label != int(label):
            raise ValueError(f"logistic reference target mismatch for {key}")
        logistic_scores[index] = probability
    if expected_keys != set(saved_predictions):
        raise ValueError("logistic reference contains predictions outside the current dataset")
    if not np.isfinite(logistic_scores).all() or np.any(
        (logistic_scores < 0.0) | (logistic_scores > 1.0)
    ):
        raise ValueError("logistic reference contains invalid probabilities")
    return logistic_scores, policies


def _aggregate_metrics(metrics_by_condition: list[dict[str, Any]]) -> dict[str, Any]:
    heldout = sum(item["heldout_bearings"] for item in metrics_by_condition)
    detected = sum(item["detected_bearings"] for item in metrics_by_condition)
    false_episodes = sum(item["false_episodes"] for item in metrics_by_condition)
    early_persistent = sum(
        item["early_persistent_episodes"] for item in metrics_by_condition
    )
    in_window = sum(item["in_window_episodes"] for item in metrics_by_condition)
    leads = [
        lead
        for item in metrics_by_condition
        for bearing in item["per_bearing"]
        for lead in bearing["in_window_leads"]
    ]
    precision = detected / (detected + false_episodes) if detected + false_episodes else 0.0
    recall = detected / heldout if heldout else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    ordered_leads = sorted(leads)
    return {
        "heldout_bearings": heldout,
        "detected_bearings": detected,
        "missed_bearings": heldout - detected,
        "false_episodes": false_episodes,
        "early_persistent_episodes": early_persistent,
        "in_window_episodes": in_window,
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
        "median_in_window_lead": ordered_leads[len(ordered_leads) // 2]
        if ordered_leads
        else None,
    }


def _lead_window_summary(
    scores_by_bearing: dict[str, np.ndarray],
    policies_by_bearing: dict[str, dict[str, Any]],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for window_name, (minimum_lead, maximum_lead) in LEAD_WINDOWS.items():
        detected_bearings = 0
        accepted_notifications = 0
        outside_notifications = 0
        for bearing, scores in scores_by_bearing.items():
            policy = policies_by_bearing[bearing]
            _, events = _notification_events(
                scores,
                float(policy["threshold"]),
                int(policy["confirmation"]),
                int(policy["cooldown"]),
            )
            accepted = [
                event
                for event in events
                if minimum_lead <= scores.size - 1 - event[0] <= maximum_lead
            ]
            detected_bearings += int(bool(accepted))
            accepted_notifications += len(accepted)
            outside_notifications += len(events) - len(accepted)
        result[window_name] = {
            "detected_bearings": detected_bearings,
            "accepted_notifications": accepted_notifications,
            "notifications_outside_window": outside_notifications,
        }
    return result


def _write_comparison_predictions(
    path: Path,
    bearing_ids: np.ndarray,
    sample_indices: np.ndarray,
    labels: np.ndarray,
    logistic_scores: np.ndarray,
    rms_scores: np.ndarray,
    selected_logistic: dict[int, dict[str, dict[str, Any]]],
    selected_rms: dict[int, dict[str, dict[str, Any]]],
) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    fieldnames = [
        "condition_id",
        "bearing_id",
        "sample_index",
        "failure_within_horizon",
        "logistic_probability",
        "horizontal_rms_z_score",
    ]
    for model_name in ("logistic", "rms"):
        for objective in OBJECTIVES:
            fieldnames.extend(
                [
                    f"{model_name}_{objective}_threshold",
                    f"{model_name}_{objective}_confirmation",
                    f"{model_name}_{objective}_cooldown",
                    f"{model_name}_{objective}_above_threshold",
                    f"{model_name}_{objective}_notification",
                ]
            )

    with temporary_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for condition_id, bearings in group_bearings_by_condition(bearing_ids.tolist()).items():
            for bearing in bearings:
                indices = np.flatnonzero(bearing_ids == bearing)
                indices = indices[np.argsort(sample_indices[indices])]
                scores = {
                    "logistic": logistic_scores[indices],
                    "rms": rms_scores[indices],
                }
                selected = {
                    "logistic": selected_logistic[condition_id],
                    "rms": selected_rms[condition_id],
                }
                notifications = {
                    model_name: {
                        objective: _notification_events(
                            scores[model_name],
                            float(selected[model_name][objective]["threshold"]),
                            int(selected[model_name][objective]["confirmation"]),
                            int(selected[model_name][objective]["cooldown"]),
                        )[0]
                        for objective in OBJECTIVES
                    }
                    for model_name in ("logistic", "rms")
                }
                for position, index in enumerate(indices):
                    row: dict[str, Any] = {
                        "condition_id": condition_id,
                        "bearing_id": bearing,
                        "sample_index": int(sample_indices[index]),
                        "failure_within_horizon": int(labels[index]),
                        "logistic_probability": f"{logistic_scores[index]:.10f}",
                        "horizontal_rms_z_score": f"{rms_scores[index]:.10f}",
                    }
                    for model_name in ("logistic", "rms"):
                        for objective in OBJECTIVES:
                            policy = selected[model_name][objective]
                            threshold = float(policy["threshold"])
                            row[f"{model_name}_{objective}_threshold"] = f"{threshold:.10g}"
                            row[f"{model_name}_{objective}_confirmation"] = int(
                                policy["confirmation"]
                            )
                            row[f"{model_name}_{objective}_cooldown"] = int(
                                policy["cooldown"]
                            )
                            row[f"{model_name}_{objective}_above_threshold"] = int(
                                scores[model_name][position] >= threshold
                            )
                            row[f"{model_name}_{objective}_notification"] = int(
                                notifications[model_name][objective][position]
                            )
                    writer.writerow(row)
    os.replace(temporary_path, path)


def run_loco_comparison(
    data_dir: Path,
    output_dir: Path,
    horizon: int,
) -> tuple[Path, Path]:
    print("Fase 1/4: estrazione feature e target", file=sys.stderr, flush=True)
    features, labels, bearing_ids, sample_indices = build_labeled_dataset(data_dir, horizon)
    condition_groups = group_bearings_by_condition(dict.fromkeys(bearing_ids.tolist()))
    if len(condition_groups) < 2:
        raise ValueError("Leave-one-condition-out needs at least two operating conditions")

    logistic_predictions_path = (
        output_dir / f"femto_loco_h{horizon}_alarm_predictions.csv"
    )
    logistic_summary_path = (
        output_dir / f"femto_loco_h{horizon}_validation_results.json"
    )
    if not logistic_predictions_path.is_file() or not logistic_summary_path.is_file():
        raise FileNotFoundError(
            "run scripts/validate_loco_policies.py first; the comparison reuses its "
            f"LOCO predictions and policies ({logistic_predictions_path}, {logistic_summary_path})"
        )
    outer_logistic, selected_logistic = _load_logistic_reference(
        logistic_predictions_path,
        logistic_summary_path,
        bearing_ids,
        sample_indices,
        labels,
        condition_groups,
        horizon,
    )
    outer_rms = np.full(labels.size, np.nan, dtype=np.float64)
    selected_rms: dict[int, dict[str, dict[str, Any]]] = {}
    fold_details: list[dict[str, Any]] = []
    condition_metrics: dict[int, dict[str, dict[str, Any]]] = {}
    condition_bearing_scores: dict[int, dict[str, dict[str, np.ndarray]]] = {}

    for condition_id, test_bearings in sorted(condition_groups.items()):
        training_bearings = [
            bearing
            for other_condition, bearings in condition_groups.items()
            if other_condition != condition_id
            for bearing in bearings
        ]
        if len(training_bearings) < 2:
            raise ValueError(f"training fold for condition {condition_id} has too few bearings")

        print(
            f"Fase 2/4: piega esterna condizione {condition_id} "
            f"({', '.join(test_bearings)})",
            file=sys.stderr,
            flush=True,
        )
        inner_rms: dict[str, np.ndarray] = {}
        for inner_bearing in training_bearings:
            inner_training = set(training_bearings) - {inner_bearing}
            inner_rms[inner_bearing] = _fit_rms_score(
                features,
                bearing_ids,
                inner_training,
                inner_bearing,
            )
        rms_thresholds = _rms_threshold_candidates(inner_rms)
        rms_thresholds = _rms_threshold_candidates(inner_rms)
        selected_rms[condition_id] = select_policies(
            inner_rms,
            horizon,
            thresholds=rms_thresholds,
        )

        test_logistic: dict[str, np.ndarray] = {}
        test_rms: dict[str, np.ndarray] = {}
        for bearing in test_bearings:
            test_rms[bearing] = _fit_rms_score(
                features,
                bearing_ids,
                set(training_bearings),
                bearing,
            )
            indices = np.flatnonzero(bearing_ids == bearing)
            indices = indices[np.argsort(sample_indices[indices])]
            test_logistic[bearing] = outer_logistic[indices]
            if test_rms[bearing].size != indices.size:
                raise RuntimeError(f"outer predictions misaligned for {bearing}")
            outer_rms[indices] = test_rms[bearing]

        condition_bearing_scores[condition_id] = {
            "logistic": test_logistic,
            "rms": test_rms,
        }
        fold: dict[str, Any] = {
            "condition_id": condition_id,
            "test_bearings": test_bearings,
            "training_bearings": training_bearings,
            "rms_threshold_candidates": list(rms_thresholds),
            "rms_threshold_quantiles": list(RMS_THRESHOLD_QUANTILES),
            "selected_policies": {
                "logistic": selected_logistic[condition_id],
                "horizontal_rms": selected_rms[condition_id],
            },
        }
        condition_metrics[condition_id] = {}
        for model_name, scores, policies in (
            ("logistic", test_logistic, selected_logistic[condition_id]),
            ("horizontal_rms", test_rms, selected_rms[condition_id]),
        ):
            condition_metrics[condition_id][model_name] = {}
            for objective, policy in policies.items():
                metrics = evaluate_policy(
                    scores,
                    float(policy["threshold"]),
                    int(policy["confirmation"]),
                    int(policy["cooldown"]),
                    horizon,
                )
                condition_metrics[condition_id][model_name][objective] = metrics
                fold.setdefault("metrics_by_model", {}).setdefault(model_name, {})[
                    objective
                ] = {key: value for key, value in metrics.items() if key != "per_bearing"}
        fold_details.append(fold)

    if not np.isfinite(outer_rms).all():
        raise RuntimeError("LOCO comparison did not produce one finite score per sample")

    print("Fase 3/4: aggregazione e salvataggio dei risultati", file=sys.stderr, flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"femto_loco_rms_baseline_h{horizon}_predictions.csv"
    json_path = output_dir / f"femto_loco_rms_baseline_h{horizon}_results.json"
    _write_comparison_predictions(
        csv_path,
        bearing_ids,
        sample_indices,
        labels,
        outer_logistic,
        outer_rms,
        selected_logistic,
        selected_rms,
    )

    results_by_model: dict[str, Any] = {}
    for model_name in ("logistic", "horizontal_rms"):
        score_key = "logistic" if model_name == "logistic" else "rms"
        by_condition: dict[int, Any] = {}
        for condition_id in sorted(condition_metrics):
            by_condition[condition_id] = {
                objective: {
                    key: value
                    for key, value in condition_metrics[condition_id][model_name][
                        objective
                    ].items()
                    if key != "per_bearing"
                }
                for objective in OBJECTIVES
            }
        overall = {
            objective: _aggregate_metrics(
                [
                    condition_metrics[condition_id][model_name][objective]
                    for condition_id in sorted(condition_metrics)
                ]
            )
            for objective in OBJECTIVES
        }
        primary_policies = {
            bearing: selected_logistic[condition_id][PRIMARY_OBJECTIVE]
            if model_name == "logistic"
            else selected_rms[condition_id][PRIMARY_OBJECTIVE]
            for condition_id, bearings in condition_groups.items()
            for bearing in bearings
        }
        scores_by_bearing = {
            bearing: condition_bearing_scores[condition_id][score_key][bearing]
            for condition_id, bearings in condition_groups.items()
            for bearing in bearings
        }
        lead_summary = _lead_window_summary(scores_by_bearing, primary_policies)
        results_by_model[model_name] = {
            "metrics_by_condition": by_condition,
            "overall_metrics": overall,
            "lead_window_sensitivity_exploratory": lead_summary,
            "policy_by_condition": {
                condition_id: (
                    selected_logistic[condition_id]
                    if model_name == "logistic"
                    else selected_rms[condition_id]
                )
                for condition_id in sorted(condition_groups)
            },
        }

    summary = {
        "method": (
            "Paired leave-one-condition-out comparison. Archived logistic results and a single-feature "
            "horizontal RMS score use identical outer folds. Threshold, confirmation, and cooldown "
            "for the logistic model are reused from the reference run; RMS policy selection uses "
            "only inner held-out training bearings."
        ),
        "logistic_reference": {
            "predictions_csv": logistic_predictions_path.name,
            "validation_json": logistic_summary_path.name,
            "weighted_loss": False,
            "training_sampling": "none",
        },
        "primary_metric": (
            "balanced_f1 event metrics in the final horizon acquisitions; the 10-50 lead window "
            "is exploratory only and is not used for policy selection."
        ),
        "event_definition": (
            "An above-threshold run ends at the first below-threshold sample. A notification is "
            "confirmed at the Nth consecutive above-threshold sample; cooldown suppresses later "
            "notifications whose confirmation is less than the configured number of acquisitions "
            "after the previous notification. An early persistent notification is reported separately."
        ),
        "baseline": {
            "feature": "horizontal_rms",
            "score": "training-fold z-score of horizontal_rms",
            "threshold_quantiles": list(RMS_THRESHOLD_QUANTILES),
            "thresholds_are_derived_from": "inner out-of-fold RMS scores only",
        },
        "horizon_acquisitions": horizon,
        "logistic_candidate_thresholds": list(THRESHOLDS),
        "candidate_confirmation_samples": list(CONFIRMATION_SAMPLES),
        "candidate_cooldowns_samples": list(COOLDOWNS),
        "objectives": OBJECTIVES,
        "condition_to_bearings": condition_groups,
        "samples": int(labels.size),
        "positive_samples": int(labels.sum()),
        "results_by_model": results_by_model,
        "fold_details": fold_details,
        "predictions_csv": csv_path.name,
    }
    with json_path.open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, allow_nan=False)
        output.write("\n")

    print(f"Campioni: {labels.size:,} | positivi: {int(labels.sum()):,}")
    for model_name, result in results_by_model.items():
        metrics = result["overall_metrics"][PRIMARY_OBJECTIVE]
        print(
            f"{model_name}: rilevati={metrics['detected_bearings']}/"
            f"{metrics['heldout_bearings']}, falsi={metrics['false_episodes']}, "
            f"F1 evento={metrics['event_f1']:.1%}"
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
    run_loco_comparison(data_dir, output_dir, args.horizon)


if __name__ == "__main__":
    main()
