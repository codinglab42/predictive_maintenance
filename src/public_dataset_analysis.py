"""Exploratory, group-separated analyses for C-MAPSS and XJTU-SY."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

C_MAPSS_FEATURE_NAMES = (
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    *(f"sensor_{index}" for index in range(1, 22)),
)
XJTU_SIGNAL_FEATURES = (
    "rms",
    "std",
    "peak_to_peak",
    "max_abs",
    "kurtosis",
)
XJTU_FEATURE_NAMES = (
    "elapsed_minutes",
    "speed_rpm",
    "radial_load_kn",
    *(f"{axis}_{feature}" for axis in ("horizontal", "vertical") for feature in XJTU_SIGNAL_FEATURES),
)
XJTU_MODEL_FEATURES = {
    "elapsed_only": (0,),
    "vibration_only": tuple(range(3, 13)),
    "elapsed_plus_vibration": (0, *range(3, 13)),
    "elapsed_context_vibration": tuple(range(13)),
}
_XJTU_BEARING = re.compile(r"^Bearing([1-3])_(\d+)$")
_XJTU_CONDITION = re.compile(r"^(35|37\.5|40)Hz(12|11|10)kN$")
_XJTU_SETTINGS = {
    "35Hz12kN": (2100.0, 12.0),
    "37.5Hz11kN": (2250.0, 11.0),
    "40Hz10kN": (2400.0, 10.0),
}


def _read_cmapss_file(path: Path) -> np.ndarray:
    try:
        rows = np.loadtxt(path, dtype=np.float64, ndmin=2)
    except (OSError, ValueError) as error:
        raise ValueError(f"could not read C-MAPSS file {path}: {error}") from error
    if rows.shape[1] != 26 or not np.isfinite(rows).all():
        raise ValueError(f"invalid C-MAPSS matrix in {path}: expected 26 finite columns")
    if not np.equal(rows[:, 0], np.floor(rows[:, 0])).all():
        raise ValueError(f"non-integer unit IDs in {path}")
    return rows


def load_cmapss_subset(
    data_dir: Path, subset: str, fd: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one official C-MAPSS subset and derive training RUL labels."""
    if subset not in {"train", "test"} or fd not in {"FD001", "FD002", "FD003", "FD004"}:
        raise ValueError("subset must be train/test and FD must be FD001 through FD004")
    rows = _read_cmapss_file(data_dir / f"{subset}_{fd}.txt")
    units = rows[:, 0].astype(np.int64)
    cycles = rows[:, 1]
    unique_units = np.unique(units)
    for unit in unique_units:
        unit_cycles = cycles[units == unit]
        if np.any(np.diff(unit_cycles) <= 0):
            raise ValueError(f"cycles are not strictly increasing for unit {unit} in {subset}_{fd}")

    if subset == "train":
        end_cycles = {
            unit: float(cycles[units == unit][-1]) for unit in unique_units
        }
        rul = np.asarray([end_cycles[unit] - cycle for unit, cycle in zip(units, cycles)])
        return rows, units, rul

    rul_path = data_dir / f"RUL_{fd}.txt"
    try:
        test_rul = np.loadtxt(rul_path, dtype=np.float64, ndmin=1)
    except (OSError, ValueError) as error:
        raise ValueError(f"could not read C-MAPSS RUL labels {rul_path}: {error}") from error
    if test_rul.size != unique_units.size or not np.isfinite(test_rul).all():
        raise ValueError(
            f"{rul_path} has {test_rul.size} labels for {unique_units.size} test units"
        )
    if not np.array_equal(unique_units, np.arange(1, unique_units.size + 1)):
        raise ValueError(f"test unit IDs are not contiguous and one-based in {fd}")
    return rows, units, test_rul


def _fit_ridge(
    train_features: np.ndarray, train_target: np.ndarray, test_features: np.ndarray, alpha: float
) -> np.ndarray:
    if alpha < 0.0:
        raise ValueError("ridge alpha must be non-negative")
    means = train_features.mean(axis=0)
    scales = train_features.std(axis=0)
    scales[scales == 0.0] = 1.0
    x_train = (train_features - means) / scales
    x_test = (test_features - means) / scales
    target_mean = float(train_target.mean())
    centered_target = train_target - target_mean
    gram = x_train.T @ x_train
    gram.flat[:: gram.shape[0] + 1] += alpha
    right_hand_side = x_train.T @ centered_target
    try:
        coefficients = np.linalg.solve(gram, right_hand_side)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(gram, right_hand_side, rcond=None)[0]
    return x_test @ coefficients + target_mean


def _unit_weighted_mae(
    actual: np.ndarray, predicted: np.ndarray, unit_ids: np.ndarray
) -> float:
    return float(
        np.mean(
            [
                np.mean(np.abs(actual[unit_ids == unit] - predicted[unit_ids == unit]))
                for unit in np.unique(unit_ids)
            ]
        )
    )


def _regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    errors = predicted - actual
    return {
        "mae": float(np.mean(np.abs(errors))),
        "median_absolute_error": float(np.median(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
    }


def _grouped_cv_ridge(
    features: np.ndarray,
    target: np.ndarray,
    units: np.ndarray,
    alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0),
    folds: int = 5,
) -> tuple[float, np.ndarray]:
    unique_units = np.unique(units)
    if unique_units.size < folds:
        raise ValueError(f"at least {folds} independent units are required for grouped CV")
    unit_fold = {unit: index % folds for index, unit in enumerate(unique_units)}
    fold_ids = np.asarray([unit_fold[unit] for unit in units])
    results: list[tuple[float, float, np.ndarray]] = []
    for alpha in alphas:
        oof = np.full(target.shape, np.nan, dtype=np.float64)
        for fold in range(folds):
            validation = fold_ids == fold
            training = ~validation
            oof[validation] = _fit_ridge(
                features[training], target[training], features[validation], alpha
            )
        results.append((alpha, _unit_weighted_mae(target, oof, units), oof))
    best_alpha, _, best_oof = min(results, key=lambda result: result[1])
    return best_alpha, best_oof


def analyze_cmapss(data_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run five-fold engine-grouped ridge validation and official test scoring."""
    summary: dict[str, Any] = {
        "dataset": "NASA C-MAPSS",
        "task": "remaining useful life regression in cycles",
        "feature_names": list(C_MAPSS_FEATURE_NAMES),
        "validation": "5-fold grouped by engine ID; alpha selected by mean per-engine MAE",
        "rul_target": "train: last training cycle minus current cycle; test: supplied official RUL",
        "subsets": {},
    }
    prediction_rows: list[dict[str, Any]] = []
    for fd in ("FD001", "FD002", "FD003", "FD004"):
        train_rows, train_units, train_rul = load_cmapss_subset(data_dir, "train", fd)
        test_rows, test_units, official_rul = load_cmapss_subset(data_dir, "test", fd)
        test_final_indices = np.asarray(
            [np.flatnonzero(test_units == unit)[-1] for unit in np.unique(test_units)]
        )
        test_final_rows = test_rows[test_final_indices]
        test_final_units = test_units[test_final_indices]

        model_results: dict[str, Any] = {}
        for model_name, feature_indices in (
            ("cycle_only", (1,)),
            ("cycle_settings_sensors", tuple(range(1, 26))),
        ):
            feature_indices_array = np.asarray(feature_indices)
            alpha, oof_predictions = _grouped_cv_ridge(
                train_rows[:, feature_indices_array],
                train_rul,
                train_units,
            )
            test_predictions = _fit_ridge(
                train_rows[:, feature_indices_array],
                train_rul,
                test_final_rows[:, feature_indices_array],
                alpha,
            )
            model_results[model_name] = {
                "selected_alpha": alpha,
                "grouped_oof": _regression_metrics(train_rul, oof_predictions),
                "grouped_oof_mean_per_engine_mae": _unit_weighted_mae(
                    train_rul, oof_predictions, train_units
                ),
                "official_test": _regression_metrics(official_rul, test_predictions),
                "train_engines": int(np.unique(train_units).size),
                "test_engines": int(test_final_units.size),
            }
            if model_name == "cycle_settings_sensors":
                prediction_rows.extend(
                    {
                        "subset": fd,
                        "unit": int(unit),
                        "actual_rul_cycles": float(actual),
                        "predicted_rul_cycles": float(predicted),
                        "error_cycles": float(predicted - actual),
                    }
                    for unit, actual, predicted in zip(
                        test_final_units, official_rul, test_predictions
                    )
                )
        summary["subsets"][fd] = model_results
    return summary, prediction_rows


def _discover_xjtu_bearings(data_dir: Path) -> list[tuple[str, str, Path, float, float]]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"XJTU-SY directory not found: {data_dir}")
    bearings: list[tuple[str, str, Path, float, float]] = []
    expected_condition_id = {"35Hz12kN": 1, "37.5Hz11kN": 2, "40Hz10kN": 3}
    for condition_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        condition_match = _XJTU_CONDITION.fullmatch(condition_dir.name)
        settings = _XJTU_SETTINGS.get(condition_dir.name)
        if condition_match is None or settings is None:
            raise ValueError(f"unrecognized XJTU-SY condition directory: {condition_dir.name}")
        for bearing_dir in sorted(path for path in condition_dir.iterdir() if path.is_dir()):
            match = _XJTU_BEARING.fullmatch(bearing_dir.name)
            if match is None:
                raise ValueError(f"unrecognized XJTU-SY bearing directory: {bearing_dir}")
            if int(match.group(1)) != expected_condition_id[condition_dir.name]:
                raise ValueError(
                    f"bearing {bearing_dir.name} does not match condition {condition_dir.name}"
                )

            files = list(bearing_dir.glob("*.csv"))
            if not files:
                raise ValueError(f"no CSV acquisitions found in {bearing_dir}")
            bearings.append((condition_dir.name, bearing_dir.name, bearing_dir, *settings))
    if not bearings:
        raise ValueError(f"no XJTU-SY bearing directories found under {data_dir}")
    return bearings


def _xjtu_file_sequence(bearing_dir: Path) -> list[Path]:
    files = list(bearing_dir.glob("*.csv"))
    if any(not path.stem.isdigit() for path in files):
        raise ValueError(f"XJTU-SY acquisition names must be integer CSV filenames: {bearing_dir}")
    files.sort(key=lambda path: int(path.stem))
    identifiers = [int(path.stem) for path in files]
    if identifiers != list(range(1, len(files) + 1)):
        raise ValueError(f"XJTU-SY acquisition IDs are not continuous from 1: {bearing_dir}")
    return files


def _read_xjtu_features(path: Path) -> np.ndarray:
    try:
        content = path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        raise ValueError(f"could not read XJTU-SY signal {path}: {error}") from error
    header, separator, body = content.partition("\n")
    if not separator or header.strip() != (
        "Horizontal_vibration_signals,Vertical_vibration_signals"
    ):
        raise ValueError(f"invalid XJTU-SY CSV header in {path}")
    values = np.fromstring(body.strip().replace("\n", ",").replace("\r", ""), sep=",")
    if values.size != 32768 * 2:
        raise ValueError(
            f"invalid XJTU-SY signal sample count in {path}: "
            f"expected {32768 * 2}, got {values.size}"
        )
    signal = values.reshape(32768, 2)
    if signal.shape != (32768, 2) or not np.isfinite(signal).all():
        raise ValueError(f"invalid XJTU-SY signal shape or values in {path}: {signal.shape}")
    features: list[float] = []
    for channel in signal.T:
        centered = channel - np.mean(channel)
        variance = float(np.mean(np.square(centered)))
        features.extend(
            (
                float(np.sqrt(np.mean(np.square(channel)))),
                float(np.sqrt(variance)),
                float(np.ptp(channel)),
                float(np.max(np.abs(channel))),
                float(np.mean(np.power(centered, 4)) / (variance * variance) - 3.0)
                if variance > 0.0
                else 0.0,
            )
        )
    row = np.asarray(features, dtype=np.float64)
    if not np.isfinite(row).all():
        raise ValueError(f"non-finite features extracted from {path}")
    return row


def _xjtu_data_signature(
    data_dir: Path,
    files_by_bearing: dict[str, list[Path]],
) -> str:
    manifest = [
        (
            path.relative_to(data_dir).as_posix(),
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for files in files_by_bearing.values()
        for path in files
    ]
    encoded = json.dumps(manifest, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_xjtu_feature_cache(
    cache_path: Path,
    signature: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if not cache_path.is_file():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as cache:
            cached_signature = str(cache["signature"].item())
            if cached_signature != signature:
                raise ValueError(
                    f"XJTU-SY feature cache does not match the current dataset: {cache_path}; "
                    "rerun with --refresh-xjtu-cache"
                )
            result = (
                cache["features"].copy(),
                cache["target"].copy(),
                cache["units"].copy(),
                cache["conditions"].copy(),
                cache["elapsed"].copy(),
            )
    except (OSError, KeyError, ValueError) as error:
        if isinstance(error, ValueError) and "does not match" in str(error):
            raise
        raise ValueError(f"could not read XJTU-SY feature cache {cache_path}: {error}") from error
    return result


def _write_xjtu_feature_cache(
    cache_path: Path,
    signature: str,
    features: np.ndarray,
    target: np.ndarray,
    units: np.ndarray,
    conditions: np.ndarray,
    elapsed: np.ndarray,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_name(cache_path.name + ".tmp")
    try:
        with temporary_path.open("wb") as output:
            np.savez_compressed(
                output,
                signature=np.asarray(signature),
                features=features,
                target=target,
                units=units,
                conditions=conditions,
                elapsed=elapsed,
            )
        os.replace(temporary_path, cache_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _xjtu_rul_baseline(
    bearings: list[tuple[str, str, Path, float, float]],
    counts_by_bearing: dict[str, int],
    held_out_id: str,
    elapsed: np.ndarray,
) -> np.ndarray:
    held_out_condition = next(item[0] for item in bearings if item[1] == held_out_id)
    train_lengths = [
        counts_by_bearing[item[1]]
        for item in bearings
        if item[1] != held_out_id and item[0] == held_out_condition
    ]
    if not train_lengths:
        raise ValueError(f"no training bearings for condition of {held_out_id}")
    expected_life = float(np.median(train_lengths) - 1)
    return np.maximum(expected_life - elapsed, 0.0)


def analyze_xjtu(
    data_dir: Path,
    progress_callback: Callable[[int, int, str], None] | None = None,
    feature_cache_path: Path | None = None,
    refresh_cache: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare XJTU-SY feature sets using leave-one-bearing-out validation."""
    bearings = _discover_xjtu_bearings(data_dir)
    files_by_bearing = {
        item[1]: _xjtu_file_sequence(item[2]) for item in bearings
    }
    total = sum(len(files) for files in files_by_bearing.values())
    counts_by_bearing = {bearing_id: len(files) for bearing_id, files in files_by_bearing.items()}
    signature = _xjtu_data_signature(data_dir, files_by_bearing)
    cached = (
        None
        if refresh_cache or feature_cache_path is None
        else _load_xjtu_feature_cache(feature_cache_path, signature)
    )
    bearing_metadata: list[dict[str, Any]] = []
    if cached is None:
        feature_rows: list[np.ndarray] = []
        target_rows: list[float] = []
        unit_rows: list[str] = []
        condition_rows: list[str] = []
        elapsed_rows: list[float] = []
        processed = 0

        for condition, bearing_id, _, speed_rpm, load_kn in bearings:
            files = files_by_bearing[bearing_id]
            count = len(files)
            for position, path in enumerate(files):
                vibration_features = _read_xjtu_features(path)
                feature_rows.append(
                    np.asarray(
                        (position, speed_rpm, load_kn, *vibration_features),
                        dtype=np.float64,
                    )
                )
                target_rows.append(float(count - position - 1))
                unit_rows.append(bearing_id)
                condition_rows.append(condition)
                elapsed_rows.append(float(position))
                processed += 1
                if progress_callback is not None and (
                    processed % 100 == 0 or processed == total
                ):
                    progress_callback(processed, total, bearing_id)

        features = np.vstack(feature_rows)
        target = np.asarray(target_rows, dtype=np.float64)
        units = np.asarray(unit_rows, dtype=str)
        conditions = np.asarray(condition_rows, dtype=str)
        elapsed = np.asarray(elapsed_rows, dtype=np.float64)
        if feature_cache_path is not None:
            _write_xjtu_feature_cache(
                feature_cache_path, signature, features, target, units, conditions, elapsed
            )
    else:
        features, target, units, conditions, elapsed = cached
        expected_units = set(counts_by_bearing)
        expected_units_in_order = np.concatenate(
            [np.repeat(bearing_id, counts_by_bearing[bearing_id]) for _, bearing_id, *_ in bearings]
        )
        expected_conditions_in_order = np.concatenate(
            [
                np.repeat(condition, counts_by_bearing[bearing_id])
                for condition, bearing_id, *_ in bearings
            ]
        )
        expected_elapsed = np.concatenate(
            [np.arange(counts_by_bearing[bearing_id]) for _, bearing_id, *_ in bearings]
        ).astype(np.float64)
        expected_target = np.concatenate(
            [
                np.arange(counts_by_bearing[bearing_id] - 1, -1, -1)
                for _, bearing_id, *_ in bearings
            ]
        ).astype(np.float64)
        if (
            features.shape != (total, len(XJTU_FEATURE_NAMES))
            or target.shape != (total,)
            or units.shape != (total,)
            or conditions.shape != (total,)
            or elapsed.shape != (total,)
            or set(units.tolist()) != expected_units
            or not np.isfinite(features).all()
            or not np.isfinite(target).all()
            or not np.array_equal(units, expected_units_in_order)
            or not np.array_equal(conditions, expected_conditions_in_order)
            or not np.array_equal(elapsed, expected_elapsed)
            or not np.array_equal(target, expected_target)
        ):
            raise ValueError(
                f"XJTU-SY feature cache has invalid shapes, values, or bearing order: "
                f"{feature_cache_path}"
            )
        if progress_callback is not None:
            progress_callback(total, total, "cached")

    start = 0
    for condition, bearing_id, _, _, _ in bearings:
        end = start + counts_by_bearing[bearing_id]
        bearing_metadata.append(
            {
                "condition": condition,
                "bearing": bearing_id,
                "acquisitions": counts_by_bearing[bearing_id],
                "life_minutes_between_first_and_last": counts_by_bearing[bearing_id] - 1,
                "feature_row_start": start,
                "feature_row_end": end,
            }
        )
        start = end

    model_predictions = {
        model_name: np.full(target.shape, np.nan, dtype=np.float64)
        for model_name in XJTU_MODEL_FEATURES
    }
    baseline_predictions = np.full(target.shape, np.nan, dtype=np.float64)

    for _, bearing_id, _, _, _ in bearings:
        test_mask = units == bearing_id
        train_mask = ~test_mask
        for model_name, feature_indices in XJTU_MODEL_FEATURES.items():
            predicted = _fit_ridge(
                features[train_mask][:, feature_indices],
                target[train_mask],
                features[test_mask][:, feature_indices],
                1.0,
            )
            model_predictions[model_name][test_mask] = np.maximum(predicted, 0.0)
        baseline_predictions[test_mask] = _xjtu_rul_baseline(
            bearings, counts_by_bearing, bearing_id, elapsed[test_mask]
        )

    if any(
        not np.isfinite(prediction).all() for prediction in model_predictions.values()
    ) or not np.isfinite(baseline_predictions).all():
        raise RuntimeError("XJTU-SY leave-one-bearing-out validation left missing predictions")

    def model_report(
        actual: np.ndarray, predicted: np.ndarray, groups: np.ndarray
    ) -> dict[str, Any]:
        group_maes = [
            float(np.mean(np.abs(actual[groups == group] - predicted[groups == group])))
            for group in np.unique(groups)
        ]
        return {
            "metrics": _regression_metrics(actual, predicted),
            "mean_per_bearing_mae": float(np.mean(group_maes)),
            "bearing_maes": {
                str(group): float(
                    np.mean(np.abs(actual[groups == group] - predicted[groups == group]))
                )
                for group in np.unique(groups)
            },
        }

    all_variant_reports = {
        name: model_report(target, prediction, units)
        for name, prediction in model_predictions.items()
    }
    all_variant_reports["condition_life_baseline"] = model_report(
        target, baseline_predictions, units
    )
    full_prediction = model_predictions["elapsed_context_vibration"]
    summary = {
        "dataset": "XJTU-SY",
        "task": "remaining observed lifetime regression in minutes",
        "rul_target": "number of one-minute sampling intervals remaining until the last record",
        "feature_names": list(XJTU_FEATURE_NAMES),
        "validation": (
            "leave one complete bearing out; fixed standardized ridge alpha=1.0; "
            "all feature variants use identical bearing folds"
        ),
        "model_variants": {
            name: {
                "feature_names": [XJTU_FEATURE_NAMES[index] for index in indices],
                "results": all_variant_reports[name],
            }
            for name, indices in XJTU_MODEL_FEATURES.items()
        },
        "baseline": {
            "description": "median observed life of other bearings in the same operating condition",
            "results": all_variant_reports["condition_life_baseline"],
        },
        "limitations": [
            "last recorded sample is treated as end of life per run-to-failure protocol",
            "minutes are derived from the one-minute sampling cadence",
            "the bearing-specific split reuses the same three laboratory operating conditions",
            "one-minute cadence and last-record end point are dataset-derived proxies",
            "these results do not validate industrial replacement decisions",
        ],
        "bearing_count": len(bearings),
        "condition_count": len({item[0] for item in bearings}),
        "acquisition_count": int(target.size),
        "overall": {
            "feature_ridge": _regression_metrics(target, full_prediction),
            "feature_ridge_mean_per_bearing_mae": all_variant_reports[
                "elapsed_context_vibration"
            ]["mean_per_bearing_mae"],
            "condition_median_life_baseline": _regression_metrics(
                target, baseline_predictions
            ),
            "condition_median_life_baseline_mean_per_bearing_mae": all_variant_reports[
                "condition_life_baseline"
            ]["mean_per_bearing_mae"],
        },
        "per_condition": {},
        "bearings": [],
    }
    for condition in sorted(set(conditions)):
        mask = conditions == condition
        summary["per_condition"][condition] = {
            "bearing_count": int(np.unique(units[mask]).size),
            "acquisition_count": int(mask.sum()),
            "feature_ridge": _regression_metrics(target[mask], full_prediction[mask]),
            "condition_median_life_baseline": _regression_metrics(
                target[mask], baseline_predictions[mask]
            ),
            "model_variants": {
                name: model_report(target[mask], prediction[mask], units[mask])
                for name, prediction in model_predictions.items()
            },
        }
    prediction_rows: list[dict[str, Any]] = []
    for metadata in bearing_metadata:
        mask = units == metadata["bearing"]
        feature_mae = float(np.mean(np.abs(target[mask] - full_prediction[mask])))
        baseline_mae = float(np.mean(np.abs(target[mask] - baseline_predictions[mask])))
        row = {
            **metadata,
            "feature_ridge": _regression_metrics(target[mask], full_prediction[mask]),
            "feature_ridge_mae": feature_mae,
            "condition_median_life_baseline": _regression_metrics(
                target[mask], baseline_predictions[mask]
            ),
            "condition_median_life_baseline_mae": baseline_mae,
            "feature_ridge_beats_baseline": feature_mae < baseline_mae,
            "model_variants": {
                name: _regression_metrics(target[mask], prediction[mask])
                | {"mae": float(np.mean(np.abs(target[mask] - prediction[mask])))}
                for name, prediction in model_predictions.items()
            },
        }
        summary["bearings"].append(row)
        prediction_rows.extend(
            {
                "condition": metadata["condition"],
                "bearing": metadata["bearing"],
                "acquisition": index + 1,
                "elapsed_minutes": float(elapsed[index]),
                "actual_rul_minutes": float(target[index]),
                "predicted_rul_minutes": float(full_prediction[index]),
                "baseline_rul_minutes": float(baseline_predictions[index]),
                **{
                    f"predicted_{name}_rul_minutes": float(prediction[index])
                    for name, prediction in model_predictions.items()
                },
            }
            for index in np.flatnonzero(mask)
        )
    summary["overall"]["feature_ridge_bearings_better_than_baseline"] = sum(
        row["feature_ridge_beats_baseline"] for row in summary["bearings"]
    )
    return summary, prediction_rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(data, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
