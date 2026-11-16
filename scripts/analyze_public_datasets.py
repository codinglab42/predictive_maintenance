#!/usr/bin/env python3
"""Run independent exploratory RUL analyses on C-MAPSS and XJTU-SY."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.public_dataset_analysis import analyze_cmapss, analyze_xjtu, write_csv, write_json


DEFAULT_DATA_ROOT = Path("/mnt/shared/predictive-maintenance/raw")
DEFAULT_OUTPUT_DIR = Path("/mnt/shared/predictive-maintenance/processed/public_datasets")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-cmapss", action="store_true")
    parser.add_argument("--skip-xjtu", action="store_true")
    parser.add_argument(
        "--refresh-xjtu-cache",
        action="store_true",
        help="re-extract all XJTU-SY acquisitions instead of reusing the feature cache",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.skip_cmapss and args.skip_xjtu:
        raise SystemExit("at least one dataset must be analyzed")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_cmapss:
        print(f"Analyzing C-MAPSS in {args.data_root / 'cmapss'}")
        summary, predictions = analyze_cmapss(args.data_root / "cmapss")
        write_json(args.output_dir / "cmapss_rul_analysis.json", summary)
        write_csv(args.output_dir / "cmapss_rul_test_predictions.csv", predictions)
        for fd, result in summary["subsets"].items():
            full = result["cycle_settings_sensors"]
            cycle = result["cycle_only"]
            print(
                f"  {fd}: official test MAE {cycle['official_test']['mae']:.2f} cycles "
                f"(cycle only), {full['official_test']['mae']:.2f} cycles "
                f"(settings + sensors); grouped validation MAE "
                f"{full['grouped_oof_mean_per_engine_mae']:.2f}"
            )

    if not args.skip_xjtu:
        print(f"Analyzing XJTU-SY in {args.data_root / 'xjtu_sy'}")

        def report_progress(processed: int, total: int, bearing_id: str) -> None:
            if bearing_id == "cached":
                print(f"  reused cached features for {processed}/{total} acquisitions")
            else:
                print(
                    f"  extracted {processed}/{total} acquisitions "
                    f"(current: {bearing_id})"
                )

        summary, predictions = analyze_xjtu(
            args.data_root / "xjtu_sy",
            progress_callback=report_progress,
            feature_cache_path=args.output_dir / "xjtu_sy_features_cache.npz",
            refresh_cache=args.refresh_xjtu_cache,
        )
        write_json(args.output_dir / "xjtu_sy_rul_analysis.json", summary)
        write_csv(args.output_dir / "xjtu_sy_rul_oof_predictions.csv", predictions)
        print(
            f"  {summary['bearing_count']} bearings, {summary['acquisition_count']} acquisitions"
        )
        for model_name, result in summary["model_variants"].items():
            print(
                f"  {model_name}: mean per-bearing MAE "
                f"{result['results']['mean_per_bearing_mae']:.2f} min"
            )
        print(
            "  condition_life_baseline: mean per-bearing MAE "
            f"{summary['baseline']['results']['mean_per_bearing_mae']:.2f} min"
        )


if __name__ == "__main__":
    main()
