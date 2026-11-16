from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.public_dataset_analysis import (
    _fit_ridge,
    _read_xjtu_features,
    analyze_xjtu,
    load_cmapss_subset,
)


class PublicDatasetAnalysisTests(unittest.TestCase):
    def test_cmapss_train_rul_uses_each_engine_last_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            rows = np.zeros((5, 26), dtype=np.float64)
            rows[:, 0] = [1, 1, 1, 2, 2]
            rows[:, 1] = [1, 2, 4, 1, 3]
            np.savetxt(root / "train_FD001.txt", rows)

            loaded, units, rul = load_cmapss_subset(root, "train", "FD001")

        self.assertEqual(loaded.shape, (5, 26))
        np.testing.assert_array_equal(units, [1, 1, 1, 2, 2])
        np.testing.assert_array_equal(rul, [3, 2, 0, 2, 0])

    def test_cmapss_test_rul_labels_align_with_contiguous_unit_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            rows = np.zeros((3, 26), dtype=np.float64)
            rows[:, 0] = [1, 1, 2]
            rows[:, 1] = [1, 2, 1]
            np.savetxt(root / "test_FD001.txt", rows)
            np.savetxt(root / "RUL_FD001.txt", [8, 12])

            _, units, rul = load_cmapss_subset(root, "test", "FD001")

        np.testing.assert_array_equal(units, [1, 1, 2])
        np.testing.assert_array_equal(rul, [8, 12])

    def test_ridge_uses_training_statistics_and_supports_constant_features(self) -> None:
        train = np.asarray([[0.0, 5.0], [1.0, 5.0], [2.0, 5.0]])
        target = np.asarray([1.0, 3.0, 5.0])
        test = np.asarray([[3.0, 5.0]])

        prediction = _fit_ridge(train, target, test, alpha=0.0)

        np.testing.assert_allclose(prediction, [7.0], atol=1e-10)

    def test_xjtu_reader_skips_header_and_extracts_ten_features(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "1.csv"
            time = np.arange(32768, dtype=np.float64)
            signal = np.column_stack((np.sin(time / 100), np.cos(time / 100)))
            with path.open("w", encoding="utf-8") as output:
                output.write("Horizontal_vibration_signals,Vertical_vibration_signals\n")
                np.savetxt(output, signal, delimiter=",")

            features = _read_xjtu_features(path)

        self.assertEqual(features.shape, (10,))
        self.assertTrue(np.isfinite(features).all())
        self.assertGreater(features[0], 0.0)
        self.assertGreater(features[5], 0.0)

    def test_xjtu_reader_rejects_invalid_header_and_sample_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "1.csv"
            path.write_text("unexpected,header\n1,2\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "invalid XJTU-SY CSV header"):
                _read_xjtu_features(path)

            path.write_text(
                "Horizontal_vibration_signals,Vertical_vibration_signals\n1,2\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(ValueError, "invalid XJTU-SY signal sample count"):
                _read_xjtu_features(path)

    def test_xjtu_analysis_leaves_out_whole_bearings_and_writes_complete_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir) / "data"
            cache_path = Path(temporary_dir) / "processed" / "features.npz"
            root.mkdir()
            settings = (
                ("35Hz12kN", 1),
                ("37.5Hz11kN", 2),
                ("40Hz10kN", 3),
            )
            time = np.arange(32768, dtype=np.float64)
            signal = np.column_stack((np.sin(time / 100), np.cos(time / 100)))
            for condition, condition_id in settings:
                for bearing_index in (1, 2):
                    bearing_dir = root / condition / f"Bearing{condition_id}_{bearing_index}"
                    bearing_dir.mkdir(parents=True)
                    for acquisition in (1, 2):
                        with (bearing_dir / f"{acquisition}.csv").open(
                            "w", encoding="utf-8"
                        ) as output:
                            output.write(
                                "Horizontal_vibration_signals,Vertical_vibration_signals\n"
                            )
                            np.savetxt(output, signal * acquisition, delimiter=",")

            summary, predictions = analyze_xjtu(root, feature_cache_path=cache_path)
            cached_progress: list[tuple[int, int, str]] = []
            cached_summary, cached_predictions = analyze_xjtu(
                root,
                progress_callback=lambda processed, total, bearing: cached_progress.append(
                    (processed, total, bearing)
                ),
                feature_cache_path=cache_path,
            )
            cache_exists = cache_path.is_file()
            with (root / "35Hz12kN" / "Bearing1_1" / "1.csv").open(
                "a", encoding="utf-8"
            ) as output:
                output.write("\n")
            with self.assertRaisesRegex(ValueError, "does not match"):
                analyze_xjtu(root, feature_cache_path=cache_path)

        self.assertEqual(summary["bearing_count"], 6)
        self.assertEqual(summary["acquisition_count"], 12)
        self.assertIn("feature_ridge_bearings_better_than_baseline", summary["overall"])
        self.assertTrue(all("feature_ridge_beats_baseline" in row for row in summary["bearings"]))
        self.assertEqual(len(predictions), 12)
        self.assertNotIn("mean_nasa_score", summary["overall"]["feature_ridge"])
        self.assertIn("elapsed_plus_vibration", summary["model_variants"])
        self.assertEqual(len(summary["model_variants"]), 4)
        self.assertEqual(predictions, cached_predictions)
        self.assertEqual(summary["model_variants"], cached_summary["model_variants"])
        self.assertEqual(cached_progress, [(12, 12, "cached")])
        self.assertTrue(cache_exists)


if __name__ == "__main__":
    unittest.main()
