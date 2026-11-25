from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np

from scripts.train_ims_model import (
    THRESHOLD_CANDIDATES,
    _baseline_normalize_features,
    _build_sliding_windows,
    _confirmed_alarms,
    _load_dataset,
    _metrics,
    _rolling_mean_by_bearing,
    _threshold_diagnostics,
    _validate_probabilities,
)


class IMSTrainingTests(unittest.TestCase):
    def test_baseline_normalization_uses_first_thirty_chronological_samples_per_bearing(
        self,
    ) -> None:
        timestamps = np.asarray(
            [f"2003-10-22 12:00:{second:02d}" for second in range(31)]
        )
        values = np.arange(31, dtype=np.float64)
        first_bearing = np.repeat((values + 10.0)[:, np.newaxis], 6, axis=1)
        second_bearing = np.repeat((values + 500.0)[:, np.newaxis], 6, axis=1)
        features = np.vstack((first_bearing, second_bearing))
        bearing_ids = np.concatenate(
            (np.repeat("bearing_a", 31), np.repeat("bearing_b", 31))
        )
        all_timestamps = np.tile(timestamps, 2)
        reversed_rows = np.arange(features.shape[0] - 1, -1, -1)

        normalized, baselines = _baseline_normalize_features(
            features[reversed_rows],
            bearing_ids[reversed_rows],
            all_timestamps[reversed_rows],
        )

        np.testing.assert_allclose(baselines["bearing_a"], np.full(6, 24.5))
        np.testing.assert_allclose(baselines["bearing_b"], np.full(6, 514.5))
        first_by_time = {
            timestamp: row
            for timestamp, row in zip(
                all_timestamps[reversed_rows][bearing_ids[reversed_rows] == "bearing_a"],
                normalized[bearing_ids[reversed_rows] == "bearing_a"],
                strict=True,
            )
        }
        second_by_time = {
            timestamp: row
            for timestamp, row in zip(
                all_timestamps[reversed_rows][bearing_ids[reversed_rows] == "bearing_b"],
                normalized[bearing_ids[reversed_rows] == "bearing_b"],
                strict=True,
            )
        }
        for timestamp in timestamps:
            np.testing.assert_allclose(first_by_time[timestamp], second_by_time[timestamp])

    def test_baseline_normalization_requires_thirty_samples_per_bearing(self) -> None:
        with self.assertRaisesRegex(ValueError, "fewer than 30 baseline samples"):
            _baseline_normalize_features(
                np.zeros((29, 6)),
                np.repeat("bearing_a", 29),
                np.asarray([f"2003-10-22 12:00:{second:02d}" for second in range(29)]),
            )

    def test_metrics_use_strictly_greater_than_seven_tenths(self) -> None:
        precision, recall, f1 = _metrics(
            np.asarray([1, 1, 0]),
            np.asarray([0.7, 0.8, 0.9]),
        )

        self.assertAlmostEqual(precision, 0.5)
        self.assertAlmostEqual(recall, 0.5)
        self.assertAlmostEqual(f1, 0.5)

    def test_sliding_windows_keep_bearings_separate_and_label_the_final_step(self) -> None:
        features = np.asarray([[index] * 6 for index in range(6)], dtype=np.float64)
        labels = np.asarray([0, 0, 1, 0, 1, 1])
        bearings = np.asarray(["bearing_a"] * 3 + ["bearing_b"] * 3)
        timestamps = np.asarray(
            ["2003-10-22 12:00:00", "2003-10-22 12:10:00", "2003-10-22 12:20:00"] * 2
        )

        windows, window_labels, window_bearings, _, file_indices = _build_sliding_windows(
            features, labels, bearings, timestamps, time_steps=2
        )

        self.assertEqual(windows.shape, (4, 2, 6))
        np.testing.assert_array_equal(window_bearings, ["bearing_a", "bearing_a", "bearing_b", "bearing_b"])
        np.testing.assert_array_equal(window_labels, [0, 1, 1, 1])
        np.testing.assert_array_equal(file_indices, [2, 3, 2, 3])
        self.assertTrue(np.all(windows[:2, :, 0] < 3))
        self.assertTrue(np.all(windows[2:, :, 0] >= 3))

    def test_rolling_mean_and_persistence_are_scoped_to_each_bearing(self) -> None:
        probabilities = np.asarray([0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
        bearings = np.asarray(["bearing_a"] * 3 + ["bearing_b"] * 3)
        timestamps = np.asarray(
            ["2003-10-22 12:00:00", "2003-10-22 12:10:00", "2003-10-22 12:20:00"] * 2
        )

        smoothed = _rolling_mean_by_bearing(probabilities, bearings, timestamps, window=2)
        alarms = _confirmed_alarms(smoothed, bearings, timestamps, threshold=0.5, persistence=2)

        np.testing.assert_allclose(smoothed, [0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
        np.testing.assert_array_equal(alarms, [False, True, True, False, False, False])

    def test_threshold_diagnostics_report_file_index_and_healthy_alarm_episodes(self) -> None:
        bearings = np.asarray(
            ["2nd_test_bearing_1"] * 5 + ["2nd_test_bearing_2"] * 5
        )
        timestamps = np.asarray(
            [f"2003-10-22 12:{index:02d}:00" for index in range(5)] * 2
        )
        labels = np.asarray([0, 0, 1, 1, 1] + [0] * 5)
        probabilities = np.full(10, 0.99)
        file_indices = np.asarray([15, 16, 17, 18, 19] * 2)

        results, selected = _threshold_diagnostics(
            labels, probabilities, bearings, timestamps, file_indices
        )

        self.assertEqual(len(results), len(THRESHOLD_CANDIDATES))
        self.assertEqual(selected["first_alarm_bearing_1_file"], 17)
        self.assertEqual(selected["healthy_false_alarm_episodes"], 1)
        self.assertEqual(
            selected["false_alarm_episodes_by_bearing"]["2nd_test_bearing_2"],
            1,
        )

    def test_probability_validation_requires_one_finite_score_per_sample(self) -> None:
        with self.assertRaisesRegex(ValueError, "one probability"):
            _validate_probabilities([0.2, 1.2], 2, "model")

    def test_cached_dataset_requires_consistent_six_feature_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "features.npz"
            np.savez_compressed(
                path,
                features=np.zeros((2, 6)),
                labels=np.asarray([0, 1]),
                bearing_ids=np.asarray(["bearing_1", "bearing_1"]),
                timestamps=np.asarray(["2003-10-22 12:00:00", "2003-10-22 12:10:00"]),
            )

            features, labels, bearing_ids, timestamps = _load_dataset(path)

        self.assertEqual(features.shape, (2, 6))
        np.testing.assert_array_equal(labels, [0, 1])
        self.assertEqual(bearing_ids.size, 2)
        self.assertEqual(timestamps.size, 2)


if __name__ == "__main__":
    unittest.main()
