from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from src.ims_data import (
    FEATURE_NAMES,
    build_labeled_dataset,
    extract_channel_features,
)


class IMSDataTests(unittest.TestCase):
    def test_feature_order_and_constant_channel_statistics(self) -> None:
        features = extract_channel_features(np.full(16, 3.0))

        self.assertEqual(features.shape, (len(FEATURE_NAMES),))
        np.testing.assert_allclose(features, [3.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    def test_feature_moments_use_population_standard_deviation_and_excess_kurtosis(self) -> None:
        features = extract_channel_features(np.asarray([0.0, 0.0, 0.0, 1.0, 2.0]))

        np.testing.assert_allclose(
            features,
            [1.0, 0.8, -0.921875, 0.84375, 2.0, 2.0],
        )

    def test_first_series_averages_per_channel_features_and_labels_failure_tail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            series_dir = Path(temporary_dir) / "1st_test"
            series_dir.mkdir()
            timestamps = [
                datetime(2003, 10, 22, 12, 0, 0) + timedelta(minutes=10 * index)
                for index in range(4)
            ]
            for index in reversed(range(4)):
                channels = [
                    np.full(16, float(column + index))
                    for column in range(8)
                ]
                np.savetxt(
                    series_dir / timestamps[index].strftime("%Y.%m.%d.%H.%M.%S"),
                    np.column_stack(channels),
                )

            features, labels, bearing_ids, sample_timestamps = build_labeled_dataset(
                Path(temporary_dir),
                "1st_test",
            )

        self.assertEqual(features.shape, (16, len(FEATURE_NAMES)))
        np.testing.assert_array_equal(
            sample_timestamps[::4],
            [timestamp.isoformat(sep=" ") for timestamp in timestamps],
        )
        self.assertEqual(bearing_ids[0], "1st_test_bearing_1")
        for bearing_number in range(1, 5):
            bearing_rows = bearing_ids == f"1st_test_bearing_{bearing_number}"
            expected_labels = [0, 0, 0, int(bearing_number in {3, 4})]
            np.testing.assert_array_equal(labels[bearing_rows], expected_labels)

        first_bearing_first_sample = features[0]
        np.testing.assert_allclose(
            first_bearing_first_sample,
            np.mean(
                [
                    extract_channel_features(np.full(16, 0.0)),
                    extract_channel_features(np.full(16, 1.0)),
                ],
                axis=0,
            ),
        )

    def test_second_and_third_series_use_one_channel_per_bearing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            for series_name in ("2nd_test", "3rd_test"):
                series_dir = root / series_name
                series_dir.mkdir()
                for index in range(4):
                    signal = np.column_stack(
                        [np.full(16, float(index + bearing)) for bearing in range(4)]
                    )
                    path = series_dir / (
                        datetime(2003, 10, 22, 12, 0, 0)
                        .replace(minute=index * 10)
                        .strftime("%Y.%m.%d.%H.%M.%S")
                    )
                    np.savetxt(path, signal)

                features, labels, bearing_ids, _ = build_labeled_dataset(root, series_name)

                self.assertEqual(features.shape, (16, len(FEATURE_NAMES)))
                failed_bearing = 1 if series_name == "2nd_test" else 3
                for bearing_number in range(1, 5):
                    bearing_rows = bearing_ids == f"{series_name}_bearing_{bearing_number}"
                    expected = [0, 0, 0, int(bearing_number == failed_bearing)]
                    np.testing.assert_array_equal(labels[bearing_rows], expected)

if __name__ == "__main__":
    unittest.main()
