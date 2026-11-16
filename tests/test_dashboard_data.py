from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.dashboard_data import (
    load_dashboard_data,
    position_at_progress,
    status_at_position,
)


class DashboardDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.features_path = self.root / "features.npz"
        self.predictions_path = self.root / "predictions.csv"

        np.savez_compressed(
            self.features_path,
            features=np.zeros((4, 12), dtype=np.float64),
            bearing_ids=np.asarray(["Bearing1_1"] * 4),
            sample_indices=np.asarray([1, 2, 3, 4]),
        )
        rows = [
            (1, 0.1, 0.3, 0, 0),
            (2, 0.4, 0.3, 1, 0),
            (3, 0.6, 0.3, 1, 1),
            (4, 0.2, 0.3, 0, 0),
        ]
        with self.predictions_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(
                [
                    "bearing_id",
                    "condition_id",
                    "sample_index",
                    "probability",
                    "balanced_f1_threshold",
                    "balanced_f1_above_threshold",
                    "balanced_f1_notification",
                ]
            )
            for sample_index, probability, threshold, above, notification in rows:
                writer.writerow(
                    [
                        "Bearing1_1",
                        1,
                        sample_index,
                        probability,
                        threshold,
                        above,
                        notification,
                    ]
                )

    def test_dashboard_replay_latches_alert_until_sequence_acknowledgement(self) -> None:
        series = load_dashboard_data(self.features_path, self.predictions_path)["Bearing1_1"]

        self.assertEqual(
            [status_at_position(series, position) for position in range(4)],
            ["green", "yellow", "red", "red"],
        )

    def test_progress_maps_sequence_endpoints_and_rejects_invalid_values(self) -> None:
        series = load_dashboard_data(self.features_path, self.predictions_path)["Bearing1_1"]

        self.assertEqual(position_at_progress(series, 0), 0)
        self.assertEqual(position_at_progress(series, 100), 3)
        with self.assertRaises(ValueError):
            position_at_progress(series, 101)

    def test_loader_rejects_mismatched_feature_and_prediction_samples(self) -> None:
        with self.predictions_path.open("a", encoding="utf-8") as output:
            output.write("Bearing1_1,1,5,0.2,0.3,0,0\n")

        with self.assertRaisesRegex(ValueError, "samples do not match"):
            load_dashboard_data(self.features_path, self.predictions_path)


if __name__ == "__main__":
    unittest.main()
