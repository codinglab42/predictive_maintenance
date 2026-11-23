from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from backend import (
    BearingReading,
    _load_cached_dashboard_data,
    build_maintenance_reports,
    classify_probability,
)


class DashboardBackendTests(unittest.TestCase):
    def test_dashboard_uses_threshold_shared_by_metrics_and_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            features_path = root / "features.npz"
            predictions_path = root / "predictions.csv"
            metrics_path = root / "metrics.json"
            timestamps = np.asarray(["2003-10-22 12:00:00", "2003-10-22 12:10:00"])
            bearing_ids = np.asarray(
                [f"2nd_test_bearing_{index}" for index in range(1, 5) for _ in range(2)]
            )
            np.savez_compressed(
                features_path,
                features=np.zeros((8, 6)),
                labels=np.zeros(8, dtype=np.int64),
                bearing_ids=bearing_ids,
                timestamps=np.tile(timestamps, 4),
            )
            with predictions_path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.writer(output)
                writer.writerow(
                    ["bearing_id", "timestamp", "label", "probability", "alert_threshold"]
                )
                writer.writerows(
                    (
                        bearing_id,
                        timestamp,
                        0,
                        (sample_index + 1) / 10.0,
                        0.85,
                    )
                    for sample_index, (bearing_id, timestamp) in enumerate(
                        (bearing_id, timestamp)
                        for bearing_id in np.unique(bearing_ids)
                        for timestamp in timestamps
                    )
                )
            metrics_path.write_text(
                json.dumps(
                    {
                        "threshold": 0.85,
                        "models": {
                            model_name: {
                                "precision": 0.5,
                                "recall": 0.5,
                                "f1_score": 0.5,
                            }
                            for model_name in (
                                "neural_network_cpp",
                                "logistic_regression",
                            )
                        },
                    }
                ),
                encoding="utf-8",
            )

            data = _load_cached_dashboard_data.__wrapped__(
                str(features_path),
                str(predictions_path),
                str(metrics_path),
                ((1, 1), (1, 1), (1, 1)),
            )
            with predictions_path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.writer(output)
                writer.writerow(
                    ["bearing_id", "timestamp", "label", "probability", "alert_threshold"]
                )
                writer.writerows(
                    (bearing_id, timestamp, 0, float(sample_index % 2), 0.85)
                    for sample_index, (bearing_id, timestamp) in enumerate(
                        (bearing_id, timestamp)
                        for bearing_id in np.unique(bearing_ids)
                        for timestamp in timestamps
                    )
                )
            with self.assertRaisesRegex(ValueError, "class-like scores"):
                _load_cached_dashboard_data.__wrapped__(
                    str(features_path),
                    str(predictions_path),
                    str(metrics_path),
                    ((2, 1), (2, 1), (1, 1)),
                )

        self.assertEqual(data.selected_threshold, 0.85)

    def test_alarm_states_respect_prealarm_and_strict_threshold_boundaries(self) -> None:
        self.assertEqual(classify_probability(0.449, 0.70), "operational")
        self.assertEqual(classify_probability(0.45, 0.70), "prealarm")
        self.assertEqual(classify_probability(0.70, 0.70), "prealarm")
        self.assertEqual(classify_probability(0.7001, 0.70), "critical")

    def test_maintenance_reports_include_only_critical_bearings(self) -> None:
        readings = (
            BearingReading(
                "2nd_test_bearing_1",
                "2004-02-13 08:00:00",
                0.82,
                18.0,
                0.25,
                1.4,
                1,
                "critical",
            ),
            BearingReading(
                "2nd_test_bearing_2",
                "2004-02-13 08:00:00",
                0.52,
                48.0,
                0.17,
                0.8,
                0,
                "prealarm",
            ),
        )

        csv_report, text_report = build_maintenance_reports(readings, 0.70)

        self.assertIsNotNone(csv_report)
        self.assertIsNotNone(text_report)
        rows = list(csv.DictReader(io.StringIO(csv_report or "")))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["bearing_id"], "2nd_test_bearing_1")
        self.assertIn("Main blade spindle (front)", rows[0]["component"])
        self.assertIn("human assessment", rows[0]["recommended_action"])
        self.assertIn("2004-02-13 08:00:00", text_report or "")

    def test_maintenance_report_is_unavailable_without_critical_bearings(self) -> None:
        reading = BearingReading(
            "2nd_test_bearing_1",
            "2004-02-13 08:00:00",
            0.70,
            30.0,
            0.25,
            1.4,
            1,
            "prealarm",
        )
        self.assertEqual(build_maintenance_reports((reading,), 0.70), (None, None))


if __name__ == "__main__":
    unittest.main()
