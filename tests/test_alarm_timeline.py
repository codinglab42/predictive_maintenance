from __future__ import annotations

import unittest

import numpy as np

from analyze_alarm_timeline import _episode_rows
from src.femto_data import FEATURE_NAMES


class AlarmTimelineTests(unittest.TestCase):
    def test_episode_rows_classify_false_persistent_and_in_window_alarms(self) -> None:
        probabilities = np.array([0.1, 0.8, 0.8, 0.1, 0.1, 0.8, 0.8, 0.8, 0.1, 0.8])
        notifications = np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 1])
        features = np.arange(probabilities.size * len(FEATURE_NAMES)).reshape(
            probabilities.size, len(FEATURE_NAMES)
        )

        episodes, assignments = _episode_rows(
            "Bearing1_1",
            np.arange(1, probabilities.size + 1),
            probabilities,
            notifications,
            features,
            threshold=0.5,
            horizon=3,
        )

        self.assertEqual(
            [episode["classification"] for episode in episodes],
            ["false_before_window", "early_persistent", "confirmed_in_window"],
        )
        self.assertEqual(
            [episode["distance_to_end_at_notification"] for episode in episodes],
            [8, 4, 0],
        )
        np.testing.assert_array_equal(assignments, [0, 1, 1, 0, 0, 2, 2, 2, 0, 3])
        self.assertEqual(episodes[0]["samples_in_episode"], 2)
        self.assertEqual(episodes[1]["distance_to_end_at_end"], 2)
        self.assertEqual(episodes[2]["max_horizontal_rms"], features[9, 0])


if __name__ == "__main__":
    unittest.main()
