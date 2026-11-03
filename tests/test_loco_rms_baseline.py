from __future__ import annotations

import unittest

import numpy as np

from scripts.compare_loco_rms_baseline import (
    RMS_FEATURE_INDEX,
    _fit_rms_score,
    _rms_threshold_candidates,
)


class LocoRmsBaselineTests(unittest.TestCase):
    def test_rms_score_uses_only_training_bearings(self) -> None:
        features = np.zeros((6, 12), dtype=np.float64)
        features[:4, RMS_FEATURE_INDEX] = [1.0, 2.0, 3.0, 4.0]
        features[4:, RMS_FEATURE_INDEX] = [100.0, 101.0]
        groups = np.array(["train_a"] * 2 + ["train_b"] * 2 + ["test"] * 2)

        scores = _fit_rms_score(
            features,
            groups,
            {"train_a", "train_b"},
            "test",
        )

        expected = (np.array([100.0, 101.0]) - 2.5) / np.std([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(scores, expected)

    def test_rms_threshold_candidates_are_unique_and_finite(self) -> None:
        candidates = _rms_threshold_candidates(
            {
                "Bearing1_1": np.array([0.0, 0.0, 1.0, 1.0]),
                "Bearing2_1": np.array([0.0, 1.0, 2.0]),
            }
        )

        self.assertEqual(candidates, tuple(sorted(set(candidates))))
        self.assertTrue(np.isfinite(candidates).all())


if __name__ == "__main__":
    unittest.main()
