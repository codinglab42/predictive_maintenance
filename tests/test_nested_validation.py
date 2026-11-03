from __future__ import annotations

import unittest

import numpy as np

from validate_nested_policies import (
    _notification_events,
    evaluate_policy,
    select_policies,
)


class NestedValidationTests(unittest.TestCase):
    def test_confirmation_and_cooldown_emit_at_most_one_notification_per_run(self) -> None:
        probabilities = np.array([0.9, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9, 0.9])

        notifications, events = _notification_events(
            probabilities,
            threshold=0.5,
            confirmation=2,
            cooldown=4,
        )

        np.testing.assert_array_equal(notifications, [0, 1, 0, 0, 0, 0, 0, 1])
        self.assertEqual(events, [(1, 1), (7, 7)])

    def test_event_metrics_distinguish_false_persistent_and_in_window(self) -> None:
        probabilities = {
            "false": np.array([0.1, 0.9, 0.9, 0.1, 0.1, 0.1]),
            "persistent": np.array([0.1, 0.1, 0.9, 0.9, 0.9, 0.9]),
            "detected": np.array([0.1, 0.1, 0.1, 0.1, 0.9, 0.9]),
        }

        metrics = evaluate_policy(
            probabilities,
            threshold=0.5,
            confirmation=2,
            cooldown=0,
            horizon=2,
        )

        self.assertEqual(metrics["detected_bearings"], 1)
        self.assertEqual(metrics["missed_bearings"], 2)
        self.assertEqual(metrics["false_episodes"], 1)
        self.assertEqual(metrics["early_persistent_episodes"], 1)
        self.assertEqual(metrics["in_window_episodes"], 1)
        self.assertEqual(metrics["event_precision"], 0.5)
        self.assertEqual(metrics["event_recall"], 1 / 3)
        self.assertEqual(metrics["median_in_window_lead"], 0)
        self.assertEqual(metrics["per_bearing"][2]["in_window_leads"], [0])

    def test_policy_selection_marks_unmet_inner_recall_constraint(self) -> None:
        probabilities = {
            "Bearing1_1": np.full(30, 0.1),
            "Bearing1_2": np.full(30, 0.1),
        }

        selected = select_policies(probabilities, horizon=20)

        for policy in selected.values():
            self.assertEqual(policy["threshold"], 0.1)
            self.assertEqual(policy["confirmation"], 1)
            self.assertEqual(policy["cooldown"], 0)
        self.assertTrue(selected["precision_at_80pct_recall"]["inner_constraint_fallback"])
        self.assertFalse(selected["sensitivity_first"]["inner_constraint_fallback"])

    def test_policy_selection_accepts_data_derived_thresholds(self) -> None:
        probabilities = {
            "Bearing1_1": np.array([0.1, 0.1, 0.9, 0.9]),
            "Bearing1_2": np.array([0.2, 0.2, 0.8, 0.8]),
        }

        selected = select_policies(
            probabilities,
            horizon=2,
            thresholds=(0.25, 0.75),
        )

        self.assertIn(selected["balanced_f1"]["threshold"], (0.25, 0.75))


if __name__ == "__main__":
    unittest.main()
