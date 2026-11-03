from __future__ import annotations

import unittest

from src.femto_data import group_bearings_by_condition, infer_condition_id


class LocoValidationTests(unittest.TestCase):
    def test_condition_id_matches_bearing_prefix(self) -> None:
        self.assertEqual(infer_condition_id("Bearing1_1"), 1)
        self.assertEqual(infer_condition_id("Bearing2_4"), 2)
        self.assertEqual(infer_condition_id("Bearing3_2"), 3)

    def test_bearings_are_grouped_by_condition(self) -> None:
        grouped = group_bearings_by_condition(
            [
                "Bearing3_2",
                "Bearing1_1",
                "Bearing2_1",
                "Bearing1_2",
                "Bearing2_2",
                "Bearing3_1",
            ]
        )
        self.assertEqual(
            grouped,
            {
                1: ["Bearing1_1", "Bearing1_2"],
                2: ["Bearing2_1", "Bearing2_2"],
                3: ["Bearing3_1", "Bearing3_2"],
            },
        )


if __name__ == "__main__":
    unittest.main()
