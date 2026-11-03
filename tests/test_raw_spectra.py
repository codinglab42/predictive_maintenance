from __future__ import annotations

import unittest

import numpy as np

from analyze_raw_spectra import (
    _bearing_characteristic_frequencies,
    _frequency_peak_rows,
    _periodogram,
    _select_sample_groups,
)


class RawSpectraTests(unittest.TestCase):
    def test_selects_disjoint_alarm_groups_and_excludes_terminal_part_of_persistent(self) -> None:
        timeline_rows = [
            {
                "sample_index": str(index),
                "distance_to_end_acquisitions": str(300 - index),
                "above_threshold": "0",
            }
            for index in range(1, 301)
        ]
        timeline_rows[3]["above_threshold"] = "1"
        timeline_rows[4]["above_threshold"] = "1"
        timeline_rows[5]["above_threshold"] = "1"
        episodes = [
            {
                "classification": "false_before_window",
                "start_sample_index": "4",
                "end_sample_index": "6",
            },
            {
                "classification": "early_persistent",
                "start_sample_index": "270",
                "end_sample_index": "300",
            },
        ]

        groups = _select_sample_groups(timeline_rows, episodes, horizon=20)

        self.assertEqual(groups["confirmed_false_episodes"], [4, 5, 6])
        self.assertEqual(groups["persistent_episode_before_h20"], list(range(270, 281)))
        self.assertEqual(groups["final_h20_window"], list(range(281, 301)))
        self.assertEqual(len(groups["far_below_threshold"]), 97)

    def test_periodogram_finds_known_tone_at_ten_hz_resolution(self) -> None:
        sample_rate = 25_600.0
        sample_count = 2_560
        time = np.arange(sample_count) / sample_rate
        signal = np.column_stack(
            (np.sin(2 * np.pi * 1_000 * time), np.sin(2 * np.pi * 2_000 * time))
        )

        frequencies, power = _periodogram(signal)

        self.assertEqual(frequencies[1] - frequencies[0], 10.0)
        self.assertEqual(frequencies[np.argmax(power[:, 0])], 1_000.0)
        self.assertEqual(frequencies[np.argmax(power[:, 1])], 2_000.0)

    def test_nsk_6307du_scenario_frequencies_use_provisional_geometry(self) -> None:
        frequencies = _bearing_characteristic_frequencies()

        self.assertAlmostEqual(frequencies["shaft_1x"], 30.0)
        self.assertAlmostEqual(frequencies["FTF"], 11.4809, places=3)
        self.assertAlmostEqual(frequencies["BSF"], 60.417, places=2)
        self.assertAlmostEqual(frequencies["BPFO"], 91.847, places=2)
        self.assertAlmostEqual(frequencies["BPFI"], 148.153, places=2)

    def test_frequency_peak_rows_report_harmonic_bins_and_reference_ratios(self) -> None:
        frequencies = np.arange(0.0, 401.0, 10.0)
        reference = np.ones((frequencies.size, 2))
        target = np.full((frequencies.size, 2), 2.0)
        target[8, :] = 1.5
        target[10, :] = 1.5
        groups = {
            "far_below_threshold": [1, 2, 3],
            "confirmed_false_episodes": [4, 5],
        }

        rows = _frequency_peak_rows(
            "Bearing1_1",
            groups,
            frequencies,
            {
                "far_below_threshold": reference,
                "confirmed_false_episodes": target,
            },
        )
        bpfo_fundamental = next(
            row
            for row in rows
            if row["group"] == "confirmed_false_episodes"
            and row["channel"] == "horizontal"
            and row["frequency_name"] == "BPFO"
            and row["harmonic_order"] == 1
        )

        self.assertEqual(bpfo_fundamental["nearest_bin_frequency_hz"], 90.0)
        self.assertEqual(bpfo_fundamental["peak_within_one_bin_hz"], 90.0)
        self.assertEqual(bpfo_fundamental["psd_ratio_vs_far_reference"], 2.0)
        self.assertEqual(bpfo_fundamental["sample_count"], 2)


if __name__ == "__main__":
    unittest.main()
