from __future__ import annotations

import errno
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from download_femto import _publish_dataset, safe_extract_zip
from src.femto_data import (
    FEATURE_NAMES,
    build_labeled_dataset,
    extract_signal_features,
    read_signal_features,
)


class FemtoDataTests(unittest.TestCase):
    def test_extract_signal_features_returns_finite_two_channel_features(self) -> None:
        time = np.arange(32, dtype=np.float64)
        signal = np.column_stack((np.sin(time), np.cos(time)))

        features = extract_signal_features(signal)

        self.assertEqual(features.shape, (len(FEATURE_NAMES),))
        self.assertTrue(np.isfinite(features).all())
        self.assertGreater(features[0], 0.0)
        self.assertGreater(features[6], 0.0)

    def test_read_signal_features_uses_acceleration_columns_from_femto_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "acc_00001.csv"
            time = np.arange(16, dtype=np.float64)
            signal = np.column_stack((np.sin(time), np.cos(time)))
            metadata = np.column_stack(
                (np.zeros((16, 3), dtype=np.float64), time * 1000)
            )
            np.savetxt(path, np.column_stack((metadata, signal)), delimiter=",")

            features = read_signal_features(path)

        np.testing.assert_allclose(features, extract_signal_features(signal))

    def test_labeled_dataset_counts_records_when_acquisition_ids_have_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            learning_bearing = root / "Learning_set" / "Bearing1_1"
            duplicate_learning_bearing = (
                root / "Test_set" / "Training_set" / "Learning_set" / "Bearing1_1"
            )
            test_bearing = root / "Test_set" / "Bearing1_2"
            learning_bearing.mkdir(parents=True)
            duplicate_learning_bearing.mkdir(parents=True)
            test_bearing.mkdir(parents=True)
            signal = np.column_stack((np.arange(16), np.arange(16) * 2))
            progress: list[tuple[int, int, str]] = []
            for sample_id in (1, 2, 4, 5):
                sample = signal + sample_id
                np.savetxt(learning_bearing / f"acc_{sample_id:05}.csv", sample, delimiter=",")
                np.savetxt(
                    duplicate_learning_bearing / f"acc_{sample_id:05}.csv",
                    sample,
                    delimiter=",",
                )
            np.savetxt(test_bearing / "acc_00001.csv", signal, delimiter=",")

            features, labels, groups, sample_ids = build_labeled_dataset(
                root,
                horizon=3,
                progress_callback=lambda processed, total, bearing: progress.append(
                    (processed, total, bearing)
                ),
            )

        self.assertEqual(features.shape, (4, len(FEATURE_NAMES)))
        np.testing.assert_array_equal(labels, [0, 1, 1, 1])
        np.testing.assert_array_equal(groups, ["Bearing1_1"] * 4)
        np.testing.assert_array_equal(sample_ids, [1, 2, 4, 5])
        self.assertEqual(
            progress,
            [
                (1, 4, "Bearing1_1"),
                (2, 4, "Bearing1_1"),
                (3, 4, "Bearing1_1"),
                (4, 4, "Bearing1_1"),
            ],
        )

    def test_safe_extract_rejects_parent_directory_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            archive_path = root / "unsafe.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../outside.txt", "not allowed")

            with self.assertRaisesRegex(ValueError, "unsafe path"):
                safe_extract_zip(archive_path, root / "extracted")
            self.assertFalse((root / "outside.txt").exists())

    def test_publish_dataset_copies_when_directory_rename_is_not_permitted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            staging_dir = root / "staging"
            dataset_dir = root / "dataset"
            staging_dir.mkdir()
            (staging_dir / "sample.csv").write_text("sample", encoding="utf-8")

            with patch(
                "download_femto.os.replace",
                side_effect=OSError(errno.EPERM, "Operation not permitted"),
            ):
                _publish_dataset(staging_dir, dataset_dir)

            self.assertEqual((dataset_dir / "sample.csv").read_text(encoding="utf-8"), "sample")


if __name__ == "__main__":
    unittest.main()
