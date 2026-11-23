from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.predictor import (
    _validate_probability_output,
    load_prediction_config,
    predict_features,
)
from src.femto_data import FEATURE_NAMES
from src.model_artifacts import sha256_file


class PredictionConfigTests(unittest.TestCase):
    def _write_artifacts(self, directory: Path, **metadata_overrides: object) -> tuple[Path, Path]:
        model_path = directory / "model.bin"
        model_path.write_bytes(b"test model")
        metadata_path = directory / "preprocessing.json"
        metadata: dict[str, object] = {
            "format_version": 2,
            "feature_names": FEATURE_NAMES,
            "means": [0.0] * len(FEATURE_NAMES),
            "scales": [1.0] * len(FEATURE_NAMES),
            "threshold": 0.5,
            "horizon_acquisitions": 20,
            "model_sha256": sha256_file(model_path),
        }
        metadata.update(metadata_overrides)
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        return model_path, metadata_path

    def test_loads_config_and_verifies_model_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            model_path, metadata_path = self._write_artifacts(Path(temporary_dir))
            expected_sha256 = sha256_file(model_path)

            config = load_prediction_config(model_path, metadata_path)

            np.testing.assert_array_equal(config.means, np.zeros(len(FEATURE_NAMES)))
            np.testing.assert_array_equal(config.scales, np.ones(len(FEATURE_NAMES)))
            self.assertEqual(config.threshold, 0.5)
            self.assertEqual(config.horizon_acquisitions, 20)
            self.assertEqual(config.model_sha256, expected_sha256)
            self.assertTrue(config.checksum_verified)

    def test_rejects_model_and_metadata_from_different_training_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            model_path, metadata_path = self._write_artifacts(Path(temporary_dir))
            model_path.write_bytes(b"different model")

            with self.assertRaisesRegex(ValueError, "checksum does not match"):
                load_prediction_config(model_path, metadata_path)

    def test_rejects_invalid_threshold_and_horizon(self) -> None:
        for metadata_override, message in (
            ({"threshold": float("nan")}, "alert threshold is invalid"),
            ({"horizon_acquisitions": 0}, "failure horizon is invalid"),
        ):
            with self.subTest(metadata_override=metadata_override):
                with tempfile.TemporaryDirectory() as temporary_dir:
                    model_path, metadata_path = self._write_artifacts(
                        Path(temporary_dir), **metadata_override
                    )

                    with self.assertRaisesRegex(ValueError, message):
                        load_prediction_config(model_path, metadata_path)

    def test_allows_legacy_metadata_without_checksum_as_unverified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            model_path, metadata_path = self._write_artifacts(Path(temporary_dir))
            expected_sha256 = sha256_file(model_path)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            del metadata["model_sha256"]
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            config = load_prediction_config(model_path, metadata_path)

            self.assertEqual(config.model_sha256, expected_sha256)
            self.assertFalse(config.checksum_verified)

    def test_rejects_invalid_model_probability_outputs(self) -> None:
        for output, message in (
            ([0.2, 0.8], "unexpected number"),
            ([float("nan")], "invalid probability"),
            ([1.2], "invalid probability"),
        ):
            with self.subTest(output=output):
                with self.assertRaisesRegex(ValueError, message):
                    _validate_probability_output(output)

    def test_accepts_a_single_valid_model_probability(self) -> None:
        self.assertEqual(_validate_probability_output([[0.75]]), 0.75)

    def test_prediction_uses_shared_validation_and_threshold_override(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            model_path, metadata_path = self._write_artifacts(Path(temporary_dir))
            config = load_prediction_config(model_path, metadata_path)

            class Model:
                def load(self, path: str) -> None:
                    self.loaded_path = path

                def predict(self, features: np.ndarray) -> np.ndarray:
                    self.features = features
                    return np.asarray([[0.75]])

            model = Model()
            core = SimpleNamespace(LogisticRegression=lambda: model)
            with patch("src.predictor.load_ml_core", return_value=core):
                result = predict_features(
                    np.zeros(len(FEATURE_NAMES)),
                    config,
                    threshold_override=0.7,
                )

        self.assertEqual(result.probability, 0.75)
        self.assertEqual(result.threshold, 0.7)
        self.assertEqual(result.threshold_source, "override")
        self.assertEqual(result.horizon_acquisitions, 20)
        self.assertEqual(result.model_sha256, config.model_sha256)
        self.assertTrue(result.checksum_verified)
        self.assertEqual(model.loaded_path, str(config.model_path))
        self.assertEqual(model.features.shape, (1, len(FEATURE_NAMES)))


if __name__ == "__main__":
    unittest.main()
