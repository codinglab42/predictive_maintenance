import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.util import Inches

from scripts import generate_presentation as presentation


class PresentationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.fit = self.root / "fit.json"
        self.inference = self.root / "inference.csv"
        self.fit.write_text(json.dumps({"summary": {
            "linear": {"regularization": [{"lambda": 0.1, "time_ms": 10}]},
            "logistic": {"sample_scaling": [{"samples": 10000, "time_ms": 20}]},
            "neural": {"sample_scaling": [{"samples": 10000, "time_ms": 30}]},
            "sequences": {"recurrent": [
                {"type": name, "time_ms": time}
                for name, time in (("SimpleRNN", 40), ("LSTM", 50), ("GRU", 60))
            ]},
        }}), encoding="utf-8")
        with self.inference.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["model", "batch_size", "calls", "run", "us_per_item"])
            for model in presentation.CORE_MODELS:
                for run in range(1, 6):
                    writer.writerow([model, 64 if model in ("SimpleRNN", "LSTM", "GRU") else 1024,
                                     50, run, run])

    def test_core_loader_uses_medians_and_rejects_missing_runs(self):
        results = presentation.load_core_benchmarks(self.fit, self.inference)
        self.assertEqual(len(results), 6)
        self.assertEqual(results[0]["fit_ms"], 10)
        self.assertEqual(results[0]["inference_us"], 3)
        rows = self.inference.read_text(encoding="utf-8").splitlines()
        self.inference.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "five inference measurements"):
            presentation.load_core_benchmarks(self.fit, self.inference)

    def test_slide_structure_and_footer(self):
        model = {"precision": 0.8, "recall": 0.7, "f1_score": 0.75, "threshold": 0.5,
                 "training_seconds": 1.0, "inference_seconds": 0.1,
                 "first_alarm_bearing_1_file": 700, "healthy_false_alarm_episodes": 0,
                 "false_alarm_episodes_by_bearing": {}}
        metrics = {"models": {name: model for name in presentation.MODEL_ORDER},
                   "active_model": "gru_cpp", "test_samples": 3880, "full_test_samples": 3936,
                   "gru_network": {"time_steps": 15}}
        output = self.root / "demo.pptx"
        screenshot = self.root / "screenshot.png"
        Image.new("RGB", (1027, 768), "blue").save(screenshot)
        with patch.object(presentation, "SCREENSHOT_PATH", screenshot):
            presentation.build_presentation(metrics, presentation.load_core_benchmarks(self.fit, self.inference), output)
        deck = Presentation(str(output))
        self.assertEqual(len(deck.slides), 10)
        texts = ["\n".join(shape.text for shape in slide.shapes if shape.has_text_frame)
                 for slide in deck.slides]
        for index, text in enumerate(texts, 1):
            self.assertIn("Maurizio Penna  •  Deda Speed  •  26/11/2026", text)
            self.assertIn(f"{index:02d} / 10", text)
        self.assertIn("Evoluzione del workflow", texts[1])
        self.assertNotIn("Evoluzione del workflow", texts[9])
        self.assertIn("Roadmap", texts[9])
        self.assertIn("non copre ancora tutte le funzionalità della libreria", texts[2])
        self.assertIn("Progetto applicativo Python", texts[3])
        self.assertNotIn("Auto-Zero", texts[3])
        self.assertIn("Auto-Zero", texts[4])
        self.assertIn("Confine tra i livelli", texts[4])
        self.assertIn("GRU / INFERENZA MISURATA", texts[4])
        pictures = [shape for shape in deck.slides[3].shapes if shape.shape_type == MSO_SHAPE_TYPE.PICTURE]
        self.assertEqual(len(pictures), 1)
        self.assertEqual(pictures[0].image.blob, screenshot.read_bytes())
        self.assertEqual({shape.text for shape in deck.slides[3].shapes if shape.has_text_frame and shape.text}, {
            presentation.APPLICATION_TITLE, "Maurizio Penna  •  Deda Speed  •  26/11/2026", "04 / 10"
        })
        self.assertTrue(any(shape.has_table and len(shape.table.rows) == 7 for shape in deck.slides[2].shapes))

    def test_screenshot_is_centered_and_preserves_aspect_ratio(self):
        for width, height in ((1200, 300), (300, 1200)):
            with self.subTest(width=width, height=height):
                image_path = self.root / f"{width}x{height}.png"
                Image.new("RGB", (width, height)).save(image_path)
                deck = Presentation()
                deck.slide_width = Inches(presentation.WIDTH)
                deck.slide_height = Inches(presentation.HEIGHT)
                presentation.slide_application_screenshot(deck, image_path)
                picture = next(shape for shape in deck.slides[0].shapes if shape.shape_type == MSO_SHAPE_TYPE.PICTURE)
                self.assertAlmostEqual(picture.width / picture.height, width / height, places=5)
                self.assertLessEqual(picture.width, Inches(11.9))
                self.assertLessEqual(picture.height, Inches(5.55))
                self.assertLessEqual(abs(2 * picture.left + picture.width - deck.slide_width), 1)
                self.assertLessEqual(abs(2 * (picture.top - Inches(1.43)) + picture.height - Inches(5.55)), 1)
                self.assertEqual((picture.crop_left, picture.crop_right, picture.crop_top, picture.crop_bottom),
                                 (0, 0, 0, 0))


if __name__ == "__main__":
    unittest.main()
