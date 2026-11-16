import importlib.util
import tempfile
import unittest
from pathlib import Path

from score2midi.cli import DEFAULT_NOTE_MODEL, _load_notehead_predictor


class BundledNoteheadModelTest(unittest.TestCase):
    def test_default_model_is_bundled(self) -> None:
        self.assertTrue(DEFAULT_NOTE_MODEL.is_file())

    def test_missing_model_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            predictor, warning = _load_notehead_predictor(
                Path(directory) / "missing.pt"
            )

        self.assertIsNone(predictor)
        self.assertIn("기존 방식", warning)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch not installed")
class NoteheadClassifierTest(unittest.TestCase):
    def test_model_returns_two_class_scores(self) -> None:
        import torch

        from score2midi.note_classifier import NoteheadClassifier

        model = NoteheadClassifier()
        scores = model(torch.zeros(3, 1, 32, 32))

        self.assertEqual(tuple(scores.shape), (3, 2))

    def test_saved_model_can_predict_an_image(self) -> None:
        import torch
        from PIL import Image

        from score2midi.note_classifier import (
            CLASS_NAMES,
            NoteheadClassifier,
            NoteheadPredictor,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model.pt"
            image_path = root / "candidate.png"
            torch.save(
                {
                    "state_dict": NoteheadClassifier().state_dict(),
                    "class_names": CLASS_NAMES,
                },
                model_path,
            )
            Image.new("L", (32, 32), 255).save(image_path)

            label, confidence = NoteheadPredictor(model_path).predict(
                image_path
            )
            image_label, _ = NoteheadPredictor(model_path).predict(
                Image.new("L", (24, 20), 255)
            )

            self.assertIn(label, CLASS_NAMES)
            self.assertIn(image_label, CLASS_NAMES)
            self.assertGreaterEqual(confidence, 0.5)
            self.assertLessEqual(confidence, 1.0)
