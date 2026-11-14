import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch not installed")
class NoteheadClassifierTest(unittest.TestCase):
    def test_model_returns_two_class_scores(self) -> None:
        import torch

        from score2midi.note_classifier import NoteheadClassifier

        model = NoteheadClassifier()
        scores = model(torch.zeros(3, 1, 32, 32))

        self.assertEqual(tuple(scores.shape), (3, 2))

    def test_saved_model_can_predict_an_image(self) -> None:
        import tempfile
        from pathlib import Path

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

            self.assertIn(label, CLASS_NAMES)
            self.assertGreaterEqual(confidence, 0.5)
            self.assertLessEqual(confidence, 1.0)
