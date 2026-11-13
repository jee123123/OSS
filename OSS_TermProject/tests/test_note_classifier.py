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
