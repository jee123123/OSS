import unittest

from score2midi.ml_dataset import _candidate_labels


class MlDatasetTest(unittest.TestCase):
    def test_labels_candidates_by_musicxml_position(self) -> None:
        candidates = [(100, 50, 0.9), (150, 80, 0.8)]
        expected = [(102.0, 49.0)]

        labels = _candidate_labels(candidates, expected, spacing=10)

        self.assertEqual(labels, ["notehead", "review"])
