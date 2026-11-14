import unittest
import csv
import tempfile
from pathlib import Path

from PIL import Image

from score2midi.ml_dataset import (
    ReviewDataset,
    _candidate_labels,
    _context_crop,
)


class MlDatasetTest(unittest.TestCase):
    def test_context_crop_keeps_surrounding_score(self) -> None:
        image = Image.new("L", (200, 160), 255)

        crop = _context_crop(image, 100, 80, spacing=10)

        self.assertEqual(crop.size, (97, 73))

    def test_labels_candidates_by_musicxml_position(self) -> None:
        candidates = [(100, 50, 0.9), (150, 80, 0.8)]
        expected = [(102.0, 49.0)]

        labels = _candidate_labels(candidates, expected, spacing=10)

        self.assertEqual(labels, ["notehead", "review"])

    def test_review_classification_moves_file_and_updates_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review_dir = root / "review"
            review_dir.mkdir()
            image_path = review_dir / "candidate.png"
            Image.new("L", (32, 32), 255).save(image_path)

            labels_path = root / "labels.csv"
            with labels_path.open(
                "w",
                newline="",
                encoding="utf-8-sig",
            ) as file:
                writer = csv.DictWriter(file, fieldnames=("path", "label"))
                writer.writeheader()
                writer.writerow(
                    {"path": "review/candidate.png", "label": "review"}
                )

            dataset = ReviewDataset(root)
            dataset.classify(0, "notehead")

            self.assertFalse(image_path.exists())
            self.assertTrue((root / "notehead" / "candidate.png").is_file())
            reloaded = ReviewDataset(root)
            self.assertEqual(reloaded.rows[0]["label"], "notehead")
            self.assertEqual(
                reloaded.rows[0]["path"],
                "notehead/candidate.png",
            )

            reloaded.classify(0, "review")
            restored = ReviewDataset(root)
            self.assertEqual(restored.rows[0]["label"], "review")
            self.assertTrue(image_path.is_file())
