import unittest
from fractions import Fraction

import cv2
import numpy as np

from score2midi.beam_recovery import (
    BeamNote,
    detect_beamed_note_groups,
    detect_staff_lines,
    is_safe_measure_replacement,
)


class BeamRecoveryTest(unittest.TestCase):
    def test_detects_staff_lines(self) -> None:
        image = np.full((120, 240), 255, dtype=np.uint8)
        for y in (40, 48, 56, 64, 72):
            cv2.line(image, (10, y), (230, y), 0, 1)

        self.assertEqual(
            detect_staff_lines(image),
            [(40, 48, 56, 64, 72)],
        )

    def test_accepts_only_complete_measure(self) -> None:
        notes = [
            BeamNote(
                x=x,
                y=50,
                midi_pitch=60,
                beam_count=1,
                quarter_length=Fraction(1, 2),
                confidence=0.9,
            )
            for x in (20, 40, 60, 80, 100, 120)
        ]

        self.assertTrue(
            is_safe_measure_replacement(
                notes,
                Fraction(3),
                x_tolerance=5,
            )
        )
        self.assertFalse(
            is_safe_measure_replacement(
                notes[:-1],
                Fraction(3),
                x_tolerance=5,
            )
        )

    def test_detects_four_notes_on_one_beam(self) -> None:
        image = np.full((140, 260), 255, dtype=np.uint8)
        staff = (50, 60, 70, 80, 90)
        for y in staff:
            cv2.line(image, (20, y), (240, y), 0, 1)
        cv2.rectangle(image, (58, 28), (182, 33), 0, -1)
        for x, y in ((60, 78), (100, 73), (140, 68), (180, 63)):
            cv2.line(image, (x, 30), (x, y), 0, 2)
            cv2.ellipse(image, (x - 5, y), (7, 4), -15, 0, 360, 0, -1)

        groups = detect_beamed_note_groups(image, staff)

        self.assertTrue(any(len(group) == 4 for group in groups))


if __name__ == "__main__":
    unittest.main()
