import unittest
from fractions import Fraction

from music21 import note, stream, tie

from score2midi.articulation import (
    REPEATED_NOTE_GAP,
    separate_repeated_notes,
)


class ArticulationTest(unittest.TestCase):
    def test_separates_repeated_note_across_measure_boundary(self) -> None:
        score = stream.Score()
        part = stream.Part()
        first = stream.Measure(number=1)
        first.insert(3, note.Note("A4", quarterLength=1))
        second = stream.Measure(number=2)
        second.insert(0, note.Note("A4", quarterLength=1))
        part.append([first, second])
        score.insert(0, part)

        separate_repeated_notes(score)

        self.assertEqual(
            Fraction(first.notes[0].duration.quarterLength),
            Fraction(1) - REPEATED_NOTE_GAP,
        )

    def test_preserves_tie_across_measure_boundary(self) -> None:
        score = stream.Score()
        part = stream.Part()
        first = stream.Measure(number=1)
        left = note.Note("A4", quarterLength=1)
        left.tie = tie.Tie("start")
        first.insert(3, left)
        second = stream.Measure(number=2)
        right = note.Note("A4", quarterLength=1)
        right.tie = tie.Tie("stop")
        second.insert(0, right)
        part.append([first, second])
        score.insert(0, part)

        separate_repeated_notes(score)

        self.assertEqual(left.duration.quarterLength, 1)


if __name__ == "__main__":
    unittest.main()
