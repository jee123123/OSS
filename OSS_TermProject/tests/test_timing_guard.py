import unittest
from fractions import Fraction

from music21 import meter, note, stream

from score2midi.timing_guard import stabilize_note_timing


def beamed(pitch: str, x: float) -> note.Note:
    event = note.Note(pitch, quarterLength=Fraction(1, 2))
    event.beams.fill(1, "start")
    event.style.absoluteX = x
    return event


class TimingGuardTest(unittest.TestCase):
    def test_sequences_distinct_beamed_notes_with_same_onset(self) -> None:
        score = stream.Score()
        part = stream.Part()
        measure = stream.Measure(number=1)
        measure.insert(0, beamed("C4", 10))
        measure.insert(0, beamed("D4", 20))
        measure.insert(1, note.Rest(quarterLength=3))
        part.append(measure)
        score.insert(0, part)

        stabilize_note_timing(score)

        self.assertEqual(
            [Fraction(event.offset) for event in measure.notes],
            [Fraction(0), Fraction(1, 2)],
        )

    def test_preserves_vertically_aligned_notes(self) -> None:
        score = stream.Score()
        part = stream.Part()
        measure = stream.Measure(number=1)
        measure.insert(0, beamed("C4", 10))
        measure.insert(0, beamed("E4", 10))
        measure.insert(1, note.Rest(quarterLength=3))
        part.append(measure)
        score.insert(0, part)

        stabilize_note_timing(score)

        self.assertEqual(
            [Fraction(event.offset) for event in measure.notes],
            [Fraction(0), Fraction(0)],
        )

    def test_restores_non_overlapping_trailing_note(self) -> None:
        score = stream.Score()
        part = stream.Part()
        measure = stream.Measure(number=1)
        measure.timeSignature = meter.TimeSignature("4/4")
        measure.insert(0, note.Rest(quarterLength=Fraction(7, 2)))
        trailing = note.Note("C4", quarterLength=Fraction(1, 2))
        trailing.style.absoluteX = 90
        measure.insert(4, trailing)
        part.append(measure)
        score.insert(0, part)

        stabilize_note_timing(score)

        self.assertEqual(Fraction(trailing.offset), Fraction(7, 2))


if __name__ == "__main__":
    unittest.main()
