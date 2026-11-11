import unittest
from fractions import Fraction

from music21 import meter, note, stream

from score2midi.rhythm_corrector import correct_rhythm


class RhythmCorrectorTest(unittest.TestCase):
    def test_does_not_modify_tuplet_measure(self) -> None:
        score = stream.Score()
        part = stream.Part()
        measure = stream.Measure(number=1)
        measure.timeSignature = meter.TimeSignature("3/4")
        offset = Fraction(0)

        for length in (
            Fraction(1, 3),
            Fraction(1, 3),
            Fraction(1, 3),
            Fraction(1),
            Fraction(1),
            Fraction(1, 2),
        ):
            event = note.Note("C4", quarterLength=length)
            measure.insert(offset, event)
            offset += length

        part.append(measure)
        score.insert(0, part)
        before = [
            (event.offset, event.duration.quarterLength)
            for event in measure.notes
        ]

        warnings = correct_rhythm(score)

        after = [
            (event.offset, event.duration.quarterLength)
            for event in measure.notes
        ]
        self.assertEqual(after, before)
        self.assertTrue(any("셋잇단음표" in warning for warning in warnings))


if __name__ == "__main__":
    unittest.main()
