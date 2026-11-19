import unittest
from fractions import Fraction

from music21 import meter, note, stream

from score2midi.rhythm_corrector import correct_rhythm


class RhythmCorrectorTest(unittest.TestCase):
    def test_preserves_one_measure_three_eight_meter_change(self) -> None:
        score = stream.Score()
        for pitch in ("C5", "C3"):
            part = stream.Part()
            first = stream.Measure(number=1)
            first.timeSignature = meter.TimeSignature("6/8")
            first.append(note.Note(pitch, quarterLength=3))
            short = stream.Measure(number=2)
            for offset in (0, Fraction(1, 2), Fraction(1)):
                short.insert(offset, note.Note(pitch, quarterLength=Fraction(1, 2)))
            restored = stream.Measure(number=3)
            restored.timeSignature = meter.TimeSignature("6/8")
            restored.append(note.Note(pitch, quarterLength=3))
            part.append([first, short, restored])
            score.insert(0, part)

        correct_rhythm(score)

        for part in score.parts:
            short = part.measure(2)
            signatures = list(short.getElementsByClass(meter.TimeSignature))
            self.assertEqual([item.ratioString for item in signatures], ["3/8"])
            self.assertEqual(part.measure(3).offset, Fraction(9, 2))

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
