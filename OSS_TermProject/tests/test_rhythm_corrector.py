import unittest
from fractions import Fraction

from music21 import meter, note, stream

from score2midi.rhythm_corrector import correct_rhythm


class RhythmCorrectorTest(unittest.TestCase):
    def test_removes_shared_false_leading_rests(self) -> None:
        score = stream.Score()
        measures = []

        for pitch in ("A5", "F#3"):
            part = stream.Part()
            measure = stream.Measure(number=15)
            measure.timeSignature = meter.TimeSignature("4/4")
            measure.insert(0, note.Rest(quarterLength=Fraction(1, 2)))
            for index in range(8):
                measure.insert(
                    Fraction(1, 2) + Fraction(index, 2),
                    note.Note(pitch, quarterLength=Fraction(1, 2)),
                )
            part.append(measure)
            score.insert(0, part)
            measures.append(measure)

        warnings = correct_rhythm(score)

        self.assertEqual([measure.highestTime for measure in measures], [4, 4])
        self.assertEqual([measure.notes[0].offset for measure in measures], [0, 0])
        self.assertTrue(any("공통 쉼표" in warning for warning in warnings))

    def test_removes_shared_padding_created_by_leading_grace_note(self) -> None:
        score = stream.Score()

        right = stream.Part()
        right_measure = stream.Measure(number=9)
        right_measure.timeSignature = meter.TimeSignature("4/4")
        right_measure.insert(0, note.Note("F#5", quarterLength=1))
        right_measure.insert(1, note.Note("A5", quarterLength=Fraction(1, 2)))
        for offset, pitch, length in (
            (Fraction(3, 2), "G#5", Fraction(1, 2)),
            (Fraction(2), "A5", Fraction(1)),
            (Fraction(3), "E5", Fraction(1)),
            (Fraction(4), "D5", Fraction(1)),
        ):
            right_measure.insert(offset, note.Note(pitch, quarterLength=length))
        right.append(right_measure)
        score.insert(0, right)

        left = stream.Part()
        left_measure = stream.Measure(number=9)
        left_measure.timeSignature = meter.TimeSignature("4/4")
        left_measure.insert(0, note.Rest(quarterLength=1))
        for offset, pitch in enumerate(("F#3", "C#4", "F#4", "D3"), 1):
            left_measure.insert(offset, note.Note(pitch, quarterLength=1))
        left.append(left_measure)
        score.insert(0, left)

        warnings = correct_rhythm(score)

        self.assertEqual(right_measure.highestTime, 4)
        self.assertEqual(left_measure.highestTime, 4)
        self.assertEqual(right_measure.notes[0].duration.quarterLength, Fraction(1, 8))
        self.assertEqual(right_measure.notes[1].offset, Fraction(1, 8))
        self.assertEqual(left_measure.notes[0].offset, 0)
        self.assertTrue(any("장식음" in warning for warning in warnings))

    def test_repairs_leading_grace_padding_inside_single_voices(self) -> None:
        score = stream.Score()
        measures = []
        for first in (note.Note("F#5", quarterLength=1), note.Rest(quarterLength=1)):
            part = stream.Part()
            measure = stream.Measure(number=9)
            measure.timeSignature = meter.TimeSignature("4/4")
            voice = stream.Voice()
            voice.insert(0, first)
            for offset in range(1, 5):
                voice.insert(offset, note.Note("A4", quarterLength=1))
            measure.insert(0, voice)
            part.append(measure)
            score.insert(0, part)
            measures.append(measure)

        correct_rhythm(score)

        self.assertEqual([measure.highestTime for measure in measures], [4, 4])
        self.assertEqual(measures[0].recurse().notes[0].duration.quarterLength, Fraction(1, 8))
        self.assertEqual(measures[1].recurse().notes[0].offset, 0)

    def test_repairs_two_staff_voices_in_one_measure(self) -> None:
        score = stream.Score()
        part = stream.Part()
        measure = stream.Measure(number=9)
        measure.timeSignature = meter.TimeSignature("4/4")

        for first in (note.Note("F#5", quarterLength=1), note.Rest(quarterLength=1)):
            voice = stream.Voice()
            voice.insert(0, first)
            for offset in range(1, 5):
                voice.insert(offset, note.Note("A4", quarterLength=1))
            measure.insert(0, voice)

        part.append(measure)
        score.insert(0, part)

        correct_rhythm(score)

        voices = list(measure.getElementsByClass(stream.Voice))
        self.assertEqual([voice.highestTime for voice in voices], [4, 4])
        self.assertEqual(voices[0].notes[0].duration.quarterLength, Fraction(1, 8))
        self.assertEqual(voices[1].notes[0].offset, 0)

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
