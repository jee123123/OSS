import unittest

from fractions import Fraction

from music21 import chord, meter, note, stream

from score2midi.rhythm_recovery import recover_missing_short_notes


class RhythmRecoveryTest(unittest.TestCase):
    def test_recovers_sequential_triplet_from_misread_chord(self) -> None:
        primary = stream.Score()
        primary_part = stream.Part()
        primary_measure = stream.Measure(number=1)
        primary_measure.insert(0, meter.TimeSignature("4/4"))
        primary_measure.insert(0, note.Note("D3", quarterLength=0.5))
        primary_measure.insert(0.5, note.Note("A3", quarterLength=0.5))
        primary_measure.insert(1, note.Note("D4", quarterLength=1))
        primary_measure.insert(
            2,
            chord.Chord(["F2", "E3"], quarterLength=1),
        )
        primary_measure.insert(3, note.Note("B3", quarterLength=1))
        primary_part.append(primary_measure)
        primary.insert(0, primary_part)

        alternative = stream.Score()
        alternative_part = stream.Part()
        alternative_measure = stream.Measure(number=1)
        alternative_measure.insert(0, meter.TimeSignature("4/4"))
        alternative_measure.insert(0, note.Note("D3", quarterLength=0.5))
        alternative_measure.insert(0.5, note.Note("A3", quarterLength=0.5))
        alternative_measure.insert(1, note.Note("D4", quarterLength=1))
        alternative_measure.insert(
            2,
            note.Note("E3", quarterLength=Fraction(2, 3)),
        )
        alternative_measure.insert(
            Fraction(8, 3),
            note.Note("B3", quarterLength=Fraction(2, 3)),
        )
        alternative_measure.insert(
            Fraction(10, 3),
            note.Note("E4", quarterLength=Fraction(2, 3)),
        )
        alternative_part.append(alternative_measure)
        alternative.insert(0, alternative_part)

        warnings = recover_missing_short_notes(primary, alternative)

        recovered = list(primary_part.measure(1).recurse().notes)
        self.assertEqual(
            [event.pitch.nameWithOctave for event in recovered],
            ["D3", "A3", "D4", "E3", "B3", "E4"],
        )
        self.assertEqual(
            [event.offset for event in recovered[-3:]],
            [Fraction(2), Fraction(8, 3), Fraction(10, 3)],
        )
        self.assertTrue(any("짧은 음표" in warning for warning in warnings))

    def test_replaces_equal_note_count_when_rhythm_is_more_accurate(
        self,
    ) -> None:
        primary = stream.Score()
        primary_part = stream.Part()
        primary_measure = stream.Measure(number=1)
        primary_measure.insert(0, meter.TimeSignature("4/4"))
        for offset in range(5):
            primary_measure.insert(offset, note.Note("C4", quarterLength=1))
        primary_part.append(primary_measure)
        primary.insert(0, primary_part)

        alternative = stream.Score()
        alternative_part = stream.Part()
        alternative_measure = stream.Measure(number=1)
        alternative_measure.insert(0, meter.TimeSignature("4/4"))
        for offset, duration in (
            (0, 0.5),
            (0.5, 0.5),
            (1, 1),
            (2, 1),
            (3, 1),
        ):
            alternative_measure.insert(
                offset,
                note.Note("C4", quarterLength=duration),
            )
        alternative_part.append(alternative_measure)
        alternative.insert(0, alternative_part)

        warnings = recover_missing_short_notes(primary, alternative)

        recovered_measure = primary_part.measure(1)
        self.assertEqual(recovered_measure.highestTime, 4)
        self.assertTrue(any("짧은 음표" in warning for warning in warnings))

    def test_appends_trailing_measure_missing_from_primary(self) -> None:
        primary = stream.Score()
        primary_part = stream.Part()
        primary_part.append(stream.Measure(number=1))
        primary.insert(0, primary_part)

        alternative = stream.Score()
        alternative_part = stream.Part()
        alternative_part.append(stream.Measure(number=1))
        final_measure = stream.Measure(number=2)
        final_measure.append(note.Note("C4"))
        alternative_part.append(final_measure)
        alternative.insert(0, alternative_part)

        warnings = recover_missing_short_notes(primary, alternative)

        measures = list(primary_part.getElementsByClass(stream.Measure))
        self.assertEqual(len(measures), 2)
        self.assertTrue(any("마지막 마디" in warning for warning in warnings))
