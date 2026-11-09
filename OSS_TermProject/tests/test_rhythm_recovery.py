import unittest

from music21 import note, stream

from score2midi.rhythm_recovery import recover_missing_short_notes


class RhythmRecoveryTest(unittest.TestCase):
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
