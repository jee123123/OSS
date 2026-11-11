import unittest

from music21 import stream, tempo

from score2midi.cli import _set_tempo


class TempoTest(unittest.TestCase):
    def test_replaces_existing_tempo_with_selected_bpm(self) -> None:
        score = stream.Score()
        part = stream.Part()
        part.insert(2, tempo.MetronomeMark(number=80))
        score.insert(0, part)

        _set_tempo(score, 96)

        marks = list(
            score.recurse().getElementsByClass(tempo.MetronomeMark)
        )
        self.assertEqual(len(marks), 1)
        self.assertEqual(marks[0].number, 96)

    def test_rejects_invalid_bpm(self) -> None:
        with self.assertRaises(ValueError):
            _set_tempo(stream.Score(), 0)


if __name__ == "__main__":
    unittest.main()
