import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from music21 import converter, note, stream, tempo

from score2midi.cli import _set_tempo, rewrite_midi_tempo


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

    def test_rewrites_midi_with_selected_bpm(self) -> None:
        score = stream.Score()
        part = stream.Part()
        part.append(note.Note("C4", quarterLength=1))
        score.insert(0, part)

        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.mid"
            output = Path(directory) / "output.mid"
            score.write("midi", fp=str(source))

            rewrite_midi_tempo(source, output, 88)

            converted = converter.parse(str(output))
            marks = list(
                converted.recurse().getElementsByClass(tempo.MetronomeMark)
            )

        self.assertTrue(marks)
        self.assertAlmostEqual(marks[0].number, 88, delta=0.1)


if __name__ == "__main__":
    unittest.main()
