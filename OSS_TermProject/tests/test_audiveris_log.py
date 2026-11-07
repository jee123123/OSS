import unittest

from score2midi.cli import _find_unplaced_chords


class AudiverisLogTest(unittest.TestCase):
    def test_finds_and_deduplicates_unplaced_chords(self) -> None:
        line = (
            "INFO [score#2] MeasureRhythm | "
            "Measure{#2P1} No timeOffset for HeadChordInter#1"
        )
        log_text = f"{line}\n{line}"

        self.assertEqual(
            _find_unplaced_chords(log_text),
            {(2, 2, 1)},
        )


if __name__ == "__main__":
    unittest.main()
