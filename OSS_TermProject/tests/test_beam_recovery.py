import unittest
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from music21 import meter, note, stream

from score2midi.beam_recovery import (
    BeamNote,
    _apply_beam_groups,
    _duration_variants,
    _has_complete_timeline,
    _measure_regions,
    detect_beamed_note_groups,
    detect_leading_eighth_rest,
    detect_staff_lines,
    deduplicate_staff_lines,
    is_safe_measure_replacement,
    select_non_overlapping_beam_groups,
)
from score2midi.beam_layout import _measure_anchors


class BeamRecoveryTest(unittest.TestCase):
    def test_measure_anchors_return_fractional_offsets(self) -> None:
        measure = stream.Measure(number=1)
        event = note.Note("C4", quarterLength=Fraction(1, 2))
        event.style.absoluteX = 42
        measure.insert(Fraction(1, 2), event)

        self.assertEqual(_measure_anchors(measure), [(42.0, Fraction(1, 2))])

    def test_measure_regions_accepts_missing_left_margin(self) -> None:
        xml = """\
<score-partwise version="4.0">
  <defaults><page-layout><page-width>1000</page-width></page-layout></defaults>
  <part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1" width="200"><print new-system="yes"/></measure>
  </part>
</score-partwise>
"""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "score.musicxml"
            path.write_text(xml, encoding="utf-8")

            page_width, regions = _measure_regions(path)

        self.assertEqual(page_width, 1000)
        self.assertEqual(regions[(1, 1)].x, 0)

    def test_measure_regions_combines_layout_from_split_parts(self) -> None:
        xml = """\
<score-partwise version="4.0">
  <defaults><page-layout><page-width>1000</page-width></page-layout></defaults>
  <part-list>
    <score-part id="P1"><part-name>Upper</part-name></score-part>
    <score-part id="P2"><part-name>Lower</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="1" width="0"/>
    <measure number="2" width="200"/>
  </part>
  <part id="P2">
    <measure number="1" width="100"/>
    <measure number="2" width="0"/>
  </part>
</score-partwise>
"""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "score.musicxml"
            path.write_text(xml, encoding="utf-8")

            _, regions = _measure_regions(path)

        self.assertEqual(regions[(1, 1)].width, 100)
        self.assertEqual(regions[(1, 2)].x, 100)
        self.assertEqual(regions[(1, 2)].width, 200)

    def test_detects_staff_lines(self) -> None:
        image = np.full((120, 240), 255, dtype=np.uint8)
        for y in (40, 48, 56, 64, 72):
            cv2.line(image, (10, y), (230, y), 0, 1)

        self.assertEqual(
            detect_staff_lines(image),
            [(40, 48, 56, 64, 72)],
        )

    def test_detects_leading_eighth_rest_shape(self) -> None:
        image = np.full((140, 220), 255, dtype=np.uint8)
        staff = (50, 60, 70, 80, 90)
        for y in staff:
            cv2.line(image, (20, y), (200, y), 0, 1)
        cv2.rectangle(image, (55, 63), (65, 68), 0, -1)

        self.assertTrue(
            detect_leading_eighth_rest(
                image,
                staff,
                left=20,
                first_note_x=100,
            )
        )

    def test_deduplicates_overlapping_staff_candidates(self) -> None:
        staves = [
            (40, 50, 60, 70, 80),
            (42, 50, 62, 70, 80),
            (120, 130, 140, 150, 160),
        ]

        self.assertEqual(
            deduplicate_staff_lines(staves, expected_count=2),
            [(40, 50, 60, 70, 80), (120, 130, 140, 150, 160)],
        )

    def test_accepts_only_complete_measure(self) -> None:
        notes = [
            BeamNote(
                x=x,
                y=50,
                midi_pitch=60,
                beam_count=1,
                quarter_length=Fraction(1, 2),
                confidence=0.9,
            )
            for x in (20, 40, 60, 80, 100, 120)
        ]

        self.assertTrue(
            is_safe_measure_replacement(
                notes,
                Fraction(3),
                x_tolerance=5,
            )
        )
        self.assertFalse(
            is_safe_measure_replacement(
                notes[:-1],
                Fraction(3),
                x_tolerance=5,
            )
        )

    def test_detects_four_notes_on_one_beam(self) -> None:
        image = np.full((140, 260), 255, dtype=np.uint8)
        staff = (50, 60, 70, 80, 90)
        for y in staff:
            cv2.line(image, (20, y), (240, y), 0, 1)
        cv2.rectangle(image, (58, 28), (182, 33), 0, -1)
        for x, y in ((60, 78), (100, 73), (140, 68), (180, 63)):
            cv2.line(image, (x, 30), (x, y), 0, 2)
            cv2.ellipse(image, (x - 5, y), (7, 4), -15, 0, 360, 0, -1)

        groups = detect_beamed_note_groups(image, staff)

        self.assertTrue(any(len(group) == 4 for group in groups))

        rejected = detect_beamed_note_groups(
            image,
            staff,
            notehead_predictor=lambda _: ("other", 0.95),
        )
        fallback = detect_beamed_note_groups(
            image,
            staff,
            notehead_predictor=lambda _: ("other", 0.89),
        )

        self.assertEqual(rejected, [])
        self.assertTrue(any(len(group) == 4 for group in fallback))

    def test_detects_two_notes_on_one_beam(self) -> None:
        image = np.full((140, 220), 255, dtype=np.uint8)
        staff = (50, 60, 70, 80, 90)
        for y in staff:
            cv2.line(image, (20, y), (200, y), 0, 1)
        cv2.rectangle(image, (58, 28), (142, 33), 0, -1)
        for x, y in ((60, 78), (140, 68)):
            cv2.line(image, (x, 30), (x, y), 0, 2)
            cv2.ellipse(image, (x - 5, y), (7, 4), -15, 0, 360, 0, -1)

        groups = detect_beamed_note_groups(image, staff)

        self.assertTrue(any(len(group) == 2 for group in groups))

    def test_detects_broken_slanted_beam(self) -> None:
        image = np.full((140, 240), 255, dtype=np.uint8)
        staff = (50, 60, 70, 80, 90)
        for y in staff:
            cv2.line(image, (20, y), (220, y), 0, 1)
        cv2.line(image, (58, 27), (116, 33), 0, 5)
        cv2.line(image, (124, 34), (182, 40), 0, 5)
        for x, y in ((60, 78), (120, 73), (180, 68)):
            beam_y = 28 + round((x - 60) * 12 / 120)
            cv2.line(image, (x, beam_y), (x, y), 0, 2)
            cv2.ellipse(image, (x - 5, y), (7, 4), -15, 0, 360, 0, -1)

        groups = detect_beamed_note_groups(image, staff)

        self.assertTrue(any(len(group) == 3 for group in groups))

    def test_reads_mixed_eighth_and_sixteenth_beams(self) -> None:
        image = np.full((140, 260), 255, dtype=np.uint8)
        staff = (50, 60, 70, 80, 90)
        for y in staff:
            cv2.line(image, (20, y), (240, y), 0, 1)
        cv2.rectangle(image, (58, 28), (182, 33), 0, -1)
        cv2.rectangle(image, (138, 38), (182, 41), 0, -1)
        for x, y in ((60, 78), (100, 73), (140, 68), (180, 63)):
            cv2.line(image, (x, 30), (x, y), 0, 2)
            cv2.ellipse(image, (x - 5, y), (7, 4), -15, 0, 360, 0, -1)

        groups = detect_beamed_note_groups(image, staff)
        group = max(groups, key=len)

        self.assertEqual(
            [item.quarter_length for item in group],
            [
                Fraction(1, 2),
                Fraction(1, 2),
                Fraction(1, 4),
                Fraction(1, 4),
            ],
        )

    def test_removes_duplicate_and_subset_beam_groups(self) -> None:
        def beam_note(x: int, confidence: float = 0.8) -> BeamNote:
            return BeamNote(
                x=x,
                y=50,
                midi_pitch=60,
                beam_count=2,
                quarter_length=Fraction(1, 4),
                confidence=confidence,
            )

        large = [beam_note(x) for x in (10, 20, 30, 40)]
        duplicate = [beam_note(x, 0.7) for x in (10, 20, 30, 40)]
        subset = [beam_note(x, 0.9) for x in (20, 30)]
        separate = [beam_note(x) for x in (60, 70)]

        selected = select_non_overlapping_beam_groups(
            [subset, duplicate, separate, large],
            x_tolerance=2,
        )

        self.assertEqual(
            [[note.x for note in group] for group in selected],
            [[10, 20, 30, 40], [60, 70]],
        )

    def test_prefers_separate_small_groups_over_merged_large_group(self) -> None:
        def beam_note(x: int) -> BeamNote:
            return BeamNote(
                x=x,
                y=50,
                midi_pitch=60,
                beam_count=2,
                quarter_length=Fraction(1, 4),
                confidence=0.8,
            )

        merged = [beam_note(x) for x in (10, 20, 30, 40, 50, 60)]
        first = [beam_note(x) for x in (10, 20)]
        second = [beam_note(x) for x in (30, 40, 50, 60)]

        selected = select_non_overlapping_beam_groups(
            [merged, first, second],
            x_tolerance=2,
        )

        self.assertEqual(
            [[note.x for note in group] for group in selected],
            [[10, 20], [30, 40, 50, 60]],
        )

    def test_merges_overlapping_partial_beam_groups(self) -> None:
        def beam_note(x: int) -> BeamNote:
            return BeamNote(
                x=x,
                y=50,
                midi_pitch=60,
                beam_count=2,
                quarter_length=Fraction(1, 4),
                confidence=0.8,
            )

        first = [beam_note(x) for x in (10, 20, 30)]
        second = [beam_note(x) for x in (20, 30, 40)]

        selected = select_non_overlapping_beam_groups(
            [first, second],
            x_tolerance=2,
        )

        self.assertEqual(
            [[note.x for note in group] for group in selected],
            [[10, 20, 30, 40]],
        )

    def test_splits_group_at_unrealistically_large_gap(self) -> None:
        def beam_note(x: int) -> BeamNote:
            return BeamNote(
                x=x,
                y=50,
                midi_pitch=60,
                beam_count=2,
                quarter_length=Fraction(1, 4),
                confidence=0.8,
            )

        selected = select_non_overlapping_beam_groups(
            [[beam_note(x) for x in (10, 20, 70, 80)]],
            x_tolerance=2,
            maximum_gap=25,
        )

        self.assertEqual(
            [[note.x for note in group] for group in selected],
            [[10, 20], [70, 80]],
        )

    def test_recovers_pair_only_present_in_oversized_group(self) -> None:
        def beam_note(x: int) -> BeamNote:
            return BeamNote(
                x=x,
                y=50,
                midi_pitch=60,
                beam_count=1,
                quarter_length=Fraction(1, 2),
                confidence=0.8,
            )

        oversized = [beam_note(x) for x in (10, 20, 40, 50, 60, 70)]
        known = [beam_note(x) for x in (40, 50, 60, 70)]

        selected = select_non_overlapping_beam_groups(
            [oversized, known],
            x_tolerance=2,
            maximum_gap=25,
        )

        self.assertEqual(
            [[note.x for note in group] for group in selected],
            [[10, 20], [40, 50, 60, 70]],
        )

    def test_prefers_higher_confidence_slice_from_oversized_group(
        self,
    ) -> None:
        def beam_note(x: int, confidence: float) -> BeamNote:
            return BeamNote(
                x=x,
                y=50,
                midi_pitch=60,
                beam_count=1,
                quarter_length=Fraction(1, 2),
                confidence=confidence,
            )

        oversized = [
            beam_note(x, 0.9)
            for x in (12, 22, 40, 50, 60, 70)
        ]
        weak_pair = [beam_note(x, 0.5) for x in (10, 20)]
        known = [beam_note(x, 0.8) for x in (40, 50, 60, 70)]

        selected = select_non_overlapping_beam_groups(
            [oversized, weak_pair, known],
            x_tolerance=2,
            maximum_gap=15,
        )

        self.assertEqual(
            [[note.x for note in group] for group in selected],
            [[12, 22], [40, 50, 60, 70]],
        )

    def test_keeps_high_confidence_quarter_after_oversized_beam(self) -> None:
        def beam_note(x: int, confidence: float = 0.8) -> BeamNote:
            return BeamNote(
                x=x,
                y=50,
                midi_pitch=60,
                beam_count=1,
                quarter_length=Fraction(1, 2),
                confidence=confidence,
            )

        oversized = [beam_note(x) for x in (10, 20, 30, 40, 60)]
        known = [beam_note(x) for x in (10, 20, 30, 40)]

        selected = select_non_overlapping_beam_groups(
            [oversized, known],
            x_tolerance=2,
            maximum_gap=25,
        )

        self.assertEqual(
            [[note.x for note in group] for group in selected],
            [[10, 20, 30, 40], [60]],
        )
        self.assertEqual(selected[1][0].quarter_length, Fraction(1))

    def test_replaces_long_note_with_complete_sixteenth_group(self) -> None:
        measure = stream.Measure(number=1)
        measure.insert(0, meter.TimeSignature("4/4"))
        leading_rest = note.Rest(quarterLength=Fraction(1, 2))
        leading_rest.style.absoluteX = 5
        measure.insert(0, leading_rest)
        long_note = note.Note("C4", quarterLength=1)
        long_note.style.absoluteX = 20
        measure.insert(Fraction(1, 2), long_note)
        final_note = note.Note("F4", quarterLength=Fraction(5, 2))
        final_note.style.absoluteX = 80
        measure.insert(Fraction(3, 2), final_note)
        group = [
            BeamNote(
                x=x,
                y=50,
                midi_pitch=60 + index,
                beam_count=2,
                quarter_length=Fraction(1, 4),
                confidence=0.9,
            )
            for index, x in enumerate((20, 30, 40, 50))
        ]

        added, adjusted, container = _apply_beam_groups(
            measure,
            [group],
            left=0,
            scale=1,
            region_width=100,
            x_tolerance=3,
            bar_duration=Fraction(4),
            individual_durations=True,
        )

        self.assertEqual((added, adjusted), (3, 1))
        self.assertTrue(_has_complete_timeline(container, Fraction(4)))
        self.assertEqual(
            [Fraction(event.offset) for event in container.notes],
            [
                Fraction(1, 2),
                Fraction(3, 4),
                Fraction(1),
                Fraction(5, 4),
                Fraction(3, 2),
            ],
        )

    def test_duration_variants_toggle_ambiguous_beam_level(self) -> None:
        group = [
            BeamNote(
                x=10,
                y=50,
                midi_pitch=60,
                beam_count=1,
                quarter_length=Fraction(1, 2),
                confidence=0.8,
            ),
            BeamNote(
                x=20,
                y=50,
                midi_pitch=62,
                beam_count=2,
                quarter_length=Fraction(1, 4),
                confidence=0.9,
            ),
        ]

        variants = _duration_variants([group])

        self.assertTrue(
            any(
                [candidate.quarter_length for candidate in variant[0]]
                == [Fraction(1, 4), Fraction(1, 4)]
                for variant in variants
            )
        )


if __name__ == "__main__":
    unittest.main()
