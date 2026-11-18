"""Build and select beamed-note candidates from detected image shapes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from statistics import median
from typing import Callable

import numpy as np
from PIL import Image

from score2midi.beam_image import (
    _beam_components,
    _gray_array,
    _ink_mask,
    _midi_pitch,
    _notehead_candidates,
    _remove_staff_lines,
    _stem_components,
    _stem_for_note,
    _beam_count,
    deduplicate_staff_lines,
    detect_leading_eighth_rest,
    detect_staff_lines,
)

@dataclass(frozen=True)
class BeamNote:
    x: int
    y: int
    midi_pitch: int
    beam_count: int
    quarter_length: Fraction
    confidence: float


def detect_beamed_notes(
    image: Image.Image | np.ndarray,
    staff_lines: tuple[int, int, int, int, int],
    clef: str = "treble",
    key_signature: int = 0,
) -> list[BeamNote]:
    return [
        note
        for group in detect_beamed_note_groups(
            image,
            staff_lines,
            clef,
            key_signature,
        )
        for note in group
    ]


def detect_beamed_note_groups(
    image: Image.Image | np.ndarray,
    staff_lines: tuple[int, int, int, int, int],
    clef: str = "treble",
    key_signature: int = 0,
    notehead_predictor: Callable[
        [Image.Image], tuple[str, float]
    ] | None = None,
    model_minimum_confidence: float = 0.9,
) -> list[list[BeamNote]]:
    if clef not in {"treble", "bass"}:
        raise ValueError("clef must be 'treble' or 'bass'")
    if not 0.5 <= model_minimum_confidence <= 1:
        raise ValueError("model_minimum_confidence must be between 0.5 and 1")

    gray = _gray_array(image)
    ink = _ink_mask(gray)
    spacing = float(median(np.diff(staff_lines)))
    staffless = _remove_staff_lines(ink, staff_lines, spacing)
    stems = _stem_components(ink, spacing)
    beams, horizontal_beams = _beam_components(staffless, spacing)
    heads = _notehead_candidates(
        staffless,
        staff_lines,
        spacing,
    )
    if notehead_predictor is not None:
        page = Image.fromarray(gray)
        half_width = max(4, round(spacing * 0.9))
        half_height = max(4, round(spacing * 0.7))
        heads = [
            candidate
            for candidate in heads
            if (
                (prediction := notehead_predictor(
                    page.crop(
                        (
                            candidate[0] - half_width,
                            candidate[1] - half_height,
                            candidate[0] + half_width + 1,
                            candidate[1] + half_height + 1,
                        )
                    )
                ))[1] < model_minimum_confidence
                or prediction[0] == "notehead"
            )
        ]
    staff_middle = sum(staff_lines) / len(staff_lines)
    groups: list[list[BeamNote]] = []
    seen_stems: set[tuple[int, ...]] = set()

    for beam in beams:
        beam_x, beam_y, beam_width, beam_height, _ = beam
        beam_middle = beam_y + beam_height / 2
        if not (
            staff_lines[0] - spacing * 3
            <= beam_middle
            <= staff_lines[-1] + spacing * 3
        ):
            continue

        connected_stems = []
        for stem in stems:
            stem_x, stem_y, stem_width, stem_height, _ = stem
            stem_middle = stem_x + stem_width / 2
            vertical_gap = max(
                stem_y - (beam_y + beam_height),
                beam_y - (stem_y + stem_height),
                0,
            )
            if (
                beam_x - spacing * 0.4
                <= stem_middle
                <= beam_x + beam_width + spacing * 0.4
                and vertical_gap <= spacing * 3
                and stem_height <= spacing * 7
            ):
                connected_stems.append(stem)

        connected_stems.sort(key=lambda item: item[0])
        stem_key = tuple(round(stem[0] / spacing) for stem in connected_stems)
        if len(connected_stems) < 2 or stem_key in seen_stems:
            continue
        seen_stems.add(stem_key)

        group: list[BeamNote] = []
        for stem in connected_stems:
            stem_x, stem_y, stem_width, stem_height, _ = stem
            stem_middle = stem_x + stem_width / 2
            endpoint = (
                stem_y + stem_height
                if beam_middle < staff_middle
                else stem_y
            )
            head = min(
                (
                    candidate
                    for candidate in heads
                    if abs(candidate[0] - stem_middle) <= spacing * 0.9
                    and abs(candidate[1] - endpoint) <= spacing * 1.5
                ),
                key=lambda candidate: (
                    abs(candidate[1] - endpoint)
                    + abs(candidate[0] - stem_middle)
                ),
                default=None,
            )
            if head is None:
                continue

            x, y, confidence = head
            duration_beams = (
                horizontal_beams
                if beam in horizontal_beams
                else beams
            )
            beam_count = max(
                1,
                _beam_count(
                    stem,
                    duration_beams,
                    spacing,
                    beam_middle,
                ),
            )
            group.append(
                BeamNote(
                    x=x,
                    y=y,
                    midi_pitch=_midi_pitch(
                        y,
                        staff_lines,
                        clef,
                        key_signature,
                    ),
                    beam_count=beam_count,
                    quarter_length=(
                        Fraction(1, 2)
                        if beam_count == 1
                        else Fraction(1, 4)
                    ),
                    confidence=confidence,
                )
            )

        if len(group) >= 2:
            groups.append(sorted(group, key=lambda item: item.x))

    return groups


def is_safe_measure_replacement(
    notes: list[BeamNote],
    expected_duration: Fraction,
    x_tolerance: int,
    minimum_confidence: float = 0.45,
) -> bool:
    if not notes or any(
        note.confidence < minimum_confidence
        for note in notes
    ):
        return False

    events: list[list[BeamNote]] = []
    for note in sorted(notes, key=lambda item: item.x):
        if not events or note.x - events[-1][0].x > x_tolerance:
            events.append([note])
        else:
            events[-1].append(note)

    return sum(
        min(note.quarter_length for note in event)
        for event in events
    ) == expected_duration


def select_non_overlapping_beam_groups(
    groups: list[list[BeamNote]],
    x_tolerance: float,
    maximum_gap: float | None = None,
) -> list[list[BeamNote]]:
    normalized = []
    for group in groups:
        unique = []
        for candidate in sorted(
            group,
            key=lambda item: item.confidence,
            reverse=True,
        ):
            if any(
                abs(candidate.x - existing.x) <= x_tolerance
                for existing in unique
            ):
                continue
            unique.append(candidate)
        ordered = sorted(unique, key=lambda item: item.x)
        segments: list[list[BeamNote]] = []
        for candidate in ordered:
            if (
                segments
                and maximum_gap is not None
                and candidate.x - segments[-1][-1].x > maximum_gap
            ):
                segments.append([])
            if not segments:
                segments.append([])
            segments[-1].append(candidate)
        normalized.extend(segment for segment in segments if len(segment) >= 2)

    preferred = [group for group in normalized if len(group) <= 4]
    oversized = [group for group in normalized if len(group) > 4]
    pool = preferred or normalized
    selected = []
    for group in sorted(
        pool,
        key=lambda item: (
            len(item),
            sum(note.confidence for note in item) / len(item),
        ),
        reverse=True,
    ):
        overlap_index = next(
            (
                index
                for index, accepted in enumerate(selected)
                if group[0].x <= accepted[-1].x + x_tolerance
                and accepted[0].x <= group[-1].x + x_tolerance
            ),
            None,
        )
        if overlap_index is not None:
            combined = []
            for candidate in sorted(
                selected[overlap_index] + group,
                key=lambda item: item.confidence,
                reverse=True,
            ):
                if any(
                    abs(candidate.x - existing.x) <= x_tolerance
                    for existing in combined
                ):
                    continue
                combined.append(candidate)
            if len(combined) <= 4:
                selected[overlap_index] = sorted(
                    combined,
                    key=lambda item: item.x,
                )
            continue
        selected.append(group)

    selected.sort(key=lambda item: item[0].x)
    for group in oversized:
        for index, accepted in enumerate(selected):
            lower = (
                float("-inf")
                if index == 0
                else (selected[index - 1][-1].x + accepted[0].x) / 2
            )
            upper = (
                float("inf")
                if index + 1 == len(selected)
                else (accepted[-1].x + selected[index + 1][0].x) / 2
            )
            challenger = [
                candidate
                for candidate in group
                if lower < candidate.x <= upper
            ]
            if not 2 <= len(challenger) <= 4:
                continue
            challenger_confidence = sum(
                candidate.confidence for candidate in challenger
            ) / len(challenger)
            accepted_confidence = sum(
                candidate.confidence for candidate in accepted
            ) / len(accepted)
            if challenger_confidence > accepted_confidence:
                selected[index] = challenger

    if maximum_gap is not None:
        for group in oversized:
            remaining = [
                candidate
                for candidate in group
                if not any(
                    abs(candidate.x - existing.x) <= x_tolerance
                    for accepted in selected
                    for existing in accepted
                )
            ]
            segments: list[list[BeamNote]] = []
            for candidate in remaining:
                if (
                    segments
                    and candidate.x - segments[-1][-1].x > maximum_gap
                ):
                    segments.append([])
                if not segments:
                    segments.append([])
                segments[-1].append(candidate)
            for segment in segments:
                if len(segment) == 1:
                    candidate = segment[0]
                    if candidate.confidence >= 0.75:
                        selected.append(
                            [
                                replace(
                                    candidate,
                                    beam_count=0,
                                    quarter_length=Fraction(1),
                                )
                            ]
                        )
                    continue
                if len(segment) < 2:
                    continue
                if any(
                    segment[0].x <= accepted[-1].x + x_tolerance
                    and accepted[0].x <= segment[-1].x + x_tolerance
                    for accepted in selected
                ):
                    continue
                selected.append(segment)

    return sorted(selected, key=lambda item: item[0].x)





