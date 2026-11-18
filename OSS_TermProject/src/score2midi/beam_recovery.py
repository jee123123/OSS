"""Recover beamed notes omitted by OMR and place them on the score timeline."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Callable

import numpy as np
from music21 import note, stream
from PIL import Image, ImageSequence

from score2midi.beam_detection import (
    BeamNote,
    detect_beamed_note_groups,
    detect_beamed_notes,
    is_safe_measure_replacement,
    select_non_overlapping_beam_groups,
)
from score2midi.beam_image import (
    _gray_array,
    _ink_mask,
    _notehead_candidates,
    _remove_staff_lines,
    deduplicate_staff_lines,
    detect_leading_eighth_rest,
    detect_staff_lines,
)
from score2midi.beam_layout import (
    _measure_anchors,
    _measure_regions,
    _staff_bounds,
)

def _has_complete_timeline(
    container: stream.Stream,
    bar_duration: Fraction,
) -> bool:
    intervals: dict[Fraction, Fraction] = {}
    for event in container.notesAndRests:
        start = Fraction(event.offset)
        end = start + Fraction(event.duration.quarterLength)
        intervals[start] = max(intervals.get(start, start), end)

    cursor = Fraction(0)
    for start, end in sorted(intervals.items()):
        if start != cursor:
            return False
        cursor = end

    return cursor == bar_duration


def _apply_beam_groups(
    measure: stream.Measure,
    groups: list[list[BeamNote]],
    left: float,
    scale: float,
    region_width: float,
    x_tolerance: float,
    bar_duration: Fraction,
    individual_durations: bool,
    add_leading_eighth_rest: bool = False,
) -> tuple[int, int, stream.Stream]:
    voices = list(measure.getElementsByClass(stream.Voice))
    container = voices[0] if voices else measure
    records: list[list[object]] = []
    for event in container.notesAndRests:
        absolute_x = event.style.absoluteX
        local_x = (
            float(absolute_x)
            if absolute_x is not None
            else region_width * float(Fraction(event.offset) / bar_duration)
        )
        records.append([local_x, event])

    if add_leading_eighth_rest and records:
        leading_rest = note.Rest(quarterLength=Fraction(1, 2))
        leading_x = min(float(record[0]) for record in records) - x_tolerance
        leading_rest.style.absoluteX = leading_x
        records.append([leading_x, leading_rest])

    added = 0
    adjusted = 0

    for group in sorted(groups, key=lambda item: item[0].x):
        ordered_group = sorted(group, key=lambda item: item.x)
        uniform_duration = min(
            candidate.quarter_length
            for candidate in ordered_group
        )
        positions = [
            (candidate.x - left) / scale
            for candidate in ordered_group
        ]
        first_x = min(positions) - x_tolerance
        last_x = max(positions) + x_tolerance
        replaced = [
            record
            for record in records
            if first_x <= float(record[0]) <= last_x
            and not record[1].isRest
        ]
        used_events: set[int] = set()
        replacements: list[list[object]] = []

        for candidate, local_x in zip(ordered_group, positions):
            duration = (
                candidate.quarter_length
                if individual_durations
                else uniform_duration
            )
            matching_record = min(
                (
                    record
                    for record in replaced
                    if id(record[1]) not in used_events
                    and abs(float(record[0]) - local_x) <= x_tolerance
                ),
                key=lambda record: abs(float(record[0]) - local_x),
                default=None,
            )

            if matching_record is not None:
                event = matching_record[1]
                used_events.add(id(event))
                adjusted += 1
            else:
                event = note.Note(candidate.midi_pitch)
                added += 1

            event.duration.quarterLength = duration
            event.style.absoluteX = local_x
            replacements.append([local_x, event])

        replaced_ids = {id(record[1]) for record in replaced}
        records = [
            record
            for record in records
            if id(record[1]) not in replaced_ids
        ]
        records.extend(replacements)

    for event in list(container.notesAndRests):
        container.remove(event)

    offset = Fraction(0)
    for _, event in sorted(records, key=lambda record: float(record[0])):
        container.insert(offset, event)
        offset += Fraction(event.duration.quarterLength)

    return added, adjusted, container


def _duration_variants(
    groups: list[list[BeamNote]],
) -> list[list[list[BeamNote]]]:
    candidates = [
        (group_index, note_index, candidate)
        for group_index, group in enumerate(groups)
        for note_index, candidate in enumerate(group)
        if candidate.quarter_length in {Fraction(1, 4), Fraction(1, 2)}
    ]
    variants = [groups]
    toggles = []
    for count in (1, 2):
        toggles.extend(combinations(candidates, count))
    toggles.sort(key=lambda items: sum(item[2].confidence for item in items))

    for items in toggles[:32]:
        variant = [list(group) for group in groups]
        for group_index, note_index, candidate in items:
            duration = (
                Fraction(1, 2)
                if candidate.quarter_length == Fraction(1, 4)
                else Fraction(1, 4)
            )
            variant[group_index][note_index] = replace(
                candidate,
                beam_count=1 if duration == Fraction(1, 2) else 2,
                quarter_length=duration,
            )
        variants.append(variant)

    return variants


def recover_unplaced_beamed_notes(
    score: stream.Score,
    page_document: Path,
    musicxml: Path,
    unplaced_chords: set[tuple[int, int, int]],
    minimum_confidence: float = 0.5,
    notehead_predictor: Callable[
        [Image.Image], tuple[str, float]
    ] | None = None,
    model_minimum_confidence: float = 0.9,
) -> tuple[list[str], set[tuple[int, int, int]]]:
    if not unplaced_chords:
        return [], set()

    _, regions = _measure_regions(musicxml)
    warnings: list[str] = []
    recovered: set[tuple[int, int, int]] = set()

    with Image.open(page_document) as document:
        pages = [frame.copy() for frame in ImageSequence.Iterator(document)]

    try:
        staves_by_page = {}
        for page_number, page in enumerate(pages, 1):
            system_count = 1 + max(
                (
                    item.system
                    for item in regions.values()
                    if item.page == page_number
                ),
                default=-1,
            )
            staves_by_page[page_number] = deduplicate_staff_lines(
                detect_staff_lines(page),
                system_count * len(score.parts),
            )

        for key in sorted(unplaced_chords):
            page_number, local_number, part_number = key
            region = regions.get((page_number, local_number))
            if region is None or not 1 <= part_number <= len(score.parts):
                continue

            page = pages[page_number - 1]
            staves = staves_by_page[page_number]
            system_count = 1 + max(
                (
                    item.system
                    for item in regions.values()
                    if item.page == page_number
                ),
                default=-1,
            )
            if len(staves) != system_count * len(score.parts):
                continue

            staff_index = region.system * len(score.parts) + part_number - 1
            staff = staves[staff_index]
            part = score.parts[part_number - 1]
            measure = part.measure(region.global_number)
            if measure is None:
                continue
            if any(
                event.duration.tuplets
                for event in measure.recurse().notesAndRests
            ):
                continue

            voices = list(measure.getElementsByClass(stream.Voice))
            if len(voices) > 1:
                continue

            system_regions = [
                item
                for item in regions.values()
                if item.page == page_number
                and item.system == region.system
            ]
            system_start = min(item.x for item in system_regions)
            system_width = sum(item.width for item in system_regions)
            staff_left, staff_right = _staff_bounds(page, staff)
            scale = (staff_right - staff_left) / system_width
            left = staff_left + (region.x - system_start) * scale
            right = left + region.width * scale
            clef_name = "treble" if part_number == 1 else "bass"
            key_signature = measure.getContextByClass("KeySignature")
            sharps = key_signature.sharps if key_signature else 0
            spacing = float(median(np.diff(staff)))
            x_tolerance = spacing / scale * 0.65
            anchors = _measure_anchors(measure)
            prediction_cache: dict[tuple[int, int], tuple[str, float]] = {}

            def is_notehead_candidate(candidate: BeamNote) -> bool:
                if notehead_predictor is None:
                    return True
                prediction = prediction_cache.get((candidate.x, candidate.y))
                if prediction is None:
                    half_width = max(4, round(spacing * 0.9))
                    half_height = max(4, round(spacing * 0.7))
                    prediction = notehead_predictor(
                        page.crop(
                            (
                                candidate.x - half_width,
                                candidate.y - half_height,
                                candidate.x + half_width + 1,
                                candidate.y + half_height + 1,
                            )
                        )
                    )
                    prediction_cache[(candidate.x, candidate.y)] = prediction
                if (
                    prediction[1] < model_minimum_confidence
                    or prediction[0] == "notehead"
                ):
                    return True
                local_x = (candidate.x - left) / scale
                return any(
                    abs(anchor_x - local_x) <= x_tolerance
                    for anchor_x, _ in anchors
                )

            groups = []
            for group in detect_beamed_note_groups(
                    page,
                    staff,
                    clef_name,
                    sharps,
                ):
                candidates = [
                    candidate
                    for candidate in group
                    if left <= candidate.x <= right
                    and is_notehead_candidate(candidate)
                ]
                if (
                    len(candidates) >= 2
                    and sum(item.confidence for item in candidates)
                    / len(candidates)
                    >= minimum_confidence
                ):
                    groups.append(candidates)
            groups = select_non_overlapping_beam_groups(
                groups,
                spacing * 0.8,
                spacing * 3.8,
            )
            if anchors:
                first_note_x = anchors[0][0]
                groups = [
                    group
                    for group in groups
                    if max(
                        (candidate.x - left) / scale
                        for candidate in group
                    )
                    >= first_note_x - x_tolerance
                ]
            if not groups:
                continue

            bar_duration = Fraction(measure.barDuration.quarterLength)
            ordered_events = sorted(
                measure.recurse().notesAndRests,
                key=lambda event: Fraction(event.getOffsetInHierarchy(measure)),
            )
            add_leading_eighth_rest = (
                bool(ordered_events)
                and not ordered_events[0].isRest
                and detect_leading_eighth_rest(
                    page,
                    staff,
                    left,
                    min(candidate.x for group in groups for candidate in group),
                )
            )
            original_measure = deepcopy(measure)
            measure_offset = measure.offset
            accepted = None

            attempts = [
                (variant, True)
                for variant in _duration_variants(groups)
            ]
            attempts.append((groups, False))
            for candidate_groups, individual_durations in attempts:
                trial = deepcopy(original_measure)
                added, adjusted, container = _apply_beam_groups(
                    trial,
                    candidate_groups,
                    left,
                    scale,
                    region.width,
                    x_tolerance,
                    bar_duration,
                    individual_durations,
                    add_leading_eighth_rest,
                )
                if (
                    (added or adjusted)
                    and _has_complete_timeline(container, bar_duration)
                ):
                    accepted = trial, added, adjusted
                    break

            if accepted is None:
                continue

            replacement, added, adjusted = accepted
            part.remove(measure)
            part.insert(measure_offset, replacement)
            recovered.add(key)
            warnings.append(
                f"{page_number}쪽 {local_number}번째 마디 "
                f"파트 {part_number}: 빔 음표 {added}개를 복구하고 "
                f"{adjusted}개의 박자를 교정했습니다."
            )
    finally:
        for page in pages:
            page.close()

    return warnings, recovered


