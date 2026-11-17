"""Detect beamed black notes that OMR may omit."""

from copy import deepcopy
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import combinations
from statistics import median
from pathlib import Path
from typing import Callable
from xml.etree import ElementTree
from zipfile import ZipFile, is_zipfile

import cv2
import numpy as np
from music21 import note, stream
from PIL import Image, ImageSequence

_PITCH_CLASSES = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}
_LETTERS = ("C", "D", "E", "F", "G", "A", "B")
_SHARP_ORDER = "FCGDAEB"
_FLAT_ORDER = "BEADGCF"


@dataclass(frozen=True)
class BeamNote:
    x: int
    y: int
    midi_pitch: int
    beam_count: int
    quarter_length: Fraction
    confidence: float


@dataclass(frozen=True)
class _MeasureRegion:
    page: int
    local_number: int
    global_number: int
    system: int
    x: float
    width: float


def _gray_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("L"))

    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    _, ink = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    return ink


def _bands(values: np.ndarray, threshold: int) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = None

    for index, value in enumerate(np.append(values, 0)):
        if value >= threshold and start is None:
            start = index
        elif value < threshold and start is not None:
            bands.append((start, index - 1))
            start = None

    return bands


def detect_staff_lines(
    image: Image.Image | np.ndarray,
) -> list[tuple[int, int, int, int, int]]:
    ink = _ink_mask(_gray_array(image))
    height, width = ink.shape
    horizontal = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(15, width // 20), 1),
        ),
    )
    projection = np.count_nonzero(horizontal, axis=1)
    rows = [
        (start + end) // 2
        for start, end in _bands(
            projection,
            max(10, round(width * 0.1)),
        )
    ]

    if len(rows) < 5:
        connected = cv2.morphologyEx(
            ink,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (max(5, width // 100), 1),
            ),
        )
        projection = np.count_nonzero(connected, axis=1)
        rows = [
            (start + end) // 2
            for start, end in _bands(
                projection,
                max(10, round(width * 0.35)),
            )
        ]

    groups: list[tuple[int, int, int, int, int]] = []
    maximum_spacing = min(80, max(12, height * 0.05))

    for start_index, first in enumerate(rows):
        for last in rows[start_index + 4:]:
            spacing = (last - first) / 4
            if not 3 <= spacing <= maximum_spacing:
                continue

            tolerance = max(2, spacing * 0.2)
            group = []

            for step in range(5):
                expected = first + spacing * step
                nearest = min(rows, key=lambda row: abs(row - expected))
                if abs(nearest - expected) > tolerance:
                    group = []
                    break
                group.append(nearest)

            if len(set(group)) != 5:
                continue

            candidate = tuple(group)
            if not any(
                abs(candidate[0] - existing[0]) <= tolerance
                for existing in groups
            ):
                groups.append(candidate)
                break

    return sorted(groups)


def deduplicate_staff_lines(
    staves: list[tuple[int, int, int, int, int]],
    expected_count: int,
) -> list[tuple[int, int, int, int, int]]:
    if len(staves) <= expected_count:
        return staves

    clusters: list[list[tuple[int, int, int, int, int]]] = []
    for staff in sorted(staves):
        if clusters and staff[0] <= max(item[-1] for item in clusters[-1]):
            clusters[-1].append(staff)
        else:
            clusters.append([staff])

    if len(clusters) != expected_count:
        return staves

    def irregularity(staff: tuple[int, int, int, int, int]) -> float:
        gaps = np.diff(staff)
        average = float(np.mean(gaps))
        return float(np.std(gaps)) / average if average else float("inf")

    return [min(cluster, key=irregularity) for cluster in clusters]


def detect_leading_eighth_rest(
    image: Image.Image | np.ndarray,
    staff_lines: tuple[int, int, int, int, int],
    left: float,
    first_note_x: float,
) -> bool:
    spacing = float(median(np.diff(staff_lines)))
    if first_note_x - left < spacing * 2:
        return False

    mask = _ink_mask(_gray_array(image))
    half_width = max(1, round(spacing * 0.22))
    for y in staff_lines:
        mask[max(0, y - half_width):y + half_width + 1] = 0

    minimum_x = left + spacing * 0.8
    maximum_x = first_note_x - spacing * 0.4
    minimum_y = staff_lines[1] - spacing * 0.4
    maximum_y = staff_lines[3] + spacing * 0.4

    return any(
        minimum_x <= x
        and x + width <= maximum_x
        and minimum_y <= y + height / 2 <= maximum_y
        and spacing * 0.7 <= width <= spacing * 1.4
        and spacing * 0.4 <= height <= spacing * 0.9
        and spacing * spacing * 0.2 <= area <= spacing * spacing * 0.75
        for x, y, width, height, area in _components(mask)
    )


def _remove_staff_lines(
    ink: np.ndarray,
    staff_lines: tuple[int, int, int, int, int],
    spacing: float,
) -> np.ndarray:
    staffless = ink.copy()
    half_width = max(1, round(spacing * 0.12))

    for y in staff_lines:
        top = max(0, y - half_width)
        bottom = min(staffless.shape[0], y + half_width + 1)
        staffless[top:bottom] = 0

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )
    return cv2.morphologyEx(
        staffless,
        cv2.MORPH_CLOSE,
        close_kernel,
    )


def _notehead_candidates(
    staffless: np.ndarray,
    staff_lines: tuple[int, int, int, int, int],
    spacing: float,
) -> list[tuple[int, int, float]]:
    head_width = max(5, round(spacing * 1.25))
    head_height = max(4, round(spacing * 0.85))
    template_width = round(head_width * 1.5)
    template_height = round(head_height * 1.7)
    template = np.zeros(
        (template_height, template_width),
        dtype=np.uint8,
    )
    cv2.ellipse(
        template,
        (template_width // 2, template_height // 2),
        (head_width // 2, head_height // 2),
        -12,
        0,
        360,
        255,
        -1,
    )
    response = cv2.matchTemplate(
        staffless,
        template,
        cv2.TM_CCOEFF_NORMED,
    )
    y_min = max(0, round(staff_lines[0] - spacing * 3))
    y_max = min(
        staffless.shape[0],
        round(staff_lines[-1] + spacing * 3),
    )
    candidates: list[tuple[int, int, float]] = []
    locations = np.argwhere(response >= 0.42)

    for top, left in locations:
        x = int(left + template_width // 2)
        y = int(top + template_height // 2)

        if not y_min <= y <= y_max:
            continue

        row = staffless[y] > 0
        run_start = x
        run_end = x
        while run_start > 0 and row[run_start - 1]:
            run_start -= 1
        while run_end + 1 < len(row) and row[run_end + 1]:
            run_end += 1
        if run_end - run_start + 1 > head_width * 1.8:
            continue

        candidates.append((x, y, float(response[top, left])))

    selected: list[tuple[int, int, float]] = []
    for candidate in sorted(
        candidates,
        key=lambda item: item[2],
        reverse=True,
    ):
        x, y, _ = candidate
        if any(
            abs(x - old_x) < spacing * 0.7
            and abs(y - old_y) <= spacing * 0.4
            for old_x, old_y, _ in selected
        ):
            continue
        selected.append(candidate)

    return sorted(selected)


def _components(
    mask: np.ndarray,
) -> list[tuple[int, int, int, int, int]]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    return [tuple(map(int, stats[index])) for index in range(1, count)]


def _stem_components(
    ink: np.ndarray,
    spacing: float,
) -> list[tuple[int, int, int, int, int]]:
    connected = cv2.morphologyEx(
        ink,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)),
    )
    vertical = cv2.morphologyEx(
        connected,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, max(5, round(spacing * 1.5))),
        ),
    )
    components = [
        component
        for component in _components(vertical)
        if component[3] >= spacing * 1.4
        and component[2] <= spacing * 0.5
    ]
    merged: list[list[int]] = []

    for x, y, width, height, area in sorted(components):
        match = next(
            (
                stem
                for stem in merged
                if abs((stem[0] + stem[2] / 2) - (x + width / 2))
                <= spacing * 0.2
                and max(
                    stem[1] - (y + height),
                    y - (stem[1] + stem[3]),
                    0,
                )
                <= spacing
            ),
            None,
        )
        if match is None:
            merged.append([x, y, width, height, area])
            continue

        right = max(match[0] + match[2], x + width)
        bottom = max(match[1] + match[3], y + height)
        match[0] = min(match[0], x)
        match[1] = min(match[1], y)
        match[2] = right - match[0]
        match[3] = bottom - match[1]
        match[4] += area

    return [tuple(stem) for stem in merged]


def _beam_components(
    staffless: np.ndarray,
    spacing: float,
) -> tuple[
    list[tuple[int, int, int, int, int]],
    list[tuple[int, int, int, int, int]],
]:
    horizontal = cv2.morphologyEx(
        staffless,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(6, round(spacing * 1.5)), 1),
        ),
    )
    components = [
        component
        for component in _components(horizontal)
        if component[2] >= spacing * 1.4
        and component[3] >= max(2, spacing * 0.12)
        and component[3] <= spacing * 1.4
    ]
    lines = cv2.HoughLinesP(
        staffless,
        1,
        np.pi / 180,
        threshold=max(8, round(spacing)),
        minLineLength=max(6, round(spacing * 1.4)),
        maxLineGap=max(3, round(spacing * 1.2)),
    )

    slanted = []
    if lines is not None:
        thickness = max(2, round(spacing * 0.25))
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            width = abs(int(x2) - int(x1)) + 1
            if width < spacing * 1.4:
                continue
            slope = abs(int(y2) - int(y1)) / width
            if not 0.03 <= slope <= 0.32:
                continue
            x = min(int(x1), int(x2))
            y = min(int(y1), int(y2)) - thickness // 2
            height = abs(int(y2) - int(y1)) + thickness
            slanted.append((x, y, width, height, width * thickness))

    components.sort(key=lambda item: item[2], reverse=True)
    selected = components.copy()
    for candidate in sorted(slanted, key=lambda item: item[2], reverse=True):
        x, y, width, height, _ = candidate
        center_y = y + height / 2
        if any(
            min(x + width, old_x + old_width) - max(x, old_x)
            >= min(width, old_width) * 0.6
            and abs(center_y - (old_y + old_height / 2))
            <= spacing * 0.35
            for old_x, old_y, old_width, old_height, _ in selected
        ):
            continue
        selected.append(candidate)

    return selected, components


def _stem_for_note(
    x: int,
    y: int,
    stems: list[tuple[int, int, int, int, int]],
    spacing: float,
) -> tuple[int, int, int, int, int] | None:
    candidates = []

    for stem in stems:
        stem_x, stem_y, width, height, _ = stem
        center_x = stem_x + width / 2
        if (
            abs(center_x - x) <= spacing * 0.9
            and stem_y - spacing * 0.8 <= y <= stem_y + height + spacing * 0.8
        ):
            candidates.append((abs(center_x - x), stem))

    if not candidates:
        return None

    return min(candidates, key=lambda item: item[0])[1]


def _beam_count(
    stem: tuple[int, int, int, int, int],
    beams: list[tuple[int, int, int, int, int]],
    spacing: float,
    reference_y: float | None = None,
) -> int:
    stem_x, stem_y, stem_width, stem_height, _ = stem
    center_x = stem_x + stem_width / 2
    connected = []

    for beam_x, beam_y, width, height, _ in beams:
        beam_middle = beam_y + height / 2
        if (
            beam_x - spacing * 0.5
            <= center_x
            <= beam_x + width + spacing * 0.5
            and beam_y <= stem_y + stem_height + spacing * 0.8
            and beam_y + height >= stem_y - spacing * 0.8
            and (
                reference_y is None
                or abs(beam_middle - reference_y) <= spacing * 1.4
            )
        ):
            connected.append(beam_middle)

    levels: list[float] = []
    for y in sorted(connected):
        if not levels or y - levels[-1] > spacing * 0.45:
            levels.append(y)

    return min(2, len(levels))


def _midi_pitch(
    y: int,
    staff_lines: tuple[int, int, int, int, int],
    clef: str,
    key_signature: int,
) -> int:
    spacing = median(np.diff(staff_lines))
    bottom_y = staff_lines[-1]
    step = round((bottom_y - y) / (spacing / 2))
    reference_letter, reference_octave = (
        ("E", 4) if clef == "treble" else ("G", 2)
    )
    reference_index = _LETTERS.index(reference_letter)
    absolute_index = reference_octave * 7 + reference_index + step
    octave, letter_index = divmod(absolute_index, 7)
    letter = _LETTERS[letter_index]
    accidental = 0

    if key_signature > 0 and letter in _SHARP_ORDER[:key_signature]:
        accidental = 1
    elif key_signature < 0 and letter in _FLAT_ORDER[:-key_signature]:
        accidental = -1

    return 12 * (octave + 1) + _PITCH_CLASSES[letter] + accidental


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


def _musicxml_root(path: Path) -> ElementTree.Element:
    if not is_zipfile(path):
        return ElementTree.parse(path).getroot()

    with ZipFile(path) as archive:
        member = next(
            name
            for name in archive.namelist()
            if name.endswith((".xml", ".musicxml"))
            and not name.startswith("META-INF/")
        )
        return ElementTree.fromstring(archive.read(member))


def _measure_regions(
    musicxml: Path,
) -> tuple[float, dict[tuple[int, int], _MeasureRegion]]:
    root = _musicxml_root(musicxml)
    page_width = float(
        root.findtext("./defaults/page-layout/page-width") or 1
    )
    first_part = root.find("part")
    if first_part is None:
        return page_width, {}

    page = 1
    system = 0
    local_number = 0
    x = 0.0
    regions: dict[tuple[int, int], _MeasureRegion] = {}

    for index, measure in enumerate(first_part.findall("measure"), 1):
        print_element = measure.find("print")
        new_page = (
            print_element is not None
            and print_element.get("new-page") == "yes"
        )
        new_system = (
            print_element is not None
            and print_element.get("new-system") == "yes"
        )

        if new_page:
            page += 1
            system = 0
            local_number = 0
        elif new_system:
            system += 1

        if index == 1 or new_page or new_system:
            x = float(
                print_element.findtext(
                    "./system-layout/system-margins/left-margin"
                )
                if print_element is not None
                else 0
            )

        local_number += 1
        width = float(measure.get("width") or 0)
        number = measure.get("number") or str(index)
        global_number = int(number) if number.isdigit() else index
        regions[(page, local_number)] = _MeasureRegion(
            page,
            local_number,
            global_number,
            system,
            x,
            width,
        )
        x += width

    return page_width, regions


def _measure_anchors(
    measure: stream.Measure,
) -> list[tuple[float, Fraction]]:
    anchors = []

    for element in measure.recurse().notes:
        absolute_x = element.style.absoluteX
        if absolute_x is None:
            continue
        anchors.append(
            (
                float(absolute_x),
                Fraction(element.getOffsetInHierarchy(measure)),
            )
        )

    return sorted(set(anchors))


def _staff_bounds(
    image: Image.Image,
    staff: tuple[int, int, int, int, int],
) -> tuple[int, int]:
    ink = _ink_mask(_gray_array(image))
    support = np.zeros(ink.shape[1], dtype=np.uint8)
    for y in staff:
        support += np.any(
            ink[max(0, y - 2):min(ink.shape[0], y + 3)] > 0,
            axis=0,
        )
    mask = cv2.morphologyEx(
        ((support >= 3) * 255).astype(np.uint8).reshape(1, -1),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1)),
    ).ravel()
    return max(_bands(mask, 255), key=lambda band: band[1] - band[0])


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
