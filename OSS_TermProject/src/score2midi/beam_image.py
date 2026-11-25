"""Low-level image processing for staff, notehead, stem, and beam shapes."""

from __future__ import annotations

from statistics import median

import cv2
import numpy as np
from PIL import Image

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
    *,
    minimum_line_ratio: float = 0.05,
) -> list[tuple[int, int, int, int, int]]:
    ink = _ink_mask(_gray_array(image))
    height, width = ink.shape
    horizontal = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(15, int(width * minimum_line_ratio)), 1),
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



