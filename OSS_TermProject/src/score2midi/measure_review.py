"""Find reported measures and crop them from the original score pages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from PIL import Image

from score2midi.beam_image import (
    _bands,
    _gray_array,
    _ink_mask,
    deduplicate_staff_lines,
    detect_staff_lines,
)
from score2midi.beam_layout import (
    _measure_regions,
    _musicxml_root,
    _staff_bounds,
)
from score2midi.navigation_vision import _staff_runs, _system_rows
from score2midi.page_editor import PageSpec, load_page_image


@dataclass(frozen=True)
class ReviewIssue:
    measure: int
    messages: tuple[str, ...]


def reported_measures(details: str, musicxml: Path) -> list[ReviewIssue]:
    try:
        _, regions = _measure_regions(musicxml)
    except (OSError, ValueError, StopIteration):
        regions = {}

    grouped: dict[int, list[str]] = {}
    for line in map(str.strip, details.splitlines()):
        if not line or "마디" not in line or "재생 순서로 전개" in line:
            continue

        local = re.search(r"(\d+)쪽\s*내\s*(\d+)번째\s*마디", line)
        if local:
            region = regions.get((int(local.group(1)), int(local.group(2))))
            measure = region.global_number if region else None
        else:
            match = re.search(r"(\d+)마디", line)
            measure = int(match.group(1)) if match else None

        if measure is not None:
            grouped.setdefault(measure, []).append(line)

    return [
        ReviewIssue(measure, tuple(messages))
        for measure, messages in sorted(grouped.items())
    ]


def load_measure_image(
    pages: list[PageSpec],
    musicxml: Path,
    measure_number: int,
    dpi: int = 160,
) -> Image.Image:
    _, regions = _measure_regions(musicxml)
    region = next(
        item for item in regions.values() if item.global_number == measure_number
    )
    if not 1 <= region.page <= len(pages):
        raise ValueError("원본 악보 페이지를 찾을 수 없습니다.")

    rows = _system_rows(musicxml)
    target_row = rows[(region.page, region.system)]
    page_regions = [
        item for item in regions.values() if item.page == region.page
    ]
    page_rows = sorted({rows[(item.page, item.system)] for item in page_regions})
    row_index = page_rows.index(target_row)
    row_regions = [
        item for item in page_regions if rows[(item.page, item.system)] == target_row
    ]
    root = _musicxml_root(musicxml)
    staff_count = sum(
        max(
            (
                int(node.text)
                for node in part.findall("./measure/attributes/staves")
                if node.text and node.text.isdigit()
            ),
            default=1,
        )
        for part in root.findall("part")
    )

    expected_count = len(page_rows) * staff_count
    with load_page_image(pages[region.page - 1], dpi=dpi) as image:
        # A long horizontal run excludes beams that otherwise merge into staff lines.
        for minimum_line_ratio in (0.125, 0.05):
            staves = deduplicate_staff_lines(
                detect_staff_lines(image, minimum_line_ratio=minimum_line_ratio),
                expected_count,
            )
            if len(staves) == expected_count and all(
                upper[-1] < lower[0] for upper, lower in zip(staves, staves[1:])
            ):
                break
        else:
            raise ValueError("원본 페이지에서 악보 줄의 위치를 확인하지 못했습니다.")

        first_staff = staves[row_index * staff_count]
        last_staff = staves[(row_index + 1) * staff_count - 1]
        spacing = float(np.median(np.diff(first_staff)))
        staff_left, staff_right = _staff_bounds(image, first_staff)
        if len({item.system for item in row_regions}) > 1:
            runs = _staff_runs(image, first_staff)
            if runs:
                staff_left, staff_right = runs[0][0], runs[-1][1]
        row_left = min(item.x for item in row_regions)
        row_width = max(item.x + item.width for item in row_regions) - row_left
        if row_width <= 0 or region.width <= 0:
            raise ValueError("마디의 가로 위치 정보가 없습니다.")
        scale = (staff_right - staff_left) / row_width
        left = staff_left + (region.x - row_left) * scale
        right = left + region.width * scale
        # Where available, align the estimated bounds to the printed barlines.
        with image.crop(
            (0, first_staff[0], image.width, last_staff[-1] + 1)
        ) as staff_area:
            columns = np.count_nonzero(_ink_mask(_gray_array(staff_area)), axis=0)
            minimum_height = round(staff_area.height * 0.85)
        barlines = [
            (start + end) / 2
            for start, end in _bands(columns, minimum_height)
        ]
        near_left = min(barlines, key=lambda x: abs(x - left), default=left)
        near_right = min(barlines, key=lambda x: abs(x - right), default=right)
        if (
            near_right > near_left
            and abs(near_left - left) <= spacing * 3
            and abs(near_right - right) <= spacing * 3
        ):
            left, right = near_left, near_right
        box = (
            max(0, round(left - spacing * 0.8)),
            max(0, round(min(first_staff) - spacing * 4)),
            min(image.width, round(right + spacing * 0.8)),
            min(image.height, round(max(last_staff) + spacing * 4)),
        )
        return image.crop(box)
