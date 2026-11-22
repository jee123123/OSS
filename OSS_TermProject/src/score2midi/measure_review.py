"""Find reported measures and crop them from the original score pages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from PIL import Image

from score2midi.beam_image import deduplicate_staff_lines, detect_staff_lines
from score2midi.beam_layout import (
    _measure_regions,
    _musicxml_root,
    _staff_bounds,
)
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
    page_width, regions = _measure_regions(musicxml)
    region = next(
        item for item in regions.values() if item.global_number == measure_number
    )
    if not 1 <= region.page <= len(pages):
        raise ValueError("원본 악보 페이지를 찾을 수 없습니다.")

    image = load_page_image(pages[region.page - 1], dpi=dpi)
    page_regions = [
        item for item in regions.values() if item.page == region.page
    ]
    system_regions = [
        item for item in page_regions if item.system == region.system
    ]
    system_count = 1 + max(item.system for item in page_regions)
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

    try:
        staves = deduplicate_staff_lines(
            detect_staff_lines(image),
            system_count * staff_count,
        )
        first_staff = staves[region.system * staff_count]
        last_staff = staves[(region.system + 1) * staff_count - 1]
        spacing = float(np.median(np.diff(first_staff)))
        staff_left, staff_right = _staff_bounds(image, first_staff)
        system_start = min(item.x for item in system_regions)
        system_width = sum(item.width for item in system_regions)
        scale = (staff_right - staff_left) / system_width
        left = staff_left + (region.x - system_start) * scale
        right = left + region.width * scale
        box = (
            max(0, round(left - spacing * 2)),
            max(0, round(min(first_staff) - spacing * 4)),
            min(image.width, round(right + spacing * 2)),
            min(image.height, round(max(last_staff) + spacing * 4)),
        )
    except (IndexError, ValueError):
        top = round(image.height * region.system / system_count)
        bottom = round(image.height * (region.system + 1) / system_count)
        left = round(image.width * region.x / page_width)
        right = round(image.width * (region.x + region.width) / page_width)
        box = (
            max(0, left - 20),
            max(0, top - 20),
            min(image.width, right + 20),
            min(image.height, bottom + 20),
        )

    try:
        return image.crop(box)
    finally:
        image.close()
