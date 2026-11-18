"""Map MusicXML measures to their positions on rendered score pages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile, is_zipfile

import cv2
import numpy as np
from music21 import stream
from PIL import Image

from score2midi.beam_image import _bands, _gray_array, _ink_mask

@dataclass(frozen=True)
class _MeasureRegion:
    page: int
    local_number: int
    global_number: int
    system: int
    x: float
    width: float


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
                (
                    print_element.findtext(
                        "./system-layout/system-margins/left-margin"
                    )
                    if print_element is not None
                    else 0
                )
                or 0
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



