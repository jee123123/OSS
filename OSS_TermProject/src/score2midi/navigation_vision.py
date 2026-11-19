from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import cv2
import numpy as np
from music21 import stream
from PIL import Image, ImageSequence

from score2midi.beam_image import (
    _bands,
    _gray_array,
    _ink_mask,
    detect_staff_lines,
)
from score2midi.beam_layout import _measure_regions, _musicxml_root
from score2midi.navigation import NavigationCommand, NavigationPlan


@dataclass(frozen=True)
class _PageRow:
    page: int
    index: int
    treble: tuple[int, int, int, int, int]
    bass: tuple[int, int, int, int, int]
    runs: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _VisualMark:
    kind: str
    page: int
    row: int
    x: float
    y: float


def _deduplicate_staves(staves):
    clusters = []
    for staff in sorted(staves):
        if clusters and staff[0] <= max(item[-1] for item in clusters[-1]):
            clusters[-1].append(staff)
        else:
            clusters.append([staff])

    def irregularity(staff):
        gaps = np.diff(staff)
        average = float(np.mean(gaps))
        return float(np.std(gaps)) / average if average else float("inf")

    return [min(cluster, key=irregularity) for cluster in clusters]


def _staff_runs(
    image: Image.Image,
    staff: tuple[int, int, int, int, int],
) -> tuple[tuple[int, int], ...]:
    ink = _ink_mask(_gray_array(image))
    support = np.zeros(ink.shape[1], dtype=np.uint8)
    for y in staff:
        support += np.any(
            ink[max(0, y - 2):min(ink.shape[0], y + 3)] > 0,
            axis=0,
        )
    mask = ((support >= 3) * 255).astype(np.uint8)
    runs = [
        run
        for run in _bands(mask, 255)
        if run[1] - run[0] >= image.width * 0.12
    ]
    return tuple(runs)


def _page_rows(page_number: int, image: Image.Image) -> list[_PageRow]:
    staves = _deduplicate_staves(detect_staff_lines(image))
    if len(staves) % 2:
        staves = staves[:-1]
    return [
        _PageRow(
            page_number,
            index,
            staves[index * 2],
            staves[index * 2 + 1],
            _staff_runs(image, staves[index * 2]),
        )
        for index in range(len(staves) // 2)
    ]


def _nearest_row(rows: list[_PageRow], y: float) -> _PageRow | None:
    return min(
        rows,
        key=lambda row: abs(y - (row.treble[0] + row.bass[-1]) / 2),
        default=None,
    )


def _detect_symbols(
    page_number: int,
    image: Image.Image,
    rows: list[_PageRow],
) -> list[_VisualMark]:
    ink = _ink_mask(_gray_array(image))
    _, _, stats, _ = cv2.connectedComponentsWithStats(
        (ink * 255).astype(np.uint8),
        8,
    )
    candidates: list[_VisualMark] = []

    for row in rows:
        spacing = float(median(np.diff(row.treble)))
        for x, y, width, height, area in stats[1:]:
            center_y = y + height / 2
            if not (
                row.treble[0] - spacing * 7
                <= center_y
                <= row.treble[0] - spacing * 0.8
            ):
                continue
            if not any(
                left - spacing <= x + width / 2 <= right + spacing
                for left, right in row.runs
            ):
                continue

            height_ratio = height / spacing
            width_ratio = width / spacing
            density = area / max(1, width * height)
            kind = None
            if (
                2.5 <= height_ratio <= 4.0
                and 1.5 <= width_ratio <= 3.0
                and 0.32 <= density <= 0.58
            ):
                kind = "segno"
            elif (
                3.1 <= height_ratio <= 5.0
                and 2.7 <= width_ratio <= 4.8
                and 0.16 <= density <= 0.34
            ):
                kind = "coda"
            elif (
                3.1 <= height_ratio <= 5.0
                and 5.2 <= width_ratio <= 8.5
                and 0.16 <= density <= 0.34
            ):
                kind = "double_coda"
            if kind is not None:
                candidates.append(
                    _VisualMark(
                        kind,
                        page_number,
                        row.index,
                        x + width / 2,
                        center_y,
                    )
                )

    segnos = [mark for mark in candidates if mark.kind == "segno"]
    consumed: set[int] = set()
    combined: list[_VisualMark] = []
    for index, mark in enumerate(segnos):
        if index in consumed:
            continue
        row = rows[mark.row]
        spacing = float(median(np.diff(row.treble)))
        partner = next(
            (
                (other_index, other)
                for other_index, other in enumerate(segnos[index + 1:], index + 1)
                if other_index not in consumed
                and other.row == mark.row
                and abs(other.y - mark.y) <= spacing
                and 0 < other.x - mark.x <= spacing * 4
            ),
            None,
        )
        if partner is None:
            combined.append(mark)
            continue
        consumed.add(partner[0])
        combined.append(
            _VisualMark(
                "double_segno",
                mark.page,
                mark.row,
                (mark.x + partner[1].x) / 2,
                (mark.y + partner[1].y) / 2,
            )
        )

    return combined + [
        mark for mark in candidates if mark.kind != "segno"
    ]


def _tessdata_path() -> Path | None:
    candidates = [
        Path(value)
        for value in (os.environ.get("TESSDATA_PREFIX"),)
        if value
    ]
    candidates.append(
        Path.home()
        / "AppData/Roaming/AudiverisLtd/audiveris/config/tessdata"
    )
    return next((path for path in candidates if path.is_dir()), None)


def _detect_text_marks(
    page_number: int,
    image: Image.Image,
    rows: list[_PageRow],
) -> tuple[list[_VisualMark], str | None]:
    try:
        from tesserocr import PSM, RIL, PyTessBaseAPI
    except ImportError:
        return [], "D.S./D.S.S./Fine OCR을 사용할 수 없어 연주 순서 인식을 건너뛰었습니다."

    tessdata = _tessdata_path()
    if tessdata is None:
        return [], "Audiveris OCR 언어 데이터를 찾지 못해 연주 순서 인식을 건너뛰었습니다."

    marks: list[_VisualMark] = []
    with PyTessBaseAPI(
        path=str(tessdata),
        lang="eng",
        psm=PSM.SPARSE_TEXT,
    ) as api:
        api.SetImage(image)
        api.Recognize()
        iterator = api.GetIterator()
        if iterator is None:
            return marks, None
        iterator.Begin()
        while True:
            text = iterator.GetUTF8Text(RIL.WORD) or ""
            key = re.sub(r"[^A-Z]", "", text.upper())
            kind = {"DS": "ds", "DSS": "dss", "FINE": "fine"}.get(key)
            box = iterator.BoundingBox(RIL.WORD)
            if kind and box is not None:
                left, top, right, bottom = box
                row = _nearest_row(rows, (top + bottom) / 2)
                if row is not None:
                    marks.append(
                        _VisualMark(
                            kind,
                            page_number,
                            row.index,
                            (left + right) / 2,
                            (top + bottom) / 2,
                        )
                    )
            if not iterator.Next(RIL.WORD):
                break
    return marks, None


def _system_rows(musicxml: Path) -> dict[tuple[int, int], int]:
    root = _musicxml_root(musicxml)
    parts = root.findall("part")
    if not parts:
        return {}
    part_measures = [part.findall("measure") for part in parts]
    measure_count = max(map(len, part_measures))
    layout_measures = [
        max(
            (
                measures[index]
                for measures in part_measures
                if index < len(measures)
            ),
            key=lambda measure: float(measure.get("width") or 0),
        )
        for index in range(measure_count)
    ]

    page = 1
    system = 0
    row = 0
    result = {(page, system): row}
    for measure in layout_measures:
        print_element = measure.find("print")
        if print_element is None:
            continue
        if print_element.get("new-page") == "yes":
            page += 1
            system = 0
            row = 0
            result[(page, system)] = row
        elif print_element.get("new-system") == "yes":
            system += 1
            distance = float(
                print_element.findtext("./system-layout/system-distance") or 0
            )
            if distance >= 0:
                row += 1
            result[(page, system)] = row
    return result


def _partition_regions(regions, run_count: int):
    if run_count <= 1:
        return [regions]
    systems = []
    for region in regions:
        if not systems or systems[-1][0].system != region.system:
            systems.append([region])
        else:
            systems[-1].append(region)
    if len(systems) == run_count:
        return systems

    widths = [region.width for region in regions]
    typical = median(widths) if widths else 1
    cut = next(
        (
            index
            for index, width in enumerate(widths[1:], 1)
            if width < typical * 0.3
        ),
        None,
    )
    if run_count == 2 and cut is not None:
        return [regions[:cut], regions[cut:]]

    result = []
    start = 0
    for group_index in range(run_count):
        end = round(len(regions) * (group_index + 1) / run_count)
        result.append(regions[start:end])
        start = end
    return result


def _map_mark_to_measure(
    mark: _VisualMark,
    rows_by_page: dict[int, list[_PageRow]],
    regions,
    system_rows: dict[tuple[int, int], int],
) -> tuple[int, bool] | None:
    rows = rows_by_page.get(mark.page, [])
    if not 0 <= mark.row < len(rows):
        return None
    row = rows[mark.row]
    row_regions = sorted(
        (
            region
            for region in regions.values()
            if region.page == mark.page
            and system_rows.get((region.page, region.system), region.system)
            == mark.row
        ),
        key=lambda region: (region.system, region.x),
    )
    if not row_regions or not row.runs:
        return None

    run_index = min(
        range(len(row.runs)),
        key=lambda index: (
            0
            if row.runs[index][0] <= mark.x <= row.runs[index][1]
            else min(
                abs(mark.x - row.runs[index][0]),
                abs(mark.x - row.runs[index][1]),
            )
        ),
    )
    groups = _partition_regions(row_regions, len(row.runs))
    if run_index >= len(groups) or not groups[run_index]:
        return None
    group = groups[run_index]
    left, right = row.runs[run_index]
    total_width = sum(region.width for region in group) or 1
    cursor = left
    boundaries = [left]
    for region in group:
        cursor += (right - left) * region.width / total_width
        boundaries.append(cursor)

    position = max(0, min(len(group) - 1, np.searchsorted(boundaries, mark.x) - 1))
    spacing = float(median(np.diff(row.treble)))
    near_start = abs(mark.x - boundaries[position]) <= spacing * 2
    if (
        position + 1 < len(group)
        and abs(mark.x - boundaries[position + 1]) <= spacing * 2
    ):
        position += 1
        near_start = True
    return group[position].global_number - 1, near_start


def detect_navigation_plan(
    score: stream.Score,
    page_document: Path,
    musicxml: Path,
) -> tuple[NavigationPlan | None, list[str]]:
    with Image.open(page_document) as document:
        pages = [frame.copy() for frame in ImageSequence.Iterator(document)]

    warnings: list[str] = []
    try:
        rows_by_page = {
            page_number: _page_rows(page_number, page)
            for page_number, page in enumerate(pages, 1)
        }
        marks: list[_VisualMark] = []
        for page_number, page in enumerate(pages, 1):
            rows = rows_by_page[page_number]
            marks.extend(_detect_symbols(page_number, page, rows))
            text_marks, warning = _detect_text_marks(page_number, page, rows)
            marks.extend(text_marks)
            if warning and warning not in warnings:
                warnings.append(warning)

        _, regions = _measure_regions(musicxml)
        system_rows = _system_rows(musicxml)
        mapped = []
        for mark in marks:
            location = _map_mark_to_measure(
                mark,
                rows_by_page,
                regions,
                system_rows,
            )
            if location is not None:
                mapped.append((mark, *location))

        segnos = tuple(sorted({measure for mark, measure, _ in mapped if mark.kind == "segno"}))
        double_segnos = tuple(sorted({measure for mark, measure, _ in mapped if mark.kind == "double_segno"}))
        commands = tuple(
            NavigationCommand(measure, mark.kind)
            for mark, measure, _ in sorted(
                (item for item in mapped if item[0].kind in {"ds", "dss"}),
                key=lambda item: item[1],
            )
        )
        fines = tuple(sorted({measure for mark, measure, _ in mapped if mark.kind == "fine"}))

        def coda_route(kind: str, command_kind: str):
            command = next(
                (item for item in commands if item.kind == command_kind),
                None,
            )
            if command is None:
                return (), ()
            candidates = sorted(
                (
                    measure - int(at_start),
                    measure,
                )
                for mark, measure, at_start in mapped
                if mark.kind == kind
            )
            before = [jump for jump, _ in candidates if jump < command.measure_index]
            after = [target for _, target in candidates if target > command.measure_index]
            return (
                (before[-1],) if before else (),
                (after[0],) if after else (),
            )

        coda_jumps, coda_targets = coda_route("coda", "ds")
        double_coda_jumps, double_coda_targets = coda_route(
            "double_coda",
            "dss",
        )
        measure_count = len(
            list(score.parts[0].getElementsByClass(stream.Measure))
        ) if score.parts else 0
        plan = NavigationPlan(
            measure_count=measure_count,
            segnos=segnos,
            double_segnos=double_segnos,
            coda_jumps=coda_jumps,
            coda_targets=coda_targets,
            double_coda_jumps=double_coda_jumps,
            double_coda_targets=double_coda_targets,
            fines=fines,
            commands=commands,
        )
        if not commands:
            return None, warnings
        warnings.append(
            "악보 이미지에서 D.S./D.S.S.와 코다 연주 순서를 자동 인식했습니다."
        )
        return plan, warnings
    finally:
        for page in pages:
            page.close()
