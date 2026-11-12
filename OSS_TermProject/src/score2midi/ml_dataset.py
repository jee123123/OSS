"""Build notehead-classification crops from a score and MusicXML."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import median

import numpy as np
from music21 import chord, clef, converter, pitch, stream
from PIL import Image

from score2midi.beam_recovery import (
    _gray_array,
    _ink_mask,
    _measure_regions,
    _notehead_candidates,
    _remove_staff_lines,
    _staff_bounds,
    detect_staff_lines,
)
from score2midi.preprocess import preprocess_page, preprocess_pdf

CROP_SIZE = 32


def _notehead_y(
    note_pitch: pitch.Pitch,
    staff: tuple[int, int, int, int, int],
    bass_clef: bool,
) -> float:
    reference = pitch.Pitch("G2" if bass_clef else "E4")
    spacing = float(median(np.diff(staff)))
    steps = note_pitch.diatonicNoteNum - reference.diatonicNoteNum
    return staff[-1] - steps * spacing / 2


def _expected_noteheads(
    measure: stream.Measure,
    left: float,
    scale: float,
    staff: tuple[int, int, int, int, int],
) -> list[tuple[float, float]]:
    current_clef = measure.getContextByClass(clef.Clef)
    bass_clef = isinstance(current_clef, clef.BassClef)
    expected = []

    for event in measure.recurse().notes:
        members = event.notes if isinstance(event, chord.Chord) else [event]
        for member in members:
            absolute_x = member.style.absoluteX
            if absolute_x is None:
                continue
            expected.append(
                (
                    left + float(absolute_x) * scale,
                    _notehead_y(member.pitch, staff, bass_clef),
                )
            )

    return expected


def _candidate_labels(
    candidates: list[tuple[int, int, float]],
    expected: list[tuple[float, float]],
    spacing: float,
) -> list[str]:
    return [
        "notehead"
        if any(
            abs(x - expected_x) <= spacing * 0.9
            and abs(y - expected_y) <= spacing * 0.65
            for expected_x, expected_y in expected
        )
        else "review"
        for x, y, _ in candidates
    ]


def _load_pages(source: Path) -> list[Image.Image]:
    if source.suffix.lower() == ".pdf":
        return preprocess_pdf(source)

    with Image.open(source) as image:
        return [preprocess_page(image)]


def _detect_expected_staves(
    page: Image.Image,
    expected_count: int,
) -> list[tuple[int, int, int, int, int]]:
    preview = page.resize(
        (max(1, page.width // 2), max(1, page.height // 2)),
        Image.Resampling.LANCZOS,
    )
    try:
        rough_staves = detect_staff_lines(preview)
    finally:
        preview.close()

    spacings = [
        float(median(np.diff(staff))) * 2
        for staff in rough_staves
        if median(np.diff(staff)) >= 5
    ]
    spacing = round(
        median(spacings) if spacings else page.height / 175
    )
    gray = _gray_array(page)
    projection = np.count_nonzero(gray < 210, axis=1).astype(float)
    smoothed = np.array(
        [
            projection[max(0, y - 2):min(len(projection), y + 3)].max()
            for y in range(len(projection))
        ]
    )
    candidates = []

    for candidate_spacing in range(max(3, spacing - 1), spacing + 2):
        for top in range(
            candidate_spacing,
            len(projection) - candidate_spacing * 5,
        ):
            rows = tuple(
                top + step * candidate_spacing
                for step in range(5)
            )
            values = np.array([smoothed[row] for row in rows])
            score = float(values.sum() - values.std() * 2)
            candidates.append((score, rows))

    selected = []
    for _, rows in sorted(candidates, reverse=True):
        center = sum(rows) / len(rows)
        if any(
            abs(center - sum(old) / len(old)) < spacing * 4
            for old in selected
        ):
            continue
        selected.append(rows)
        if len(selected) == expected_count:
            break

    return sorted(selected)


def build_notehead_dataset(
    source: Path,
    musicxml: Path,
    output_dir: Path,
) -> dict[str, int]:
    if not source.is_file():
        raise FileNotFoundError(f"악보 파일을 찾을 수 없습니다: {source}")
    if not musicxml.is_file():
        raise FileNotFoundError(f"MusicXML 파일을 찾을 수 없습니다: {musicxml}")

    labels_path = output_dir / "labels.csv"
    if labels_path.exists():
        raise FileExistsError(
            f"이미 학습 데이터가 존재합니다: {labels_path}"
        )

    score = converter.parse(str(musicxml))
    _, regions = _measure_regions(musicxml)
    if not regions:
        raise ValueError("MusicXML에서 마디 배치 정보를 찾지 못했습니다.")

    output_dir.mkdir(parents=True, exist_ok=True)
    for label in ("notehead", "review"):
        (output_dir / label).mkdir(exist_ok=True)

    pages = _load_pages(source)
    rows = []
    counts = {"notehead": 0, "review": 0}

    try:
        for page_number, page in enumerate(pages, 1):
            page_regions = [
                region
                for region in regions.values()
                if region.page == page_number
            ]
            if not page_regions:
                continue

            system_count = 1 + max(region.system for region in page_regions)
            expected_staff_count = system_count * len(score.parts)
            staves = _detect_expected_staves(page, expected_staff_count)
            if len(staves) != expected_staff_count:
                raise ValueError(
                    f"{page_number}쪽 오선은 {len(staves)}개, "
                    f"MusicXML 기준으로는 {expected_staff_count}개입니다."
                )

            gray = _gray_array(page)
            ink = _ink_mask(gray)

            for region in page_regions:
                system_regions = [
                    item
                    for item in page_regions
                    if item.system == region.system
                ]
                system_start = min(item.x for item in system_regions)
                system_width = sum(item.width for item in system_regions)

                for part_number, part in enumerate(score.parts, 1):
                    staff_index = (
                        region.system * len(score.parts) + part_number - 1
                    )
                    staff = staves[staff_index]
                    spacing = float(median(np.diff(staff)))
                    staff_left, staff_right = _staff_bounds(page, staff)
                    scale = (staff_right - staff_left) / system_width
                    left = staff_left + (region.x - system_start) * scale
                    right = left + region.width * scale
                    measure = part.measure(region.global_number)
                    if measure is None:
                        continue

                    staffless = _remove_staff_lines(ink, staff, spacing)
                    candidates = [
                        candidate
                        for candidate in _notehead_candidates(
                            staffless,
                            staff,
                            spacing,
                        )
                        if left <= candidate[0] < right
                    ]
                    expected = _expected_noteheads(
                        measure,
                        left,
                        scale,
                        staff,
                    )
                    labels = _candidate_labels(candidates, expected, spacing)

                    for index, ((x, y, confidence), label) in enumerate(
                        zip(candidates, labels),
                        1,
                    ):
                        half_width = max(4, round(spacing * 0.9))
                        half_height = max(4, round(spacing * 0.7))
                        crop = page.crop(
                            (
                                x - half_width,
                                y - half_height,
                                x + half_width + 1,
                                y + half_height + 1,
                            )
                        ).convert("L")
                        crop = crop.resize(
                            (CROP_SIZE, CROP_SIZE),
                            Image.Resampling.LANCZOS,
                        )
                        filename = (
                            f"p{page_number:03d}_m{region.global_number:03d}_"
                            f"part{part_number}_c{index:03d}.png"
                        )
                        relative_path = Path(label) / filename
                        crop.save(output_dir / relative_path)
                        counts[label] += 1
                        rows.append(
                            {
                                "path": relative_path.as_posix(),
                                "label": label,
                                "page": page_number,
                                "measure": region.global_number,
                                "part": part_number,
                                "x": x,
                                "y": y,
                                "confidence": f"{confidence:.4f}",
                            }
                        )
    finally:
        for page in pages:
            page.close()

    with labels_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "path",
                "label",
                "page",
                "measure",
                "part",
                "x",
                "y",
                "confidence",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)

    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="악보와 MusicXML에서 음표 분류 학습 데이터를 만듭니다."
    )
    parser.add_argument("source", type=Path, help="원본 PDF 또는 악보 이미지")
    parser.add_argument("musicxml", type=Path, help="검토가 끝난 MusicXML")
    parser.add_argument("output", type=Path, help="학습 데이터 저장 폴더")
    args = parser.parse_args()

    try:
        counts = build_notehead_dataset(
            args.source,
            args.musicxml,
            args.output,
        )
    except (FileNotFoundError, FileExistsError, OSError, ValueError) as error:
        parser.error(str(error))

    print(
        f"음표 {counts['notehead']}개, "
        f"검토 후보 {counts['review']}개를 저장했습니다."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
