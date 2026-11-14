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


class ReviewDataset:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.labels_path = self.root / "labels.csv"
        if not self.labels_path.is_file():
            raise FileNotFoundError(
                f"labels.csv를 찾을 수 없습니다: {self.labels_path}"
            )

        with self.labels_path.open(
            newline="",
            encoding="utf-8-sig",
        ) as file:
            reader = csv.DictReader(file)
            self.fieldnames = reader.fieldnames or []
            self.rows = list(reader)

        if not {"path", "label"}.issubset(self.fieldnames):
            raise ValueError("labels.csv에 path와 label 열이 필요합니다.")

    def review_indices(self) -> list[int]:
        return [
            index
            for index, row in enumerate(self.rows)
            if row["label"] == "review"
        ]

    def classify(self, index: int, label: str) -> None:
        if label not in {"notehead", "other", "review"}:
            raise ValueError(f"지원하지 않는 라벨입니다: {label}")

        row = self.rows[index]
        old_label = row["label"]
        old_relative = Path(row["path"])
        if old_label == label:
            return

        source = (self.root / old_relative).resolve()
        if not source.is_relative_to(self.root):
            raise ValueError(f"데이터셋 밖의 경로입니다: {source}")
        if not source.is_file():
            raise FileNotFoundError(f"후보 이미지를 찾을 수 없습니다: {source}")

        destination_dir = self.root / label
        destination_dir.mkdir(exist_ok=True)
        destination = destination_dir / source.name
        if destination.exists():
            raise FileExistsError(f"이미 파일이 존재합니다: {destination}")

        destination_relative = destination.relative_to(self.root)
        source.replace(destination)
        row["path"] = destination_relative.as_posix()
        row["label"] = label

        try:
            self._save()
        except OSError:
            destination.replace(source)
            row["path"] = old_relative.as_posix()
            row["label"] = old_label
            raise

    def _save(self) -> None:
        temporary = self.labels_path.with_suffix(".tmp")
        with temporary.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        temporary.replace(self.labels_path)


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


def _context_crop(
    page: Image.Image,
    x: int,
    y: int,
    spacing: float,
) -> Image.Image:
    half_width = max(48, round(spacing * 4.5))
    half_height = max(36, round(spacing * 3.2))
    return page.crop(
        (
            x - half_width,
            y - half_height,
            x + half_width + 1,
            y + half_height + 1,
        )
    ).convert("L")


def add_context_images(source: Path, dataset_dir: Path) -> int:
    dataset = ReviewDataset(dataset_dir)
    required = {"page", "x", "y"}
    if not required.issubset(dataset.fieldnames):
        raise ValueError("labels.csv에 page, x, y 열이 필요합니다.")

    pages = _load_pages(source)
    context_dir = dataset.root / "context"
    context_dir.mkdir(exist_ok=True)
    if "context_path" not in dataset.fieldnames:
        dataset.fieldnames.append("context_path")

    try:
        for row in dataset.rows:
            page_number = int(row["page"])
            if not 1 <= page_number <= len(pages):
                raise ValueError(f"존재하지 않는 페이지입니다: {page_number}")
            page = pages[page_number - 1]
            spacing = max(8.0, page.height / 175)
            relative_path = Path("context") / Path(row["path"]).name
            _context_crop(
                page,
                int(row["x"]),
                int(row["y"]),
                spacing,
            ).save(dataset.root / relative_path)
            row["context_path"] = relative_path.as_posix()
    finally:
        for page in pages:
            page.close()

    dataset._save()
    return len(dataset.rows)


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
    (output_dir / "context").mkdir(exist_ok=True)

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
                        filename = (
                            f"p{page_number:03d}_m{region.global_number:03d}_"
                            f"part{part_number}_c{index:03d}.png"
                        )
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
                        relative_path = Path(label) / filename
                        context_path = Path("context") / filename
                        crop.save(output_dir / relative_path)
                        _context_crop(page, x, y, spacing).save(
                            output_dir / context_path
                        )
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
                                "context_path": context_path.as_posix(),
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
                "context_path",
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
    parser.add_argument("musicxml", nargs="?", type=Path)
    parser.add_argument("output", nargs="?", type=Path)
    parser.add_argument(
        "--add-context",
        type=Path,
        metavar="DATASET",
        help="기존 데이터셋에 원본 악보 주변 이미지를 추가합니다.",
    )
    args = parser.parse_args()

    if args.add_context is not None:
        try:
            count = add_context_images(args.source, args.add_context)
        except (FileNotFoundError, OSError, ValueError) as error:
            parser.error(str(error))
        print(f"주변 이미지 {count}개를 추가했습니다.")
        return 0

    if args.musicxml is None or args.output is None:
        parser.error("MusicXML과 학습 데이터 저장 폴더가 필요합니다.")

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
