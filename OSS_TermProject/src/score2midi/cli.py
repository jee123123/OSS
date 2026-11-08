from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from music21 import converter, instrument
from PIL import Image

from score2midi.articulation import separate_repeated_notes
from score2midi.beam_recovery import recover_unplaced_beamed_notes
from score2midi.preprocess import preprocess_page, preprocess_pdf
from score2midi.rhythm_corrector import correct_rhythm
from score2midi.rhythm_recovery import recover_missing_short_notes

NO_TIME_OFFSET_PATTERN = re.compile(
    r"\[[^\]]*#(?P<sheet>\d+)\].*?"
    r"Measure\{#(?P<measure>\d+)P(?P<part>\d+)\}"
    r".*?No timeOffset"
)


def _find_unplaced_chords(
    log_text: str,
) -> set[tuple[int, int, int]]:
    return {
        (
            int(match.group("sheet")),
            int(match.group("measure")),
            int(match.group("part")),
        )
        for match in NO_TIME_OFFSET_PATTERN.finditer(log_text)
    }


def convert_score(image: Path, output: Path, audiveris: str) -> None:
    """Convert a score image to MIDI using Audiveris and music21."""
    convert_pages([image], output, audiveris)


def _run_audiveris(
    input_document: Path,
    output_dir: Path,
    audiveris_command: str,
) -> tuple[Path, set[tuple[int, int, int]]]:
    """Run Audiveris and return the generated MusicXML path."""
    output_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            audiveris_command,
            "-batch",
            "-transcribe",
            "-export",
            "-output",
            str(output_dir),
            str(input_document),
        ],
        capture_output=True,
        check=False,
    )

    stderr = result.stderr.decode(
        "utf-8", errors="replace"
    ).strip()
    stdout = result.stdout.decode(
        "utf-8", errors="replace"
    ).strip()
    log_text = f"{stdout}\n{stderr}"

    if result.returncode != 0:
        details = stderr or stdout or "자세한 오류 메시지가 없습니다."
        raise RuntimeError(f"Audiveris 실행 실패:\n{details}")

    candidates = (
        list(output_dir.rglob("*.mxl"))
        + list(output_dir.rglob("*.musicxml"))
        + list(output_dir.rglob("*.xml"))
    )

    if not candidates:
        raise FileNotFoundError(
            "Audiveris가 MusicXML 파일을 만들지 않았습니다."
        )

    return candidates[0], _find_unplaced_chords(log_text)


def convert_pages(
    images: list[Path],
    output: Path,
    audiveris: str,
) -> None:
    """Convert ordered score pages into one MIDI file."""
    if not images:
        raise ValueError("변환할 악보가 없습니다.")

    for image in images:
        if not image.is_file():
            raise FileNotFoundError(
                f"악보 파일을 찾을 수 없습니다: {image}"
            )

    pdf_inputs = [
        image for image in images if image.suffix.lower() == ".pdf"
    ]
    if pdf_inputs and len(images) != 1:
        raise ValueError(
            "PDF는 한 번에 하나만 선택할 수 있으며 "
            "이미지 파일과 함께 사용할 수 없습니다."
        )

    executable = shutil.which(audiveris)
    if executable is None and not Path(audiveris).is_file():
        raise FileNotFoundError(
            "Audiveris를 찾을 수 없습니다. "
            "--audiveris 옵션에 실행 파일 경로를 지정하세요."
        )
    audiveris_command = executable or str(Path(audiveris).resolve())

    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="score2midi-") as temp_dir:
        pages = []

        if pdf_inputs:
            pages = preprocess_pdf(pdf_inputs[0])
        else:
            for image in images:
                with Image.open(image) as source:
                    pages.append(preprocess_page(source))

        input_document = Path(temp_dir) / "score.tiff"
        try:
            pages[0].save(
                input_document,
                "TIFF",
                save_all=True,
                append_images=pages[1:],
                compression="tiff_lzw",
                dpi=(300, 300),
            )
        finally:
            for page in pages:
                page.close()

        primary_musicxml, unplaced_chords = _run_audiveris(
            input_document,
            Path(temp_dir) / "primary",
            audiveris_command,
        )
        score = converter.parse(str(primary_musicxml))
        conversion_warnings: list[str] = []
        review_warnings: list[str] = []

        if pdf_inputs:
            try:
                high_resolution_pages = preprocess_pdf(
                    pdf_inputs[0],
                    dpi=400,
                )
                high_resolution_document = (
                    Path(temp_dir) / "score-400dpi.tiff"
                )

                try:
                    high_resolution_pages[0].save(
                        high_resolution_document,
                        "TIFF",
                        save_all=True,
                        append_images=high_resolution_pages[1:],
                        compression="tiff_lzw",
                        dpi=(400, 400),
                    )
                finally:
                    for page in high_resolution_pages:
                        page.close()

                high_resolution_musicxml, _ = _run_audiveris(
                    high_resolution_document,
                    Path(temp_dir) / "high-resolution",
                    audiveris_command,
                )
                high_resolution_score = converter.parse(
                    str(high_resolution_musicxml)
                )
                conversion_warnings.extend(
                    recover_missing_short_notes(
                        score,
                        high_resolution_score,
                    )
                )
            except (FileNotFoundError, RuntimeError, OSError) as error:
                conversion_warnings.append(
                    "400DPI 보조 인식은 실패했지만 기본 변환은 "
                    f"계속했습니다: {error}"
                )

            try:
                alternative_musicxml, _ = _run_audiveris(
                    pdf_inputs[0].resolve(),
                    Path(temp_dir) / "alternative",
                    audiveris_command,
                )
                alternative_score = converter.parse(
                    str(alternative_musicxml)
                )
                conversion_warnings.extend(
                    recover_missing_short_notes(
                        score,
                        alternative_score,
                    )
                )
            except (FileNotFoundError, RuntimeError, OSError) as error:
                conversion_warnings.append(
                    "원본 PDF 보조 인식은 실패했지만 기본 변환은 "
                    f"계속했습니다: {error}"
                )

        try:
            beam_warnings, recovered_chords = recover_unplaced_beamed_notes(
                score,
                input_document,
                primary_musicxml,
                unplaced_chords,
            )
            conversion_warnings.extend(beam_warnings)
        except (OSError, ValueError, IndexError, StopIteration) as error:
            recovered_chords = set()
            conversion_warnings.append(
                f"빔 음표 보조 복구를 건너뛰었습니다: {error}"
            )

        for sheet, measure, part_number in sorted(
            unplaced_chords - recovered_chords
        ):
            if part_number == 1:
                part_name = "오른손"
            elif part_number == 2:
                part_name = "왼손"
            else:
                part_name = f"파트 {part_number}"

            review_warnings.append(
                f"{sheet}쪽 내 {measure}번째 마디 {part_name}: "
                "음표의 박자 위치를 정하지 못해 일부 음이 "
                "누락됐을 수 있습니다."
            )

        conversion_warnings.extend(correct_rhythm(score))
        conversion_warnings.extend(separate_repeated_notes(score))
        for part in score.parts:
            detected_instruments = list(
                part.recurse().getElementsByClass(instrument.Instrument)
            )
            if detected_instruments:
                for detected in detected_instruments:
                    detected.midiProgram = 0
            else:
                part.insert(0, instrument.Piano())
        score.write("midi", fp=str(output))

        details = conversion_warnings + review_warnings
        if details:
            output.with_suffix(".txt").write_text(
                "\n".join(details),
                encoding="utf-8",
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="악보 이미지 또는 PDF에서 MIDI 파일을 만듭니다."
    )
    parser.add_argument("image", type=Path, help="입력 악보 이미지 또는 PDF")
    parser.add_argument("output", type=Path, help="저장할 MIDI 파일")
    parser.add_argument(
        "--audiveris",
        default="audiveris",
        help="Audiveris 실행 파일 경로",
    )
    args = parser.parse_args()

    try:
        convert_score(args.image, args.output, args.audiveris)
    except (FileNotFoundError, RuntimeError, OSError) as error:
        print(f"오류: {error}", file=sys.stderr)
        return 1

    print(f"MIDI 파일을 저장했습니다: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
