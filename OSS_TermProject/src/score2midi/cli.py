from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from music21 import converter, instrument, tempo
from PIL import Image, ImageChops

from score2midi.articulation import separate_repeated_notes
from score2midi.beam_recovery import recover_unplaced_beamed_notes
from score2midi.navigation import prepare_playback_score
from score2midi.navigation_vision import detect_navigation_plan
from score2midi.preprocess import (
    preprocess_page,
    preprocess_pdf,
)
from score2midi.part_consolidation import consolidate_split_piano_parts
from score2midi.rhythm_corrector import correct_rhythm
from score2midi.timing_guard import stabilize_note_timing
from score2midi.rhythm_recovery import recover_missing_short_notes

NO_TIME_OFFSET_PATTERN = re.compile(
    r"\[[^\]]*#(?P<sheet>\d+)\].*?"
    r"Measure\{#(?P<measure>\d+)P(?P<part>\d+)\}"
    r".*?No timeOffset"
)
DEFAULT_NOTE_MODEL = Path(__file__).with_name("models") / "notehead.pt"


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


def _set_tempo(score, bpm: int) -> None:
    if not 20 <= bpm <= 300:
        raise ValueError("BPM은 20에서 300 사이여야 합니다.")

    for mark in list(
        score.recurse().getElementsByClass(tempo.MetronomeMark)
    ):
        if mark.activeSite is not None:
            mark.activeSite.remove(mark)
    score.insert(0, tempo.MetronomeMark(number=bpm))


def _force_piano(score) -> None:
    for part in score.parts:
        for detected in list(
            part.recurse().getElementsByClass(instrument.Instrument)
        ):
            if detected.activeSite is not None:
                detected.activeSite.remove(detected)
        part.insert(0, instrument.Piano())


def rewrite_midi_tempo(source: Path, output: Path, bpm: int) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    score = converter.parse(str(source))
    _set_tempo(score, bpm)
    _force_piano(score)
    output.parent.mkdir(parents=True, exist_ok=True)
    score.write("midi", fp=str(output))


def _load_notehead_predictor(note_model: Path | None):
    if note_model is None:
        return None, None

    try:
        if not note_model.is_file():
            raise FileNotFoundError(note_model)
        from score2midi.note_classifier import NoteheadPredictor

        return NoteheadPredictor(note_model).predict, None
    except Exception as error:
        return (
            None,
            "AI 음표 모델을 불러오지 못해 기존 방식으로 변환했습니다: "
            f"{error}",
        )


def convert_score(
    image: Path,
    output: Path,
    audiveris: str,
    bpm: int = 120,
    note_model: Path | None = DEFAULT_NOTE_MODEL,
) -> None:
    """Convert a score image to MIDI using Audiveris and music21."""
    convert_pages([image], output, audiveris, bpm, note_model)


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
    bpm: int = 120,
    note_model: Path | None = DEFAULT_NOTE_MODEL,
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

    notehead_predictor, model_warning = _load_notehead_predictor(note_model)

    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="score2midi-") as temp_dir:
        pages = []
        extended_pages = []

        if pdf_inputs:
            pages = preprocess_pdf(pdf_inputs[0])
            extended_pages = preprocess_pdf(
                pdf_inputs[0],
                extend_final_system=True,
            )
        else:
            for image in images:
                with Image.open(image) as source:
                    pages.append(preprocess_page(source))
                    extended_pages.append(
                        preprocess_page(
                            source,
                            extend_final_system=True,
                        )
                    )

        extension_changed = any(
            ImageChops.difference(primary, extended).getbbox()
            is not None
            for primary, extended in zip(pages, extended_pages)
        )

        input_document = Path(temp_dir) / "score.tiff"
        extended_document = Path(temp_dir) / "score-extended.tiff"
        try:
            pages[0].save(
                input_document,
                "TIFF",
                save_all=True,
                append_images=pages[1:],
                compression="tiff_lzw",
                dpi=(300, 300),
            )
            if extension_changed and not pdf_inputs:
                extended_pages[0].save(
                    extended_document,
                    "TIFF",
                    save_all=True,
                    append_images=extended_pages[1:],
                    compression="tiff_lzw",
                    dpi=(300, 300),
                )
        finally:
            for page in pages + extended_pages:
                page.close()

        primary_musicxml, unplaced_chords = _run_audiveris(
            input_document,
            Path(temp_dir) / "primary",
            audiveris_command,
        )
        score = converter.parse(str(primary_musicxml))
        conversion_warnings = [model_warning] if model_warning else []
        review_warnings: list[str] = []
        beam_sources = [(input_document, primary_musicxml)]

        if extension_changed and not pdf_inputs:
            try:
                extended_musicxml, _ = _run_audiveris(
                    extended_document,
                    Path(temp_dir) / "extended",
                    audiveris_command,
                )
                extended_score = converter.parse(str(extended_musicxml))
                conversion_warnings.extend(
                    recover_missing_short_notes(score, extended_score)
                )
            except (FileNotFoundError, RuntimeError, OSError) as error:
                conversion_warnings.append(
                    "마지막 오선 보조 인식은 실패했지만 기본 변환은 "
                    f"계속했습니다: {error}"
                )

        if pdf_inputs and (unplaced_chords or extension_changed):
            try:
                high_resolution_pages = preprocess_pdf(
                    pdf_inputs[0],
                    dpi=400,
                    extend_final_system=extension_changed,
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
                correct_rhythm(high_resolution_score)
                conversion_warnings.extend(
                    recover_missing_short_notes(
                        score,
                        high_resolution_score,
                    )
                )
                beam_sources.append(
                    (high_resolution_document, high_resolution_musicxml)
                )
            except (FileNotFoundError, RuntimeError, OSError) as error:
                conversion_warnings.append(
                    "400DPI 보조 인식은 실패했지만 기본 변환은 "
                    f"계속했습니다: {error}"
                )

        try:
            recovered_chords = set()
            for beam_document, beam_musicxml in beam_sources:
                remaining_chords = unplaced_chords - recovered_chords
                if not remaining_chords:
                    break
                beam_warnings, recovered = recover_unplaced_beamed_notes(
                    score,
                    beam_document,
                    beam_musicxml,
                    remaining_chords,
                    notehead_predictor=notehead_predictor,
                )
                recovered_chords.update(recovered)
                conversion_warnings.extend(beam_warnings)
        except (OSError, ValueError, IndexError, StopIteration) as error:
            recovered_chords = set()
            conversion_warnings.append(
                f"빔 음표 보조 복구를 건너뛰었습니다: {error}"
            )

        part_mapping, part_warnings = consolidate_split_piano_parts(score)
        conversion_warnings.extend(part_warnings)

        for sheet, measure, part_number in sorted(
            unplaced_chords - recovered_chords
        ):
            consolidated_part = part_mapping.get(part_number, part_number)
            if consolidated_part == 1:
                part_name = "오른손"
            elif consolidated_part == 2:
                part_name = "왼손"
            else:
                part_name = f"파트 {part_number}"

            review_warnings.append(
                f"{sheet}쪽 내 {measure}번째 마디 {part_name}: "
                "음표의 박자 위치를 정하지 못해 일부 음이 "
                "누락됐을 수 있습니다."
            )

        conversion_warnings.extend(correct_rhythm(score))
        conversion_warnings.extend(stabilize_note_timing(score))
        conversion_warnings.extend(separate_repeated_notes(score))
        _set_tempo(score, bpm)
        _force_piano(score)
        musicxml_output = output.with_suffix(".musicxml")
        score.write("musicxml", fp=str(musicxml_output))
        score = converter.parse(str(musicxml_output))
        serialized_warnings = correct_rhythm(score)
        repaired_measures = {
            match.group(1)
            for warning in serialized_warnings
            if (
                match := re.match(
                    r"(\d+)마디: (?:장식음|잘못 추가된 앞쪽 공통 쉼표)",
                    warning,
                )
            )
        }
        conversion_warnings = [
            warning
            for warning in conversion_warnings
            if not any(
                re.match(rf"{measure}마디 파트 .*초과", warning)
                for measure in repaired_measures
            )
        ]
        for warning in serialized_warnings:
            if warning not in conversion_warnings:
                conversion_warnings.append(warning)
        _force_piano(score)
        score.write("musicxml", fp=str(musicxml_output))
        navigation_plan = None
        try:
            navigation_plan, navigation_detection_warnings = (
                detect_navigation_plan(
                    score,
                    input_document,
                    primary_musicxml,
                )
            )
            conversion_warnings.extend(navigation_detection_warnings)
        except (OSError, ValueError, IndexError, StopIteration) as error:
            conversion_warnings.append(
                f"연주 순서 기호 인식을 건너뛰었습니다: {error}"
            )
        if navigation_plan is None and part_warnings:
            conversion_warnings.append(
                "연주 순서 기호를 찾지 못해 MIDI를 지면 순서대로 "
                "생성했습니다. 저장된 MusicXML을 검토하세요."
            )
        playback_score, navigation_warnings = prepare_playback_score(
            score,
            navigation_plan,
        )
        conversion_warnings.extend(navigation_warnings)
        _force_piano(playback_score)
        playback_score.write("midi", fp=str(output))

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
    parser.add_argument(
        "--bpm",
        type=int,
        default=120,
        help="출력 MIDI 속도(기본값: 120)",
    )
    parser.add_argument(
        "--note-model",
        type=Path,
        default=DEFAULT_NOTE_MODEL,
        help="기본 내장 모델 대신 사용할 PyTorch 모델(.pt)",
    )
    args = parser.parse_args()

    try:
        convert_score(
            args.image,
            args.output,
            args.audiveris,
            args.bpm,
            args.note_model,
        )
    except (FileNotFoundError, RuntimeError, OSError, ValueError) as error:
        print(f"오류: {error}", file=sys.stderr)
        return 1

    print(f"MIDI 파일을 저장했습니다: {args.output}")
    print(
        "MusicXML 파일을 저장했습니다: "
        f"{args.output.with_suffix('.musicxml')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
