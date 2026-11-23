"""Re-recognize one score page and replace a selected measure safely."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import shutil
import tempfile

from music21 import converter, meter, stream

from score2midi.beam_layout import _measure_regions
from score2midi.cli import _force_piano, _run_audiveris, _set_tempo
from score2midi.navigation import prepare_playback_score
from score2midi.page_editor import PageSpec, load_page_image
from score2midi.preprocess import preprocess_page
from score2midi.rhythm_corrector import correct_rhythm


@dataclass(frozen=True)
class MeasureStats:
    notes: int
    rests: int
    duration_error: Fraction


@dataclass(frozen=True)
class RetryCandidate:
    measure: int
    local_number: int
    musicxml: Path
    current: MeasureStats
    candidate: MeasureStats

    @property
    def safe(self) -> bool:
        return self.candidate.duration_error == 0

    @property
    def comparison(self) -> str:
        verdict = (
            "적용 가능"
            if self.safe
            else "적용 불가: 기존 마디 길이와 일치하지 않음"
        )
        return (
            f"기존 결과: 음표 {self.current.notes}개, 쉼표 "
            f"{self.current.rests}개, 박자 오차 "
            f"{float(self.current.duration_error):g}박\n"
            f"재인식 결과: 음표 {self.candidate.notes}개, 쉼표 "
            f"{self.candidate.rests}개, 박자 오차 "
            f"{float(self.candidate.duration_error):g}박\n판정: {verdict}"
        )


def _part_measure(
    part: stream.Part,
    measure_number: int,
    local_number: int,
) -> stream.Measure:
    direct = part.measure(measure_number)
    if direct is not None:
        return direct
    measures = list(part.getElementsByClass(stream.Measure))
    if not 1 <= local_number <= len(measures):
        raise ValueError("재인식 결과에서 대상 마디를 찾을 수 없습니다.")
    return measures[local_number - 1]


def measure_stats(
    score: stream.Score,
    measure_number: int,
    local_number: int,
    expected_durations: tuple[Fraction, ...] | None = None,
) -> MeasureStats:
    notes = rests = 0
    error = Fraction(0)
    for index, part in enumerate(score.parts):
        measure = _part_measure(part, measure_number, local_number)
        notes += len(list(measure.recurse().notes))
        rests += len(list(measure.recurse().getElementsByClass("Rest")))
        expected = (
            expected_durations[index]
            if expected_durations is not None
            else Fraction(measure.barDuration.quarterLength)
        )
        error += abs(Fraction(measure.highestTime) - expected)
    return MeasureStats(notes, rests, error)


def recognize_measure_candidate(
    pages: list[PageSpec],
    current_musicxml: Path,
    measure_number: int,
    audiveris: str,
) -> RetryCandidate:
    _, regions = _measure_regions(current_musicxml)
    region = next(
        item for item in regions.values() if item.global_number == measure_number
    )
    if not 1 <= region.page <= len(pages):
        raise ValueError("재인식할 원본 페이지를 찾을 수 없습니다.")

    current_score = converter.parse(str(current_musicxml))
    with tempfile.TemporaryDirectory(prefix="score2midi-retry-") as directory:
        directory = Path(directory)
        source = load_page_image(pages[region.page - 1], dpi=400)
        processed = preprocess_page(source)
        document = directory / "page.tiff"
        try:
            processed.save(document, "TIFF", compression="tiff_lzw", dpi=(400, 400))
        finally:
            source.close()
            if processed is not source:
                processed.close()

        recognized, _ = _run_audiveris(document, directory / "audiveris", audiveris)
        candidate_score = converter.parse(str(recognized))
        if len(candidate_score.parts) != len(current_score.parts):
            raise ValueError(
                "재인식 결과의 오른손·왼손 구조가 기존 결과와 달라 "
                "적용할 수 없습니다."
            )
        for current_part, candidate_part in zip(
            current_score.parts,
            candidate_score.parts,
        ):
            current_measure = _part_measure(
                current_part,
                measure_number,
                region.local_number,
            )
            candidate_measure = _part_measure(
                candidate_part,
                measure_number,
                region.local_number,
            )
            signature = current_measure.getContextByClass(meter.TimeSignature)
            if signature is not None:
                candidate_measure.timeSignature = meter.TimeSignature(
                    signature.ratioString
                )
        correct_rhythm(candidate_score)
        normalized = directory / "candidate.musicxml"
        candidate_score.write("musicxml", fp=str(normalized))
        candidate_score = converter.parse(str(normalized))
        correct_rhythm(candidate_score)

    output = current_musicxml.with_name(
        f"{current_musicxml.stem}.measure-{measure_number}-retry.musicxml"
    )
    candidate_score.write("musicxml", fp=str(output))
    expected_durations = tuple(
        Fraction(
            _part_measure(part, measure_number, region.local_number)
            .barDuration.quarterLength
        )
        for part in current_score.parts
    )
    return RetryCandidate(
        measure_number,
        region.local_number,
        output,
        measure_stats(current_score, measure_number, region.local_number),
        measure_stats(
            candidate_score,
            measure_number,
            region.local_number,
            expected_durations,
        ),
    )


def apply_retry_candidate(
    current_musicxml: Path,
    midi_output: Path,
    retry: RetryCandidate,
    bpm: int,
) -> None:
    score = converter.parse(str(current_musicxml))
    candidate_score = converter.parse(str(retry.musicxml))
    if len(score.parts) != len(candidate_score.parts):
        raise ValueError("재인식 결과의 파트 구성이 달라 적용할 수 없습니다.")

    for part, candidate_part in zip(score.parts, candidate_score.parts):
        current = _part_measure(part, retry.measure, retry.local_number)
        candidate_measure = _part_measure(
            candidate_part,
            retry.measure,
            retry.local_number,
        )
        if Fraction(candidate_measure.highestTime) != Fraction(
            current.barDuration.quarterLength
        ):
            raise ValueError(
                "재인식 마디의 총박자가 기존 박자표와 달라 적용하지 않았습니다."
            )
        replacement = deepcopy(candidate_measure)
        replacement.number = current.number
        replacement.numberSuffix = current.numberSuffix
        offset = current.offset
        part.remove(current)
        part.insert(offset, replacement)

    correct_rhythm(score)
    _set_tempo(score, bpm)
    _force_piano(score)

    xml_backup = current_musicxml.with_name(
        f"{current_musicxml.stem}.before-retry.musicxml"
    )
    midi_backup = midi_output.with_name(f"{midi_output.stem}.before-retry.mid")
    if not xml_backup.exists():
        shutil.copy2(current_musicxml, xml_backup)
    if midi_output.exists() and not midi_backup.exists():
        shutil.copy2(midi_output, midi_backup)

    score.write("musicxml", fp=str(current_musicxml))
    playback, _ = prepare_playback_score(score)
    _force_piano(playback)
    playback.write("midi", fp=str(midi_output))

    report = midi_output.with_suffix(".txt")
    line = f"{retry.measure}마디: 400DPI 재인식 결과를 적용했습니다."
    try:
        details = report.read_text(encoding="utf-8")
    except FileNotFoundError:
        details = ""
    if line not in details:
        report.write_text(
            f"{details.rstrip()}\n{line}".lstrip(),
            encoding="utf-8",
        )
