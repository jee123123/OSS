"""Recover short notes from independent OMR results."""

from copy import deepcopy
from fractions import Fraction

from music21 import stream

SHORT_NOTE_LIMIT = Fraction(1, 2)
MIN_SHORT_NOTE_GAIN = 1
MAX_EXTRA_RHYTHM_ERROR = Fraction(1, 2)


def _short_note_count(measure: stream.Measure) -> int:
    return sum(
        Fraction(element.duration.quarterLength) <= SHORT_NOTE_LIMIT
        for element in measure.recurse().notes
    )


def _note_event_count(measure: stream.Measure) -> int:
    return sum(1 for _ in measure.recurse().notes)


def _measure_rhythm_error(measure: stream.Measure) -> Fraction:
    expected = Fraction(measure.barDuration.quarterLength)
    actual = Fraction(measure.highestTime)
    return abs(actual - expected)


def recover_missing_short_notes(
    primary_score: stream.Score,
    alternative_score: stream.Score,
) -> list[str]:
    warnings: list[str] = []

    for part_number, (primary_part, alternative_part) in enumerate(
        zip(primary_score.parts, alternative_score.parts),
        start=1,
    ):
        primary_measures = list(
            primary_part.getElementsByClass(stream.Measure)
        )
        alternative_measures = list(
            alternative_part.getElementsByClass(stream.Measure)
        )

        for primary_measure, alternative_measure in zip(
            primary_measures,
            alternative_measures,
        ):
            recovered_count = (
                _short_note_count(alternative_measure)
                - _short_note_count(primary_measure)
            )

            if (
                recovered_count < MIN_SHORT_NOTE_GAIN
                or _note_event_count(alternative_measure)
                <= _note_event_count(primary_measure)
            ):
                continue

            primary_error = _measure_rhythm_error(primary_measure)
            alternative_error = _measure_rhythm_error(
                alternative_measure
            )
            allowed_extra_error = (
                Fraction(0)
                if recovered_count == 1
                else MAX_EXTRA_RHYTHM_ERROR
            )

            if alternative_error > primary_error + allowed_extra_error:
                continue

            measure_offset = primary_measure.offset
            replacement = deepcopy(alternative_measure)

            primary_part.remove(primary_measure)
            primary_part.insert(measure_offset, replacement)

            warnings.append(
                f"{primary_measure.number}마디 파트 {part_number}: "
                f"보조 인식 결과에서 짧은 음표 "
                f"{recovered_count}개를 복구했습니다."
            )

        for alternative_measure in alternative_measures[
            len(primary_measures):
        ]:
            if _note_event_count(alternative_measure) == 0:
                continue

            primary_part.append(deepcopy(alternative_measure))
            warnings.append(
                f"{alternative_measure.number}마디 파트 {part_number}: "
                "보조 인식 결과에서 누락된 마지막 마디를 복구했습니다."
            )

    return warnings
