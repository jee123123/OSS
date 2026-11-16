"""Recover short notes from independent OMR results."""

from copy import deepcopy
from fractions import Fraction

from music21 import stream

SHORT_NOTE_LIMIT = Fraction(2, 3)
MIN_SHORT_NOTE_GAIN = 1


def _short_note_count(measure: stream.Measure) -> int:
    return sum(
        Fraction(element.duration.quarterLength) <= SHORT_NOTE_LIMIT
        for element in measure.recurse().notes
    )


def _note_event_count(measure: stream.Measure) -> int:
    return sum(1 for _ in measure.recurse().notes)


def _measure_rhythm_error(measure: stream.Measure) -> Fraction:
    expected = Fraction(measure.barDuration.quarterLength) - Fraction(
        measure.paddingLeft or 0
    )
    actual = Fraction(measure.highestTime)
    return abs(actual - expected)


def _measure_key(measure: stream.Measure) -> tuple[str, str]:
    return str(measure.number), str(measure.numberSuffix or "")


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
        alternative_by_key = {
            _measure_key(measure): measure
            for measure in alternative_measures
        }

        for primary_measure in primary_measures:
            alternative_measure = alternative_by_key.get(
                _measure_key(primary_measure)
            )
            if alternative_measure is None:
                continue

            primary_count = _note_event_count(primary_measure)
            alternative_count = _note_event_count(alternative_measure)
            recovered_count = (
                _short_note_count(alternative_measure)
                - _short_note_count(primary_measure)
            )

            if (
                recovered_count < MIN_SHORT_NOTE_GAIN
                or alternative_count < primary_count
            ):
                continue

            primary_error = _measure_rhythm_error(primary_measure)
            alternative_error = _measure_rhythm_error(
                alternative_measure
            )

            if alternative_error != 0:
                continue

            if (
                alternative_count == primary_count
                and alternative_error >= primary_error
            ):
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

        primary_keys = {_measure_key(measure) for measure in primary_measures}
        last_primary_key = (
            _measure_key(primary_measures[-1])
            if primary_measures
            else None
        )
        last_primary_index = next(
            (
                index
                for index, measure in enumerate(alternative_measures)
                if _measure_key(measure) == last_primary_key
            ),
            len(alternative_measures),
        )

        for alternative_measure in alternative_measures[
            last_primary_index + 1:
        ]:
            if _measure_key(alternative_measure) in primary_keys:
                continue
            if _note_event_count(alternative_measure) == 0:
                continue

            primary_part.append(deepcopy(alternative_measure))
            warnings.append(
                f"{alternative_measure.number}마디 파트 {part_number}: "
                "보조 인식 결과에서 누락된 마지막 마디를 복구했습니다."
            )

    return warnings
