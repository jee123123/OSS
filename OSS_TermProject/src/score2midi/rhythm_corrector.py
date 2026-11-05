"""Synchronize recognized MusicXML parts on one measure timeline."""

from collections import Counter
from fractions import Fraction

from music21 import stream

GRID_DIVISORS = (4, 3, 8)
SNAP_TOLERANCE = Fraction(1, 32)


def _snap_to_grid(value) -> Fraction:
    """Snap only very small recognition errors to common note grids."""
    original = Fraction(value)

    candidates = [
        Fraction(round(float(original) * divisor), divisor)
        for divisor in GRID_DIVISORS
    ]
    nearest = min(
        candidates,
        key=lambda candidate: abs(candidate - original),
    )

    if abs(nearest - original) <= SNAP_TOLERANCE:
        return nearest

    return original


def _snap_measure_contents(measure: stream.Measure) -> None:
    """Correct tiny offset and duration errors without rewriting notes."""
    voices = list(measure.getElementsByClass(stream.Voice))
    containers = voices or [measure]

    for container in containers:
        for element in container.notesAndRests:
            element.offset = _snap_to_grid(element.offset)

            if element.duration.quarterLength > 0:
                element.duration.quarterLength = _snap_to_grid(
                    element.duration.quarterLength
                )


def _choose_measure_duration(
    measures: list[stream.Measure],
    measure_index: int,
    warnings: list[str],
) -> Fraction:
    """Choose one shared duration for the same measure in every part."""
    bar_durations = [
        Fraction(measure.barDuration.quarterLength)
        for measure in measures
    ]

    counts = Counter(bar_durations)
    duration = max(
        counts,
        key=lambda value: (counts[value], value),
    )

    if len(counts) > 1:
        values = ", ".join(
            f"{float(value):g}박"
            for value in sorted(counts)
        )
        warnings.append(
            f"{measures[0].number}마디: "
            f"파트별 박자표 길이가 다릅니다({values}). "
            f"{float(duration):g}박으로 통일했습니다."
        )

    # Anacrusis/pickup measure is used only when every part agrees.
    if measure_index == 0:
        recognized_durations = [
            bar_duration - Fraction(measure.paddingLeft or 0)
            for bar_duration, measure in zip(
                bar_durations,
                measures,
            )
        ]

        if len(set(recognized_durations)) == 1:
            duration = recognized_durations[0]
        elif any(measure.paddingLeft for measure in measures):
            values = ", ".join(
                f"{float(value):g}박"
                for value in recognized_durations
            )
            warnings.append(
                f"첫 마디의 파트별 길이가 다릅니다({values}). "
                f"{float(duration):g}박으로 통일했습니다."
            )

    return duration


def correct_rhythm(score: stream.Score) -> list[str]:
    """Place every part's measures on one shared timeline."""
    warnings: list[str] = []
    parts = list(score.parts)

    part_measures = [
        list(part.getElementsByClass(stream.Measure))
        for part in parts
    ]

    measure_count = max(
        (len(measures) for measures in part_measures),
        default=0,
    )

    timeline = Fraction(0)

    for measure_index in range(measure_count):
        current_measures = [
            measures[measure_index]
            for measures in part_measures
            if measure_index < len(measures)
        ]

        if not current_measures:
            continue

        expected_duration = _choose_measure_duration(
            current_measures,
            measure_index,
            warnings,
        )

        for part_number, (part, measures) in enumerate(
            zip(parts, part_measures),
            start=1,
        ):
            if measure_index >= len(measures):
                continue

            measure = measures[measure_index]
            _snap_measure_contents(measure)

            actual_duration = Fraction(measure.highestTime)

            if actual_duration > expected_duration:
                overflow = actual_duration - expected_duration
                warnings.append(
                    f"{measure.number}마디 파트 {part_number}: "
                    f"인식된 내용이 {float(overflow):g}박 깁니다. "
                    "다음 마디 시작점은 공통 위치로 고정했습니다."
                )

            # The essential fix: both hands use exactly the same start.
            part.setElementOffset(measure, timeline)

        timeline += expected_duration

    return warnings
