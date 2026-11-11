"""Synchronize recognized MusicXML parts on one measure timeline."""

from collections import Counter
from fractions import Fraction

from music21 import stream

GRID_DIVISORS = (4, 3, 8)
SNAP_TOLERANCE = Fraction(1, 32)
ALLOWED_DURATIONS = frozenset(
    {
        Fraction(1, 8),
        Fraction(1, 6),
        Fraction(1, 4),
        Fraction(1, 3),
        Fraction(3, 8),
        Fraction(1, 2),
        Fraction(2, 3),
        Fraction(3, 4),
        Fraction(1),
        Fraction(3, 2),
        Fraction(2),
        Fraction(3),
        Fraction(4),
    }
)


def _snap_to_grid(value) -> Fraction:
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
    voices = list(measure.getElementsByClass(stream.Voice))
    containers = voices or [measure]

    for container in containers:
        for element in container.notesAndRests:
            element.offset = _snap_to_grid(element.offset)

            if element.duration.quarterLength > 0:
                element.duration.quarterLength = _snap_to_grid(
                    element.duration.quarterLength
                )


def _contains_tuplets(measure: stream.Measure) -> bool:
    return any(
        element.duration.tuplets
        for element in measure.recurse().notesAndRests
    )


def _fit_overfull_container(
    container: stream.Stream,
    expected_duration: Fraction,
) -> bool:
    events = list(container.notesAndRests)
    actual_duration = Fraction(container.highestTime)
    overflow = actual_duration - expected_duration

    if overflow <= 0:
        return True

    def reduced_duration(element) -> Fraction:
        return Fraction(element.duration.quarterLength) - overflow

    dotted_candidates = [
        element
        for element in events
        if element.duration.dots
        and reduced_duration(element) in ALLOWED_DURATIONS
        and reduced_duration(element)
        == Fraction(element.duration.quarterLength) * Fraction(2, 3)
    ]
    candidate = dotted_candidates[-1] if dotted_candidates else None

    if candidate is None:
        first_outside = next(
            (
                index
                for index, element in enumerate(events)
                if Fraction(element.offset) >= expected_duration
            ),
            len(events),
        )
        candidates = events[:first_outside]
        candidate = next(
            (
                element
                for element in reversed(candidates)
                if reduced_duration(element) in ALLOWED_DURATIONS
            ),
            None,
        )

    if candidate is None:
        return False

    old_end = (
        Fraction(candidate.offset)
        + Fraction(candidate.duration.quarterLength)
    )
    candidate.duration.quarterLength = reduced_duration(candidate)

    for element in events:
        if element is candidate:
            continue

        if Fraction(element.offset) >= old_end:
            element.offset = Fraction(element.offset) - overflow

    return True


def _choose_measure_duration(
    measures: list[stream.Measure],
    measure_index: int,
    warnings: list[str],
) -> Fraction:
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
            if _contains_tuplets(measure):
                warnings.append(
                    f"{measure.number}마디 파트 {part_number}: "
                    "셋잇단음표가 있어 자동 리듬 보정을 건너뛰었습니다."
                )
                part.setElementOffset(measure, timeline)
                continue

            _snap_measure_contents(measure)
            voices = list(measure.getElementsByClass(stream.Voice))
            containers = voices or [measure]

            for voice_number, container in enumerate(containers, start=1):
                actual_duration = Fraction(container.highestTime)

                if actual_duration <= expected_duration:
                    continue

                overflow = actual_duration - expected_duration

                if _fit_overfull_container(
                    container,
                    expected_duration,
                ):
                    warnings.append(
                        f"{measure.number}마디 파트 {part_number} "
                        f"보이스 {voice_number}: "
                        f"{float(overflow):g}박을 자동 보정했습니다."
                    )
                else:
                    warnings.append(
                        f"{measure.number}마디 파트 {part_number} "
                        f"보이스 {voice_number}: "
                        f"{float(overflow):g}박이 초과되지만 "
                        "안전하게 보정할 수 없어 유지했습니다."
                    )

            part.setElementOffset(measure, timeline)

        timeline += expected_duration

    return warnings
