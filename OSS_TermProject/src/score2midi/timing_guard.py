"""Conservative safeguards for collapsed and trailing note onsets."""

from collections import defaultdict
from fractions import Fraction

from music21 import note, stream

MIN_DISTINCT_X = 1.0
MAX_RECOVERABLE_NOTE = Fraction(1)


def _containers(measure: stream.Measure) -> list[stream.Stream]:
    voices = list(measure.getElementsByClass(stream.Voice))
    return voices or [measure]


def _x_position(event) -> float | None:
    value = event.style.absoluteX
    return float(value) if value is not None else None


def _repair_collapsed_beam_group(
    container: stream.Stream,
    bar_duration: Fraction,
) -> int:
    events = list(container.notesAndRests)
    by_offset: dict[Fraction, list] = defaultdict(list)

    for event in events:
        by_offset[Fraction(event.offset)].append(event)

    repaired = 0
    offsets = sorted(by_offset)

    for index, start in enumerate(offsets):
        group = by_offset[start]

        if len(group) < 2 or not all(
            isinstance(event, note.Note)
            and bool(event.beams)
            and Fraction(event.duration.quarterLength) <= Fraction(1, 2)
            and _x_position(event) is not None
            for event in group
        ):
            continue

        ordered = sorted(group, key=_x_position)
        positions = [_x_position(event) for event in ordered]

        if any(
            right - left <= MIN_DISTINCT_X
            for left, right in zip(positions, positions[1:])
        ):
            continue

        next_offset = (
            offsets[index + 1]
            if index + 1 < len(offsets)
            else bar_duration
        )
        total_duration = sum(
            (
                Fraction(event.duration.quarterLength)
                for event in ordered
            ),
            Fraction(0),
        )

        if start + total_duration > next_offset:
            continue

        cursor = start
        for event in ordered:
            event.offset = cursor
            cursor += Fraction(event.duration.quarterLength)

        repaired += len(ordered) - 1

    return repaired


def _restore_trailing_note(
    container: stream.Stream,
    bar_duration: Fraction,
) -> int:
    sounding = [
        event
        for event in container.notes
        if _x_position(event) is not None
    ]

    if not sounding:
        return 0

    trailing = max(sounding, key=_x_position)
    start = Fraction(trailing.offset)
    duration = Fraction(trailing.duration.quarterLength)

    if start < bar_duration or not 0 < duration <= MAX_RECOVERABLE_NOTE:
        return 0

    target = bar_duration - duration
    if target < 0:
        return 0

    previous_end = max(
        (
            Fraction(event.offset)
            + Fraction(event.duration.quarterLength)
            for event in container.notesAndRests
            if event is not trailing
            and _x_position(event) is not None
            and _x_position(event) < _x_position(trailing)
        ),
        default=Fraction(0),
    )

    if previous_end > target:
        return 0

    trailing.offset = target
    return 1


def stabilize_note_timing(score: stream.Score) -> list[str]:
    collapsed_count = 0
    trailing_count = 0

    for measure in score.recurse().getElementsByClass(stream.Measure):
        bar_duration = Fraction(measure.barDuration.quarterLength)

        for container in _containers(measure):
            collapsed_count += _repair_collapsed_beam_group(
                container,
                bar_duration,
            )
            trailing_count += _restore_trailing_note(
                container,
                bar_duration,
            )

    warnings = []
    if collapsed_count:
        warnings.append(
            f"같은 시각으로 겹친 빔 음표 {collapsed_count}개를 "
            "가로 위치 순서대로 복원했습니다."
        )
    if trailing_count:
        warnings.append(
            f"마디 밖으로 밀린 마지막 음표 {trailing_count}개를 "
            "마디 안으로 복원했습니다."
        )
    return warnings
