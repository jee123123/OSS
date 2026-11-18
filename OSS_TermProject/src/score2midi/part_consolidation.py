from __future__ import annotations

from copy import deepcopy
from statistics import median

from music21 import instrument, stream


def _measure_key(measure: stream.Measure) -> tuple[str, str]:
    return str(measure.number), str(measure.numberSuffix or "")


def _occupied_measure_keys(part: stream.Part) -> set[tuple[str, str]]:
    return {
        _measure_key(measure)
        for measure in part.getElementsByClass(stream.Measure)
        if any(measure.recurse().notes)
    }


def _median_pitch(part: stream.Part) -> float | None:
    pitches: list[float] = []
    for element in part.recurse().notes:
        if hasattr(element, "pitches"):
            pitches.extend(float(pitch.midi) for pitch in element.pitches)
        elif hasattr(element, "pitch"):
            pitches.append(float(element.pitch.midi))
    return median(pitches) if pitches else None


def _groups_are_complementary(parts: list[stream.Part]) -> bool:
    occupied = [_occupied_measure_keys(part) for part in parts]
    combined: set[tuple[str, str]] = set()

    for keys in sorted(occupied, key=len, reverse=True):
        if not keys:
            continue
        overlap = len(keys & combined) / max(1, len(keys))
        if overlap > 0.25:
            return False
        combined.update(keys)

    return bool(combined)


def _merge_parts(parts: list[stream.Part], part_id: str) -> stream.Part:
    base = max(parts, key=lambda part: len(_occupied_measure_keys(part)))
    merged = deepcopy(base)
    merged.id = part_id
    merged.partName = "Piano"

    source_measures = [
        {
            _measure_key(measure): measure
            for measure in part.getElementsByClass(stream.Measure)
        }
        for part in parts
    ]

    for current in list(merged.getElementsByClass(stream.Measure)):
        key = _measure_key(current)
        candidates = [
            measures[key]
            for measures in source_measures
            if key in measures
        ]
        nonempty = [
            measure for measure in candidates if any(measure.recurse().notes)
        ]
        if not nonempty:
            continue

        replacement = max(
            nonempty,
            key=lambda measure: sum(1 for _ in measure.recurse().notes),
        )
        if replacement is current:
            continue

        offset = current.offset
        merged.remove(current)
        merged.insert(offset, deepcopy(replacement))

    for detected in list(
        merged.recurse().getElementsByClass(instrument.Instrument)
    ):
        detected.midiProgram = 0
    if not merged.recurse().getElementsByClass(instrument.Instrument):
        merged.insert(0, instrument.Piano())

    return merged


def consolidate_split_piano_parts(
    score: stream.Score,
) -> tuple[dict[int, int], list[str]]:
    parts = list(score.parts)
    if len(parts) < 4 or len(parts) % 2:
        return {index: index for index in range(1, len(parts) + 1)}, []

    pitched = [(_median_pitch(part), index, part) for index, part in enumerate(parts)]
    if any(pitch is None for pitch, _, _ in pitched):
        return {index: index for index in range(1, len(parts) + 1)}, []

    pitched.sort(key=lambda item: item[0])
    split = len(pitched) // 2
    low = [part for _, _, part in pitched[:split]]
    high = [part for _, _, part in pitched[split:]]

    if not _groups_are_complementary(high) or not _groups_are_complementary(low):
        return {index: index for index in range(1, len(parts) + 1)}, []

    mapping: dict[int, int] = {}
    for _, original_index, part in pitched:
        mapping[original_index + 1] = 1 if part in high else 2

    merged_high = _merge_parts(high, "Piano-Right")
    merged_low = _merge_parts(low, "Piano-Left")

    for part in parts:
        score.remove(part)
    score.insert(0, merged_high)
    score.insert(0, merged_low)

    return mapping, [
        f"Audiveris가 나눈 {len(parts)}개 파트를 "
        "음역과 마디 연속성을 기준으로 오른손·왼손 2개 파트로 병합했습니다."
    ]
