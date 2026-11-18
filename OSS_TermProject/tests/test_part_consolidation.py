from music21 import note, stream

from score2midi.part_consolidation import consolidate_split_piano_parts


def _part(pitch: str, occupied: set[int]) -> stream.Part:
    part = stream.Part()
    for number in range(1, 7):
        measure = stream.Measure(number=number)
        if number in occupied:
            measure.append(note.Note(pitch, quarterLength=1))
        else:
            measure.append(note.Rest(quarterLength=1))
        part.append(measure)
    return part


def test_consolidates_complementary_split_piano_parts() -> None:
    score = stream.Score()
    score.append(_part("C5", {1, 2, 5}))
    score.append(_part("C3", {1, 2, 5}))
    score.append(_part("D5", {3, 4, 6}))
    score.append(_part("D3", {3, 4, 6}))

    mapping, warnings = consolidate_split_piano_parts(score)

    assert len(score.parts) == 2
    assert mapping == {1: 1, 2: 2, 3: 1, 4: 2}
    assert warnings
    assert [
        measure.number
        for measure in score.parts[0].getElementsByClass(stream.Measure)
        if any(measure.recurse().notes)
    ] == [1, 2, 3, 4, 5, 6]


def test_keeps_simultaneous_four_part_score() -> None:
    score = stream.Score()
    for pitch in ("C6", "C5", "C4", "C3"):
        score.append(_part(pitch, {1, 2, 3, 4}))

    mapping, warnings = consolidate_split_piano_parts(score)

    assert len(score.parts) == 4
    assert mapping == {1: 1, 2: 2, 3: 3, 4: 4}
    assert warnings == []
