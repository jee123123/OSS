from pathlib import Path
from tempfile import TemporaryDirectory

from music21 import harmony, note, repeat, stream

from score2midi.navigation import (
    NavigationCommand,
    NavigationPlan,
    expand_navigation,
    prepare_playback_score,
    resolve_playback_order,
)


def test_resolves_dal_segno_al_coda() -> None:
    plan = NavigationPlan(
        measure_count=8,
        segnos=(1,),
        coda_marks=(4, 7),
        commands=(NavigationCommand(5, "ds_al_coda"),),
    )

    order, warnings = resolve_playback_order(plan)

    assert order == [0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 7]
    assert warnings == []


def test_resolves_dal_segno_al_fine() -> None:
    plan = NavigationPlan(
        measure_count=7,
        segnos=(1,),
        fines=(3,),
        commands=(NavigationCommand(5, "ds_al_fine"),),
    )

    order, warnings = resolve_playback_order(plan)

    assert order == [0, 1, 2, 3, 4, 5, 1, 2, 3]
    assert warnings == []


def test_resolves_double_dal_segno() -> None:
    plan = NavigationPlan(
        measure_count=7,
        segnos=(1,),
        double_segnos=(2,),
        commands=(NavigationCommand(5, "dss"),),
    )

    order, warnings = resolve_playback_order(plan)

    assert order == [0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6]
    assert warnings == []


def test_resolves_two_stage_ds_and_dss_coda_route() -> None:
    plan = NavigationPlan(
        measure_count=12,
        segnos=(1,),
        double_segnos=(5,),
        coda_jumps=(3,),
        coda_targets=(7,),
        double_coda_jumps=(8,),
        double_coda_targets=(10,),
        commands=(
            NavigationCommand(6, "ds"),
            NavigationCommand(9, "dss"),
        ),
    )

    order, warnings = resolve_playback_order(plan)

    assert order == [
        0, 1, 2, 3, 4, 5, 6,
        1, 2, 3, 7, 8, 9,
        5, 6, 7, 8, 10, 11,
    ]
    assert warnings == []


def test_missing_segno_keeps_linear_order() -> None:
    plan = NavigationPlan(
        measure_count=4,
        commands=(NavigationCommand(2, "ds"),),
    )

    order, warnings = resolve_playback_order(plan)

    assert order == [0, 1, 2, 3]
    assert len(warnings) == 1


def test_invalid_navigation_can_still_be_written_to_midi() -> None:
    score = stream.Score()
    part = stream.Part()
    for number in range(1, 5):
        measure = stream.Measure(number=number)
        measure.append(note.Note("C4", quarterLength=1))
        part.append(measure)
    part.measure(3).insert(0, repeat.DalSegno())
    score.append(part)

    expanded, warnings = expand_navigation(score)

    assert warnings
    assert not list(
        expanded.recurse().getElementsByClass(repeat.RepeatExpression)
    )
    with TemporaryDirectory() as directory:
        output = Path(directory) / "fallback.mid"
        expanded.write("midi", fp=str(output))
        assert output.stat().st_size > 0


def test_chord_names_are_not_played_as_hanging_midi_notes() -> None:
    score = stream.Score()
    part = stream.Part()
    measure = stream.Measure(number=1)
    measure.insert(0, harmony.ChordSymbol("Dm"))
    measure.insert(0, note.Note("F4", quarterLength=1))
    part.append(measure)
    score.append(part)

    playback, warnings = prepare_playback_score(score)

    assert list(score.recurse().getElementsByClass(harmony.ChordSymbol))
    assert not list(
        playback.recurse().getElementsByClass(harmony.ChordSymbol)
    )
    assert warnings[-1] == "코드 이름 1개를 MIDI 연주에서 제외했습니다."
    with TemporaryDirectory() as directory:
        output = Path(directory) / "no-hanging-chord.mid"
        playback.write("midi", fp=str(output))
        assert output.stat().st_size > 0


def test_expands_all_parts_with_one_navigation_route() -> None:
    score = stream.Score()
    for octave in (5, 3):
        part = stream.Part()
        for number in range(1, 9):
            measure = stream.Measure(number=number)
            measure.append(note.Note(f"C{octave}", quarterLength=1))
            part.append(measure)
        score.append(part)

    source = score.parts[0]
    source.measure(2).insert(0, repeat.Segno())
    source.measure(5).insert(0, repeat.Coda())
    source.measure(6).insert(0, repeat.DalSegnoAlCoda())
    source.measure(8).insert(0, repeat.Coda())

    expanded, warnings = expand_navigation(score)

    assert [len(list(part.getElementsByClass(stream.Measure))) for part in expanded.parts] == [11, 11]
    assert len(list(score.parts[0].getElementsByClass(stream.Measure))) == 8
    assert warnings[-1].endswith("11마디의 재생 순서로 전개했습니다.")
    assert not list(
        expanded.recurse().getElementsByClass(repeat.RepeatExpression)
    )
    with TemporaryDirectory() as directory:
        output = Path(directory) / "navigation.mid"
        expanded.write("midi", fp=str(output))
        assert output.stat().st_size > 0
