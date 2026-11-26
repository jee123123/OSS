from copy import deepcopy
from fractions import Fraction

import pytest
from music21 import chord, clef, converter, harmony, meter, note, stream, tie

from score2midi.measure_edit import measure_pitches, rhythm_gaps, save_pitch_edit
from score2midi.measure_preview import render_measure_preview
from score2midi.measure_review import MeasureImageLayout
from score2midi.navigation import NavigationCommand, NavigationPlan


def test_edit_changes_one_chord_member_and_preserves_timing_and_repeats(tmp_path):
    original = stream.Score()
    part = stream.Part()
    first = stream.Measure(number=1)
    first.timeSignature = meter.TimeSignature("4/4")
    first.insert(0, harmony.ChordSymbol("C"))
    first.insert(1, chord.Chord(["C4", "G4"], quarterLength=3))
    second = stream.Measure(number=2)
    second.append(note.Note("E4", quarterLength=4))
    part.insert(0, first)
    part.insert(4, second)
    original.insert(0, part)
    draft = deepcopy(original)

    pitches = measure_pitches(draft, 1)
    assert [row.pitch_label for row in pitches] == ["C4", "G4"]
    pitches[0].shift(1)
    assert render_measure_preview(draft, 1, 120) != render_measure_preview(original, 1, 120)

    output = tmp_path / "edited.mid"
    plan = NavigationPlan(2, commands=(NavigationCommand(1, "dc"),))
    xml = save_pitch_edit(draft, output, 120, plan)

    saved = converter.parse(str(xml))
    saved_notes = measure_pitches(saved, 1)
    assert [row.member.pitch.midi for row in saved_notes] == [61, 67]
    assert [row.member.pitch.midi for row in measure_pitches(original, 1)] == [60, 67]
    assert {row.event.getOffsetInHierarchy(row.measure) for row in saved_notes} == {1}
    assert {row.event.quarterLength for row in saved_notes} == {3}
    assert measure_pitches(saved, 2)[0].pitch_label == "E4"
    midi = converter.parse(str(output))
    repeated = list(midi.recurse().getElementsByClass(chord.Chord))
    assert [frozenset(p.midi for p in item.pitches) for item in repeated] == [
        {61, 67}, {61, 67},
    ]

    before = xml.read_bytes(), output.read_bytes()
    with pytest.raises(FileExistsError):
        save_pitch_edit(draft, output, 92, plan)
    assert (xml.read_bytes(), output.read_bytes()) == before


def test_rejects_pitch_changes_that_would_break_a_tie_or_duplicate_a_chord_pitch():
    score = stream.Score()
    part = stream.Part()
    measure = stream.Measure(number=1)
    tied = note.Note("D4")
    tied.tie = tie.Tie("start")
    measure.append([tied, chord.Chord(["F4", "F#4"])])
    part.append(measure)
    score.insert(0, part)

    linked, lower, upper = measure_pitches(score, 1)
    with pytest.raises(ValueError, match="붙임줄"):
        linked.shift(1)
    with pytest.raises(ValueError, match="화음"):
        lower.shift(1)
    assert (linked.pitch_label, lower.pitch_label, upper.pitch_label) == ("D4", "F4", "F♯4")


def test_highlights_individual_chord_members_on_their_original_staff():
    score = stream.Score()
    for staff_clef, pitches in ((clef.TrebleClef(), ["E4", "F5"]), (clef.BassClef(), ["G2"])):
        part = stream.Part()
        bar = stream.Measure(number=1)
        bar.insert(0, staff_clef)
        event = chord.Chord(pitches)
        for member in event.notes:
            member.style.absoluteX = 30
        bar.insert(0, event)
        part.append(bar)
        score.insert(0, part)
    layout = MeasureImageLayout(10, 2, ((20, 30, 40, 50, 60), (100, 110, 120, 130, 140)))
    assert [
        row.original_image_position(layout) for row in measure_pitches(score, 1)
    ] == [(75, 60), (75, 20), (75, 140)]


def test_rhythm_edit_updates_a_whole_chord_and_preserves_later_bar_onsets(tmp_path):
    score = stream.Score()
    part = stream.Part()
    first = stream.Measure(number=1)
    first.timeSignature = meter.TimeSignature("4/4")
    first.insert(0, note.Note("C4", quarterLength=2))
    first.insert(2, chord.Chord(["E4", "G4"], quarterLength=2))
    second = stream.Measure(number=2)
    second.insert(0, note.Note("F4", quarterLength=4))
    part.insert(0, first)
    part.insert(4, second)
    score.insert(0, part)

    _, lower, upper = measure_pitches(score, 1)
    with pytest.raises(ValueError, match="벗어납니다"):
        upper.change_timing(Fraction(7, 2), Fraction(1))
    lower.change_timing(Fraction(3, 2), Fraction(1, 2))
    with pytest.raises(ValueError, match="겹칩니다"):
        rhythm_gaps([lower])
    lower.change_timing(Fraction(5, 2), Fraction(1, 2))
    assert upper.timing == lower.timing

    output = tmp_path / "rhythm.mid"
    plan = NavigationPlan(2, commands=(NavigationCommand(1, "dc"),))
    xml = save_pitch_edit(score, output, 120, plan, timing_edits=[lower])
    assert not first.getElementsByClass(note.Rest)
    saved = converter.parse(str(xml))
    rests = saved.parts[0].measure(1).getElementsByClass(note.Rest)
    assert [(rest.offset, rest.quarterLength) for rest in rests] == [(2, 0.5), (3, 1)]
    midi = converter.parse(str(output))
    assert [
        event.getOffsetInHierarchy(midi) for event in midi.recurse().notes
        if isinstance(event, note.Note) and event.pitch.name == "F"
    ] == [4, 12]
