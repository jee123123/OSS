"""Edit a recognized note and export a separate score."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory

from music21 import beam, chord, clef, defaults, harmony, note, stream

from score2midi.cli import _force_piano, _set_tempo
from score2midi.measure_review import MeasureImageLayout
from score2midi.navigation import NavigationPlan, prepare_playback_score


@dataclass(frozen=True)
class MeasurePitch:
    part_index: int
    measure: stream.Measure
    event: note.Note | chord.Chord
    member: note.Note
    voice: stream.Stream

    @property
    def pitch_label(self) -> str:
        return self.member.pitch.nameWithOctave.replace("-", "♭").replace("#", "♯")

    def shift(self, semitones: int) -> None:
        if self.member.tie is not None:
            raise ValueError("붙임줄로 연결된 음은 이 창에서 바꿀 수 없습니다.")
        target = self.member.pitch.transpose(semitones)
        if not 0 <= target.ps <= 127:
            raise ValueError("MIDI 음높이 범위를 벗어났습니다.")
        if isinstance(self.event, chord.Chord) and any(
            other is not self.member and other.pitch.midi == target.midi
            for other in self.event.notes
        ):
            raise ValueError("화음에 같은 높이의 음이 이미 있습니다.")
        self.member.pitch = target

    @property
    def timing(self) -> tuple[Fraction, Fraction]:
        return (
            Fraction(self.event.getOffsetInHierarchy(self.measure)),
            Fraction(self.event.quarterLength),
        )

    @property
    def voice_offset(self) -> Fraction:
        if self.voice is self.measure:
            return Fraction(0)
        return Fraction(self.voice.getOffsetBySite(self.measure))

    @property
    def bar_length(self) -> Fraction:
        return Fraction(
            self.measure.barDuration.quarterLength
            - self.measure.paddingLeft
            - self.measure.paddingRight
        )

    def change_timing(self, start: Fraction, length: Fraction) -> None:
        if self.event.duration.isGrace:
            raise ValueError("꾸밈음의 박자는 이 창에서 바꿀 수 없습니다.")
        if start < self.voice_offset or length <= 0:
            raise ValueError(f"시작 위치는 {self.voice_offset} 이상, 길이는 0보다 커야 합니다.")
        if start + length > self.bar_length:
            raise ValueError(f"음표가 {self.bar_length}박인 마디를 벗어납니다.")
        if any(
            value != old and defaults.divisionsPerQuarter % value.denominator
            for value, old in zip((start, length), self.timing)
        ):
            raise ValueError("저장할 수 있는 박자 값을 입력해 주세요. 셋잇단음표는 1/3처럼 입력할 수 있습니다.")
        self.voice.setElementOffset(self.event, start - self.voice_offset)
        self.event.quarterLength = length

    def original_image_position(
        self, layout: MeasureImageLayout,
    ) -> tuple[float, float] | None:
        x = self.member.style.absoluteX
        if x is None:
            x = self.event.style.absoluteX
        staff = self.part_index - 1
        if x is None or staff >= len(layout.staves):
            return None
        y = self.member.style.absoluteY
        if y is not None:
            return layout.note_position(staff, float(x), 0, staff_y=float(y))
        active_clef = self.event.getContextByClass(clef.Clef)
        if active_clef is None or active_clef.lowestLine is None:
            return None
        steps = self.member.pitch.diatonicNoteNum - active_clef.lowestLine
        return layout.note_position(staff, float(x), steps)


def measure_pitches(score: stream.Score, number: int) -> list[MeasurePitch]:
    pitches = []
    for part_index, part in enumerate(score.parts, 1):
        measure = part.measure(number)
        if measure is None:
            continue
        for voice in (measure, *measure.getElementsByClass(stream.Voice)):
            for event in voice.notes:
                if isinstance(event, harmony.Harmony):
                    continue
                if isinstance(event, note.Note):
                    pitches.append(MeasurePitch(part_index, measure, event, event, voice))
                elif isinstance(event, chord.Chord):
                    pitches.extend(
                        MeasurePitch(part_index, measure, event, member, voice)
                        for member in event.notes
                    )
    return pitches


def _neighbor_tied_pitches(entry: MeasurePitch, *, previous: bool) -> set[int]:
    """Only treat ties at a matching adjacent bar boundary as cross-bar ties."""
    part = entry.measure.getContextByClass(stream.Part)
    if part is None:
        return set()
    measures = list(part.getElementsByClass(stream.Measure))
    index = measures.index(entry.measure) + (-1 if previous else 1)
    if not 0 <= index < len(measures):
        return set()
    measure = measures[index]
    voice = measure
    if isinstance(entry.voice, stream.Voice):
        voice = next((item for item in measure.voices if item.id == entry.voice.id), measure)
    expected_ties = {"start", "continue"} if previous else {"stop", "continue"}
    boundary = (
        measure.barDuration.quarterLength - measure.paddingLeft - measure.paddingRight
        if previous else 0
    )
    result = set()
    for event in voice.notes:
        position = event.getOffsetInHierarchy(measure)
        if previous:
            position += event.quarterLength
        if position != boundary or isinstance(event, harmony.Harmony):
            continue
        members = event.notes if isinstance(event, chord.Chord) else (event,)
        result.update(
            member.pitch.midi for member in members
            if isinstance(member, note.Note)
            and member.tie is not None and member.tie.type in expected_ties
        )
    return result


def rhythm_gaps(
    edited: list[MeasurePitch],
) -> list[tuple[stream.Stream, list[tuple[Fraction, Fraction]]]]:
    """Validate the edited voices and compute their remaining rests."""
    voices = {id(entry.voice): entry for entry in edited}
    changed_events = {id(entry.event) for entry in edited}
    result = []
    for entry in voices.values():
        tied_from_previous = _neighbor_tied_pitches(entry, previous=True)
        tied_to_next = _neighbor_tied_pitches(entry, previous=False)
        end = entry.voice_offset
        gaps = []
        ties: dict[int, tuple[Fraction, bool]] = {}
        for event in entry.voice.notes:
            if isinstance(event, harmony.Harmony) or event.quarterLength <= 0:
                continue
            start = Fraction(event.getOffsetBySite(entry.voice)) + entry.voice_offset
            stop = start + Fraction(event.quarterLength)
            if start < end:
                raise ValueError(f"파트 {entry.part_index}의 같은 보이스에서 {start}박에 음표가 겹칩니다.")
            if stop > entry.bar_length:
                raise ValueError(
                    f"파트 {entry.part_index}의 같은 보이스가 {entry.bar_length}박을 초과합니다."
                )
            if start > end:
                gaps.append((end - entry.voice_offset, start - end))
            end = stop

            members = event.notes if isinstance(event, chord.Chord) else (event,)
            for member in members:
                if not isinstance(member, note.Note) or member.tie is None:
                    continue
                previous = ties.get(member.pitch.midi)
                changed = id(event) in changed_events
                if member.tie.type in {"stop", "continue"}:
                    if previous is not None and previous[0] != start:
                        raise ValueError(f"파트 {entry.part_index}의 붙임줄 사이에 빈 박자가 있습니다.")
                    if (
                        previous is None and changed and start != 0
                        and member.pitch.midi in tied_from_previous
                    ):
                        raise ValueError("앞 마디에서 이어지는 음은 마디 시작에 있어야 합니다.")
                if member.tie.type == "stop":
                    ties.pop(member.pitch.midi, None)
                else:
                    ties[member.pitch.midi] = (stop, changed)
        if any(
            changed and stop != entry.bar_length and pitch in tied_to_next
            for pitch, (stop, changed) in ties.items()
        ):
            raise ValueError("다음 마디로 이어지는 음은 현재 마디 끝까지 길이가 이어져야 합니다.")
        if end < entry.bar_length:
            gaps.append((end - entry.voice_offset, entry.bar_length - end))
        result.append((entry.voice, gaps))
    return result


def apply_rhythm_gaps(
    plans: list[tuple[stream.Stream, list[tuple[Fraction, Fraction]]]],
) -> None:
    for voice, gaps in plans:
        rests = {
            (Fraction(rest.getOffsetBySite(voice)), Fraction(rest.quarterLength)): rest
            for rest in voice.getElementsByClass(note.Rest)
        }
        for rest in list(voice.getElementsByClass(note.Rest)):
            if not any(rest is rests.get(gap) for gap in gaps):
                voice.remove(rest)
        for start, length in gaps:
            if (start, length) not in rests:
                voice.insert(start, note.Rest(quarterLength=length))
        # The original beam groups may no longer match the edited durations.
        for event in voice.notes:
            if isinstance(event, (note.Note, chord.Chord)) and not isinstance(event, harmony.Harmony):
                event.beams = beam.Beams()


def save_pitch_edit(
    draft: stream.Score,
    midi_output: Path,
    bpm: int,
    navigation_plan: NavigationPlan | None,
    *,
    timing_edits: list[MeasurePitch] | None = None,
) -> Path:
    if midi_output.suffix.lower() != ".mid":
        raise ValueError("저장할 파일 이름은 .mid로 끝나야 합니다.")
    xml_output = midi_output.with_suffix(".musicxml")
    for destination in (midi_output, xml_output):
        if destination.exists():
            raise FileExistsError(f"이미 있는 파일입니다. 새 이름을 선택하세요.\n{destination}")

    plans = rhythm_gaps(timing_edits or [])
    memo = {}
    edit = deepcopy(draft, memo)
    apply_rhythm_gaps([(memo[id(voice)], gaps) for voice, gaps in plans])
    _set_tempo(edit, bpm)
    _force_piano(edit)
    playback, _ = prepare_playback_score(edit, navigation_plan)
    _force_piano(playback)

    with TemporaryDirectory(prefix=".score2midi-edit-", dir=midi_output.parent) as directory:
        staged_midi = Path(directory) / "score.mid"
        staged_xml = Path(directory) / "score.musicxml"
        edit.write("musicxml", fp=str(staged_xml))
        playback.write("midi", fp=str(staged_midi))
        created = []
        try:
            for source, destination in ((staged_xml, xml_output), (staged_midi, midi_output)):
                with source.open("rb") as contents, destination.open("xb") as file:
                    created.append(destination)
                    copyfileobj(contents, file)
        except OSError:
            for destination in created:
                destination.unlink(missing_ok=True)
            raise
    return xml_output
