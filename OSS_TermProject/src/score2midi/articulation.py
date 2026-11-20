"""Adjust MIDI articulation without changing note onsets."""

from fractions import Fraction

from music21 import chord, note, stream

REPEATED_NOTE_GAP = Fraction(1, 64)


def _pitched_notes(element) -> dict[int, note.Note]:
    if isinstance(element, note.Note):
        return {element.pitch.midi: element}

    if isinstance(element, chord.Chord):
        return {
            member.pitch.midi: member
            for member in element.notes
        }

    return {}


def _forms_true_tie(
    left_note: note.Note,
    right_note: note.Note,
) -> bool:
    left_type = getattr(left_note.tie, "type", None)
    right_type = getattr(right_note.tie, "type", None)

    return (
        left_type in {"start", "continue"}
        and right_type in {"continue", "stop"}
    )


def separate_repeated_notes(score: stream.Score) -> list[str]:
    corrected_count = 0

    for part in score.parts:
        events = list(part.recurse().notes)
        starts: dict[Fraction, list] = {}

        for event in events:
            start = Fraction(event.getOffsetInHierarchy(part))
            starts.setdefault(start, []).append(event)

        for left in events:
            left_start = Fraction(left.getOffsetInHierarchy(part))
            duration = Fraction(left.duration.quarterLength)
            left_end = left_start + duration
            right_events = starts.get(left_end, [])

            if not right_events:
                continue

            left_notes = _pitched_notes(left)
            should_separate = False

            for right in right_events:
                right_notes = _pitched_notes(right)
                shared_pitches = left_notes.keys() & right_notes.keys()

                if shared_pitches and not any(
                    _forms_true_tie(
                        left_notes[pitch],
                        right_notes[pitch],
                    )
                    for pitch in shared_pitches
                ):
                    should_separate = True
                    break

            if not should_separate:
                continue

            gap = min(REPEATED_NOTE_GAP, duration / 8)

            if duration <= gap:
                continue

            left.duration.quarterLength = duration - gap
            corrected_count += 1

    if corrected_count == 0:
        return []

    return [
        f"붙어서 연주되던 반복음 {corrected_count}곳을 "
        "분리했습니다."
    ]
