from pathlib import Path
from tempfile import TemporaryDirectory
from fractions import Fraction

from music21 import meter, note, stream

from score2midi.measure_retry import (
    RetryCandidate,
    apply_retry_candidate,
    measure_stats,
)


def _score(pitch: str) -> stream.Score:
    score = stream.Score()
    for octave in (5, 3):
        part = stream.Part()
        measure = stream.Measure(number=1)
        measure.timeSignature = meter.TimeSignature("4/4")
        measure.insert(0, note.Note(f"{pitch}{octave}", quarterLength=4))
        part.append(measure)
        score.insert(0, part)
    return score


def test_replaces_only_selected_measure_and_writes_backups() -> None:
    with TemporaryDirectory() as directory:
        directory = Path(directory)
        current_xml = directory / "score.musicxml"
        candidate_xml = directory / "candidate.musicxml"
        midi = directory / "score.mid"
        _score("C").write("musicxml", fp=str(current_xml))
        _score("C").write("midi", fp=str(midi))
        _score("D").write("musicxml", fp=str(candidate_xml))
        candidate_score = _score("D")
        retry = RetryCandidate(
            1,
            1,
            candidate_xml,
            measure_stats(_score("C"), 1, 1),
            measure_stats(candidate_score, 1, 1),
        )

        apply_retry_candidate(current_xml, midi, retry, 120)

        from music21 import converter

        result = converter.parse(str(current_xml))
        assert {event.pitch.name for event in result.recurse().notes} == {"D"}
        assert (directory / "score.before-retry.musicxml").is_file()
        assert (directory / "score.before-retry.mid").is_file()


def test_candidate_uses_existing_meter_for_duration_error() -> None:
    candidate = _score("D")
    measure = candidate.parts[0].measure(1)
    measure.append(note.Note("E5", quarterLength=1))

    stats = measure_stats(
        candidate,
        1,
        1,
        (Fraction(4), Fraction(4)),
    )

    assert stats.duration_error == 1
