from io import BytesIO
import wave

import numpy as np
from music21 import meter, note, stream

from score2midi.measure_preview import RELEASE_SECONDS, SAMPLE_RATE, render_measure_preview
from score2midi.measure_retry import MeasureStats, RetryCandidate


def write_score(path, start_number, pitches):
    score = stream.Score()
    for octave, onset in ((5, 0), (3, 2)):
        part = stream.Part()
        for number, name in enumerate(pitches, start_number):
            measure = stream.Measure(number)
            if number == start_number:
                measure.timeSignature = meter.TimeSignature("4/4")
            measure.insert(onset, note.Note(f"{name}{octave}", quarterLength=2))
            part.insert((number - start_number) * 4, measure)
        score.insert(0, part)
    score.write("musicxml", fp=str(path))


def test_preview_mixes_both_parts_at_the_requested_tempo(tmp_path):
    musicxml = tmp_path / "score.musicxml"
    write_score(musicxml, 1, ["C", "D", "E"])
    with wave.open(BytesIO(render_measure_preview(musicxml, 2, 120)), "rb") as wav:
        assert wav.getframerate() == SAMPLE_RATE
        assert wav.getnframes() == round((2 + RELEASE_SECONDS) * SAMPLE_RATE)
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    for start, pitch in ((0.2, "D5"), (1.2, "D3")):
        section = samples[round(start * SAMPLE_RATE):round((start + 0.5) * SAMPLE_RATE)]
        frequencies = np.fft.rfftfreq(len(section), 1 / SAMPLE_RATE)
        peak = frequencies[np.argmax(np.abs(np.fft.rfft(section)))]
        assert abs(peak - note.Note(pitch).pitch.frequency) <= 2


def test_preview_uses_candidate_page_position_and_original_neighbors(tmp_path):
    current = tmp_path / "current.musicxml"
    candidate = tmp_path / "candidate.musicxml"
    expected = tmp_path / "expected.musicxml"
    write_score(current, 3, ["C", "D", "E"])
    write_score(candidate, 1, ["F", "G", "A", "B"])
    write_score(expected, 3, ["C", "G", "E"])
    stats = MeasureStats(2, 0, 0)
    retry = RetryCandidate(4, 2, candidate, stats, stats)

    assert render_measure_preview(
        current, 4, 120, include_neighbors=True, retry=retry
    ) == render_measure_preview(expected, 4, 120, include_neighbors=True)
