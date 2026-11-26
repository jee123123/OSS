from io import BytesIO
from struct import pack
import wave

import numpy as np
from music21 import meter, note, stream

from score2midi.measure_preview import RELEASE_SECONDS, SAMPLE_RATE, render_measure_preview
from score2midi.measure_retry import MeasureStats, RetryCandidate
from score2midi.piano_samples import _read_piano_bank


def test_preview_reads_and_renders_the_grand_piano_sample(tmp_path, monkeypatch):
    def chunk(tag, data):
        return pack("<4sI", tag, len(data)) + data + b"\x00" * (len(data) % 2)

    def list_chunk(kind, *chunks):
        return chunk(b"LIST", kind + b"".join(chunks))

    piano_wave = list_chunk(
        b"wave",
        chunk(b"fmt ", pack("<HHIIHH", 1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16)),
        chunk(b"data", pack("<4h", 0, 8192, 16384, 4096)),
    )
    region = list_chunk(
        b"rgn ",
        chunk(b"rgnh", pack("<6H", 0, 127, 0, 127, 0, 0)),
        chunk(b"wlnk", pack("<HHII", 0, 0, 1, 0)),
        chunk(b"wsmp", pack("<IHhiII", 20, 74, 0, 0, 0, 1) + pack("<4I", 16, 0, 1, 2)),
    )
    instrument = list_chunk(
        b"ins ", chunk(b"insh", pack("<III", 1, 0, 0)), list_chunk(b"lrgn", region)
    )
    piano = _read_piano_bank(chunk(
        b"RIFF", b"DLS " + list_chunk(b"lins", instrument)
        + chunk(b"ptbl", pack("<III", 8, 1, 0)) + list_chunk(b"wvpl", piano_wave)
    ))
    np.testing.assert_allclose(
        piano[74].render(74, SAMPLE_RATE, 6), [0, 0.25, 0.5, 0.25, 0.5, 0.25]
    )
    monkeypatch.setattr("score2midi.measure_preview.load_piano_samples", lambda: piano)
    musicxml = tmp_path / "score.musicxml"
    write_score(musicxml, 1, ["D"])
    with wave.open(BytesIO(render_measure_preview(musicxml, 1, 120)), "rb") as wav:
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        # At the native pitch the sample repeats these two positive values.
        assert 0 < samples[101] < samples[102]
        assert 0 < samples[103] < samples[104]
        assert samples[102] > 10000


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
        spectrum = np.abs(np.fft.rfft(section * np.hanning(len(section))))
        fundamental = np.abs(frequencies - note.Note(pitch).pitch.frequency) <= 2
        assert spectrum[fundamental].max() >= spectrum.max() * 0.2


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
