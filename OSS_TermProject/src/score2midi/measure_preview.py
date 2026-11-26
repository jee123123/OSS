"""Render a short piano WAV for the measure review screen."""

from __future__ import annotations

from copy import deepcopy
from io import BytesIO
from pathlib import Path
import wave

import numpy as np
from music21 import chord, converter, harmony, note, stream

from score2midi.measure_retry import RetryCandidate, _part_measure
from score2midi.piano_samples import load_piano_samples

SAMPLE_RATE = 22050
RELEASE_SECONDS = 0.06


def _synthetic_tone(pitch: float, t: np.ndarray) -> np.ndarray:
    """Keep preview playback available when the system sound bank is absent."""
    frequency = 440 * 2 ** ((pitch - 69) / 12)
    tone = np.zeros(t.size)
    for harmonic, gain, decay in (
        (1, 0.66, 1.8), (2, 0.24, 3), (3, 0.13, 5), (5, 0.05, 9)
    ):
        partial = frequency * harmonic * (1 + 0.0003 * (harmonic - 1) ** 2)
        if partial < SAMPLE_RATE / 2:
            tone += gain * np.sin(2 * np.pi * partial * t) * np.exp(-decay * t)
    return tone


def render_measure_preview(
    musicxml: Path | stream.Score,
    measure_number: int,
    bpm: int,
    *,
    include_neighbors: bool = False,
    retry: RetryCandidate | None = None,
) -> bytes:
    if not 20 <= bpm <= 300:
        raise ValueError("BPM은 20에서 300 사이여야 합니다.")

    score = musicxml if isinstance(musicxml, stream.Score) else converter.parse(str(musicxml))
    parts = list(score.parts)
    if not parts:
        raise ValueError("미리듣기에 사용할 악보 파트가 없습니다.")
    measures = list(parts[0].getElementsByClass(stream.Measure))
    selected = parts[0].measure(measure_number)
    if selected is None:
        raise ValueError(f"{measure_number}마디를 찾을 수 없습니다.")
    index = measures.index(selected)
    if include_neighbors:
        measures = measures[max(0, index - 1):index + 2]
    else:
        measures = [selected]

    candidate_parts = []
    if retry is not None:
        candidate_parts = list(converter.parse(str(retry.musicxml)).parts)
        if retry.measure != measure_number or len(candidate_parts) != len(parts):
            raise ValueError("기존 마디와 재인식 결과의 파트가 일치하지 않습니다.")

    starts = []
    beats = 0.0
    for measure in measures:
        starts.append(beats)
        beats += float(
            measure.barDuration.quarterLength
            - measure.paddingLeft
            - measure.paddingRight
        )

    events = []
    for i, part in enumerate(parts):
        excerpt = stream.Stream()
        for start, reference in zip(starts, measures):
            measure = part.measure(reference.measureNumberWithSuffix())
            if retry is not None and reference is selected:
                measure = _part_measure(candidate_parts[i], None, retry.local_number)
            if measure is None:
                continue
            for event in measure.recurse().notes:
                if isinstance(event, (note.Note, chord.Chord)) and not isinstance(
                    event, harmony.Harmony
                ) and event.quarterLength > 0:
                    excerpt.insert(
                        start + float(event.getOffsetInHierarchy(measure)),
                        deepcopy(event),
                    )
        excerpt.stripTies(inPlace=True)
        for event in excerpt.notes:
            pitches = (event.pitch,) if isinstance(event, note.Note) else event.pitches
            events.extend(
                (float(event.offset), float(event.quarterLength), pitch.ps)
                for pitch in pitches
            )

    seconds_per_beat = 60 / bpm
    # Keep overlong events audible so the preview exposes bar-line collisions.
    beats = max(beats, max((start + length for start, length, _ in events), default=0))
    duration = beats * seconds_per_beat + RELEASE_SECONDS
    if not 0 < duration <= 120:
        raise ValueError("미리듣기 구간이 너무 깁니다.")

    samples = np.zeros(round(duration * SAMPLE_RATE), dtype=np.float32)
    release = round(RELEASE_SECONDS * SAMPLE_RATE)
    piano = load_piano_samples() if events else {}
    for start, length, pitch in events:
        offset = round(start * seconds_per_beat * SAMPLE_RATE)
        held = max(1, round(length * seconds_per_beat * SAMPLE_RATE))
        size = min(held + release, len(samples) - offset)
        if offset < 0 or size <= 0:
            continue
        t = np.arange(size) / SAMPLE_RATE
        sample = piano.get(round(pitch))
        if sample is None:
            tone = _synthetic_tone(pitch, t)
        else:
            tone = sample.render(pitch, SAMPLE_RATE, size)
            # Apply a decay to the sustained sample loop.
            tone *= np.exp(-0.8 * t)
        tone *= np.minimum(t / 0.004, 1)
        if size > held:
            tone[held:] *= 1 - np.arange(size - held) / release
        samples[offset:offset + size] += tone

    samples *= 0.85 / max(1, float(np.max(np.abs(samples))))
    output = BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
        wav.writeframes((samples * 32767).astype("<i2").tobytes())
    return output.getvalue()
