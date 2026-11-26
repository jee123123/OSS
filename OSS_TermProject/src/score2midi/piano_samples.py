"""Load the grand piano samples supplied by Windows for WAV previews."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from os import environ
from pathlib import Path
from struct import error as StructError, unpack_from

import numpy as np


@dataclass(frozen=True)
class PianoSample:
    pcm: np.ndarray
    rate: int
    root_pitch: float
    loop_start: int = 0
    loop_end: int = 0

    def render(self, pitch: float, rate: int, length: int) -> np.ndarray:
        step = self.rate / rate * 2 ** ((pitch - self.root_pitch) / 12)
        positions = np.arange(length) * step
        if self.loop_end > self.loop_start:
            tail = positions >= self.loop_end
            positions[tail] = self.loop_start + (
                (positions[tail] - self.loop_start) % (self.loop_end - self.loop_start)
            )
            pcm = np.append(self.pcm[:self.loop_end], self.pcm[self.loop_start])
        else:
            pcm = self.pcm
        return np.interp(positions, np.arange(pcm.size), pcm, right=0)


def _chunks(data: bytes) -> Iterator[tuple[int, bytes, bytes]]:
    """Yield RIFF chunk offsets, tags and payloads, including LIST subtypes."""
    offset = 0
    while offset + 8 <= len(data):
        tag, size = unpack_from("<4sI", data, offset)
        end = offset + 8 + size
        if end > len(data):
            raise ValueError("Invalid DLS chunk length")
        payload = data[offset + 8:end]
        if tag == b"LIST":
            tag, payload = payload[:4], payload[4:]
        yield offset, tag, payload
        offset = end + (size & 1)


def _fields(data: bytes) -> dict[bytes, bytes]:
    return {tag: payload for _, tag, payload in _chunks(data)}


def _read_piano_bank(data: bytes) -> dict[int, PianoSample]:
    if data[:4] != b"RIFF" or data[8:12] != b"DLS ":
        raise ValueError("A DLS sound bank was expected")
    collection = _fields(data[12:])
    pool = collection[b"ptbl"]
    header_size, count = unpack_from("<II", pool)
    offsets = unpack_from(f"<{count}I", pool, header_size)
    waves = {
        offset: payload
        for offset, tag, payload in _chunks(collection[b"wvpl"])
        if tag == b"wave"
    }

    for _, tag, payload in _chunks(collection[b"lins"]):
        if tag != b"ins ":
            continue
        instrument = _fields(payload)
        _, bank, program = unpack_from("<III", instrument[b"insh"])
        if (bank, program) == (0, 0):
            break
    else:
        raise ValueError("The DLS bank does not contain the GM grand piano")

    samples = {}
    for _, tag, payload in _chunks(instrument[b"lrgn"]):
        if tag not in (b"rgn ", b"rgn2"):
            continue
        region = _fields(payload)
        low, high = unpack_from("<HH", region[b"rgnh"])
        wave_index, = unpack_from("<I", region[b"wlnk"], 8)
        wave = _fields(waves[offsets[wave_index]])
        format_tag, channels, rate, _, _, bits = unpack_from("<HHIIHH", wave[b"fmt "])
        # The Windows grand piano uses mono, 16-bit PCM for every region.
        if (format_tag, channels, bits) != (1, 1, 16) or rate <= 0:
            raise ValueError("Unsupported piano sample format")

        settings = region[b"wsmp"] if b"wsmp" in region else wave[b"wsmp"]
        header_size, root, cents, attenuation, _, loops = unpack_from("<IHhiII", settings)
        gain = 10 ** (attenuation / (65536 * 200))
        pcm = np.frombuffer(wave[b"data"], dtype="<i2").astype(np.float32) * (gain / 32768)
        if not pcm.size:
            raise ValueError("Empty piano sample")

        loop_start = loop_end = 0
        if loops:
            _, loop_type, start, length = unpack_from("<IIII", settings, header_size)
            if loop_type == 0 and length > 0 and start + length <= pcm.size:
                loop_start, loop_end = start, start + length
        sample = PianoSample(pcm, rate, root - cents / 100, loop_start, loop_end)
        for key in range(low, min(high, 127) + 1):
            samples[key] = sample
    return samples


@lru_cache(maxsize=1)
def load_piano_samples() -> dict[int, PianoSample]:
    windows = environ.get("SystemRoot")
    if windows:
        bank = Path(windows) / "System32" / "drivers" / "gm.dls"
        try:
            return _read_piano_bank(bank.read_bytes())
        except (OSError, ValueError, KeyError, IndexError, StructError):
            pass
    return {}
