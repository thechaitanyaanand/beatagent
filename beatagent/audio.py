"""Audio I/O that keeps the input's sample rate and bit depth, plus click audition."""
from pathlib import Path

import numpy as np
import soundfile as sf


def load(path):
    """-> (float32 [samples, channels], sample_rate, info). info remembers format/subtype."""
    audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
    i = sf.info(str(path))
    return audio, sr, {"format": i.format, "subtype": i.subtype}


def save(path, audio, sr, info):
    """Write with the input's format and bit depth. MP3 has no bit depth: it becomes 16-bit WAV
    (re-encoding to MP3 would add a second generation of loss and encoder delay)."""
    path = Path(path)
    fmt, subtype = info["format"], info["subtype"]
    if fmt not in ("WAV", "FLAC", "AIFF"):
        fmt, subtype = "WAV", "PCM_16"
    path = path.with_suffix({"WAV": ".wav", "FLAC": ".flac", "AIFF": ".aiff"}[fmt])
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    sf.write(str(path), np.clip(audio, -1, 1), sr, format=fmt, subtype=subtype)
    return path, peak


def click(sr, freq, ms=30):
    n = int(sr * ms / 1000)
    t = np.arange(n) / sr
    return np.sin(2 * np.pi * freq * t) * np.exp(-t * 150)


def click_audition(audio, sr, grid):
    """Original + a click on every beat, a higher, louder click on downbeats."""
    out = audio * 0.7
    beat, down = click(sr, 1500) * 0.35, click(sr, 3000) * 0.6
    for t, p in zip(grid.times, grid.pos):
        c = down if p == 1 else beat
        i = int(round(t * sr))
        if 0 <= i < len(out):
            seg = c[: len(out) - i]
            out[i : i + len(seg)] += seg[:, None]
    return out
