"""Beat-match track B to master track A.

ratio  = A's tempo / B's tempo over the overlap (B is time-stretched by it, pitch kept)
offset = where B's first downbeat must start so it lands on A's chosen downbeat.
Rubber Band adds a small latency; we measure it from B's kicks instead of assuming it.
"""
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

from .audio import load, save
from .grid import analyze


def kick_onsets(mono, sr):
    """Kick attack times: steepest rise of the <150 Hz envelope (zero-phase filter, no delay)."""
    lp = sosfiltfilt(butter(4, 150, "lowpass", fs=sr, output="sos"), mono)
    w = max(1, int(0.002 * sr))
    e = np.sqrt(np.convolve(lp**2, np.ones(w) / w, "same"))
    d = np.diff(e, prepend=0)
    d[: w * 5] = 0  # a file that starts mid-sound has a fake attack at t=0
    pk, _ = find_peaks(d, distance=int(0.1 * sr))
    if not len(pk):
        return pk / sr
    h = d[pk]
    # measured on A7: kick attacks ~2x steeper than the bassline's; 0.6 splits them
    pk = pk[h > 0.6 * np.percentile(h, 90)]
    # onset = half-rise point (base -> local peak); steepest-rise alone gets pulled
    # around by whatever bass is ringing underneath
    pre, post = int(0.02 * sr), int(0.03 * sr)
    out = []
    for p in pk:
        a, b = max(p - pre, 0), min(p + post, len(e))
        base, top = e[a], e[a:b].max()
        out.append(a + int(np.argmax(e[a:b] >= base + 0.5 * (top - base))))
    return np.array(out) / sr


def match(a, b, window=0.05):
    """Signed offsets b - a for each a with a b within the window."""
    if not len(a) or not len(b):
        return np.array([])
    j = np.clip(np.searchsorted(b, a), 1, len(b) - 1)
    near = np.where(np.abs(b[j - 1] - a) < np.abs(b[j] - a), b[j - 1], b[j])
    d = near - a
    return d[np.abs(d) <= window]


def local_tempo(grid, t0, t1):
    ibi = np.diff(grid.times[(grid.times >= t0) & (grid.times <= t1)])
    if len(ibi) < 4:
        raise ValueError("not enough beats in the overlap to measure tempo")
    return 60 / float(np.median(ibi))


def stretch(path, ratio, sr):
    out = Path(tempfile.mkdtemp()) / "stretched.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-af",
                    # window=short: default window smeared every other kick by ~25 ms (measured)
                    f"rubberband=tempo={ratio:.8f}:transients=crisp:detector=percussive:window=short",
                    "-ar", str(sr), "-c:a", "pcm_f32le", str(out)], check=True)
    return load(out)[0]


def beatmatch(a_path, b_path, out_dir, start_bar=33, seconds=60.0, grid_a=None, grid_b=None):
    A, sr, info = load(a_path)
    B, srb, _ = load(b_path)
    ga = grid_a or analyze(audio=A, sr=sr)
    gb = grid_b or analyze(audio=B, sr=srb)
    starts = ga.times[(ga.bars == start_bar) & (ga.pos == 1)]
    if not len(starts):
        raise ValueError(f"track A has bars 1-{ga.n_bars}; bar {start_bar} does not exist")
    tA = float(starts[0])
    tB = float(gb.times[gb.pos == 1][0])  # B enters from its own bar 1
    bpm_a = local_tempo(ga, tA, tA + seconds)
    bpm_b = local_tempo(gb, tB, tB + seconds)
    ratio = bpm_a / bpm_b
    # ponytail: one constant ratio; a variable-tempo B would need a piecewise stretch map
    S = stretch(b_path, ratio, sr)

    # same detector on B before and after stretching, so the detector's own bias cancels
    before = kick_onsets(B.mean(axis=1), srb) / ratio
    lat = match(before, kick_onsets(S.mean(axis=1), sr), 0.03)
    latency = float(np.median(lat)) if len(lat) else 0.0
    b0 = tB / ratio + latency  # where B's bar 1 sits inside the stretched file

    n = int(seconds * sr)
    iA, iB = int(round(tA * sr)), int(round(b0 * sr))
    a_seg = np.zeros((n, A.shape[1]), np.float32)
    seg = A[iA:iA + n]
    a_seg[: len(seg)] = seg
    b_seg = np.zeros_like(a_seg)
    seg = S[iB:iB + n]
    b_seg[: len(seg)] = seg if S.shape[1] == A.shape[1] else seg.mean(axis=1, keepdims=True)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {k: str(save(out_dir / f"sync_{k}", x, sr, info)[0])
             for k, x in [("mix", 0.5 * (a_seg + b_seg)), ("a", a_seg), ("b", b_seg)]}
    return {
        "a_bpm": round(bpm_a, 3), "b_bpm": round(bpm_b, 3), "stretch_ratio": round(ratio, 6),
        "a_start_bar": start_bar, "a_start_time": round(tA, 4), "b_bar1_time": round(tB, 4),
        "offset_seconds": round(tA - b0, 4),
        "rubberband_latency_ms": round(latency * 1000, 2),
        "note": "offset = where B's (stretched) t=0 sits on A's timeline; B's bar 1 lands on A's bar "
                f"{start_bar}",
        "files": files,
    }
