"""Beat grid analysis.

Beat This! finds beats and downbeats (20 ms frames). Code then makes the grid
trustworthy:
  1. fit straight lines through stretches of steady tempo (kills frame jitter),
  2. shift each line onto the measured onsets (kills model bias),
  3. drop beats at the edges that the audio does not support (ambient intros),
  4. derive bar positions and time signature from detected downbeats.

Conventions (also written into every grid JSON):
  - beat numbers are 0-based: beat 0 is the first downbeat, so in 4/4 beat 16 is the
    first beat of bar 5. Pickup beats before it are -1, -2, ... and belong to bar 0.
  - bar numbers are 1-based; pos is 1..beats_per_bar.
  - times are seconds from the start of the file.
"""
from dataclasses import dataclass, field
from functools import lru_cache

import librosa
import numpy as np
import soundfile as sf

AR = 22050       # analysis rate
HOP = 64         # 2.9 ms onset frames
TOL = 0.015      # a beat within 15 ms of a straight line belongs to it
MIN_SEG = 16     # beats needed before we call a stretch "steady"
SNAP = 0.035     # search window for the onset under a beat
ONSET_BIAS = 0.005  # calibration knob: onset-strength peaks land ~5 ms after the true attack
                    # (measured on clips with known onsets; re-measure on real annotations)

CONVENTIONS = ("beat numbers 0-based, beat 0 = first downbeat (4/4: beat 16 = bar 5.1), pickup beats < 0 in bar 0; "
               "bars 1-based; pos 1..beats_per_bar; times in seconds from file start; "
               "beat ranges are half-open (beats 16-32 = start of 16 to start of 32), bar ranges inclusive")


@dataclass
class Grid:
    times: np.ndarray            # beat times, seconds
    pos: np.ndarray              # position in bar, 1..bpb
    bpb: int
    duration: float
    time_signature: str = "4/4"
    confidence: float = 1.0
    origin: np.ndarray = None    # which beat of the turn-start grid each beat came from
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.times = np.asarray(self.times, float)
        self.pos = np.asarray(self.pos, int)
        if self.origin is None:
            self.origin = np.arange(len(self.times))

    @property
    def bars(self):
        """1-based bar number per beat; beats before the first downbeat are bar 0."""
        return np.cumsum(self.pos == 1)

    @property
    def numbers(self):
        """0-based beat number per beat; beat 0 = first downbeat."""
        first = int(np.argmax(self.pos == 1)) if (self.pos == 1).any() else 0
        return np.arange(len(self.times)) - first

    @property
    def n_bars(self):
        return int(self.bars.max()) if len(self.times) else 0

    def segments(self):
        return [{"start_beat": int(self.numbers[i]), "end_beat": int(self.numbers[j - 1]),
                 "start_time": round(float(self.times[i]), 4), "bpm": round(60 / p, 3)}
                for i, j, p in steady_segments(self.times)]

    def tempo_changes(self):
        """Beat slots where one steady segment hands over to a clearly different tempo."""
        segs = steady_segments(self.times)
        return [b[0] for a, b in zip(segs, segs[1:]) if abs(a[2] / b[2] - 1) > 0.01]

    def to_json(self):
        segs = self.segments()
        ibi = np.diff(self.times)
        return {
            "tempo": round(60 / float(np.median(ibi)), 3) if len(ibi) else None,
            "time_signature": self.time_signature,
            "beats_per_bar": self.bpb,
            "constant_tempo": len(segs) == 1 and segs[0]["end_beat"] - segs[0]["start_beat"] + 1 >= 0.95 * len(self.times),
            "confidence": round(float(self.confidence), 3),
            "duration": round(self.duration, 4),
            "n_beats": len(self.times),
            "n_bars": self.n_bars,
            "segments": segs,
            "beats": [{"beat": int(n), "time": round(float(t), 5), "bar": int(b), "pos": int(p)}
                      for n, t, b, p in zip(self.numbers, self.times, self.bars, self.pos)],
            "downbeats": [round(float(t), 5) for t in self.times[self.pos == 1]],
            "conventions": CONVENTIONS,
            **self.meta,
        }


def steady_segments(t):
    """Greedy split of beat times into straight lines (constant tempo).
    Returns [(start, end, period)], end exclusive. Lone outliers inside a run are tolerated."""
    t = np.asarray(t, float)
    out, i, n = [], 0, len(t)
    while i + MIN_SEG <= n:
        k = np.arange(i, i + MIN_SEG)
        p = np.polyfit(k, t[i:i + MIN_SEG], 1)
        if np.abs(np.polyval(p, k) - t[i:i + MIN_SEG]).max() > TOL:
            i += 1
            continue
        j = i + MIN_SEG
        while j < n:
            ahead = [m for m in range(j, min(j + 4, n)) if abs(t[m] - np.polyval(p, m)) < TOL]
            if not ahead:
                break
            j = ahead[0] + 1
            if (j - i) % 8 == 0:
                q = np.polyfit(np.arange(i, j), t[i:j], 1)
                # refitting lets a line crawl along a slow drift; stop once it no longer fits
                if (np.abs(np.polyval(q, np.arange(i, j)) - t[i:j]) > TOL).mean() > 0.1:
                    j -= 8
                    break
                p = q
        p = np.polyfit(np.arange(i, j), t[i:j], 1)
        out.append((i, j, float(p[0])))
        i = j
    return out


@lru_cache(1)
def _tracker():
    from beat_this.inference import Audio2Beats
    return Audio2Beats(device="cpu", dbn=False)


def onset_envelope(mono, sr):
    y = librosa.resample(mono, orig_sr=sr, target_sr=AR)
    env = librosa.onset.onset_strength(y=y, sr=AR, hop_length=HOP, n_fft=512)
    return env / (np.median(env) + 1e-9)


def analyze(path=None, audio=None, sr=None):
    if audio is None:
        audio, sr = sf.read(path, always_2d=True, dtype="float32")
    mono = audio.mean(axis=1)
    beats, downs = _tracker()(mono, sr)
    beats = np.asarray(beats, float)
    if not len(beats):
        return Grid([], [], 4, len(mono) / sr, "4/4", 0.0,
                    meta={"source": str(path) if path else None, "sample_rate": sr})
    env = onset_envelope(mono, sr)
    env_t = np.arange(len(env)) * HOP / AR

    # snap each beat to the strongest onset nearby; "support" = a real onset is there
    snap, strength = np.empty_like(beats), np.empty_like(beats)
    for k, b in enumerate(beats):
        w = np.flatnonzero(np.abs(env_t - b) <= SNAP)
        m = w[np.argmax(env[w])] if len(w) else int(round(b * AR / HOP))
        snap[k], strength[k] = env_t[min(m, len(env_t) - 1)] - ONSET_BIAS, env[min(m, len(env) - 1)]
    support = strength > 3.0

    # trim unsupported beats at the edges (no drums = no evidence = no grid)
    run = np.convolve(support, np.ones(4), "valid") == 4
    if run.any():
        lo = int(np.argmax(run))
        hi = len(run) - int(np.argmax(run[::-1])) + 3
        beats, snap, support = beats[lo:hi], snap[lo:hi], support[lo:hi]

    # best per-beat estimate: the measured onset where there is one, the tracker elsewhere
    beats = np.where(support, snap, beats)

    # steady stretches that sit on the same line (same tempo, phase continues) are
    # one stretch with a gap, e.g. a breakdown where the tracker wobbled or missed beats.
    # A short wobbly stretch between two that agree is dropped and re-gridded.
    def line(g):
        idx = np.concatenate([np.arange(i, j) for i, j, _ in g])
        k = np.round((beats[idx] - beats[g[0][0]]) / g[0][2])
        P, a = np.polyfit(k, beats[idx], 1)
        return P, a

    def size(g):
        return sum(j - i for i, j, _ in g)

    groups = []
    for s in steady_segments(beats):
        merged = False
        for back in (1, 2):
            if len(groups) < back or (back == 2 and size(groups[-1]) >= 32):
                break
            P, a = line(groups[-back])
            t = beats[s[0]:s[1]]
            r = t - a - P * np.round((t - a) / P)
            if abs(s[2] / P - 1) < 0.002 and np.median(np.abs(r)) < TOL / 2:
                if back == 2:
                    groups.pop()
                groups[-1].append(s)
                merged = True
                break
        if not merged:
            groups.append([s])

    # rebuild: each group becomes an exact line, phase-locked to measured onsets, and
    # extended over neighbouring beats for as long as real onsets confirm it.
    # Everything else (drift, the seam of a tempo change) keeps onset-snapped times.
    duration = len(mono) / sr

    def onset_near(t):
        w = np.abs(env_t - t) <= TOL
        return w.any() and env[w].max() > 3.0

    pts, claimed = [], np.zeros(len(beats), bool)  # (time, region, steady)
    for r, g in enumerate(groups):
        idx = np.concatenate([np.arange(i, j) for i, j, _ in g])
        k = np.round((beats[idx] - beats[g[0][0]]) / g[0][2])
        P, a = np.polyfit(k, beats[idx], 1)
        sup = support[idx]
        a += np.median(snap[idx][sup] - (P * k[sup] + a)) if sup.sum() >= 8 else 0.0
        k0, k1 = 0, int(k.max())
        while a + P * (k1 + 1) < duration and onset_near(a + P * (k1 + 1)):
            k1 += 1
        while a + P * (k0 - 1) > 0 and onset_near(a + P * (k0 - 1)):
            k0 -= 1
        pts += [(a + P * kk, r, True) for kk in range(k0, k1 + 1)]
        claimed[g[0][0]:g[-1][1]] = True
    gap = 0.3 * float(np.median(np.diff(beats))) if len(beats) > 1 else 0.1
    line_t = np.array([p[0] for p in pts])
    for k in np.flatnonzero(~claimed):
        if not len(line_t) or np.abs(line_t - beats[k]).min() > gap:
            pts.append((beats[k], -1, False))
    pts.sort()
    pts = [p for n, p in enumerate(pts) if n == 0 or p[0] - pts[n - 1][0] > gap]
    times = np.array([p[0] for p in pts])
    region = np.array([p[1] for p in pts])
    steady = np.array([p[2] for p in pts])

    # bars: beats per bar from the tracker's downbeat spacing, phase voted per region
    # (a tempo change can shift the bar phase, so one global vote is not enough)
    idx = np.array([int(np.argmin(np.abs(times - d))) for d in downs if len(times) and np.abs(times - d).min() < 0.07], int)
    gaps = np.diff(idx)
    ok = gaps[(gaps >= 2) & (gaps <= 12)]
    bpb = int(np.bincount(ok).argmax()) if len(ok) else 4
    pos = np.zeros(len(times), int)
    starts = np.flatnonzero(np.r_[True, region[1:] != region[:-1]])
    for s, e in zip(starts, np.r_[starts[1:], len(times)]):
        votes = idx[(idx >= s) & (idx < e)] - s
        if len(votes) >= 2 or s == 0:
            phase = int(np.bincount(votes % bpb, minlength=bpb).argmax()) if len(votes) else 0
            pos[s:e] = (np.arange(e - s) - phase) % bpb + 1
        else:  # too little evidence: keep counting from the previous bar
            pos[s:e] = (pos[s - 1] + np.arange(e - s)) % bpb + 1

    # compound meter? onset energy on beat thirds vs beat quarters/halves
    def at(*fracs):
        out = []
        for f in fracs:
            q = np.round((times[:-1] + f * np.diff(times)) * AR / HOP).astype(int)
            w = np.clip(q[:, None] + np.arange(-4, 5), 0, len(env) - 1)  # +-12 ms
            out.append(env[w].max(axis=1).mean())
        return np.mean(out)
    compound = len(times) > 8 and at(1 / 3, 2 / 3) > 1.5 * at(1 / 4, 1 / 2, 3 / 4)
    ts = {2: "6/8", 3: "9/8", 4: "12/8"}.get(bpb, f"{bpb * 3}/8") if compound else f"{bpb}/4"

    confidence = float(support.mean() * (0.5 + 0.5 * steady.mean())) if len(times) else 0.0
    return Grid(times, pos, bpb, len(mono) / sr, ts, confidence,
                meta={"source": str(path) if path else None, "sample_rate": sr})
