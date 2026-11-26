"""Synthetic stand-in for the Soundverse test pack (A1-A7, P1-P4).

Every clip is generated from code, so ground truth is exact: a beat is at the
sample where we placed it. Seeded, so reruns are bit-identical.

Ground truth files use the common Ballroom/GTZAN format so the real pack can
be dropped in later with no code change:
  <id>.beats : one line per beat, "time<TAB>position_in_bar" (or just "time")
  <id>.mid   : reference notes for P1-P3
"""
import subprocess
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf
from scipy.signal import resample_poly

SR = 44100
OUT = Path(__file__).parent / "testpack"
rng = np.random.default_rng(7)


# ---------- tiny synth kit ----------
def env(n, decay):
    return np.exp(-np.arange(n) / SR * decay)


def kick(g=1.0):
    n = int(0.4 * SR)
    t = np.arange(n) / SR
    f = 45 + 110 * np.exp(-t * 35)
    return g * np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 7)


def noise_hp(n):  # second difference of white noise = crude high-pass
    return np.diff(rng.standard_normal(n + 2), 2)


def snare(g=1.0):
    n = int(0.25 * SR)
    t = np.arange(n) / SR
    return g * (0.25 * noise_hp(n) * env(n, 20) + 0.5 * np.sin(2 * np.pi * 190 * t) * env(n, 35))


def hat(open_=False, g=1.0):
    n = int((0.22 if open_ else 0.05) * SR)
    return g * 0.08 * noise_hp(n) * env(n, 14 if open_ else 80)


def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def tone(freq, dur, wave="saw", decay=3.0, attack=0.005, g=1.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    if wave == "sine":
        x = np.sin(2 * np.pi * freq * t)
    elif wave == "piano":
        x = sum(np.sin(2 * np.pi * k * freq * t) * np.exp(-t * decay * k**0.5) / k**1.5 for k in range(1, 9))
    else:  # band-limited saw
        x = sum(np.sin(2 * np.pi * k * freq * t) / k for k in range(1, min(20, int(SR / 2 / freq))))
        x *= np.exp(-t * decay)
    a = max(1, int(attack * SR))
    x[:a] *= np.linspace(0, 1, a)
    r = min(n, int(0.01 * SR))
    x[n - r:] *= np.linspace(1, 0, r)
    return g * x


def dholak(stroke, g=1.0):
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    if stroke == "dha":  # low bend + ring
        f = 85 + 40 * np.exp(-t * 25)
        x = np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 9) + 0.4 * np.sin(2 * np.pi * 410 * t) * env(n, 18)
    else:
        x = 0.6 * np.sin(2 * np.pi * (420 if stroke == "na" else 300) * t) * env(n, 22) + 0.1 * noise_hp(n) * env(n, 60)
    return g * x


def place(buf, x, t):
    i = int(round(t * SR))
    if 0 <= i < len(buf):
        x = x[: len(buf) - i]
        buf[i : i + len(x)] += x


def pad(buf, roots, t, dur, g=0.05):
    for r in roots:
        place(buf, tone(midi_hz(r), dur, "sine", attack=0.3), t)


# ---------- writers ----------
def write_audio(name, y, fmt="wav", subtype="PCM_16", sr=SR):
    y = y / (np.abs(y).max() + 1e-9) * 0.9 + 1e-4 * rng.standard_normal(len(y))
    if sr != SR:
        y = resample_poly(y, sr // 100, SR // 100)
    path = OUT / f"{name}.{fmt}"
    if fmt == "mp3":
        wav = OUT / f"{name}.tmp.wav"
        sf.write(wav, y, sr, subtype="PCM_16")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-b:a", "192k", str(path)], check=True)
        wav.unlink()
        # MP3 encoders add delay; measure it rather than trust it.
        back, _ = sf.read(path)
        lag = int(np.argmax(np.correlate(back[: SR * 4], y[: SR * 2], "valid")))
        assert lag == 0, f"{name}: mp3 decode offset {lag} samples, ground truth would be wrong"
    else:
        sf.write(path, y, sr, subtype=subtype)
    return path


def write_beats(name, beats, pos=None):
    lines = [f"{t:.6f}\t{p}" if pos else f"{t:.6f}" for t, p in zip(beats, pos or beats)]
    (OUT / f"{name}.beats").write_text("\n".join(lines) + "\n")


def write_midi(name, bpm, tracks):
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    for prog, is_drum, notes in tracks:
        inst = pretty_midi.Instrument(program=prog, is_drum=is_drum)
        inst.notes = [pretty_midi.Note(int(v * 127), p, s, s + d) for s, d, p, v in notes]
        pm.instruments.append(inst)
    pm.write(str(OUT / f"{name}.mid"))


# ---------- A clips: beat tracking ----------
def house(bpm, bars, lead, roots, breakdown=(), tail=2.0):
    spb = 60 / bpm
    y = np.zeros(int((lead + bars * 4 * spb + tail) * SR))
    beats, pos = [], []
    K, C, HO, HC = kick(), snare(0.7), hat(True), hat()
    for b in range(bars * 4):
        t, bar, p = lead + b * spb, b // 4, b % 4
        beats.append(t)
        pos.append(p + 1)
        root = roots[(bar // 2) % len(roots)]
        if p == 0:
            pad(y, [root + 24, root + 28, root + 31], t, 4 * spb, g=0.05)
        if breakdown and breakdown[0] <= bar + 1 <= breakdown[1]:
            continue
        place(y, K, t)
        place(y, HO, t + spb / 2)
        place(y, HC, t + spb / 4)
        place(y, HC, t + 3 * spb / 4)
        if p in (1, 3):
            place(y, C, t)
        place(y, tone(midi_hz(root), spb * 0.4, "saw", decay=6, g=0.3), t + spb / 2)
    return y, beats, pos


def a2_hiphop():  # half-time feel: snare on beat 3 only, fast hats with 32nd rolls
    bpm, bars, lead = 88, 32, 0.25
    spb = 60 / bpm
    y = np.zeros(int((lead + bars * 4 * spb + 2) * SR))
    beats, pos = [], []
    for bar in range(bars):
        t0 = lead + bar * 4 * spb
        for p in range(4):
            beats.append(t0 + p * spb)
            pos.append(p + 1)
        for k in (0, 1.5, 2.75 if bar % 2 else 3.5):
            place(y, kick(), t0 + k * spb)
            place(y, tone(midi_hz(33), spb * 0.9, "sine", g=0.5), t0 + k * spb)  # 808-ish
        place(y, snare(1.0), t0 + 2 * spb)
        step = 8 if bar % 2 else 4  # 32nds on odd bars, 16ths on even
        for i in range(4 * step):
            place(y, hat(g=0.8 if i % 2 == 0 else 0.5), t0 + i * spb * 4 / (4 * step))
    return y, beats, pos


def a3_live():  # no click: tempo wanders +-4 BPM, players are humanised
    beats, t = [], 0.4
    while t < 120:
        beats.append(t)
        t += 60 / (100 + 4 * np.sin(2 * np.pi * t / 40))
    y = np.zeros(int((beats[-1] + 3) * SR))
    roots = [45, 41, 36, 43]
    for i, t in enumerate(beats):
        bar, p = i // 4, i % 4
        h = lambda s: t + rng.normal(0, s)  # noqa: E731
        if p in (0, 2):
            place(y, kick(0.9), h(0.004))
            place(y, tone(midi_hz(roots[bar % 4]), 0.5, "saw", decay=4, g=0.25), h(0.010))
        else:
            place(y, snare(0.8), h(0.004))
        place(y, hat(True, 0.6), h(0.006))
        nxt = beats[i + 1] if i + 1 < len(beats) else t + 0.6
        place(y, hat(True, 0.4), h(0.006) + (nxt - t) / 2)
        r = roots[bar % 4] + 12
        for n in (r, r + 4, r + 7):
            place(y, tone(midi_hz(n), 0.5, "piano", decay=5, g=0.12), h(0.012))
    return y, beats


def a4_ambient_intro():  # 45 s of arrhythmic pad, then a 120 BPM beat
    y = np.zeros(int((45 + 64 + 2) * SR))
    for t, r in zip([0, 7.3, 16.1, 23.0, 31.7, 38.2], [48, 53, 45, 50, 48, 43]):
        pad(y, [r, r + 7, r + 12, r + 16], t, 9.0, g=0.08)
    beat, beats, pos = house(120, 32, 0, [36, 41, 43, 38])
    y[int(45 * SR) : int(45 * SR) + len(beat)] += beat[: len(y) - int(45 * SR)]
    return y, [45 + b for b in beats], pos


def a5_tempo_change():  # 120 BPM for 32 bars, then 132 BPM for 32 bars
    a, ba, pa = house(120, 32, 0.1, [36, 39, 41, 43], tail=0)
    b, bb, pb = house(132, 32, 0, [38, 41, 43, 45])
    start = 0.1 + 32 * 4 * 0.5
    y = np.zeros(int(round(start * SR)) + len(b))
    y[: len(a)] += a[: len(y)]
    y[int(round(start * SR)) : int(round(start * SR)) + len(b)] += b
    return y, ba + [start + t for t in bb], pa + pb


def a6_dadra():  # 6/8 at dotted-quarter = 70; reference beat = dotted quarter (2 per bar)
    eighth = 60 / 70 / 3
    bars, lead = 36, 0.3
    y = np.zeros(int((lead + bars * 6 * eighth + 2) * SR))
    beats, pos = [], []
    theka = ["dha", "dhin", "na", "dha", "tin", "na"]
    melody = [62, 64, 66, 69, 67, 66, 64, 62]
    for bar in range(bars):
        t0 = lead + bar * 6 * eighth
        beats += [t0, t0 + 3 * eighth]
        pos += [1, 2]
        for i, s in enumerate(theka):
            place(y, dholak("dha" if s in ("dha", "dhin") else s, 1.0 if i in (0, 3) else 0.6), t0 + i * eighth)
        place(y, tone(midi_hz(melody[bar % 8]), 3 * eighth, "sine", attack=0.03, g=0.25), t0)
        place(y, tone(midi_hz(melody[(bar + 3) % 8] - 12), 3 * eighth, "piano", g=0.15), t0 + 3 * eighth)
    return y, beats, pos


# ---------- P clips: recreation (exactly N bars, tails wrapped so they loop) ----------
def loop_clip(name, bpm, bars, tracks, ref=True):
    spb = 60 / bpm
    n = int(round(bars * 4 * spb * SR))
    y = np.zeros(n + 2 * SR)
    voices = {"kick": kick, "snare": snare, "hat": lambda g: hat(False, g)}
    midi = []
    for kind, notes in tracks:
        for s, d, p, v in notes:
            t = s * spb
            if kind == "drums":
                place(y, voices[{36: "kick", 38: "snare", 42: "hat"}[p]](v), t)
            else:
                wave, g = {"bass": ("saw", 0.35), "keys": ("piano", 0.18), "lead": ("saw", 0.12)}[kind]
                place(y, tone(midi_hz(p), d * spb, wave, decay=4 if kind == "bass" else 2, g=g * v), t)
        midi.append((0 if kind != "bass" else 33, kind == "drums", [(s * spb, (0.1 if kind == "drums" else d * spb), p, v) for s, d, p, v in notes]))
    y[: 2 * SR] += y[n:]  # wrap decays onto the start so the loop is seamless
    write_audio(name, y[:n])
    if ref:
        write_midi(name, bpm, midi)


def drums(bars, kicks, snares, hats_per_beat=2, vel=1.0):
    notes = []
    for bar in range(bars):
        o = bar * 4
        notes += [(o + k, 0.25, 36, vel) for k in kicks]
        notes += [(o + s, 0.25, 38, 0.9 * vel) for s in snares]
        notes += [(o + i / hats_per_beat, 0.1, 42, 0.6 if i % 2 else 0.8) for i in range(4 * hats_per_beat)]
    return notes


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    y, b, p = house(124, 186, 0.37, [36, 41, 43, 39], breakdown=(97, 112))
    write_audio("A1", y), write_beats("A1", b, p)
    y, b, p = a2_hiphop()
    write_audio("A2", y, "flac", "PCM_24"), write_beats("A2", b, p)
    y, b = a3_live()
    write_audio("A3", y, "mp3"), write_beats("A3", b)
    y, b, p = a4_ambient_intro()
    write_audio("A4", y), write_beats("A4", b, p)
    y, b, p = a5_tempo_change()
    write_audio("A5", y, sr=48000), write_beats("A5", b, p)
    y, b, p = a6_dadra()
    write_audio("A6", y), write_beats("A6", b, p)
    y, b, p = house(128, 80, 0.81, [38, 43, 45, 41])
    write_audio("A7", y), write_beats("A7", b, p)

    loop_clip("P1", 100, 2, [("drums", drums(2, [0, 2.5], [1, 3]))])
    bass = [(i * 0.5, 0.45, [40, 40, 43, 45, 40, 47, 45, 43][i % 8], 0.9) for i in range(32)]
    loop_clip("P2", 110, 4, [("drums", drums(4, [0, 1.5, 2], [1, 3], 4)), ("bass", bass)])
    chords = [(bar * 4 + h, 2, n, 0.7) for bar, r in enumerate([48, 53, 55, 48, 57, 53, 55, 55]) for h in (0, 2) for n in (r, r + 4, r + 7)]
    mel = [(i * 0.5 + (0.5 if i % 3 == 2 else 0), 0.5, [72, 74, 76, 79, 77, 76, 74, 72][i % 8], 0.9) for i in range(0, 64, 2)]
    loop_clip("P3", 90, 8, [("keys", chords), ("keys", mel)])
    dense = [("drums", drums(8, [0, 1, 2, 3], [1, 3], 4)), ("bass", [(i * 0.5, 0.4, 36 + [0, 0, 7, 5][i % 4], 0.9) for i in range(64)]),
             ("keys", [(bar * 4, 4, n, 0.6) for bar in range(8) for n in (60, 63, 67, 70)]),
             ("lead", [(i * 0.25, 0.25, 72 + [0, 3, 7, 10, 12, 10, 7, 3][i % 8], 0.7) for i in range(128)])]
    loop_clip("P4", 120, 8, dense, ref=False)
    print("test pack written to", OUT)


if __name__ == "__main__":
    main()
