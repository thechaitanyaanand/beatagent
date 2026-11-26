"""Audio clip -> score JSON -> self-contained Web Audio page.

Fidelity order from the brief: rhythm, then pitch, then timbre.
  rhythm: tempo comes from the loop length (an N-bar loop has an exact tempo) and
          every onset is measured from the audio;
  pitch:  Basic Pitch on the harmonic layer;
  timbre: a handful of synth parameters estimated from the spectrum.
The page is a fixed template that reads the score. No LLM-written code runs in the browser.
"""
import json
import tempfile
import warnings
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

from .audio import load
from .grid import analyze

TEMPLATE = Path(__file__).with_name("player.html")
KICK, SNARE, HAT = 36, 38, 42


def loop_tempo(duration, bpb, estimate):
    """An N-bar loop (N in 1..16) has tempo 60*bpb*N/duration; pick N nearest the tracker's estimate."""
    cands = [(abs(np.log2(60 * bpb * n / duration / estimate)), n) for n in (1, 2, 3, 4, 6, 8, 12, 16)]
    n = min(cands)[1]
    return 60 * bpb * n / duration, n


def band_env(y, sr, lo, hi, hop):
    """Rise of the energy in one band, normalised so the loudest hit is 1."""
    if lo == 0:
        sos = butter(4, hi, "lowpass", fs=sr, output="sos")
    elif hi >= sr / 2:
        sos = butter(4, lo, "highpass", fs=sr, output="sos")
    else:
        sos = butter(4, [lo, hi], "bandpass", fs=sr, output="sos")
    x = sosfiltfilt(sos, y)
    e = np.sqrt(np.convolve(x ** 2, np.ones(256) / 256, "same"))[::hop]
    d = np.maximum(0, np.diff(e, prepend=e[0]))
    # ponytail: normalised by the loudest hit; one freak transient would mute the rest
    return d / (d.max() + 1e-12)


def drums(perc, sr, spb):
    """Peaks per band on the percussive layer: low = kick, mid = snare, high (no snare there) = hat.
    Bands are peak-picked separately so a quiet hat isn't lost under the onset list of a loud kick."""
    from scipy.signal import find_peaks
    hop = 128
    gap = int(0.06 * sr / hop)
    low, mid, high = (band_env(perc, sr, lo, hi, hop) for lo, hi in ((0, 150), (300, 3000), (6000, sr / 2)))
    hits = {}
    for pitch, env in ((KICK, low), (SNARE, mid)):
        pk, pr = find_peaks(env, height=0.25, distance=gap)
        hits[pitch] = list(zip(pk, pr["peak_heights"]))
    # snares are bright too and would set the high band's scale; judge hats only against
    # high-band peaks that are not snares
    sn = np.array([f for f, _ in hits[SNARE]])
    pk, pr = find_peaks(high, height=0.01, distance=gap)
    cand = [(f, v) for f, v in zip(pk, pr["peak_heights"]) if not len(sn) or np.abs(sn - f).min() > 3]
    top = max((v for _, v in cand), default=1.0)
    hits[HAT] = [(f, v / top) for f, v in cand if v / top > 0.25]
    return sorted(({"start": round(f * hop / sr / spb, 4), "duration": 0.25, "pitch": p, "velocity": round(float(min(1, v)), 3)}
                   for p, hs in hits.items() for f, v in hs), key=lambda n: n["start"])


def pitched(harm, sr, spb, path_hint="clip"):
    from basic_pitch.inference import predict
    tmp = Path(tempfile.mkdtemp()) / f"{path_hint}_harm.wav"
    sf.write(tmp, harm, sr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, events = predict(tmp, onset_threshold=0.5, frame_threshold=0.3, minimum_note_length=120)  # shorter = kick bleed read as bass
    return [{"start": round(s / spb, 4), "duration": round(max(0.05, (e - s) / spb), 4), "pitch": int(p),
             "velocity": round(float(min(1.0, a * 1.5)), 3)} for s, e, p, a, *_ in sorted(events)]


def brightness(harm, sr, lo, hi):
    """Spectral centroid of a register -> a lowpass cutoff that sounds about as bright."""
    S = np.abs(librosa.stft(harm, n_fft=4096))
    f = librosa.fft_frequencies(sr=sr, n_fft=4096)
    c = float(librosa.feature.spectral_centroid(S=S, freq=f).mean())
    return int(np.clip(c * 1.5, lo, hi))


def transcribe(path):
    audio, sr, _ = load(path)
    mono = audio.mean(axis=1)
    g = analyze(audio=audio, sr=sr)
    est = g.to_json()["tempo"] or 120.0
    # a 2-bar loop is too short to count beats per bar reliably: 4/4 unless 3/4 or compound was measured
    bpb = g.bpb if g.bpb == 3 or g.time_signature.endswith("/8") else 4
    tempo, bars = loop_tempo(len(mono) / sr, bpb, est)
    spb = 60 / tempo
    harm, perc = librosa.effects.hpss(mono, margin=2.0)

    tracks = []
    dn = drums(perc, sr, spb)
    # a piano has percussive attacks too; only call it drums if the percussive layer carries real energy
    perc_ratio = float(np.sqrt(np.mean(perc ** 2)) / (np.sqrt(np.mean(harm ** 2)) + 1e-9))
    if dn and perc_ratio > 0.35:
        tracks.append({"name": "drums", "kind": "drums", "gain": 1.0, "mute": False, "synth": {}, "notes": dn})
    notes = pitched(harm, sr, spb, Path(path).stem)
    bass = [n for n in notes if n["pitch"] < 52]
    keys = [n for n in notes if n["pitch"] >= 52]
    if bass:
        tracks.append({"name": "bass", "kind": "bass", "gain": 0.8, "mute": False,
                       "synth": {"wave": "sawtooth", "cutoff": brightness(sosfiltfilt(butter(4, 300, "lowpass", fs=sr, output="sos"), harm), sr, 200, 2500),
                                 "q": 6, "attack": 0.005, "release": 0.06}, "notes": bass})
    if keys:
        tracks.append({"name": "keys", "kind": "keys", "gain": 0.5, "mute": False,
                       "synth": {"wave": "triangle", "cutoff": brightness(harm, sr, 800, 8000), "q": 1,
                                 "attack": 0.004, "release": 0.25, "decay": 1.2}, "notes": keys})
    return {
        "tempo": round(tempo, 4), "time_signature": g.time_signature if bpb == g.bpb else "4/4",
        "bars": bars, "loop_beats": bars * bpb, "beats_per_bar": bpb, "source": Path(path).name,
        "note_units": "start and duration are in beats from the loop start; pitch is MIDI; drums use GM (36 kick, 38 snare, 42 hat)",
        "perc_to_harm_ratio": round(perc_ratio, 3),
        "tracks": tracks,
    }


def render_page(score, out_path):
    html = TEMPLATE.read_text(encoding="utf-8")
    data = json.dumps(score).replace("</", "<\\/")  # can't close the script tag from inside data
    Path(out_path).write_text(html.replace("/*__SCORE__*/null", data), encoding="utf-8")
    return out_path


def score_notes_seconds(score):
    """(intervals [s], pitches [MIDI]) for evaluation, one loop."""
    spb = 60 / score["tempo"]
    ivs, ps = [], []
    for t in score["tracks"]:
        for n in t["notes"]:
            ivs.append([n["start"] * spb, (n["start"] + n["duration"]) * spb])
            ps.append(n["pitch"])
    return np.array(ivs).reshape(-1, 2), np.array(ps)
