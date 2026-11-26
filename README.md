# Beat Agent

One chat-driven agent that does three things with audio:

1. **Beat grid and sync.** Tempo, beats, downbeats and meter accurate enough to beat-match two
   tracks, plus a click audition and a 60 s beat-matched overlap.
2. **Chat editing on the grid.** "Cut beat 16–32 and repeat it twice" becomes typed ops, applied
   by code with clean splices, an exact echo, and undo/redo.
3. **Audio to Web Audio.** A 2–16 bar clip becomes a score JSON and a self-contained HTML page
   that plays it with synthesis only.

The rule throughout: **the LLM interprets, code measures and executes.** Tempo, beat times and
notes come from signal processing and models, never from the LLM.

## Run it

```bash
cp .env.example .env              # add GEMINI_API_KEY
docker compose up                 # builds the test pack, runs all evals, serves http://localhost:8000
```

Without Docker (Python 3.12/3.13, ffmpeg with Rubber Band on PATH):

```bash
pip install -r requirements.txt
pip install --no-deps basic-pitch      # its pinned deps don't build on new Python; ONNX model works
playwright install chromium
python run.py            # test pack + evals + UI on :8000
python run.py evals      # evals only -> evals/results.md, evals/out/
python run.py serve      # UI only
python run.py test       # unit tests
```

Evals replay LLM replies from `evals/out/llm_cache.json`, so they run without a key and cost
nothing. Delete that file to call Gemini live.

**Test pack.** No pack came with the brief, so `evals/make_clips.py` generates a synthetic stand-in
(A1–A7, P1–P4, same roles) with exact ground truth. To use the real pack, put its files in
`evals/testpack/` as `A1.wav`, `A1.beats`, `P1.mid` and so on, then re-run. No code changes needed.

## Architecture

```
chat ──► agent.py (Gemini, JSON schema) ──► typed ops ──► edits.py (resolve, splice, carry grid) ──► audio + grid
                       ▲                                        ▲
             context: grid summary + energy table          grid.py (Beat This! + line fit + onset lock)
                                                                ▲
audio ──────────────────────────────────────────────────────────┘ ──► sync.py (Rubber Band via ffmpeg)

clip ──► recreate.py (HPSS, band peaks, Basic Pitch) ──► score JSON ──► player.html template ──► render.py (headless Chromium)
```

| file | does |
|---|---|
| `beatagent/grid.py` | Beat This! detection, then exact lines through steady stretches, phase-locked to measured onsets; bars and meter |
| `beatagent/sync.py` | stretch ratio and offset from the grids, Rubber Band stretch, latency measured from the kicks |
| `beatagent/edits.py` | typed ops, range resolution, splicing with crossfade, grid carried through edits, undo/redo |
| `beatagent/agent.py` | chat → ops via Gemini, range cross-check, echo; chat → score changes for recreations |
| `beatagent/recreate.py`, `player.html`, `render.py` | clip → score → page → offline WAV |
| `beatagent/server.py`, `index.html` | web UI: waveform + grid, two-track sync, chat, undo/redo, downloads, recreation with mixer |
| `evals/run.py` | every metric in the brief → `evals/results.md` |

**Conventions** (also in every grid JSON):

- **Beats count from 0**, and beat 0 is the first downbeat. In 4/4, beat 16 = bar 5.1.
- **Bars count from 1.**
- **Beat ranges are half-open:** "16–32" = 16 beats.
- **Bar ranges are inclusive:** "bars 5–8" = 4 bars.
- Every edit is echoed as beats, bar.beat and seconds.

## Status

Numbers from `evals/results.md` on the **synthetic** pack. Real music will score lower.

| | status | evidence |
|---|---|---|
| Beat grid JSON (tempo, beats, bars, downbeats, meter, confidence, constant/variable) | ✅ works | `evals/out/part1/*/grid.json` |
| Beat F ≥ 0.90 on steady clips | ✅ A1 0.999, A4 0.992, A6 1.000, A7 0.998 · ❌ **A2 0.667** | A2 is tracked at double tempo (176 vs 88 BPM, half-time trap); AMLt 0.996 shows the phase is right |
| Downbeat F ≥ 0.80 | ✅ all clips with downbeats except **A2 (0.667)** | same cause |
| Drift ≤ 10 ms on the 6-min clip | ✅ max 0.41 ms, all 744 reference beats matched | |
| 6-min analysis < 60 s | ✅ ~7 s on a laptop CPU | |
| Click audition | ✅ works | `click_audition.wav` per clip |
| Sync: kick median ≤ 5 ms, max ≤ 15 ms | ✅ 0.29 ms / 0.68 ms (A7 → A1 from bar 33) | `evals/out/part1/sync_A7_to_A1/` |
| Variable-tempo tracks (A3 drift, A5 change) | ⚠️ partial: A5 perfect, A3 beat F 0.945 with worst beat 61 ms; Beat This loses lock in the slowest stretch | |
| Typed edit ops, echo, undo/redo, graceful errors | ✅ works | `evals/out/part2/*/transcript.md` |
| Clean splices, same sample rate and bit depth | ✅ unit-tested; 24-bit FLAC stays 24-bit FLAC | `tests/test_edits.py` |
| Grid stays true after edits | ✅ re-analysing the edited audio matches the carried grid (F 1.000, 0.03 ms) on steady clips | A3: F 0.917 (tracker, not the edit) |
| Edits on a variable-tempo grid | ✅ works ("last bar before the tempo change" on A5) | session 3 |
| Preview player with waveform + grid | ✅ works | web UI |
| Score JSON, playable page, 2+ parts, offline render | ✅ works, 0 network requests | `evals/out/part3/*` |
| Note F (onset-only, ±50 ms) | ⚠️ P1 0.64, P2 0.63, P3 0.84; snares weakest (P2 0.33) | |
| Rendered onset timing vs original | ✅ onset F 0.86–0.99, median offset 0–6 ms | |
| Chat changes to recreation, per-track mixer | ✅ works ("mute the drums", "brighter bass", "play it at 140") | web UI |
| Docker | ⚠️ `Dockerfile` and compose written but **not built or tested** on this machine | |

### Top three known problems
1. **Half-time tracks come out at double tempo (A2).** Beat F 0.667, and "the first 4 bars" then means
   2 real bars. No automatic fix, because any rule that fixes this breaks real 170 BPM tracks. The
   missing piece is a "the tempo is half that" command.
2. **Synthetic test pack.** Exact truth, but clean audio. The 5 ms onset-bias calibration and the drum
   band split were measured on these clips and must be re-measured on real annotations.
3. **Drum transcription is simple** (HPSS + band peaks). Snares and hats get confused, and kick bleed
   becomes short bass notes. Demucs plus a drum transcription model is the upgrade.

Also: the default LLM is Gemini Flash-Lite, because full Flash was overloaded and Pro was over quota
on the supplied keys. Its parses are guarded by a code check, but a stronger model would push back
better.

See `DECISIONS.md` for the calls and assumptions, `DEVLOG.md` for how they were reached, and
`AI_USAGE.md` for where the AI was wrong.
