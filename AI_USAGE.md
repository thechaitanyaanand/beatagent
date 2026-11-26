# AI usage

## Tools
- **Claude Code** (Anthropic) was used as an implementation assistant for writing modules, the synthetic
  test pack, the evaluation harness, and initial doc drafts under my architectural direction.
- **Antigravity / Gemini** was used in the final review pass to cross-check every module against the
  brief and catch edge-case bugs.
- **Gemini 3.5 Flash-Lite** runs inside the product itself to translate chat messages into typed edit ops
  and score adjustments.

## Design ownership vs. what I delegated
I researched and compared the candidate approaches before choosing the architecture and conventions:
Beat This! vs. madmom/BeatNet/librosa for tracking; ffmpeg's built-in Rubber Band filter vs. `pyrubberband`;
HPSS + band-specific peak picking + Basic Pitch (ONNX) vs. Demucs/MT3 for Part 3; a fixed score-driven
Web Audio template vs. LLM-generated JavaScript; 0-based vs. 1-based beat indexing; and equal-gain vs.
equal-power splice crossfades. Those trade-offs and product decisions are documented in `DECISIONS.md`.

I delegated boilerplate and module code generation, deterministic test-clip synthesis (`evals/make_clips.py`),
evaluation harness wiring (`evals/run.py`), and parameter sweeps (such as benchmarking Rubber Band window
modes) to the AI coding assistant, reviewing the outputs and measurements at each stage.

## What I reviewed line by line
I read and verified all key component files across the codebase:
- `beatagent/grid.py` (`Grid`, `steady_segments`, `analyze`: onset snapping, bias calibration, segment merging across breakdowns, per-region downbeat voting, compound-meter check, and empty-beat handling)
- `beatagent/sync.py` (`kick_onsets` half-rise detector, `local_tempo`, `stretch` with `window=short`, and `beatmatch` latency measurement and downbeat alignment)
- `beatagent/edits.py` (`resolve`, `_target_slot`, `describe`, `_render` raised-cosine equal-gain splice crossfading, `_carry` grid propagation, `execute`, and `Session` origin tracking + replay undo/redo)
- `beatagent/agent.py` (`SYSTEM`, `SCHEMA`, `_env`/`api_key`, `call_llm` retry/cache, `energy_table`, `missed_ranges`, `handle`, and `handle_score`)
- `beatagent/recreate.py` (`loop_tempo`, `band_env`, `drums`, `pitched`, `brightness`, `transcribe`, `render_page`), `beatagent/player.html` (Web Audio voices, lookahead loop scheduler, `renderOffline`), and `beatagent/render.py`
- `beatagent/server.py`, `beatagent/index.html`, `evals/run.py`, `evals/make_clips.py`, `tests/test_edits.py`, `Dockerfile`, and `docker-compose.yml`

## Where the AI was wrong, and how it was caught
Every case below was caught by measuring or by line-by-line code review:

1. **Sync latency correction was biased by the detector.** The first version measured Rubber Band
   latency as "B's stretched kicks vs B's grid". But the kick detector fires ~16 ms after the true
   attack, and the grid is bias-corrected, so the "latency" was mostly the detector's own bias, and B
   would have been shifted the wrong way. Caught by checking the detector against known onsets
   (16.3 ms bias). Fixed by comparing B's kicks before and after stretching with the same detector.
2. **Equal-power crossfade added +3 dB at every splice.** It's the textbook choice, but both sides of a
   beat-aligned splice are usually the same loop, so the signals are correlated and the weights sum to
   1.41. Caught by a unit test. The test itself was also wrong at first: its 220 Hz tone was phase-aligned
   across the cut, so even a hard cut passed. Moved the tone to 221 Hz, confirmed a hard cut now fails,
   then switched to an equal-gain raised-cosine fade.
3. **1-based beat numbering made the brief's own command sound wrong.** "Cut beat 16–32" became a loop
   starting on bar 4 beat 4. Caught by reading the eval transcript's echo ("bar 4 beat 4 → bar 8 beat 4").
   Switched to 0-based beats, so it's exactly bars 5–8.
4. **The LLM silently dropped the end of "beat 16–32"** and repeated one beat. The echo made it obvious
   ("Repeated beat 16 (1 beat…)"). Fixed by making schema fields required and adding a code check that
   every range the user typed appears in the ops.
5. **Mid-track `Gain` fade-out ramp read back an overwritten array element.** In `edits.py`, ramping the
   start edge (`g[A:A + r] = np.linspace(1, g[A], r)`) overwrote `g[A]` to `1.0`, so the end-edge ramp
   (`np.linspace(g[A], 1, r)`) stayed flat at `1.0` and stepped at `B - r`. Caught during line-by-line
   review of `edits.py`, verified with a new unit test in `tests/test_edits.py`, and fixed by saving
   the target linear gain before applying the ramps.
6. **Smaller ones:** a stray `cat >` in a shell command hung the first baseline run; the time-signature
   check sampled one exact point and called house music 12/8; a catch-all FastAPI route swallowed
   `/api/recreate`; the UI didn't hide a previous track's recreation panel when loading a new track; and
   `Dockerfile` lacked `torchaudio` and `--extra-index-url` on the CPU wheel install. Each was caught
   either by a failing run or by line-by-line review.
