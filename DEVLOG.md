# Dev log

Short entries, in the order things happened. Each one says what was decided and why.
Numbers quoted are from the run at that moment; final numbers are in `evals/results.md`.

## Setup

- **No test pack was provided, so I built one.** `evals/make_clips.py` generates A1–A7 and
  P1–P4 from code with the same roles the brief describes (6-min house, half-time hip-hop,
  drifting live band, ambient intro, tempo change, 6/8 dholak, second house track, drum/bass/piano
  loops). Since we placed every hit, ground truth is exact. Files use the Ballroom `.beats`
  format and real `.mid`, so the real pack drops in without code changes.
  Downside: synthetic audio is cleaner than real music, so these numbers are optimistic.
- **Clips exercise the I/O rules too**: A2 is 24-bit FLAC, A3 is MP3, A5 is 48 kHz.
- **MP3 encoder delay was checked, not assumed.** The generator cross-correlates the decoded
  MP3 with the source and refuses to write ground truth if they're offset. (Offset was 0.)
- **Audio is not committed.** It's deterministic (seeded) and rebuilt by one command; the
  6-min WAV alone is 32 MB.
- **Rubber Band via ffmpeg**, which already has the filter compiled in. No extra binary.
- **Basic Pitch installed with `--no-deps`** and run on ONNX: its pinned deps don't build on
  Python 3.13.

## Part 1 — beat grid

- **Beat This! as the tracker** (CPU, 6-min track in ~5 s). Raw baseline on our pack: beat F
  0.67–1.0, but A1's worst beat was 38 ms off, which fails the 10 ms drift target, and the
  half-time hip-hop came out at double tempo.
- **Steady stretches become exact straight lines.** Beat This works in 20 ms frames; fitting a
  line through a steady run averages that jitter away. Drift on the 6-min clip went from 38 ms
  max to 0.4 ms max.
- **Lines are phase-locked to measured onsets**, not to the model's output. Each beat is
  snapped to the strongest onset within ±35 ms; the line is shifted by the median gap.
- **Onset detector lag is a calibration constant (5 ms).** Every steady clip showed the same
  +5 ms; it's the onset-strength window, not the music. Measured, written down as a knob in
  `grid.py`, to be re-measured on real annotations.
- **Breakdowns don't split the grid.** In A1's drumless breakdown the tracker wobbled and the
  grid split in two. Fix: stretches with the same tempo whose phase continues are merged,
  skipping a short wobbly stretch in between; beats in the gap are regenerated from the line.
- **Lines extend only while real onsets confirm them.** Fixed a spurious beat at A5's tempo
  change (40 ms → 0.4 ms).
- **A line may not "crawl" along a drift.** Refitting while extending let a line follow a slow
  tempo drift; now it stops once >10% of its beats miss the line.
- **No drums, no grid.** Unsupported beats at the start/end are trimmed (A4's ambient intro
  had 20 invented beats). A DJ app might extrapolate into the intro; I chose not to guess.
- **Bar phase is voted per steady region**, because a tempo change can shift the bar. A5's
  downbeat F went 0.52 → 1.0.
- **Time signature from measurement**: beats per bar from the model's downbeat spacing;
  6/8 vs 2/4 decided by onset energy on beat-thirds vs beat-quarters. The first version
  compared one exact sample point and flipped house to 12/8; using a ±12 ms max fixed it.
- **Known misses, kept honest**: A2 is tracked at 176 BPM vs 88 reference (the half-time trap;
  F 0.67). A3 (live, drifting 96–104 BPM): Beat This loses lock in the slowest part, F 0.945.

## Part 1 — sync

- **Stretch ratio from local tempo over the overlap**, offset from downbeats: B's bar 1 lands
  on A's bar 33.
- **Kick detector for measuring sync**: <150 Hz envelope, steepest rise, then the half-rise
  point. Steepest rise alone was pulled around by the bassline underneath (2–18 ms noise).
  Half-rise is within 0.2 ms of ground truth on the originals.
- **Caught my own bug**: Rubber Band latency was first measured as "B's kicks vs B's grid", which
  mixed the detector's 16 ms bias into the correction. Now it's "B's kicks before vs after
  stretching" with the same detector, so the bias cancels.
- **Rubber Band `window=short`**: the default window smeared every other kick by ~25 ms
  (measured across 7 option combos). Short window: max 0.34 ms. Trade-off: slightly rougher
  sustained tones, which is fine for beat-matching.
- Result on A7→A1 from bar 33: median kick offset 0.29 ms, max 0.68 ms.

## Part 2 — chat editing

- **Ops are typed (pydantic) and addressed in beats/bars only**: remove, keep, repeat, move,
  insert, fade, gain. A range is `{unit, anchor, start, end, count}` where anchor is absolute,
  first, last, before_tempo_change or all. The LLM fills that in; it never sees or computes seconds.
- **Range conventions**: beat ranges are half-open positions ("beats 16–32" = 16 beats, 4 bars),
  bar ranges are inclusive ("bars 5–8" = 4 bars). These are the readings a DJ would expect, and
  every echo states the exact beats, bars and seconds so a wrong reading is visible.
- **"Repeat it twice" = plays twice in total.** Genuinely ambiguous (could be 3 plays). Chosen
  so "repeat twice" and "loop four times" follow one rule: the number is the total. Echo says
  "it now plays 2 times in a row".
- **"Cut X" alone = remove; "cut X and repeat it" = take X and repeat in place.**
- **"Beat 0" means the user counts from zero**: "beat 0 to beat 8" = our beats 1–9 (8 beats).
  The agent says so in its reading.
- **All numbers in one message refer to the track before that message.** "Remove bar 2 and bar 5"
  removes the bars the user saw, not the bar that became 5. Implemented by tagging every beat
  with where it came from and following it through each step.
- **Edits re-render from the original on undo/redo**, nothing is destructive.
  History is a list of turns plus a pointer.
- **The grid is carried through the splice**, not re-analysed: each beat goes wherever its sample
  went. So the exported grid always matches the edited audio exactly.
- **Splices cut on the beat sample and crossfade in the 8 ms *before* it**, so the beat's attack
  is untouched. First version used equal-power; the test caught a +3 dB bump, because both sides
  of a beat-aligned splice are usually the same loop (correlated). Switched to equal-gain with a
  raised-cosine shape (no corners). The test was also wrong at first: its 220 Hz tone happened to
  be phase-aligned across the cut, so even a hard cut passed. Changed to 221 Hz and verified that
  a hard cut now fails the test.
- **Output keeps sample rate and bit depth.** MP3 input becomes 16-bit WAV (re-encoding would add
  loss and encoder delay).
- **Fades go to/from silence**; a fade-out in the middle silences what follows, and the echo says so.
- **Errors are replies, not crashes**: out-of-range bars, no tempo change on a constant track,
  moving a range into itself, an edit that leaves no audio. Nothing changes when refused.

## Part 2 — the LLM

- **Gemini, called over plain HTTPS with a JSON response schema** (no SDK; one POST). The user
  supplied a Gemini key. Key read from env or `.env`; the provided `.env` spells it
  `GEMINI-API-KEY`, which isn't a valid env var name, so the loader accepts both.
- **Model: `gemini-3.5-flash-lite`.** Full Flash models returned 503 "high demand" for minutes
  and Pro is over quota on this key. Lite answers in ~2 s. Configurable via `BEAT_AGENT_MODEL`.
  Retries with backoff on 429/5xx and timeouts.
- **The model kept dropping the end of "beat 16–32"** and repeated a single beat. The echo made
  it obvious. Two fixes: (1) every range field is required-but-nullable in the schema, plus three
  worked examples in the prompt; (2) a code check: every "X–Y" the user typed must appear in the
  ops, else the model gets one retry with a note, else we refuse and ask. A wrong edit is never
  applied silently.
- **Vague requests get a measured suggestion.** The prompt includes a per-4-bar table of level and
  low-end energy (computed by code). "Chop out the boring part" → "Did you mean bars 97–112
  (low end −5 dB)?", which is exactly A1's breakdown. It asks; it doesn't cut.
- **Out-of-scope requests are refused with what we can do instead** ("make the vocals louder").
- **LLM replies are cached by prompt hash** (`BEAT_AGENT_CACHE`), so evals replay at zero cost and
  identically; a reviewer can delete the cache to run live.

## Part 3 — audio to Web Audio

- **The LLM does not write JavaScript.** The page is a fixed template (`beatagent/player.html`)
  that reads a score JSON. Generated code running in a user's browser could do anything
  (network calls, exfiltration, infinite loops); data can't. It also keeps analysis and playback
  decoupled: the score is the interface.
- **Score times are in beats**, so "play it at 140" is a tempo change, not a re-transcription.
- **Tempo of a loop comes from its length**: an N-bar loop is exactly 60·beats·N/duration BPM.
  The tracker's estimate is only used to pick N. P1's first run came out as 4 bars at 100 BPM
  (the 2-bar clip was too short for the beats-per-bar vote); now 4/4 unless 3/4 or compound
  was measured.
- **HPSS splits drums from pitched parts**; Demucs would be better on real mixes but is a heavy
  install for a 2–16 bar loop. Noted as the first upgrade.
- **Drums: peaks picked per band** (low = kick, mid = snare, high = hat). First version used one
  onset list and labelled each onset; quiet hats under kicks vanished (hat F = 0). Then hats were
  still lost because the bright snare set the high band's scale; now hats are judged only against
  non-snare high peaks (P1 hat F 0 → 0.72).
- **Pitched: Basic Pitch on the harmonic layer**, split into bass (< E3) and keys by register.
  Minimum note length 120 ms, because kick bleed showed up as short bass notes.
- **No drums when the percussive layer is weak** (percussive/harmonic RMS < 0.35): piano attacks
  were becoming a fake drum track on P3 (note F 0.76 → 0.84).
- **Seeded noise in the page**, so offline renders are identical run to run.
- **Offline render** runs the page's own scheduler in an `OfflineAudioContext` in headless
  Chromium (Playwright), and records any network request the page makes (expected: none).

## Evals, UI, packaging

- **One harness, one report.** `evals/run.py` writes every artifact the brief lists (grid, click
  audition, edited audio, sync render, pages, offline renders) and `evals/results.md`.
- **"Grid stays true" is measured, not assumed**: after each session the edited audio is re-analysed
  from scratch and compared with the grid we carried through the edits (F 1.000, 0.03 ms median on
  steady clips).
- **Per-drum F in Part 3**, so a missing hat can't hide behind good piano notes.
- **Web UI is one FastAPI file and one HTML page**, vanilla JS: waveform with the grid drawn on it,
  click-audition toggle, chat with tagged replies (edit / refused / clarify), undo/redo, downloads,
  recreation with mixer and score chat. Sessions live in memory (single user).
- **`python run.py` is the one command** (no `make` on Windows); Docker runs the same thing.
  The Dockerfile is written but not built here.

## Switch to 0-based beats (after reading the first full transcript)

- With 1-based beats, the brief's "cut beat 16–32" echoed as "bar 4 beat 4 → bar 8 beat 4", an off-bar
  loop. With beat 0 = first downbeat, it's exactly bars 5–8, and "beat 0 to beat 8" is exactly bars 1–2.
  The brief says "beat 0", so its author counts from zero. Switched; bars stay 1-based. Echo uses
  DAW-style bar.beat ("5.1") so it can't be confused with global beat numbers.

## Validation pass

- **The model's "reading" invented a number** ("the last bar before the tempo change (bar 33)") that
  contradicted the code's echo (bar 32.1 → 33.1). Readings now only restate numbers the user said;
  exact numbers come from code.
- **Snare band moved from 300–3000 Hz to 150–1000 Hz**: hat noise was landing in the snare band.
  P2 note F 0.56 → 0.63 and render onset F 0.81 → 0.99; P1 dipped 0.68 → 0.64. Kept, net better.
- **Paid Gemini key added later.** Preferred automatically (`GEMINI_API_KEY_PRO`), but full Flash
  was still overloaded and Pro still over quota, so Flash-Lite stays the default.
- **Route bug**: a catch-all `/api/{action}/{sid}` swallowed `/api/recreate`; undo/redo are explicit
  routes now. Found by the headless UI smoke test.
