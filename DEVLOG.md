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
