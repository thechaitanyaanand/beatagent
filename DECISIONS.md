# Decisions

The ten calls that shaped this build. The step-by-step history is in `DEVLOG.md`.

## 1. Build a synthetic test pack instead of waiting for the real one
- **Options:** wait for the pack; use public annotated sets (Ballroom, GTZAN); generate clips from code.
- **Choice:** generate A1–A7 and P1–P4 from code, in the roles the brief describes, with exact ground truth.
- **Trade-off:** ground truth is perfect, but the audio is far cleaner than real music, so every number here
  is optimistic. Files use the standard `.beats` and `.mid` formats, so the real pack drops in unchanged.
- **Would change my mind:** getting the real pack. Re-run `python run.py evals` and trust those numbers instead.

## 2. Beat This! plus line fitting, not a tracker alone
- **Options:** librosa (the brief's baseline); madmom/BeatNet; Beat This! alone; Beat This! plus post-processing.
- **Choice:** Beat This! for detection. Code then fits exact lines through steady stretches, locks them to
  measured onsets, merges stretches across breakdowns, and extends a line only while onsets confirm it.
- **Trade-off:** steady tracks get sub-millisecond grids (A1 max error 0.4 ms over 6 minutes, against
  38 ms raw). Drifting live material gets no smoothing, only onset snapping, so A3 stays the weakest clip.
- **Would change my mind:** real recordings with gentle drift where a line fit hurts. Then use a smooth
  tempo curve (spline) instead of lines.

## 3. Beats count from 0, bars from 1
- **Options:** 1-based beats (my first version); 0-based beats.
- **Choice:** beat 0 is the first downbeat, so in 4/4 beat 16 = bar 5.1. Bars are 1-based, as musicians
  count them. Beat ranges are half-open ("16–32" = 16 beats); bar ranges are inclusive ("bars 5–8" = 4 bars).
- **Why:** with 1-based beats, the brief's "cut beat 16–32" became an off-bar loop starting on bar 4
  beat 4. With 0-based beats it is exactly bars 5–8, and "beat 0 to beat 8" is exactly bars 1–2. The
  brief's own wording ("beat 0") points the same way.
- **Would change my mind:** users who think in 1-based beats. The echo always shows bar.beat and
  seconds, so a misreading is visible either way.

## 4. The LLM fills a small typed schema; code does everything else
- **Choice:** Gemini returns `{action, reading, ops[]}`, where each op addresses beats/bars with an
  anchor (absolute, first, last, before_tempo_change, all). Pydantic validates it. Code resolves the
  ranges, checks bounds, cuts the audio and writes the echo with exact times.
- **Guard:** every "X–Y" range the user typed must appear in the ops, or the model gets one retry and then
  we ask the user. This was added after the model dropped the end of "16–32".
- **Trade-off:** the op set is small, so anything outside it is refused. That's intended.

## 5. Echo, then apply (not confirm, then apply)
- **Options:** ask for confirmation before every edit; apply and echo with undo available.
- **Choice:** apply and echo, because every edit is undoable and nothing is destructive. Vague requests
  ("the boring part") are the exception: the agent proposes a measured candidate and asks first.
- **Would change my mind:** slow edits on long files, where a wrong edit costs time.

## 6. All numbers in one message refer to the track the user saw
- "Remove bar 2 and bar 5" removes the two bars the user meant, not the bar that became 5. Every beat is
  tagged with its origin and followed through each step. If an earlier step deleted the target, we refuse.

## 7. Splices: cut on the beat sample, crossfade in the 8 ms before it, equal-gain
- **Choice:** the attack after the cut is untouched. Equal-gain with a raised-cosine shape, because the
  material on both sides of a beat splice is usually the same loop (correlated), and equal-power bumped it
  by +3 dB. The grid is carried through the splice instead of being re-analysed, so it matches the audio
  by construction. The evals re-analyse the edited audio to check this.

## 8. Rubber Band through ffmpeg with `window=short`, latency measured
- **Choice:** ffmpeg already has the Rubber Band filter, so there is no extra binary. The default window
  smeared every other kick by about 25 ms (measured over 7 option combinations); the short window gives
  0.3 ms. Latency is measured by comparing B's kicks before and after stretching with the same detector,
  so the detector's own bias cancels out.
- **Limit:** one constant ratio. A variable-tempo B would need a piecewise stretch map (not built).

## 9. The LLM never writes JavaScript
- **Choice:** the recreation page is a fixed synth template that reads a score JSON. LLM-written code
  running in a user's browser could make network requests, leak data or hang the tab; data can't. It
  also keeps analysis and playback separate: the score is the contract. Chat changes to a recreation
  ("mute the drums", "play it at 140") edit the score, never code.

## 10. Gemini Flash-Lite, plain HTTPS, cached
- **Choice:** the user supplied Gemini keys. Full Flash models returned 503 "high demand" for minutes at a
  time, and Pro is over quota on these keys, so the default is `gemini-3.5-flash-lite` (about 2 s per turn,
  well under the $10 budget). It's one POST with a JSON schema, so there is no SDK. Replies are cached by
  prompt hash, so evals replay identically and for free.

## Assumptions where the brief or a command was unclear
- "Repeat it twice" = 2 plays in total; "loop it four times" = 4 plays in total. The number is always the total.
- A standalone "cut X" = remove X. "Cut X and repeat it" = take X and repeat it in place.
- "Fade out the last 8 beats" includes the tail after the last beat. A fade-out mid-track silences
  everything after it, and the echo says so.
- "Loop the first 4 bars four times, then let the track play on" = repeat bars 1–4 to 4 plays in total,
  with the rest untouched. The lead-in before bar 1 is kept.
- "The last bar before the tempo change" = the last complete bar ending at or before the first point
  where one steady tempo hands over to a clearly different one (>1%). A drifting track has no such point,
  and we say so.
- "Sync A7 to A1 starting at A1's bar 33" = A7's bar 1 lands on A1's bar 33.1, rendered for 60 s.
- "Chop out the boring part" is a matter of taste, so the agent proposes the lowest-energy stretch and asks.
- In 6/8 the beat is the dotted quarter (2 per bar), which is what people tap.
- An ambient intro with no percussion gets no grid. A DJ app might extrapolate one; we don't guess.
- MP3 input is exported as 16-bit WAV, since MP3 has no bit depth and re-encoding adds loss and delay.
- A loop clip for Part 3 is trimmed to whole bars (the brief says "2–16 bars"), so tempo comes from its length.

## Chose not to build
- Automatic half/double tempo correction (A2's half-time trap). Any rule that fixes A2 breaks real 170 BPM
  drum & bass. The right fix is a user "the tempo is half that" command, which is not built.
- Variable-tempo sync (piecewise stretch), Demucs separation, per-beat confidence, a public deployment,
  multi-user sessions.
