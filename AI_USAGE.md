# AI usage

## Tools
- **Claude Code** (Anthropic) wrote most of the code, the synthetic test pack, the evals and the first
  drafts of these docs, working in this repo under my direction.
- **Gemini 3.5 Flash-Lite** is part of the product: it turns chat messages into edit ops at runtime.

## What I delegated
Code generation for every module, test-pack synthesis, the eval harness, and option sweeps (for example,
measuring Rubber Band settings). Choices about what to build, which readings of ambiguous commands to
use, and what counts as "done" were made in conversation and are recorded in `DECISIONS.md`.

## What I reviewed line by line
> **To fill in before submitting:** list the files and functions you actually read line by line, for
> example `edits.py` (range resolution, splice/crossfade, origin tracking) and `grid.py` (line fitting
> and merging). Be honest: the brief grades the review, not abstinence.

## Where the AI was wrong, and how it was caught
Every case below was caught by measuring, not by reading.

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
5. **Smaller ones:** a stray `cat >` in a shell command hung the first baseline run for 10 minutes, and
   it looked like Beat This was slow. The time-signature check sampled one exact point and called house
   music 12/8. A catch-all FastAPI route swallowed `/api/recreate`. Each was caught by a run that
   produced an impossible number or an error.
