"""Chat message -> typed edit ops (LLM), then code validates, executes and echoes.

The LLM never sees seconds and never does arithmetic on the audio: it maps words to
beat/bar ranges. Exact times in every reply come from the grid, not from the model.
"""
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from pydantic import TypeAdapter, ValidationError

from .edits import EditError, Op

ROOT = Path(__file__).resolve().parents[1]
MODEL = os.environ.get("BEAT_AGENT_MODEL", "gemini-3.5-flash-lite")  # full Flash was 503-overloaded on our key
CACHE = os.environ.get("BEAT_AGENT_CACHE")  # JSON file: replay identical prompts with no API call

SYSTEM = """You turn one chat message about an audio track into edit operations on its beat grid.
You only interpret. Code resolves beats/bars to exact times, validates ranges and edits the audio.

Grid conventions (the user sees these too):
- Beats are numbered from 1; beat 1 is the first downbeat. Bars are numbered from 1.
- Beat ranges are half-open positions: "beats 16-32" = start of beat 16 to start of beat 32 (16 beats) -> start=16, end=32.
- Bar ranges are inclusive: "bars 5-8" -> start=5, end=8. "bar 5" -> start=5, end=5.
- "first N beats/bars" -> anchor=first, count=N. "last N ..." -> anchor=last, count=N (includes the tail).
- "the N bars/beats before the tempo change" -> anchor=before_tempo_change, count=N.
- whole track -> anchor=all.
- All numbers in one message refer to the track as it is BEFORE this message.

Operations:
- remove(range); keep(range) = keep only that, drop the rest;
- repeat(range, total_plays) = the range plays total_plays times in a row in place, then the track continues;
- move(range, to) / insert(range, to) = cut/copy the range to before/after a beat or bar;
- fade(range, direction in|out) = fade to/from silence over the range;
- gain(range, db).

Readings we have decided (say which one you used in "reading"):
- "repeat it twice" / "play it twice" = total_plays 2. "loop it four times" = total_plays 4.
- A standalone "cut X" means remove X. "cut X and repeat/loop it" means take X and repeat it in place.
- If the user numbers from 0 ("beat 0 to beat 8"), they count from zero: beat 0 is our beat 1, so
  "beat 0 to beat 8" = start 1, end 9 (the first 8 beats). Say so.
- "undo" / "redo" -> action undo / redo.

When to not edit:
- Subjective or vague targets ("the boring part", "the bad bit"): action=clarify. Use the energy
  table to propose a concrete candidate range with its numbers and ask the user to confirm.
- Requests outside what these operations can do (pitch, EQ, new sounds, other tracks): action=reject,
  and say what you can do instead.
- Do NOT reject numbers you think are out of range; emit the op and code will check it.

Examples (range objects):
- "cut beat 16-32 and repeat it twice" -> repeat, {unit: beat, anchor: absolute, start: 16, end: 32, count: null}, total_plays 2
- "remove bar 5" -> remove, {unit: bar, anchor: absolute, start: 5, end: 5, count: null}
- "fade out the last 8 beats" -> fade out, {unit: beat, anchor: last, start: null, end: null, count: 8}

Output JSON only. "reading" = one plain sentence restating what you will do, in beats/bars.
"message" = what to tell the user for clarify/reject (empty for edits)."""

RANGE = {"type": "object", "properties": {
    "unit": {"type": "string", "enum": ["beat", "bar"]},
    "anchor": {"type": "string", "enum": ["absolute", "first", "last", "before_tempo_change", "all"]},
    "start": {"type": "integer", "nullable": True},
    "end": {"type": "integer", "nullable": True},
    "count": {"type": "integer", "nullable": True}},
    # every field required (nullable): the small model silently dropped "end" when it was optional
    "required": ["unit", "anchor", "start", "end", "count"]}
OP = {"type": "object", "properties": {
    "op": {"type": "string", "enum": ["remove", "keep", "repeat", "move", "insert", "fade", "gain"]},
    "range": RANGE,
    "total_plays": {"type": "integer", "nullable": True},
    "db": {"type": "number", "nullable": True},
    "direction": {"type": "string", "enum": ["in", "out"], "nullable": True},
    "to": {"type": "object", "nullable": True, "properties": {
        "unit": {"type": "string", "enum": ["beat", "bar"]}, "number": {"type": "integer"},
        "side": {"type": "string", "enum": ["before", "after"]}}, "required": ["unit", "number", "side"]}},
    "required": ["op", "range"]}
SCHEMA = {"type": "object", "properties": {
    "action": {"type": "string", "enum": ["edit", "undo", "redo", "clarify", "reject"]},
    "reading": {"type": "string"},
    "message": {"type": "string"},
    "ops": {"type": "array", "items": OP}},
    "required": ["action", "reading", "message", "ops"],
    "propertyOrdering": ["action", "reading", "message", "ops"]}

OPS = TypeAdapter(list[Op])


class LLMError(RuntimeError):
    pass


def api_key():
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            name, _, val = line.partition("=")
            if name.strip().replace("-", "_").upper() == "GEMINI_API_KEY":
                return val.strip().strip("'\"")
    return None


def call_llm(system, prompt):
    key = hashlib.sha256(f"{MODEL}\n{system}\n{prompt}".encode()).hexdigest()[:16]
    cache = json.loads(Path(CACHE).read_text()) if CACHE and Path(CACHE).exists() else {}
    if key in cache:
        return cache[key]
    k = api_key()
    if not k:
        raise LLMError("no GEMINI_API_KEY set (see .env.example)")
    body = {"systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json", "responseSchema": SCHEMA}}
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "x-goog-api-key": k})
    for attempt in range(4):  # busy/rate-limited models are common; back off and retry
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.load(r)
            out = json.loads(data["candidates"][0]["content"]["parts"][0]["text"])
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503) and attempt < 3:
                time.sleep(2 * 2 ** attempt)
                continue
            msg = json.loads(e.read() or b"{}").get("error", {}).get("message", "")
            raise LLMError(f"HTTP {e.code} {msg[:120]}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < 3:
                time.sleep(2)
                continue
            raise LLMError(f"network: {e}") from e
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise LLMError(f"unexpected reply from the model: {e}") from e
    if CACHE:
        cache[key] = out
        Path(CACHE).write_text(json.dumps(cache, indent=1))
    return out


def energy_table(session, block=4):
    """Per block of bars: loudness and low-end (kick/bass) vs the track median. Code measures,
    the LLM only reads this to suggest a candidate for vague requests."""
    from scipy.signal import butter, sosfilt
    audio, g = session.cur
    mono = audio.mean(axis=1)
    low = sosfilt(butter(2, 150, "lowpass", fs=session.sr, output="sos"), mono)
    downs = np.flatnonzero(g.pos == 1)
    rows = []
    for i in range(0, len(downs), block):
        a = int(g.times[downs[i]] * session.sr)
        j = downs[i + block] if i + block < len(downs) else None
        b = int(g.times[j] * session.sr) if j is not None else len(mono)
        rows.append((int(g.bars[downs[i]]), int(g.bars[downs[min(i + block, len(downs)) - 1]]),
                     np.sqrt(np.mean(mono[a:b] ** 2)) + 1e-9, np.sqrt(np.mean(low[a:b] ** 2)) + 1e-9))
    if not rows:
        return "(no bars)"
    med, lmed = np.median([r[2] for r in rows]), np.median([r[3] for r in rows])
    return "\n".join(f"bars {s}-{e}: level {20 * np.log10(r / med):+.1f} dB, low end {20 * np.log10(l / lmed):+.1f} dB"
                     for s, e, r, l in rows)


def context(session):
    j = session.grid.to_json()
    nums = session.grid.numbers
    lines = [
        f"duration {j['duration']:.2f} s; tempo {j['tempo']} BPM ({'constant' if j['constant_tempo'] else 'variable'}); "
        f"time signature {j['time_signature']} ({j['beats_per_bar']} beats per bar)",
        f"beats {int(nums[0])}-{int(nums[-1])}; {j['n_bars']} bars" if len(nums) else "no beats detected",
        "tempo regions: " + "; ".join(f"from beat {s['start_beat']} at {s['bpm']} BPM" for s in j["segments"]),
        f"edits applied so far: {session.ptr} (undo available: {session.ptr > 0}, redo available: {session.ptr < len(session.turns)})",
        "energy by bars:\n" + energy_table(session),
    ]
    return "\n".join(lines)


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items() if v is not None}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    return x


def handle(session, text, history=()):
    """One chat turn. Returns {"text", "action", "ops", "ok"}; never raises for user input."""
    def reply(msg, action, ok=True, ops=None):
        return {"text": msg, "action": action, "ok": ok, "ops": ops or []}

    if not text.strip():
        return reply("Say what you'd like to change, in beats or bars (e.g. \"remove bar 5\").", "clarify", False)
    hist = "\n".join(f"{who}: {msg}" for who, msg in list(history)[-6:]) or "(none)"
    prompt = f"TRACK\n{context(session)}\n\nRECENT CHAT\n{hist}\n\nUSER MESSAGE\n{text}"
    for attempt in range(2):
        try:
            out = call_llm(SYSTEM, prompt)
        except LLMError as e:
            return reply(f"I couldn't reach the language model ({e}). Nothing was changed.", "error", False)
        action, reading = out.get("action"), out.get("reading", "").strip()
        if action != "edit":
            break
        try:
            ops = OPS.validate_python(_clean(out.get("ops") or []))
        except ValidationError as e:
            return reply(f"I couldn't turn that into a valid edit ({e.errors()[0]['msg']}). Nothing was changed.",
                         "refused", False)
        missed = missed_ranges(text, ops)
        if not missed:
            break
        # code noticed the parse dropped a number the user said; let the model look again once
        prompt += (f"\n\nNOTE FROM THE EDITOR: your previous ops did not use the range(s) "
                   f"{', '.join(f'{a}-{b}' for a, b in missed)} that the user wrote. Re-read the message.")
    else:
        return reply(f"I read your message as: {reading} But that doesn't match the range "
                     f"{missed[0][0]}–{missed[0][1]} you wrote, so I changed nothing. Could you rephrase?",
                     "clarify", False)

    if action in ("clarify", "reject"):
        return reply(out.get("message") or reading, action, action == "clarify")
    try:
        if action == "undo":
            lines = session.undo()
        elif action == "redo":
            lines = session.redo()
        else:
            if not ops:
                return reply("I read that as an edit but couldn't turn it into one. "
                             "Could you say it in beats or bars?", "clarify", False)
            lines = session.apply(ops)
            return reply("\n".join([f"Reading: {reading}"] + [f"- {x}" for x in lines]), "edit",
                         ops=[o.model_dump() for o in ops])
    except EditError as e:
        return reply((f"Reading: {reading}\n" if reading else "") + f"I didn't change anything: {e}", "refused", False)
    return reply("\n".join(lines), action)


RANGE_RE = re.compile(r"(\d+)\s*(?:-|–|—|to|through|thru|until)\s*(\d+)")


def missed_ranges(text, ops):
    """Ranges written as "X-Y" in the message that no op uses (zero-based X-1..Y-1 also counts)."""
    have = set()
    for o in ops:
        r = o.range
        if r.anchor == "absolute" and r.start is not None:
            e = r.end if r.end is not None else r.start
            have |= {(r.start, e), (r.start - 1, e - 1)}
    return [(int(a), int(b)) for a, b in RANGE_RE.findall(text) if (int(a), int(b)) not in have]
