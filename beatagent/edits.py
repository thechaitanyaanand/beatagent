"""Typed edit operations addressed in beats/bars, and a session with undo/redo.

The LLM only ever produces these objects. Everything below is deterministic:
ranges are resolved against the grid, audio is spliced at beat sample positions,
and the grid is carried through the same splice so it always matches the audio.

Range conventions (shown to the LLM and echoed to the user):
  beats: half-open positions. "beats 16-32" = from the start of beat 16 to the start of beat 32 (16 beats).
  bars:  inclusive. "bars 5-8" = 4 whole bars. "bar 5" = one bar.
"""
from typing import Annotated, Literal, Optional, Union

import numpy as np
from pydantic import BaseModel, Field

from .audio import load, save
from .grid import Grid, analyze

XFADE = 0.008  # splice crossfade, ending exactly on the beat so the attack is untouched
EDGE = 0.003   # fade where the file now starts or ends mid-sound
MAX_SECONDS = 30 * 60


class EditError(ValueError):
    """A request we refuse with a clear reason (out of range, impossible, unclear)."""


class Range(BaseModel):
    unit: Literal["beat", "bar"]
    anchor: Literal["absolute", "first", "last", "before_tempo_change", "all"] = "absolute"
    start: Optional[int] = Field(None, description="absolute only: first beat/bar number")
    end: Optional[int] = Field(None, description="absolute only: beats = position to stop at (exclusive); bars = last bar (inclusive). Omit for a single beat/bar")
    count: Optional[int] = Field(None, ge=1, description="first/last/before_tempo_change: how many beats/bars")


class Target(BaseModel):
    unit: Literal["beat", "bar"]
    number: int
    side: Literal["before", "after"] = "before"


class Remove(BaseModel):
    op: Literal["remove"]
    range: Range


class Keep(BaseModel):
    op: Literal["keep"]
    range: Range


class Repeat(BaseModel):
    op: Literal["repeat"]
    range: Range
    total_plays: int = Field(ge=2, le=64, description="how many times the range plays in total (original included)")


class Move(BaseModel):
    op: Literal["move"]
    range: Range
    to: Target


class Insert(BaseModel):
    op: Literal["insert"]
    range: Range
    to: Target


class Fade(BaseModel):
    op: Literal["fade"]
    range: Range
    direction: Literal["in", "out"]


class Gain(BaseModel):
    op: Literal["gain"]
    range: Range
    db: float = Field(ge=-60, le=24)


Op = Annotated[Union[Remove, Keep, Repeat, Move, Insert, Fade, Gain], Field(discriminator="op")]


# ---------- resolving musical ranges to beat slots ----------
# A slot is an index into grid.times. -1 = start of file, len(times) = end of file.

def _t(grid, slot):
    return 0.0 if slot < 0 else grid.duration if slot >= len(grid.times) else float(grid.times[slot])


def _bar_slot(grid, bar):
    hits = np.flatnonzero((grid.bars == bar) & (grid.pos == 1))
    return int(hits[0]) if len(hits) else None


def _complete_bars(grid):
    """Bar numbers that have all their beats (the last bar is often a stub before the tail)."""
    bars, counts = np.unique(grid.bars[grid.bars > 0], return_counts=True)
    return bars[counts == grid.bpb]


def resolve(grid, r: Range):
    n = len(grid.times)
    nums = grid.numbers
    if n == 0:
        raise EditError("This track has no beat grid (no beats were detected), so beat/bar edits are not possible.")
    if r.anchor == "all":
        return -1, n
    if r.anchor == "absolute" and r.start is None:
        raise EditError("I need a start beat or bar for that range.")

    if r.unit == "beat":
        lo, hi = int(nums[0]), int(nums[-1])
        if r.anchor == "absolute":
            s, e = r.start, r.end if r.end is not None else r.start + 1
            if not (lo <= s < e <= hi + 1):
                raise EditError(f"Beats {s}-{e} are outside this track: it has beats {lo}-{hi} "
                                f"(beat ranges stop before the end number, so the last usable end is {hi + 1}).")
            return int(np.flatnonzero(nums == s)[0]), (int(np.flatnonzero(nums == e)[0]) if e <= hi else n)
        c = r.count or 1
        if c > n:
            raise EditError(f"The track only has {n} beats.")
        if r.anchor == "first":
            a = int(np.flatnonzero(nums == 1)[0]) if (nums == 1).any() else 0
            if a + c > n:
                raise EditError(f"There are only {n - a} beats from beat 1.")
            return a, a + c
        if r.anchor == "last":
            return n - c, n
        c0 = _change_slot(grid)
        if c0 - c < 0:
            raise EditError(f"There are only {c0} beats before the tempo change.")
        return c0 - c, c0

    # bars
    full = _complete_bars(grid)
    if not len(full):
        raise EditError("No complete bars were found in this track.")
    lo, hi = int(full.min()), int(full.max())
    if r.anchor == "absolute":
        s, e = r.start, r.end if r.end is not None else r.start
        if not (lo <= s <= e <= hi):
            raise EditError(f"Bars {s}-{e} don't exist: this track has bars {lo}-{hi} "
                            f"({hi - lo + 1} complete bars of {grid.bpb} beats).")
    elif r.anchor == "first":
        s, e = lo, lo + (r.count or 1) - 1
        if e > hi:
            raise EditError(f"The track only has {hi - lo + 1} complete bars.")
    elif r.anchor == "last":
        s, e = hi - (r.count or 1) + 1, hi
        if s < lo:
            raise EditError(f"The track only has {hi - lo + 1} complete bars.")
        return _bar_slot(grid, s), n  # include the tail after the last bar
    else:  # before_tempo_change
        c0 = _change_slot(grid)
        downs = np.flatnonzero(grid.pos == 1)
        downs_before = downs[downs <= c0]
        k = r.count or 1
        if len(downs_before) < k + 1:
            raise EditError(f"There aren't {k} complete bars before the tempo change.")
        return int(downs_before[-k - 1]), int(downs_before[-1])
    a = _bar_slot(grid, s)
    b = _bar_slot(grid, e + 1)
    return a, (b if b is not None else n)


def _change_slot(grid):
    ch = grid.tempo_changes()
    if not ch:
        j = grid.to_json()
        kind = "constant" if j["constant_tempo"] else "gradually drifting (no single change point)"
        raise EditError(f"I don't find a tempo change in this track: the tempo is {kind}, around {j['tempo']:.1f} BPM.")
    return ch[0]


def _target_slot(grid, t: Target):
    n = len(grid.times)
    if t.unit == "beat":
        nums = grid.numbers
        num = t.number + (1 if t.side == "after" else 0)
        if num == nums[-1] + 1:
            return n
        hit = np.flatnonzero(nums == num)
        if not len(hit):
            raise EditError(f"Beat {t.number} doesn't exist (beats {nums[0]}-{nums[-1]}).")
        return int(hit[0])
    bar = t.number + (1 if t.side == "after" else 0)
    s = _bar_slot(grid, bar)
    if s is None:
        if bar == grid.n_bars + 1:
            return n
        raise EditError(f"Bar {t.number} doesn't exist (bars 1-{grid.n_bars}).")
    return s


# ---------- describing (the echo) ----------

def bar_beat(grid, slot):
    if slot >= len(grid.times):
        return "end of track"
    if slot < 0:
        return "start of file"
    return f"bar {grid.bars[slot]} beat {grid.pos[slot]}"


def describe(grid, a, b):
    nb = b - a if a >= 0 else b
    if a < 0 and b >= len(grid.times):
        return f"the whole track (0.000 s → {grid.duration:.3f} s)"
    nums = grid.numbers
    first = int(nums[a]) if a >= 0 else int(nums[0])
    end = f"beat {int(nums[b])}" if b < len(nums) else "the end of the track"
    span = f"from beat {first} up to {end}" if nb != 1 else f"beat {first}"
    return (f"{span} ({nb} beat{'s' if nb != 1 else ''}, {bar_beat(grid, a)} → {bar_beat(grid, b)}), "
            f"{_t(grid, a):.3f} s → {_t(grid, b):.3f} s ({_t(grid, b) - _t(grid, a):.3f} s)")


# ---------- executing ----------

def _render(src, pieces, sr):
    """Concatenate source pieces; crossfade into every joint over the XFADE before it."""
    pieces = [(s, e) for s, e in pieces if e > s]
    if not pieces:
        raise EditError("That edit would leave no audio at all.")
    out = np.concatenate([src[s:e] for s, e in pieces]).copy()
    L = int(XFADE * sr)
    q = 0
    joints = []
    for k, (s, e) in enumerate(pieces):
        if k and pieces[k - 1][1] != s:
            joints.append((q, s))
        q += e - s
    for q, s in joints:
        n = min(L, q, s)  # need pre-roll in the source and room in the output
        if n <= 0:
            continue
        # equal-gain (weights sum to 1): both sides of a beat-aligned splice are usually the
        # same loop, and equal-power would bump correlated material by +3 dB.
        # Raised-cosine shape so the ramp itself has no corners.
        ramp = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n, endpoint=False)))[:, None]
        out[q - n:q] = out[q - n:q] * (1 - ramp) + src[s - n:s] * ramp
    E = min(int(EDGE * sr), len(out))
    if pieces[0][0] > 0:
        out[:E] *= np.linspace(0, 1, E)[:, None]
    if pieces[-1][1] < len(src):
        out[len(out) - E:] *= np.linspace(1, 0, E)[:, None]
    return out


def _carry(grid, pieces, sr, new_len):
    """Move the grid through the same splice: each beat goes wherever its sample went."""
    bs = np.round(grid.times * sr).astype(np.int64)
    t, p, o, off = [], [], [], 0
    for s, e in pieces:
        if e <= s:
            continue
        m = (bs >= s) & (bs < e)
        t.append((off + bs[m] - s) / sr)
        p.append(grid.pos[m])
        o.append(grid.origin[m])
        off += e - s
    return Grid(np.concatenate(t), np.concatenate(p), grid.bpb, new_len / sr, grid.time_signature,
                grid.confidence, np.concatenate(o), dict(grid.meta))


def execute(audio, sr, grid, op, a, b, p=None):
    """Apply one op to slots [a, b) (and target slot p). Returns (audio, grid, echo line)."""
    S = lambda slot: int(round(_t(grid, slot) * sr))  # noqa: E731
    A, B, Ln = S(a), S(b), len(audio)
    what = describe(grid, a, b)

    if op.op in ("fade", "gain"):
        out = audio.copy()
        g = np.ones(Ln, np.float32)
        if op.op == "gain":
            r = max(1, min(int(0.005 * sr), (B - A) // 2))
            g[A:B] = 10 ** (op.db / 20)
            if A > 0:
                g[A:A + r] = np.linspace(1, g[A], r)
            if B < Ln:
                g[B - r:B] = np.linspace(g[A], 1, r)
            line = f"Gain {op.db:+.1f} dB on {what}."
        elif op.direction == "out":
            g[A:B] = np.cos(np.linspace(0, np.pi / 2, B - A))
            g[B:] = 0
            line = f"Fade out over {what}" + (f"; everything after {B / sr:.3f} s is now silent." if B < Ln else ".")
        else:
            g[A:B] = np.sin(np.linspace(0, np.pi / 2, B - A))
            g[:A] = 0
            line = f"Fade in over {what}" + (f"; everything before {A / sr:.3f} s is now silent." if A > 0 else ".")
        return out * g[:, None], grid, line

    if op.op == "remove":
        pieces, line = [(0, A), (B, Ln)], f"Removed {what}."
    elif op.op == "keep":
        pieces, line = [(A, B)], f"Kept only {what}."
    elif op.op == "repeat":
        pieces = [(0, B)] + [(A, B)] * (op.total_plays - 1) + [(B, Ln)]
        line = f"Repeated {what}: it now plays {op.total_plays} times in a row, then the track continues."
    else:
        P = S(p)
        where = f"{bar_beat(grid, p)} ({P / sr:.3f} s)"
        if op.op == "insert":
            pieces, line = [(0, P), (A, B), (P, Ln)], f"Copied {what} and inserted it at {where}."
        elif P <= A:
            pieces, line = [(0, P), (A, B), (P, A), (B, Ln)], f"Moved {what} to {where}."
        elif P >= B:
            pieces, line = [(0, A), (B, P), (A, B), (P, Ln)], f"Moved {what} to {where}."
        else:
            raise EditError("I can't move a range to a point inside itself.")
    out = _render(audio, pieces, sr)
    if len(out) > MAX_SECONDS * sr:
        raise EditError(f"That would make the track {len(out) / sr / 60:.0f} minutes long; I stop at {MAX_SECONDS // 60}.")
    return out, _carry(grid, pieces, sr, len(out)), line


class Session:
    """One track being edited. History is a list of turns; undo/redo moves a pointer and
    re-renders from the original, so nothing is ever lost."""

    def __init__(self, path, grid=None):
        self.path = path
        self.audio, self.sr, self.info = load(path)
        self.grid0 = grid or analyze(audio=self.audio, sr=self.sr)
        self.turns, self.ptr = [], 0
        self.cur = (self.audio, self.grid0)

    @property
    def grid(self):
        return self.cur[1]

    def _run(self, audio, grid, ops):
        grid = Grid(grid.times, grid.pos, grid.bpb, grid.duration, grid.time_signature, grid.confidence, None, dict(grid.meta))
        n0 = len(grid.times)
        # resolve everything against the grid the user saw, before touching audio
        plan = [(op, *resolve(grid, op.range), _target_slot(grid, op.to) if hasattr(op, "to") else None) for op in ops]
        lines = []
        for op, a, b, p in plan:
            def now(slot):
                if slot is None or slot < 0:
                    return slot
                if slot >= n0:
                    return len(grid.times)
                hit = np.flatnonzero(grid.origin == slot)
                if not len(hit):
                    raise EditError("An earlier step in this message already removed part of that range; "
                                    "please send the edits one at a time.")
                return int(hit[0])
            audio, grid, line = execute(audio, self.sr, grid, op, now(a), now(b), now(p))
            lines.append(line)
        return audio, grid, lines

    def apply(self, ops):
        audio, grid, lines = self._run(*self.cur, ops)
        self.turns = self.turns[: self.ptr] + [ops]
        self.echoes = getattr(self, "echoes", [])[: self.ptr] + [lines]
        self.ptr += 1
        self.cur = (audio, grid)
        return lines + [self.summary()]

    def _replay(self):
        audio, grid = self.audio, self.grid0
        for ops in self.turns[: self.ptr]:
            audio, grid, _ = self._run(audio, grid, ops)
        self.cur = (audio, grid)

    def undo(self):
        if self.ptr == 0:
            raise EditError("Nothing to undo.")
        self.ptr -= 1
        self._replay()
        return ["Undid: " + " ".join(self.echoes[self.ptr]), self.summary()]

    def redo(self):
        if self.ptr >= len(self.turns):
            raise EditError("Nothing to redo.")
        self.ptr += 1
        self._replay()
        return ["Redid: " + " ".join(self.echoes[self.ptr - 1]), self.summary()]

    def summary(self):
        a, g = self.cur
        d = len(a) / self.sr
        d0 = len(self.audio) / self.sr
        peak = float(np.abs(a).max()) if a.size else 0.0
        warn = f" Warning: peak is {20 * np.log10(peak):+.1f} dBFS and will clip on export." if peak > 1.0 else ""
        return (f"Track is now {int(d // 60)}:{d % 60:06.3f} long ({d - d0:+.3f} s vs original), "
                f"{len(g.times)} beats, {g.n_bars} bars.{warn}")

    def export(self, stem):
        audio, grid = self.cur
        path, _ = save(stem, audio, self.sr, self.info)
        return path, grid.to_json()
