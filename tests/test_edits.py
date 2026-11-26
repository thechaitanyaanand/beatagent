"""Executor checks on a synthetic track with a hand-made grid (no model needed)."""
import numpy as np
import pytest
import soundfile as sf

from beatagent.edits import EditError, Fade, Gain, Insert, Move, Range, Remove, Repeat, Session, Target
from beatagent.grid import Grid

SR, BPM, BARS = 44100, 120, 8


@pytest.fixture
def session(tmp_path):
    n = int((BARS * 4 * 0.5 + 1.0) * SR)
    t = np.arange(n) / SR
    y = 0.3 * np.sin(2 * np.pi * 221 * t)  # 221 Hz: the cut lands mid-cycle, so a hard splice jumps
    times = 0.25 + np.arange(BARS * 4) * 0.5
    for k, tb in enumerate(times):  # a short blip on every beat, louder on downbeats
        i = int(round(tb * SR))
        y[i:i + 200] += (0.6 if k % 4 == 0 else 0.3) * np.hanning(200)
    path = tmp_path / "t.wav"
    sf.write(path, y, SR, subtype="PCM_24")
    grid = Grid(times, np.arange(BARS * 4) % 4 + 1, 4, n / SR)
    return Session(path, grid)


def blips(audio):
    """Where the beat blips are in rendered audio (seconds)."""
    x = np.abs(audio[:, 0] - 0.3 * np.sin(2 * np.pi * 221 * np.arange(len(audio)) / SR))
    return np.flatnonzero((x[1:] > 0.05) & (x[:-1] <= 0.05)) / SR


def test_remove_bar_moves_grid_with_audio(session):
    lines = session.apply([Remove(op="remove", range=Range(unit="bar", start=2))])
    a, g = session.cur
    assert len(a) == len(session.audio) - 2 * SR
    assert len(g.times) == BARS * 4 - 4 and g.n_bars == BARS - 1
    assert "bar 2 beat 1 → bar 3 beat 1" in lines[0]


def test_repeat_and_grid_matches_audio(session):
    session.apply([Repeat(op="repeat", range=Range(unit="beat", start=5, end=9), total_plays=3)])
    a, g = session.cur
    assert len(a) == len(session.audio) + 2 * 4 * 0.5 * SR
    found = blips(a)
    assert len(found) >= len(g.times) - 1
    assert np.abs(found[:, None] - g.times[None, :]).min(axis=0).max() < 0.002  # every grid beat sits on a blip


def test_splices_do_not_click(session):
    session.apply([Remove(op="remove", range=Range(unit="beat", start=3, end=6))])
    a, _ = session.cur
    # the tone's phase jumps at the cut (1.25 s -> 2.75 s); a hard cut would spike here.
    # Compare with the same moment in the original, where beat 6 starts with no cut.
    d2 = lambda x, i: np.abs(np.diff(x[i - 400:i + 3, 0], 2)).max()  # noqa: E731
    edited = d2(a, int(round(1.25 * SR)))
    original = d2(session.audio, int(round(2.75 * SR)))
    assert edited < 1.5 * original


def test_undo_redo(session):
    n0 = len(session.cur[0])
    session.apply([Remove(op="remove", range=Range(unit="bar", start=1))])
    session.apply([Gain(op="gain", range=Range(unit="bar", anchor="all"), db=-6)])
    session.undo()
    session.undo()
    assert len(session.cur[0]) == n0
    session.redo()
    assert len(session.cur[0]) == n0 - 2 * SR
    with pytest.raises(EditError):
        session.redo(); session.redo()  # noqa: E702


def test_out_of_range_is_refused_cleanly(session):
    with pytest.raises(EditError, match="don't exist"):
        session.apply([Remove(op="remove", range=Range(unit="bar", start=200, end=210))])
    with pytest.raises(EditError, match="tempo"):
        session.apply([Repeat(op="repeat", range=Range(unit="bar", anchor="before_tempo_change", count=1), total_plays=2)])
    assert session.ptr == 0  # nothing applied


def test_fade_out_last_beats_and_move(session):
    session.apply([Fade(op="fade", range=Range(unit="beat", anchor="last", count=8), direction="out")])
    a, _ = session.cur
    assert abs(a[-1, 0]) < 1e-3
    session.apply([Move(op="move", range=Range(unit="bar", start=1), to=Target(unit="bar", number=4, side="after"))])
    assert len(session.cur[1].times) == BARS * 4


def test_two_ops_in_one_message_use_the_numbers_the_user_saw(session):
    session.apply([Remove(op="remove", range=Range(unit="bar", start=2)),
                   Remove(op="remove", range=Range(unit="bar", start=5))])
    assert len(session.cur[0]) == len(session.audio) - 4 * SR
    with pytest.raises(EditError, match="earlier step"):
        session.apply([Remove(op="remove", range=Range(unit="bar", start=1)),
                       Insert(op="insert", range=Range(unit="bar", start=1), to=Target(unit="bar", number=3))])


def test_export_keeps_rate_and_depth(session, tmp_path):
    path, grid_json = session.export(tmp_path / "out")
    info = sf.info(path)
    assert (info.samplerate, info.subtype) == (SR, "PCM_24")
    assert grid_json["n_beats"] == BARS * 4
