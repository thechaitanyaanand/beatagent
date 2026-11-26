"""Web UI: pick or upload a track, see the waveform + grid, edit by chat, recreate clips."""
import io
import json
import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel

from . import agent
from .audio import click_audition, load
from .edits import EditError, Session
from .sync import beatmatch, kick_onsets, match

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "evals" / "testpack"
WORK = Path(tempfile.mkdtemp(prefix="beatagent_"))
AUDIO_EXT = {".wav", ".flac", ".mp3", ".aiff"}
app = FastAPI()
SESSIONS = {}  # ponytail: in-memory, single process; a restart forgets open sessions


class Chat(BaseModel):
    text: str


class SyncReq(BaseModel):
    track_b: str
    start_bar: int = 33
    seconds: float = 60.0


def state(sid):
    s = SESSIONS.get(sid)
    if not s:
        raise HTTPException(404, "session not found (the server may have restarted); load the track again")
    return s


def view(sid):
    s = SESSIONS[sid]
    audio, g = s["session"].cur
    mono = np.abs(audio.mean(axis=1))
    n = 1200
    peaks = [float(c.max()) if len(c) else 0.0 for c in np.array_split(mono, n)]
    return {"sid": sid, "name": s["name"], "duration": len(audio) / s["session"].sr, "peaks": peaks,
            "beats": [[round(float(t), 4), int(p)] for t, p in zip(g.times, g.pos)],
            "grid": {k: v for k, v in g.to_json().items() if k not in ("beats", "downbeats")},
            "can_undo": s["session"].ptr > 0, "can_redo": s["session"].ptr < len(s["session"].turns),
            "version": s["session"].ptr * 1000 + len(s["history"])}


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).with_name("index.html")).read_text(encoding="utf-8")


@app.get("/api/tracks")
def tracks():
    return sorted(p.name for p in PACK.glob("*") if p.suffix.lower() in AUDIO_EXT) if PACK.exists() else []


@app.post("/api/load/{name}")
def load_track(name: str):
    path = PACK / Path(name).name
    if not path.exists():
        raise HTTPException(404, f"{name} not found in the test pack")
    return _open(path, name)


@app.post("/api/upload")
async def upload(file: UploadFile):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in AUDIO_EXT:
        raise HTTPException(400, f"unsupported file type '{ext}'; use WAV, FLAC or MP3")
    path = WORK / f"{uuid.uuid4().hex[:8]}{ext}"
    with path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        sf.info(str(path))
    except RuntimeError:
        raise HTTPException(400, "that file couldn't be read as audio")
    return _open(path, file.filename)


def _open(path, name):
    sid = uuid.uuid4().hex[:8]
    SESSIONS[sid] = {"session": Session(path), "name": name, "history": [], "path": path}
    return view(sid)


@app.post("/api/chat/{sid}")
def chat(sid: str, body: Chat):
    s = state(sid)
    r = agent.handle(s["session"], body.text[:2000], s["history"])
    s["history"] += [("user", body.text), ("agent", r["text"])]
    return {"reply": r, **view(sid)}


@app.post("/api/undo/{sid}")
def undo(sid: str):
    return _history(sid, "undo")


@app.post("/api/redo/{sid}")
def redo(sid: str):
    return _history(sid, "redo")


def _history(sid, action):
    s = state(sid)
    try:
        lines = getattr(s["session"], action)()
        reply = {"text": "\n".join(lines), "action": action, "ok": True}
    except EditError as e:
        reply = {"text": str(e), "action": action, "ok": False}
    s["history"] += [("user", action), ("agent", reply["text"])]
    return {"reply": reply, **view(sid)}


def _wav(audio, sr):
    buf = io.BytesIO()
    sf.write(buf, np.clip(audio, -1, 1), sr, format="WAV", subtype="PCM_16")
    return Response(buf.getvalue(), media_type="audio/wav")


@app.get("/api/audio/{sid}")
def audio(sid: str, v: int = 0, click: int = 0):
    s = state(sid)["session"]
    a, g = s.cur
    return _wav(click_audition(a, s.sr, g) if click else a, s.sr)


@app.get("/api/export/{sid}")
def export(sid: str):
    s = state(sid)
    path, _ = s["session"].export(WORK / f"{sid}_edited")
    return FileResponse(path, filename=f"edited_{Path(s['name']).stem}{path.suffix}")


@app.get("/api/grid/{sid}")
def grid(sid: str):
    s = state(sid)
    return Response(json.dumps(s["session"].grid.to_json(), indent=1), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="grid_{Path(s["name"]).stem}.json"'})


@app.post("/api/recreate/{sid}")
def recreate(sid: str):
    from .recreate import render_page, transcribe
    s = state(sid)
    if s["session"].ptr:
        raise HTTPException(400, "recreate works on the original clip; undo the edits or reload it")
    if len(s["session"].audio) / s["session"].sr > 40:
        raise HTTPException(400, "recreation is for short loops (2-16 bars); this track is too long")
    s["score"] = transcribe(s["path"])
    render_page(s["score"], WORK / f"{sid}.html")
    return {"url": f"/page/{sid}", "score": s["score"]}


@app.post("/api/score_chat/{sid}")
def score_chat(sid: str, body: Chat):
    from .recreate import render_page
    s = state(sid)
    if "score" not in s:
        raise HTTPException(400, "recreate the clip first")
    s["score"], text = agent.handle_score(s["score"], body.text[:2000])
    render_page(s["score"], WORK / f"{sid}.html")
    return {"url": f"/page/{sid}", "reply": text}


@app.get("/page/{sid}", response_class=HTMLResponse)
def page(sid: str):
    p = WORK / f"{Path(sid).name}.html"
    if not p.exists():
        raise HTTPException(404)
    return p.read_text(encoding="utf-8")


@app.post("/api/sync/{sid}")
def sync_tracks(sid: str, body: SyncReq):
    s = state(sid)
    b_path = PACK / Path(body.track_b).name
    if not b_path.exists():
        raise HTTPException(404, f"{body.track_b} not found in the test pack")
    try:
        r = beatmatch(s["path"], b_path, WORK / f"{sid}_sync", start_bar=body.start_bar,
                      seconds=body.seconds, grid_a=s["session"].grid0)
    except ValueError as e:
        raise HTTPException(400, str(e))
    A, sr, _ = load(r["files"]["a"])
    B, _, _ = load(r["files"]["b"])
    ka, kb = kick_onsets(A.mean(axis=1), sr), kick_onsets(B.mean(axis=1), sr)
    off = match(ka, kb) * 1000
    r |= {"kicks_a": len(ka), "kicks_matched": len(off),
          "kick_offset_median_ms": round(float(np.median(np.abs(off))), 2) if len(off) else None,
          "kick_offset_max_ms": round(float(np.abs(off).max()), 2) if len(off) else None}
    s["sync_files"] = r["files"]
    return {k: v for k, v in r.items() if k != "files"}


@app.get("/api/sync_audio/{sid}")
def sync_audio(sid: str, part: str = "mix"):
    files = state(sid).get("sync_files") or {}
    if part not in files or not Path(files[part]).exists():
        raise HTTPException(404, "run sync first")
    return FileResponse(files[part])
