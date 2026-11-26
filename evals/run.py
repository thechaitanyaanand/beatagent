"""Run every eval in the brief and write evals/results.md + evals/out/.

  python evals/run.py            # all parts
  python evals/run.py --part 1   # just one part

Clips are found by ID prefix in evals/testpack (A1.*, P1.*), so the real pack drops in.
LLM replies are cached in evals/out/llm_cache.json; delete it to call the model live.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PACK, OUT = ROOT / "evals" / "testpack", ROOT / "evals" / "out"
os.environ.setdefault("BEAT_AGENT_CACHE", str(OUT / "llm_cache.json"))

import mir_eval  # noqa: E402
import soundfile as sf  # noqa: E402

from beatagent import agent  # noqa: E402
from beatagent.audio import click_audition, load, save  # noqa: E402
from beatagent.edits import Session  # noqa: E402
from beatagent.grid import analyze  # noqa: E402
from beatagent.sync import beatmatch, kick_onsets, match  # noqa: E402

SESSIONS = [
    ("1", "A1", ["Cut beat 16–32 and repeat it twice.", "Now fade out the last 8 beats.", "Undo.", "Remove bar 5."]),
    ("2", "A2", ["Loop the first 4 bars four times, then let the track play on."]),
    ("3", "A5", ["Repeat the last bar before the tempo change twice."]),
    ("4", "A1", ["Chop out the boring part."]),
    ("5", "A3", ["Cut beat 0 to beat 8.", "Delete bars 200–210."]),
]


def clip(cid, exts=(".wav", ".flac", ".mp3", ".aiff")):
    for p in sorted(PACK.glob(f"{cid}*")):
        if p.suffix.lower() in exts:
            return p
    return None


def ref_beats(cid):
    p = PACK / f"{cid}.beats"
    if not p.exists():
        return None, None
    r = np.loadtxt(p, ndmin=2)
    return r[:, 0], (r[r[:, 1] == 1, 0] if r.shape[1] > 1 else None)


def fmt(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def part1(rows, grids):
    for cid in ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]:
        path = clip(cid)
        if not path:
            continue
        d = OUT / "part1" / cid
        d.mkdir(parents=True, exist_ok=True)
        audio, sr, info = load(path)
        t = time.time()
        g = analyze(audio=audio, sr=sr)
        secs = time.time() - t
        grids[cid] = g
        j = g.to_json() | {"source": path.name}
        (d / "grid.json").write_text(encoding="utf-8", data=json.dumps(j, indent=1))
        save(d / "click_audition", click_audition(audio, sr, g), sr, info)
        rb, rd = ref_beats(cid)
        row = {"clip": cid, "tempo": j["tempo"], "ts": j["time_signature"], "constant": j["constant_tempo"],
               "confidence": j["confidence"], "seconds": round(secs, 2), "duration": round(len(audio) / sr, 1)}
        if rb is not None:
            row["beat_f"] = mir_eval.beat.f_measure(rb, g.times)
            row["amlt"] = mir_eval.beat.continuity(rb, g.times)[3]  # allows double/half tempo
            row["down_f"] = mir_eval.beat.f_measure(rd, g.times[g.pos == 1]) if rd is not None else None
            m = mir_eval.util.match_events(rb, g.times, 0.07)
            err = np.array([g.times[j_] - rb[i] for i, j_ in m]) * 1000
            row["err_med_ms"] = float(np.median(np.abs(err))) if len(err) else None
            row["err_max_ms"] = float(np.abs(err).max()) if len(err) else None
            row["unmatched_ref"] = len(rb) - len(m)
        rows.append(row)
        print(f"part1 {cid}: {row}", flush=True)


def part1_sync(grids):
    a, b = clip("A1"), clip("A7")
    if not (a and b):
        return None
    d = OUT / "part1" / "sync_A7_to_A1"
    r = beatmatch(a, b, d, start_bar=33, seconds=60, grid_a=grids.get("A1"), grid_b=grids.get("A7"))
    A, sr = sf.read(r["files"]["a"])
    B, _ = sf.read(r["files"]["b"])
    ka, kb = kick_onsets(A.reshape(len(A), -1).mean(1), sr), kick_onsets(B.reshape(len(B), -1).mean(1), sr)
    off = match(ka, kb) * 1000
    r |= {"kicks_a": len(ka), "kicks_b": len(kb), "kicks_matched": len(off),
          "kick_offset_median_ms": float(np.median(np.abs(off))) if len(off) else None,
          "kick_offset_max_ms": float(np.abs(off).max()) if len(off) else None}
    (d / "sync.json").write_text(encoding="utf-8", data=json.dumps(r, indent=1))
    print("sync:", {k: v for k, v in r.items() if k != "files"}, flush=True)
    return r


def part2(grids):
    results = []
    for sid, cid, msgs in SESSIONS:
        path = clip(cid)
        if not path:
            continue
        d = OUT / "part2" / f"session{sid}_{cid}"
        d.mkdir(parents=True, exist_ok=True)
        s = Session(path, grids.get(cid))
        history, turns = [], []
        for m in msgs:
            r = agent.handle(s, m, history)
            history += [("user", m), ("agent", r["text"])]
            turns.append({"user": m, "action": r["action"], "ok": r["ok"], "ops": r["ops"], "reply": r["text"]})
            print(f"session {sid} > {m}\n  [{r['action']}] {r['text']}", flush=True)
        audio_path, grid_json = s.export(d / "edited")
        (d / "edited_grid.json").write_text(encoding="utf-8", data=json.dumps(grid_json, indent=1))
        (d / "transcript.json").write_text(encoding="utf-8", data=json.dumps(turns, indent=1, ensure_ascii=False))
        md = [f"# Session {sid} on {cid}\n"] + [f"**User:** {t['user']}\n\n**Agent ({t['action']}):**\n\n{t['reply']}\n" for t in turns]
        (d / "transcript.md").write_text(encoding="utf-8", data="\n".join(md))
        # does the exported grid match the edited audio? re-analyse it from scratch and compare
        check = {}
        if s.ptr:
            fresh = analyze(audio=s.cur[0], sr=s.sr)
            exp = s.grid.times
            m = mir_eval.util.match_events(exp, fresh.times, 0.07)
            dev = np.array([abs(fresh.times[j] - exp[i]) for i, j in m]) * 1000
            check = {"grid_vs_reanalysis_f": mir_eval.beat.f_measure(exp, fresh.times),
                     "grid_vs_reanalysis_median_ms": float(np.median(dev)) if len(dev) else None}
            save(d / "edited_click_audition", click_audition(s.cur[0], s.sr, s.grid), s.sr, s.info)
        info = sf.info(str(audio_path))
        src = sf.info(str(path))
        results.append({"session": sid, "clip": cid, "turns": turns, "edits_applied": s.ptr,
                        "sample_rate_kept": info.samplerate == src.samplerate,
                        "format": f"{src.format}/{src.subtype} -> {info.format}/{info.subtype}", **check})
    (OUT / "part2" / "sessions.json").write_text(encoding="utf-8", data=json.dumps(results, indent=1, ensure_ascii=False))
    return results


def part3(rows):
    import pretty_midi
    import librosa
    from beatagent.recreate import render_page, score_notes_seconds, transcribe
    from beatagent.render import render_offline
    for cid in ["P1", "P2", "P3", "P4"]:
        path = clip(cid)
        if not path:
            continue
        d = OUT / "part3" / cid
        d.mkdir(parents=True, exist_ok=True)
        score = transcribe(path)
        (d / "score.json").write_text(encoding="utf-8", data=json.dumps(score, indent=1))
        page = render_page(score, d / f"{cid}.html")
        pcm, sr, reqs = render_offline(page, d / "render_2loops.wav", loops=2)
        second = pcm[len(pcm) // 2:]  # loop 2 carries loop 1's tails, like the original clip
        orig, osr = sf.read(path)
        orig = orig.reshape(len(orig), -1).mean(1).astype(np.float32)
        row = {"clip": cid, "tempo": score["tempo"], "bars": score["bars"], "network_requests": len(reqs),
               "tracks": ", ".join(f"{t['name']} {len(t['notes'])}" for t in score["tracks"])}
        if (PACK / f"{cid}.mid").exists():
            pm = pretty_midi.PrettyMIDI(str(PACK / f"{cid}.mid"))
            ri = np.array([[n.start, n.end] for i in pm.instruments for n in i.notes])
            rp = np.array([n.pitch for i in pm.instruments for n in i.notes])
            ei, ep = score_notes_seconds(score)
            hz = mir_eval.util.midi_to_hz
            p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
                ri, hz(rp), ei, hz(ep), onset_tolerance=0.05, offset_ratio=None) if len(ei) else (0, 0, 0, 0)
            row |= {"note_f": f, "note_p": p, "note_r": r}
            drums = [i for i in pm.instruments if i.is_drum]
            if drums:  # per drum voice, so a missing hat can't hide behind good pitched notes
                for name, gm in (("kick", 36), ("snare", 38), ("hat", 42)):
                    rr = ri[rp == gm, 0]
                    if len(rr):
                        row[f"{name}_f"] = mir_eval.onset.f_measure(rr, ei[ep == gm, 0], 0.05)[0]
        oo = librosa.onset.onset_detect(y=orig, sr=osr, units="time")
        ro = librosa.onset.onset_detect(y=second, sr=sr, units="time")
        m = mir_eval.util.match_events(oo, ro, 0.05)
        row["render_onset_f"] = mir_eval.onset.f_measure(oo, ro, 0.05)[0]
        row["render_onset_median_ms"] = float(np.median([abs(ro[j] - oo[i]) for i, j in m]) * 1000) if m else None
        rows.append(row)
        print(f"part3 {cid}: {row}", flush=True)


def write_report(p1, sync, p2, p3):
    L = ["# Eval results", "",
         "Generated by `python evals/run.py`. Clips are the synthetic stand-in pack (see README): ground truth "
         "is exact, audio is cleaner than real music, so treat these as optimistic.", ""]
    if p1:
        L += ["## Part 1: beat grid", "",
              "| clip | tempo | meter | constant | conf | beat F | AMLt | downbeat F | median err ms | max err ms | missed ref beats | analysis s |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in p1:
            L.append(f"| {r['clip']} | {r['tempo']} | {r['ts']} | {r['constant']} | {r['confidence']} | {fmt(r.get('beat_f'))} | "
                     f"{fmt(r.get('amlt'))} | {fmt(r.get('down_f'))} | {fmt(r.get('err_med_ms'), 1)} | {fmt(r.get('err_max_ms'), 1)} | "
                     f"{r.get('unmatched_ref', '—')} | {r['seconds']} (for {r['duration']} s) |")
        a1 = next((r for r in p1 if r["clip"] == "A1"), None)
        steady = [r for r in p1 if r["constant"] and r.get("beat_f") is not None]
        L += ["", "Targets:", ""]
        L.append(f"- Beat F ≥ 0.90 on steady clips: " + ", ".join(f"{r['clip']} {r['beat_f']:.3f}" for r in steady))
        L.append(f"- Downbeat F ≥ 0.80: " + ", ".join(f"{r['clip']} {fmt(r.get('down_f'))}" for r in p1 if r.get("down_f") is not None))
        if a1:
            L.append(f"- Drift on A1 (6 min): max error {fmt(a1.get('err_max_ms'), 2)} ms over every matched beat "
                     f"(target ≤ 10 ms); {a1.get('unmatched_ref')} reference beats unmatched")
            L.append(f"- Speed: A1 analysed in {a1['seconds']} s (target < 60 s)")
    if sync:
        L += ["", "## Part 1: sync (A7 → A1 from A1's bar 33, 60 s)", "",
              f"- ratio {sync['stretch_ratio']} ({sync['b_bpm']} → {sync['a_bpm']} BPM), offset {sync['offset_seconds']} s, "
              f"Rubber Band latency measured {sync['rubberband_latency_ms']} ms",
              f"- kick onsets matched {sync['kicks_matched']}/{sync['kicks_a']}: median |offset| "
              f"{fmt(sync['kick_offset_median_ms'], 2)} ms (target ≤ 5), max {fmt(sync['kick_offset_max_ms'], 2)} ms (target ≤ 15)"]
    if p2:
        L += ["", "## Part 2: chat sessions", "",
              "Full transcripts: `evals/out/part2/*/transcript.md`. \"Grid vs re-analysis\" re-runs beat tracking "
              "on the edited audio and compares it with the grid we carried through the edits.", ""]
        for s in p2:
            L.append(f"### Session {s['session']} on {s['clip']}")
            for t in s["turns"]:
                first = t["reply"].splitlines()[0] if t["reply"] else ""
                L.append(f"- *{t['user']}* → **{t['action']}**: {first}")
            if "grid_vs_reanalysis_f" in s:
                L.append(f"- grid vs re-analysis: F {s['grid_vs_reanalysis_f']:.3f}, median {fmt(s['grid_vs_reanalysis_median_ms'], 2)} ms")
            L.append(f"- output format: {s['format']}, sample rate kept: {s['sample_rate_kept']}")
            L.append("")
    if p3:
        L += ["## Part 3: recreation", "",
              "| clip | tempo | bars | tracks (notes) | note F | P | R | kick F | snare F | hat F | render onset F | render onset median ms | network requests |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in p3:
            L.append(f"| {r['clip']} | {r['tempo']} | {r['bars']} | {r['tracks']} | {fmt(r.get('note_f'))} | {fmt(r.get('note_p'))} | "
                     f"{fmt(r.get('note_r'))} | {fmt(r.get('kick_f'), 2)} | {fmt(r.get('snare_f'), 2)} | {fmt(r.get('hat_f'), 2)} | "
                     f"{fmt(r['render_onset_f'])} | {fmt(r['render_onset_median_ms'], 1)} | {r['network_requests']} |")
    (ROOT / "evals" / "results.md").write_text(encoding="utf-8", data="\n".join(L) + "\n")
    (OUT / "results.json").write_text(encoding="utf-8", data=json.dumps({"part1": p1, "sync": sync, "part2": p2, "part3": p3}, indent=1,
                                                 ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, choices=[1, 2, 3])
    a = ap.parse_args()
    if not clip("A1"):
        try:
            from make_clips import main as make
        except ImportError:
            from evals.make_clips import main as make
        make()
    OUT.mkdir(parents=True, exist_ok=True)
    p1, grids, sync, p2, p3 = [], {}, None, [], []
    if a.part in (None, 1, 2):
        part1(p1, grids)
    if a.part in (None, 1):
        sync = part1_sync(grids)
    if a.part in (None, 2):
        p2 = part2(grids)
    if a.part in (None, 3):
        part3(p3)
    write_report(p1 if a.part != 2 else [], sync, p2, p3)
    print("wrote evals/results.md")


if __name__ == "__main__":
    main()
