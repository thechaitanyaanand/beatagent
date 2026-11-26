"""One command for everything.

  python run.py           # build the test pack if missing, run all evals, then serve the UI on :8000
  python run.py evals     # evals only (writes evals/results.md and evals/out/)
  python run.py serve     # UI only
  python run.py test      # unit tests
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

if cmd in ("all", "evals"):
    subprocess.run([sys.executable, "-W", "ignore", str(ROOT / "evals" / "run.py")], check=True, env=env, cwd=ROOT)
if cmd == "test":
    sys.exit(subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT).returncode)
if cmd in ("all", "serve"):
    if not (ROOT / "evals" / "testpack").exists():
        subprocess.run([sys.executable, str(ROOT / "evals" / "make_clips.py")], check=True, cwd=ROOT)
    import uvicorn
    print("Beat Agent on http://localhost:8000")
    uvicorn.run("beatagent.server:app", host=os.environ.get("HOST", "127.0.0.1"), port=8000, app_dir=str(ROOT))
