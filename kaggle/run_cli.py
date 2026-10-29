"""Run Kaggle CLI using local .env credentials without printing their values."""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")
load_dotenv(ROOT / ".env", override=False)
# Keep CLI state in the workspace; this directory is ignored by git.
os.environ.setdefault("KAGGLE_CONFIG_DIR", str(ROOT / ".kaggle"))

if __name__ == "__main__":
    from kaggle.cli import main
    main()
