"""Convenient entry point for running the OHSLIC CLI without installing the package."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ohslic.cli import main as cli_main


if __name__ == "__main__":  # pragma: no cover
    cli_main()
