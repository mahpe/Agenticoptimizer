"""Put methods/ and campaign/ on sys.path for the package tests."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "methods", ROOT / "campaign"):
    path = str(folder)
    if path not in sys.path:
        sys.path.insert(0, path)
