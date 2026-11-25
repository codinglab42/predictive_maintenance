#!/usr/bin/env python3
"""Compatibility entry point for the StoneCut IMS web dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from web_server import main


if __name__ == "__main__":
    main()
