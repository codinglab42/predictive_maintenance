"""Load the local ml_core Python binding."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType


PROJECT_ROOT = Path(__file__).resolve().parent
ML_PROJECT_ROOT = PROJECT_ROOT.parent / "machineLearning"


def load_ml_core() -> ModuleType:
    """Load ml_core from the environment or a sibling machineLearning build."""
    try:
        return importlib.import_module("ml_core")
    except ModuleNotFoundError as error:
        if error.name != "ml_core":
            raise

    build_dirs = [
        ML_PROJECT_ROOT / "src",
        ML_PROJECT_ROOT / "build" / "pybinding",
        ML_PROJECT_ROOT / "build" / "lib",
    ]
    build_root = ML_PROJECT_ROOT / "build"
    if build_root.is_dir():
        build_dirs.extend(sorted(build_root.glob("lib.*")))
    for build_dir in build_dirs:
        if build_dir.is_dir() and str(build_dir) not in sys.path:
            sys.path.insert(0, str(build_dir))

    try:
        return importlib.import_module("ml_core")
    except ModuleNotFoundError as error:
        if error.name != "ml_core":
            raise
        if sys.version_info[:2] != (3, 11):
            raise RuntimeError(
                "ml_core requires Python 3.11 (>=3.11,<3.12); "
                "run this project with Python 3.11."
            ) from error
        raise ModuleNotFoundError(
            "ml_core is not installed and no compatible sibling build was found; "
            "build or install ../machineLearning first."
        ) from error
