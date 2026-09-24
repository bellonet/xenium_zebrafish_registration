#!/usr/bin/env python3
"""Remove only this project's generated build directory."""

from pathlib import Path
import shutil


build = Path(__file__).resolve().parents[1] / "build"
if build.exists():
    shutil.rmtree(build)
    print(f"removed {build}")
else:
    print(f"nothing to remove: {build}")

