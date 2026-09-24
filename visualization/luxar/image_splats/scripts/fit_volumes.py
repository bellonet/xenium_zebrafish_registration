#!/usr/bin/env python3
"""Run Luxar's image-volume -> fitted splats -> streaming LOD pipeline."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tomllib
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
LUXAR = HERE.parent / ".venv" / "bin" / "luxar"
BUILD = HERE / "build"


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "config.toml")
    args = parser.parse_args()
    with args.config.resolve().open("rb") as handle:
        config = tomllib.load(handle)
    manifest = json.loads((BUILD / "manifest.json").read_text())
    fits = BUILD / "fits"
    fits.mkdir(parents=True, exist_ok=True)

    fit = config["fit"]
    for fish in config["selection"]["fish"]:
        fish = int(fish)
        volume = Path(manifest["fish"][str(fish)]["volume"])
        raw = fits / f"fish_{fish}_fit.gsplats.zarr"
        lod = fits / f"fish_{fish}_stream.gsplats.zarr"
        for path in (raw, lod):
            if path.exists():
                shutil.rmtree(path)
        run(
            [
                str(LUXAR),
                "gsplat",
                "fit",
                str(volume),
                str(raw),
                "--axes",
                "z,y,x",
                "--preset",
                str(fit["preset"]),
                "--seeds",
                str(fit["seeds"]),
                "--iters",
                str(fit["iterations"]),
                "--device",
                str(fit["device"]),
                "--floor",
                "auto",
                "--tiling",
                "none",
                "--cull-retention",
                str(fit["cull_retention"]),
            ]
        )
        run(
            [
                str(LUXAR),
                "gsplat",
                "lod",
                str(raw),
                str(lod),
                "--recipe",
                "stream",
                "--n-lods",
                str(fit["lod_levels"]),
                "--breakpoints",
                f"equi-energy:{fit['lod_levels']}",
            ]
        )
    (fits / ".done").write_text("fitted\n")


if __name__ == "__main__":
    main()

