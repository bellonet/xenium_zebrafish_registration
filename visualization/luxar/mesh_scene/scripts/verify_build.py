#!/usr/bin/env python3
"""Fail fast when the generated scene or its provenance is incomplete."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
BUILD = PROJECT / "build"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT / "config.toml")
    args = parser.parse_args()
    with args.config.resolve().open("rb") as handle:
        config = tomllib.load(handle)
    with (BUILD / "manifest.json").open() as handle:
        manifest = json.load(handle)

    scene = BUILD / config["output"]["scene_name"]
    if not scene.is_dir() or not (scene / "zarr.json").exists():
        raise SystemExit(f"missing or invalid scene: {scene}")
    with (scene / "zarr.json").open() as handle:
        root_metadata = json.load(handle)
    viewer_ui = root_metadata["attributes"].get("viewer_config", {}).get("ui", {})
    if viewer_ui.get("show_layers") is not True:
        raise SystemExit("scene does not enable the Layers panel")
    node_metadata = root_metadata["consolidated_metadata"]["metadata"].values()
    layer_count = sum(
        entry.get("attributes", {}).get("layer") is True for entry in node_metadata
    )
    expected_layer_count = 3 + len(manifest["selected_genes"])
    if layer_count != expected_layer_count:
        raise SystemExit(
            f"scene exposes {layer_count} editable layers; "
            f"expected {expected_layer_count} shared object-type layers"
        )
    if not manifest["selected_genes"]:
        raise SystemExit("no genes selected")
    for fish in config["selection"]["fish"]:
        path = BUILD / "processed" / f"fish_{fish}.npz"
        expected = manifest["fish"][str(fish)]["sha256"]
        if digest(path) != expected:
            raise SystemExit(f"checksum mismatch: {path}")
        data = np.load(path)
        for name in ["cells", "transcripts", "tissue_vertices", "tubule_vertices"]:
            if len(data[name]) == 0:
                raise SystemExit(f"fish {fish}: empty {name}")
            if name.endswith("vertices") or name in {"cells", "transcripts"}:
                if not np.isfinite(data[name]).all():
                    raise SystemExit(f"fish {fish}: non-finite values in {name}")
        tissue_min = data["tissue_vertices"].min(axis=0)
        tissue_max = data["tissue_vertices"].max(axis=0)
        tolerance = 25.0
        for name in ["cells", "transcripts", "tubule_vertices"]:
            values = data[name]
            inside = np.all(
                (values >= tissue_min - tolerance) & (values <= tissue_max + tolerance),
                axis=1,
            )
            fraction = float(inside.mean())
            if fraction < 0.95:
                raise SystemExit(
                    f"fish {fish}: only {fraction:.1%} of {name} overlaps the tissue bounds; "
                    "check coordinate axes and registration"
                )
        # A coarse nearest-surface check catches transform-direction mistakes
        # that broad tissue bounding boxes cannot. Sampling keeps this pure-
        # NumPy check lightweight in the minimal Luxar environment.
        surface = data["tubule_vertices"]
        surface = surface[:: max(1, len(surface) // 20_000)]
        cells = data["cells"]
        nearest = []
        for start in range(0, len(cells), 32):
            chunk = cells[start : start + 32]
            distances2 = np.sum((chunk[:, None, :] - surface[None, :, :]) ** 2, axis=2)
            nearest.extend(np.sqrt(distances2.min(axis=1)).tolist())
        median_distance = float(np.median(nearest))
        if median_distance > 25.0:
            raise SystemExit(
                f"fish {fish}: median cell-to-tubule distance is "
                f"{median_distance:.1f} µm; check transform direction"
            )
        print(f"fish {fish}: median cell-to-tubule distance {median_distance:.1f} µm")
    print(f"verified {scene}")
    print(f"editable layers: {layer_count}")
    print("genes:", ", ".join(manifest["selected_genes"]))


if __name__ == "__main__":
    main()
