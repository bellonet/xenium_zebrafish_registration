#!/usr/bin/env python3
"""Verify the independent image-derived Luxar build and its semantic layers."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
BUILD = HERE / "build"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "config.toml")
    args = parser.parse_args()
    with args.config.resolve().open("rb") as handle:
        config = tomllib.load(handle)
    scene = BUILD / config["output"]["scene_name"]
    root = json.loads((scene / "zarr.json").read_text())
    entries = root["consolidated_metadata"]["metadata"]
    layer_paths = {
        path
        for path, entry in entries.items()
        if entry.get("attributes", {}).get("layer") is True
    }
    expected = {
        "DAPI tissue context — image-derived splats",
        "Pronephric distal early tubule cells",
        *{f"Gene — {gene}" for gene in config["selection"]["genes"]},
    }
    if layer_paths != expected:
        raise SystemExit(f"layer mismatch: got {sorted(layer_paths)}, want {sorted(expected)}")
    node_types = [entry.get("attributes", {}).get("type") for entry in entries.values()]
    if "gsplats" not in node_types or "points" not in node_types:
        raise SystemExit(f"expected gsplats and points, found {sorted(set(node_types))}")
    if "mesh" in node_types:
        raise SystemExit("image-derived scene unexpectedly contains a mesh")
    print(f"verified {scene}")
    print("layers:", ", ".join(sorted(layer_paths)))


if __name__ == "__main__":
    main()

