#!/usr/bin/env python3
"""Prepare downsampled registered images and aligned biological point layers."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import tomllib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile


HERE = Path(__file__).resolve().parents[1]
LUXAR_ROOT = HERE.parent
BUILD = HERE / "build"
PREPARED = BUILD / "prepared"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_shared_preparation():
    source = LUXAR_ROOT / "mesh_scene" / "scripts" / "prepare_data.py"
    spec = importlib.util.spec_from_file_location("shared_luxar_preparation", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "config.toml")
    args = parser.parse_args()
    with args.config.resolve().open("rb") as handle:
        config = tomllib.load(handle)

    shared = load_shared_preparation()
    shared.CONFIG = config
    registration = shared.load_registration_module(config)
    annotations_path = shared.REPO / "leiden10annots.csv"
    annotations = pd.read_csv(annotations_path, dtype=str)
    target = annotations[
        annotations["leiden10annots"] == config["selection"]["cell_type"]
    ].copy()
    folders = shared.run_folder_map(registration)
    ids_by_run = shared.match_target_ids_by_run(target, folders)
    boxes_path = shared.ANALYSIS / "fish_bbox_summary_tagged_2_4_5.csv"
    boxes = pd.read_csv(boxes_path)

    PREPARED.mkdir(parents=True, exist_ok=True)
    selected_genes = list(config["selection"]["genes"])
    strides = tuple(int(value) for value in config["image"]["downsample_zyx"])
    if len(strides) != 3 or any(value < 1 for value in strides):
        raise ValueError("image.downsample_zyx must contain three positive integers")

    manifest = {
        "description": "Image-derived Luxar Gaussian-splat comparison inputs",
        "config": config,
        "selected_genes": selected_genes,
        "inputs": [str(annotations_path), str(boxes_path)],
        "fish": {},
    }
    reference_shape = None

    for fish in config["selection"]["fish"]:
        fish = int(fish)
        cells, transcripts, genes = shared.prepare_points(
            registration, fish, boxes, ids_by_run, folders
        )
        gene_mask = np.isin(genes, selected_genes)
        points_path = PREPARED / f"fish_{fish}_points.npz"
        np.savez_compressed(
            points_path,
            cells=cells,
            transcripts=transcripts[gene_mask],
            genes=genes[gene_mask],
        )

        image_path = (
            shared.ANALYSIS
            / "4_registered"
            / config["registration"]["script4_experiment"]
            / str(fish)
            / config["image"]["channel"]
        )
        source = tifffile.memmap(image_path)
        source_shape = tuple(int(value) for value in source.shape)
        if reference_shape is None:
            reference_shape = source_shape
        elif source_shape != reference_shape:
            raise ValueError(
                f"registered image shapes differ: {source_shape} != {reference_shape}"
            )
        volume = np.asarray(
            source[:: strides[0], :: strides[1], :: strides[2]], dtype=np.float32
        ).copy()
        volume_path = PREPARED / f"fish_{fish}_dapi.npy"
        np.save(volume_path, volume, allow_pickle=False)
        nonzero = volume[volume > 0]
        percentiles = (
            np.percentile(nonzero, [1, 50, 99, 99.9]).tolist()
            if len(nonzero)
            else [0.0, 0.0, 0.0, 0.0]
        )
        counts = Counter(genes[gene_mask].tolist())
        manifest["inputs"].append(str(image_path))
        manifest["fish"][str(fish)] = {
            "source_image": str(image_path),
            "source_shape_zyx": source_shape,
            "prepared_shape_zyx": list(volume.shape),
            "downsample_zyx": list(strides),
            "nonzero_percentiles": percentiles,
            "cells": int(len(cells)),
            "transcripts": int(gene_mask.sum()),
            "gene_counts": dict(counts),
            "volume": str(volume_path),
            "volume_sha256": sha256(volume_path),
            "points": str(points_path),
            "points_sha256": sha256(points_path),
        }
        print(
            f"fish {fish}: image {source_shape} -> {volume.shape}; "
            f"{len(cells)} cells; {gene_mask.sum()} transcripts"
        )

    assert reference_shape is not None
    z, y, x = reference_shape
    xy = float(config["registration"]["xy_pixel_um"])
    z_spacing = float(config["registration"]["z_spacing_um"])
    manifest["source_shape_zyx"] = list(reference_shape)
    manifest["image_center_xyz_um"] = [
        (x - 1) * xy / 2.0,
        -(y - 1) * xy / 2.0,
        (z - 1) * z_spacing / 2.0,
    ]
    manifest["effective_spacing_zyx_um"] = [
        z_spacing * strides[0],
        xy * strides[1],
        xy * strides[2],
    ]
    manifest_path = BUILD / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (PREPARED / ".done").write_text("prepared\n")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
