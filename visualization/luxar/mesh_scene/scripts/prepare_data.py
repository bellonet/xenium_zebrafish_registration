#!/usr/bin/env python3
"""Prepare registered point data and existing surfaces for the Luxar scene."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import struct
import sys
import tomllib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import trimesh


PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[2]
DATA = REPO.parent / "data"
ANALYSIS = REPO.parent / "analysis"
REGISTRATION = REPO / "registration"
ORGANELLA = REPO / "visualization" / "organella"
BUILD = PROJECT / "build"
PROCESSED = BUILD / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT / "config.toml")
    return parser.parse_args()


def load_registration_module(config: dict):
    """Load the pipeline's tested transform math without modifying its source."""
    source = REGISTRATION / "6_apply_registration.py"
    old_cwd = Path.cwd()
    os.chdir(REGISTRATION)
    try:
        spec = importlib.util.spec_from_file_location("zfish_apply_registration", source)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import {source}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        os.chdir(old_cwd)

    module.DATA_DIR = str(DATA)
    module.IN2_DIR = str(ANALYSIS / "2_registered")
    module.IN3_DIR = str(ANALYSIS / "3_improved_registration")
    module.IN4_DIR = str(ANALYSIS / "4_registered")
    module.SCRIPT3_EXP = config["registration"]["script3_experiment"]
    module.SCRIPT4_EXP = config["registration"]["script4_experiment"]
    module.Z_SPACING_UM = float(config["registration"]["z_spacing_um"])
    module.XENIUM_PX_UM = float(config["registration"]["xy_pixel_um"])
    module.XENIUM_PX_PER_UM = 1.0 / module.XENIUM_PX_UM
    tf_path = module.IN4_DIR + f"/{module.SCRIPT4_EXP}/1/transform.json"
    with open(tf_path) as handle:
        module.DOWNSAMPLE_XY = int(json.load(handle).get("ds_xy", 1))
    module.DS_SPACING = module.XENIUM_PX_UM * module.DOWNSAMPLE_XY
    return module


def bare_cell_id(value: str) -> str:
    value = str(value)
    return value.split("_", 1)[-1] if value.startswith("slide") else value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_mesh(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    n_vertices, n_faces = struct.unpack_from("<II", data, 0)
    minimum = np.frombuffer(data, dtype=np.float32, count=3, offset=8)
    scale = np.frombuffer(data, dtype=np.float32, count=3, offset=20)
    quantized = np.frombuffer(
        data, dtype=np.uint16, count=n_vertices * 3, offset=32
    ).reshape(n_vertices, 3)
    index_dtype = np.uint16 if n_vertices < 65536 else np.uint32
    faces = np.frombuffer(
        data,
        dtype=index_dtype,
        count=n_faces * 3,
        offset=32 + n_vertices * 3 * 2,
    ).reshape(n_faces, 3)
    vertices_zyx = quantized / 65535.0 * scale + minimum
    return vertices_zyx.astype(np.float32), faces.astype(np.uint32)


def decode_ellipsoid(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    values = np.frombuffer(data, dtype=np.float32)
    center = values[0:3]
    radii = values[3:6]
    axes = values[6:15].reshape(3, 3)
    sphere = trimesh.creation.icosphere(subdivisions=2)
    vertices = (np.asarray(sphere.vertices, dtype=np.float32) * radii) @ axes + center
    return vertices.astype(np.float32), np.asarray(sphere.faces, dtype=np.uint32)


def concatenate_meshes(parts: list[tuple[np.ndarray, np.ndarray]]):
    if not parts:
        raise ValueError("No mesh parts found")
    vertices, faces, offset = [], [], 0
    for part_vertices, part_faces in parts:
        vertices.append(part_vertices)
        faces.append(part_faces + offset)
        offset += len(part_vertices)
    return np.concatenate(vertices), np.concatenate(faces)


def load_surface(fish: int, entity: str, row_type: str):
    path = ORGANELLA / "report_meshes" / str(fish) / "geometry.parquet"
    table = pl.read_parquet(path).filter(
        (pl.col("entity_name") == entity)
        & (pl.col("row_type") == row_type)
        & pl.col("surface").is_not_null()
    )
    parts = []
    for row in table.iter_rows(named=True):
        payload = bytes(row["surface"])
        if row["surface_kind"] == "mesh":
            parts.append(decode_mesh(payload))
        else:
            parts.append(decode_ellipsoid(payload))
    vertices_display, faces = concatenate_meshes(parts)
    # Organella's established viewer convention is (first, -second, third).
    # The registration points below use the same display-space convention.
    vertices_display = vertices_display.astype(np.float32)
    vertices_display[:, 1] *= -1.0
    return vertices_display, faces, path


def run_folder_map(registration_module) -> dict[int, Path]:
    return {
        int(run): DATA / folder
        for run, folder in registration_module.RUN_FOLDERS.items()
    }


def match_target_ids_by_run(target_annotations: pd.DataFrame, folders: dict[int, Path]):
    annotated = set(target_annotations["cell_id"].map(bare_cell_id))
    result: dict[int, set[str]] = {}
    for run, folder in folders.items():
        cells = pd.read_parquet(folder / "cells.parquet", columns=["cell_id"])
        ids = set(cells["cell_id"].astype(str))
        result[run] = ids & annotated
    return result


def transform_tile(reg, frame, fish: int, row, x_col: str, y_col: str):
    """Map moving-image points into the final fixed/cropped image canvas.

    Elastix parameter maps are backward resampling maps (fixed/output to
    moving/input). Points travel moving to fixed, so every Elastix map is
    inverted here. This follows the validated `_map_coords_to_canvas` logic in
    registration/2_rigid_registration.py rather than script 6's forward use.
    """
    gnum = int(round(float(row.global_slice_num)))
    margin = CONFIG["registration"]["fish_crop_margin_px"]
    bbox_r0 = max(0.0, float(row.bbox_global_min_row) - margin)
    bbox_c0 = max(0.0, float(row.bbox_global_min_col) - margin)
    slices, canvas_h, canvas_w = reg.load_script2_meta(fish)
    slice_meta = slices[str(gnum)]

    xy = np.column_stack(
        [
            frame[x_col].to_numpy() * reg.XENIUM_PX_PER_UM
            - bbox_c0
            + float(slice_meta.get("pad_left", 0.0)),
            frame[y_col].to_numpy() * reg.XENIUM_PX_PER_UM
            - bbox_r0
            + float(slice_meta.get("pad_top", 0.0)),
        ]
    )

    dv_angle = float(slice_meta.get("dv_angle_deg", 0.0))
    if abs(dv_angle) >= 0.01:
        angle = np.deg2rad(dv_angle)
        dx = xy[:, 0] - canvas_w / 2.0
        dy = xy[:, 1] - canvas_h / 2.0
        xy = np.column_stack(
            [
                canvas_w / 2.0 + np.cos(angle) * dx + np.sin(angle) * dy,
                canvas_h / 2.0 - np.sin(angle) * dx + np.cos(angle) * dy,
            ]
        )

    elastix_file = slice_meta.get("elastix_file")
    if elastix_file:
        rotation, offset = reg.rigid2d_from_euler_file(
            str(ANALYSIS / "2_registered" / str(fish) / elastix_file)
        )
        xy = (rotation.T @ (xy.T - offset[:, None])).T

    correction = reg.load_script3_corrections(
        fish, CONFIG["registration"]["script3_experiment"]
    ).get(gnum)
    if correction is not None:
        rotation, offset = reg.rigid2d_from_params(
            correction.get("angle_deg", 0.0),
            correction.get("tx", 0.0),
            correction.get("ty", 0.0),
            tuple(correction.get("center", [canvas_w / 2.0, canvas_h / 2.0])),
        )
        xy = (rotation.T @ (xy.T - offset[:, None])).T

    x_ds = xy[:, 0] / reg.DOWNSAMPLE_XY
    y_ds = xy[:, 1] / reg.DOWNSAMPLE_XY
    z_ds = np.full(len(frame), gnum - 1.0, dtype=np.float64)

    transform4 = reg.load_script4_transform(
        fish, CONFIG["registration"]["script4_experiment"]
    )
    shift = np.asarray(transform4.get("canvas_shift", [0.0, 0.0, 0.0]))
    x_ds += shift[2]
    y_ds += shift[1]
    z_ds += shift[0]
    elastix = transform4.get("elastix")
    if elastix is not None:
        parameters = np.asarray(elastix["TransformParameters"], dtype=np.float64)
        center = np.asarray(
            elastix.get("CenterOfRotationPoint", [0.0, 0.0, 0.0]),
            dtype=np.float64,
        )
        physical_moving = np.stack(
            [x_ds * reg.DS_SPACING, y_ds * reg.DS_SPACING, z_ds * reg.Z_SPACING_UM]
        )
        transform_type = elastix.get("Transform", ["AffineTransform"])[0]
        if "Affine" in transform_type:
            matrix = parameters[:9].reshape(3, 3)
            translation = parameters[9:12]
            physical_fixed = (
                np.linalg.solve(
                    matrix,
                    physical_moving - center[:, None] - translation[:, None],
                )
                + center[:, None]
            )
        else:
            rx, ry, rz = parameters[:3]
            cx, sx = np.cos(rx), np.sin(rx)
            cy, sy = np.cos(ry), np.sin(ry)
            cz, sz = np.cos(rz), np.sin(rz)
            matrix = np.array(
                [
                    [cy * cz, cz * sx * sy - cx * sz, cx * cz * sy + sx * sz],
                    [cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx],
                    [-sy, cy * sx, cx * cy],
                ]
            )
            translation = parameters[3:6]
            physical_fixed = (
                matrix.T
                @ (physical_moving - center[:, None] - translation[:, None])
                + center[:, None]
            )
        x_ds = physical_fixed[0] / reg.DS_SPACING
        y_ds = physical_fixed[1] / reg.DS_SPACING
        z_ds = physical_fixed[2] / reg.Z_SPACING_UM

    # Script 4 crops every registered output to a shared final canvas after it
    # saves the transform. Point coordinates must receive that same translation;
    # the original script-6 helper does not currently apply it.
    crop = transform4.get("canvas_crop_box", {})
    x_cropped = x_ds - float(crop.get("col0", 0.0))
    y_cropped = y_ds - float(crop.get("row0", 0.0))
    # Match the existing Organella viewer convention: (x, -y, serial-z).
    return np.column_stack(
        [
            x_cropped * reg.DS_SPACING,
            -y_cropped * reg.DS_SPACING,
            z_ds * reg.Z_SPACING_UM,
        ]
    ).astype(np.float32)


def prepare_points(reg, fish: int, boxes: pd.DataFrame, ids_by_run, folders):
    cell_positions: list[np.ndarray] = []
    transcript_positions: list[np.ndarray] = []
    transcript_genes: list[np.ndarray] = []
    used_cells: set[tuple[int, str]] = set()
    qv_min = float(CONFIG["selection"]["qv_min"])

    for run, target_ids in ids_by_run.items():
        if not target_ids:
            continue
        folder = folders[run]
        cells = pd.read_parquet(
            folder / "cells.parquet", columns=["cell_id", "x_centroid", "y_centroid"]
        )
        cells["cell_id"] = cells["cell_id"].astype(str)
        cells = cells[cells["cell_id"].isin(target_ids)].copy()
        cells["x_px"] = cells["x_centroid"] / reg.XENIUM_PX_UM
        cells["y_px"] = cells["y_centroid"] / reg.XENIUM_PX_UM

        transcripts = pd.read_parquet(
            folder / "transcripts.parquet",
            columns=["cell_id", "feature_name", "x_location", "y_location", "qv", "is_gene"],
            filters=[("qv", ">=", qv_min), ("is_gene", "==", True)],
        )
        transcripts["cell_id"] = transcripts["cell_id"].astype(str)
        transcripts = transcripts[transcripts["cell_id"].isin(target_ids)]

        fish_boxes = boxes[(boxes["run"] == run) & (boxes["fish_name"] == fish)]
        fish_boxes = fish_boxes[fish_boxes["global_slice_num"].notna()]
        for row in fish_boxes.itertuples(index=False):
            in_box = (
                (cells["x_px"] >= row.bbox_global_min_col)
                & (cells["x_px"] <= row.bbox_global_max_col)
                & (cells["y_px"] >= row.bbox_global_min_row)
                & (cells["y_px"] <= row.bbox_global_max_row)
            )
            tile_cells = cells[in_box].copy()
            if tile_cells.empty:
                continue
            keys = [(run, value) for value in tile_cells["cell_id"]]
            keep = np.array([key not in used_cells for key in keys])
            tile_cells = tile_cells[keep]
            if tile_cells.empty:
                continue
            used_cells.update((run, value) for value in tile_cells["cell_id"])
            cell_positions.append(
                transform_tile(reg, tile_cells, fish, row, "x_centroid", "y_centroid")
            )
            tile_tx = transcripts[transcripts["cell_id"].isin(tile_cells["cell_id"])]
            if not tile_tx.empty:
                transcript_positions.append(
                    transform_tile(reg, tile_tx, fish, row, "x_location", "y_location")
                )
                transcript_genes.append(tile_tx["feature_name"].astype(str).to_numpy())

    cells_out = np.concatenate(cell_positions) if cell_positions else np.empty((0, 3), np.float32)
    tx_out = np.concatenate(transcript_positions) if transcript_positions else np.empty((0, 3), np.float32)
    genes_out = (
        np.asarray(np.concatenate(transcript_genes), dtype=str)
        if transcript_genes
        else np.empty(0, dtype="U1")
    )
    return cells_out, tx_out, genes_out


def package_version(module_name: str) -> str:
    module = __import__(module_name)
    return str(getattr(module, "__version__", "unknown"))


def main() -> None:
    global CONFIG
    args = parse_args()
    with args.config.resolve().open("rb") as handle:
        CONFIG = tomllib.load(handle)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    reg = load_registration_module(CONFIG)

    annotations_path = REPO / "leiden10annots.csv"
    annotations = pd.read_csv(annotations_path, dtype=str)
    target = annotations[
        annotations["leiden10annots"] == CONFIG["selection"]["cell_type"]
    ].copy()
    folders = run_folder_map(reg)
    ids_by_run = match_target_ids_by_run(target, folders)
    boxes_path = ANALYSIS / "fish_bbox_summary_tagged_2_4_5.csv"
    boxes = pd.read_csv(boxes_path)

    prepared = {}
    all_gene_counts: Counter[str] = Counter()
    mesh_inputs = []
    for fish in CONFIG["selection"]["fish"]:
        cells, transcripts, genes = prepare_points(reg, fish, boxes, ids_by_run, folders)
        tissue_v, tissue_f, tissue_source = load_surface(fish, "tissue", "file")
        tubule_v, tubule_f, tubule_source = load_surface(
            fish, "pronephric_distal_early_tubule", "instance"
        )
        prepared[int(fish)] = {
            "cells": cells,
            "transcripts": transcripts,
            "genes": genes,
            "tissue_vertices": tissue_v,
            "tissue_faces": tissue_f,
            "tubule_vertices": tubule_v,
            "tubule_faces": tubule_f,
        }
        all_gene_counts.update(genes.tolist())
        mesh_inputs.extend([tissue_source, tubule_source])

    selected_genes = [name for name, _ in all_gene_counts.most_common(CONFIG["selection"]["top_genes"])]
    manifest = {
        "description": "Registered pronephric distal early tubule Luxar inputs",
        "config": CONFIG,
        "selected_genes": selected_genes,
        "target_annotation_cells": len(target),
        "target_ids_by_run": {str(k): len(v) for k, v in ids_by_run.items()},
        "inputs": [str(annotations_path), str(boxes_path)]
        + [str(path / "transcripts.parquet") for path in folders.values()]
        + sorted({str(path) for path in mesh_inputs}),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in ["numpy", "pandas", "pyarrow", "polars", "scipy", "tifffile", "trimesh"]
        },
        "fish": {},
    }

    for fish, arrays in prepared.items():
        mask = np.isin(arrays["genes"], selected_genes)
        output = PROCESSED / f"fish_{fish}.npz"
        np.savez_compressed(
            output,
            cells=arrays["cells"],
            transcripts=arrays["transcripts"][mask],
            genes=arrays["genes"][mask],
            tissue_vertices=arrays["tissue_vertices"],
            tissue_faces=arrays["tissue_faces"],
            tubule_vertices=arrays["tubule_vertices"],
            tubule_faces=arrays["tubule_faces"],
        )
        manifest["fish"][str(fish)] = {
            "cells": int(len(arrays["cells"])),
            "transcripts_all_genes": int(len(arrays["transcripts"])),
            "transcripts_selected_genes": int(mask.sum()),
            "gene_counts": dict(Counter(arrays["genes"][mask].tolist())),
            "tissue_vertices": int(len(arrays["tissue_vertices"])),
            "tubule_vertices": int(len(arrays["tubule_vertices"])),
            "processed_file": str(output),
            "sha256": sha256(output),
        }
        print(f"fish {fish}: {len(arrays['cells'])} cells, {mask.sum()} selected transcripts")

    BUILD.mkdir(exist_ok=True)
    with (BUILD / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    print("selected genes:", ", ".join(selected_genes))
    print(f"wrote {BUILD / 'manifest.json'}")


if __name__ == "__main__":
    main()
