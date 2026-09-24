#!/usr/bin/env python3
"""Compile fitted DAPI splats plus pronephric cells and two genes into Luxar."""

from __future__ import annotations

import argparse
import json
import shutil
import tomllib
from pathlib import Path

import numpy as np
from luxar import CameraConfig, Dimensions, LuxarZarrCompiler, UIConfig, ViewerConfig


HERE = Path(__file__).resolve().parents[1]
BUILD = HERE / "build"


def colors(count: int, rgb: list[int]) -> np.ndarray:
    return np.broadcast_to(np.asarray(rgb, dtype=np.uint8), (count, 3)).copy()


def point_positions(values: np.ndarray, center: np.ndarray, shift: float) -> np.ndarray:
    result = values.astype(np.float32, copy=True) - center.astype(np.float32)
    result[:, 0] += shift
    return result


def volume_transform(
    center: np.ndarray, spacing_zyx: list[float], shift: float
) -> np.ndarray:
    z_spacing, y_spacing, x_spacing = map(float, spacing_zyx)
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 0] = x_spacing
    matrix[1, 1] = -y_spacing
    matrix[2, 2] = z_spacing
    matrix[:3, 3] = [shift - center[0], -center[1], -center[2]]
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "config.toml")
    args = parser.parse_args()
    with args.config.resolve().open("rb") as handle:
        config = tomllib.load(handle)
    manifest = json.loads((BUILD / "manifest.json").read_text())

    fish_ids = [int(value) for value in config["selection"]["fish"]]
    labels = list(config["selection"]["fish_labels"])
    genes = list(config["selection"]["genes"])
    center = np.asarray(manifest["image_center_xyz_um"], dtype=np.float64)
    source_shape = manifest["source_shape_zyx"]
    width_um = (source_shape[2] - 1) * float(config["registration"]["xy_pixel_um"])
    gap = float(config["appearance"]["comparison_gap_um"])
    shifts = np.linspace(-(width_um + gap) / 2.0, (width_um + gap) / 2.0, len(fish_ids))
    scene_width = width_um * len(fish_ids) + gap

    output = BUILD / config["output"]["scene_name"]
    if output.exists():
        shutil.rmtree(output)
    viewer = ViewerConfig(
        title="Registered DAPI splats — pronephric cells and two genes",
        background_color=config["appearance"]["background_color"],
        tone_mapping="ACES",
        exposure=0.45,
        bloom_enabled=True,
        bloom_strength=0.08,
        bloom_radius=0.75,
        bloom_threshold=0.12,
        fxaa_enabled=True,
        auto_rotate=False,
        ui=UIConfig(show_layers=True, show_rendering_controls=True, show_help=True),
        camera=CameraConfig(
            position=(0.0, -scene_width * 1.65, scene_width * 0.75),
            target=(0.0, 0.0, 0.0),
            up=(0.0, 0.0, 1.0),
            fov=48.0,
        ),
    )

    with LuxarZarrCompiler(output) as compiler:
        scene = compiler.create_scene(Dimensions.default_3d(), viewer_config=viewer)
        scene.add_text(
            "DAPI image → Gaussian splats | pronephric cells + two genes",
            position=(0.02, 0.03),
            font_size=0.032,
            anchor="top-left",
        )
        dapi = scene.add_group(
            "DAPI tissue context — image-derived splats",
            layer=True,
            colormap=config["appearance"]["dapi_colormap"],
            opacity=float(config["appearance"]["dapi_opacity"]),
            absorption=float(config["appearance"]["dapi_absorption"]),
            blending_mode="volumetric",
        )
        cells_layer = scene.add_group(
            "Pronephric distal early tubule cells",
            layer=True,
            opacity=0.82,
            blending_mode="additive",
        )
        gene_layers = {
            gene: scene.add_group(
                f"Gene — {gene}", layer=True, opacity=0.92, blending_mode="additive"
            )
            for gene in genes
        }

        for fish, label, shift in zip(fish_ids, labels, shifts, strict=True):
            dapi.add_gsplats_from_file(
                label,
                BUILD / "fits" / f"fish_{fish}_stream.gsplats.zarr",
                dim_order=["z", "y", "x"],
                transform=volume_transform(
                    center, manifest["effective_spacing_zyx_um"], float(shift)
                ),
                normalize_amplitudes=True,
            )
            data = np.load(BUILD / "prepared" / f"fish_{fish}_points.npz")
            cell_points = point_positions(data["cells"], center, float(shift))
            cells_layer.add_points(
                label,
                cell_points,
                colors=colors(len(cell_points), config["appearance"]["cell_color"]),
                radii=float(config["appearance"]["cell_radius_um"]),
            )
            for index, gene in enumerate(genes):
                mask = data["genes"] == gene
                positions = point_positions(data["transcripts"][mask], center, float(shift))
                gene_layers[gene].add_points(
                    label,
                    positions,
                    colors=colors(
                        len(positions), config["appearance"]["gene_colors"][index]
                    ),
                    radii=float(config["appearance"]["transcript_radius_um"]),
                )
            scene.add_text(
                label,
                position=(0.27 if shift < 0 else 0.73, 0.92),
                font_size=0.03,
                anchor="center",
            )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

