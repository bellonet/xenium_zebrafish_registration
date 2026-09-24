#!/usr/bin/env python3
"""Compile prepared arrays into a polished side-by-side Luxar scene."""

from __future__ import annotations

import argparse
import json
import shutil
import tomllib
from pathlib import Path

import numpy as np
from luxar import CameraConfig, Dimensions, LuxarZarrCompiler, UIConfig, ViewerConfig


PROJECT = Path(__file__).resolve().parents[1]
BUILD = PROJECT / "build"
PALETTE = np.asarray(
    [
        [255, 45, 145],
        [255, 145, 0],
        [255, 55, 45],
        [255, 220, 0],
        [45, 220, 90],
        [50, 205, 255],
        [55, 95, 255],
        [185, 75, 255],
    ],
    dtype=np.uint8,
)


def solid_color(count: int, rgb) -> np.ndarray:
    return np.broadcast_to(np.asarray(rgb, dtype=np.uint8), (count, 3)).copy()


def shifted(values: np.ndarray, center: np.ndarray, x_shift: float) -> np.ndarray:
    result = values.astype(np.float32, copy=True) - center.astype(np.float32)
    result[:, 0] += x_shift
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT / "config.toml")
    args = parser.parse_args()
    with args.config.resolve().open("rb") as handle:
        config = tomllib.load(handle)
    with (BUILD / "manifest.json").open() as handle:
        manifest = json.load(handle)

    fish_ids = [int(value) for value in config["selection"]["fish"]]
    fish_labels = config["selection"]["fish_labels"]
    gene_colors = config["appearance"].get("gene_colors", {})
    datasets = {
        fish: np.load(BUILD / "processed" / f"fish_{fish}.npz") for fish in fish_ids
    }
    all_tissue = np.concatenate([datasets[fish]["tissue_vertices"] for fish in fish_ids])
    common_center = (all_tissue.min(axis=0) + all_tissue.max(axis=0)) / 2.0
    width = float(np.ptp(all_tissue[:, 0]))
    gap = float(config["appearance"]["comparison_gap_um"])
    shifts = np.linspace(-(width + gap) / 2.0, (width + gap) / 2.0, len(fish_ids))
    scene_width = width * 2.0 + gap

    output = BUILD / config["output"]["scene_name"]
    if output.exists():
        shutil.rmtree(output)
    viewer = ViewerConfig(
        title="Pronephric distal early tubule — mutant vs WT",
        background_color=config["appearance"]["background_color"],
        tone_mapping="ACES",
        exposure=1.1,
        bloom_enabled=True,
        bloom_strength=0.12,
        bloom_radius=0.9,
        bloom_threshold=0.04,
        fxaa_enabled=True,
        allow_high_dpr=False,
        ui=UIConfig(
            show_layers=True,
            show_rendering_controls=True,
            show_help=True,
            show_overlays=True,
        ),
        camera=CameraConfig(
            position=(0.0, -scene_width * 1.25, scene_width * 0.7),
            target=(0.0, 0.0, 0.0),
            up=(0.0, 0.0, 1.0),
            fov=47.0,
        ),
    )

    with LuxarZarrCompiler(output) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d(), viewer_config=viewer)
        scene.add_text(
            "Pronephric distal early tubule — Xenium transcripts",
            position=(0.02, 0.03),
            font_size=0.034,
            anchor="top-left",
        )

        # Layers represent biological object types, not specimens. Each layer
        # contains geometry from both fish, so one Layers-panel control updates
        # the side-by-side pair together.
        tissue_layer = scene.add_group(
            "Tissue context",
            opacity=float(config["appearance"]["tissue_opacity"]),
            blending_mode="normal",
            layer=True,
        )
        tubule_layer = scene.add_group(
            "Pronephric distal early tubule",
            opacity=float(config["appearance"]["tubule_opacity"]),
            blending_mode="normal",
            layer=True,
        )
        cell_layer = scene.add_group(
            "Cells of interest — pronephric distal early tubule",
            opacity=0.72,
            blending_mode="additive",
            layer=True,
        )
        gene_layers = {
            gene: scene.add_group(
                f"Gene — {gene}",
                opacity=0.88,
                blending_mode="additive",
                layer=True,
            )
            for gene in manifest["selected_genes"]
        }

        for fish, label, x_shift in zip(fish_ids, fish_labels, shifts, strict=True):
            data = datasets[fish]
            tissue = shifted(data["tissue_vertices"], common_center, float(x_shift))
            tubule = shifted(data["tubule_vertices"], common_center, float(x_shift))
            cells = shifted(data["cells"], common_center, float(x_shift))
            transcripts = shifted(data["transcripts"], common_center, float(x_shift))

            tissue_layer.add_mesh(
                label,
                tissue,
                data["tissue_faces"],
                colors=solid_color(len(tissue), config["appearance"]["tissue_color"]),
                shading="smooth",
            )
            tubule_layer.add_mesh(
                label,
                tubule,
                data["tubule_faces"],
                colors=solid_color(len(tubule), config["appearance"]["tubule_color"]),
                shading="smooth",
            )
            if len(cells):
                cell_layer.add_points(
                    label,
                    cells,
                    colors=solid_color(len(cells), config["appearance"]["cell_color"]),
                    radii=float(config["appearance"]["cell_radius_um"]),
                )
            for index, gene in enumerate(manifest["selected_genes"]):
                mask = data["genes"] == gene
                if not mask.any():
                    continue
                points = transcripts[mask]
                gene_color = gene_colors.get(gene, PALETTE[index % len(PALETTE)])
                gene_layers[gene].add_points(
                    label,
                    points,
                    colors=solid_color(len(points), gene_color),
                    radii=float(config["appearance"]["transcript_radius_um"]),
                )
            scene.add_text(
                label,
                position=(0.27 if x_shift < 0 else 0.73, 0.92),
                font_size=0.03,
                anchor="center",
            )

    print(f"wrote {output}")


if __name__ == "__main__":
    main()
