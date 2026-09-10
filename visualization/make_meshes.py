"""
make_meshes.py
==============
Generate 3D surface meshes for focal cell types from registered zebrafish
cell_type_label volumes (script 5 consensus output).

One representative fish is chosen per group (highest within-group Dice on
dapi_blend_rigid_affine_z10):
  WT     → fish 4  (dice_tissue = 0.704)
  Mutant → fish 3  (dice_tissue = 0.655)
  (fish 1 is the registration reference — excluded from evaluation)

Pipeline per cell type per fish:
  1. Load cell_type_label.tif (uint8, Z × H × W)
  2. Downsample XY 8× with majority-vote (skimage.measure.block_reduce)
     → voxel size: XY = 1.7 µm, Z = 10 µm
  3. Extract binary mask for the label
  4. Gaussian smooth (anisotropy-aware: small σ in Z, larger in XY)
  5. Marching cubes with spacing=(10, 1.7, 1.7) → vertices in physical µm
  6. Trimesh Laplacian smooth + cleanup
  7. Save as .ply → visualization/meshes/{group}_{cell_type}.ply

Usage
-----
  python make_meshes.py
"""

import os
import re

import numpy as np
import tifffile
import trimesh
from scipy.ndimage import gaussian_filter, label as nd_label
from skimage.measure import block_reduce, marching_cubes

# ── configuration ──────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
DATA_BASE  = os.path.join(os.path.dirname(REPO_ROOT), "analysis", "5_consensus")
OUT_DIR    = os.path.join(SCRIPT_DIR, "meshes")

ANNOTS_CSV = os.path.join(REPO_ROOT, "leiden10annots.csv")

# Representative fish chosen by best within-group dice_tissue
REP_FISH = {"WT": 4, "Mutant": 3}

# Physical voxel sizes
XY_UM       = 0.2125
Z_UM        = 10.0
DS_FACTOR   = 8                        # XY downsample factor
DS_XY_UM    = XY_UM * DS_FACTOR       # 1.7 µm after downsampling

# Gaussian smooth sigma in voxel units — target ~5 µm physical in each axis:
#   Z:  5 µm / 10 µm per voxel = 0.5 voxels
#   XY: 5 µm / 1.7 µm per voxel ≈ 3.0 voxels
SIGMA = (0.5, 3.0, 3.0)

SMOOTH_THRESHOLD = 0.15   # iso-level on smoothed float mask (loose — recover thin structures)
MC_ISO           = 0.25   # marching cubes iso-level
LAPLACIAN_ITERS  = 5

# Fallback sphere radius (µm) per voxel when marching cubes fails entirely.
# Each downsampled voxel ≈ DS_FACTOR² × XY_UM² × Z_UM volume.
# Sphere radius = cube_root(n_voxels × DS_FACTOR² × voxel_vol / (4/3 π)) capped.
MIN_SPHERE_RADIUS_UM = 5.0    # smallest sphere (µm) placed per isolated cluster
MAX_SPHERE_RADIUS_UM = 30.0   # cap so sparse blood doesn't balloon

FOCAL_TYPES = [
    "blood",
    "hematopoietic cell",
    "cranial vasculature",
    "spinal cord",
    "pronephric distal early tubule",
]

# ── label map ──────────────────────────────────────────────────────────────────

def build_label_map(annots_csv: str) -> dict:
    """1-based alphabetical label IDs — matches 2_rigid_registration.py."""
    import csv
    cell_types = set()
    with open(annots_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cell_types.add(row["leiden10annots"])
    return {ct: i + 1 for i, ct in enumerate(sorted(cell_types))}


# ── processing ─────────────────────────────────────────────────────────────────

def load_label_vol(fish: int) -> np.ndarray:
    path = os.path.join(DATA_BASE, str(fish), "cell_type_label.tif")
    print(f"  loading fish {fish}: {path}")
    return tifffile.imread(path)


def downsample_labels(vol: np.ndarray, factor: int) -> np.ndarray:
    """Majority-vote XY downsampling for label volumes (no colour mixing)."""
    return block_reduce(vol, block_size=(1, factor, factor), func=np.max).astype(np.uint8)


def _spheres_fallback(mask_bin: np.ndarray) -> trimesh.Trimesh:
    """Place one sphere per connected cluster at its centroid.
    Radius is proportional to cube root of cluster voxel count (physical µm),
    clamped to [MIN_SPHERE_RADIUS_UM, MAX_SPHERE_RADIUS_UM].
    Vertices are in physical µm (same space as marching-cubes meshes).
    """
    labeled, n_clusters = nd_label(mask_bin)
    meshes = []
    voxel_vol_um3 = Z_UM * DS_XY_UM * DS_XY_UM
    for i in range(1, n_clusters + 1):
        coords = np.argwhere(labeled == i)   # (N, 3) in voxel indices (Z, Y, X)
        centroid_vox = coords.mean(axis=0)
        # Convert to physical µm: (Z, Y, X) × spacing
        centroid_um = centroid_vox * np.array([Z_UM, DS_XY_UM, DS_XY_UM])
        n_vox = len(coords)
        vol_um3 = n_vox * voxel_vol_um3
        radius = np.clip(
            (3 * vol_um3 / (4 * np.pi)) ** (1 / 3),
            MIN_SPHERE_RADIUS_UM,
            MAX_SPHERE_RADIUS_UM,
        )
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius)
        sphere.apply_translation(centroid_um)
        meshes.append(sphere)
    return trimesh.util.concatenate(meshes)


def make_mesh(vol_ds: np.ndarray, label_id: int) -> trimesh.Trimesh | None:
    """Binary mask → smoothed → marching cubes → cleaned trimesh.
    Falls back to cluster spheres if marching cubes cannot run (too sparse).
    """
    mask_bin = (vol_ds == label_id)
    n_vox = int(mask_bin.sum())
    if n_vox == 0:
        print(f"    label {label_id}: no voxels — skipping")
        return None

    # Anisotropy-aware smoothing
    mask_smooth = gaussian_filter(mask_bin.astype(np.float32), sigma=SIGMA)

    # Threshold to recover thin structures (loose threshold before MC)
    mask_thresholded = mask_smooth > SMOOTH_THRESHOLD
    if not mask_thresholded.any():
        # Structure too sparse to survive smoothing — place spheres at raw voxels
        print(f"    label {label_id}: sparse ({n_vox} voxels) — sphere fallback")
        return _spheres_fallback(mask_bin)

    # Second pass smooth on binary → float for marching cubes iso surface
    mask_smooth2 = gaussian_filter(mask_thresholded.astype(np.float32), sigma=SIGMA)

    if mask_smooth2.max() < MC_ISO:
        print(f"    label {label_id}: max smoothed value {mask_smooth2.max():.3f} < "
              f"MC_ISO {MC_ISO} — sphere fallback")
        return _spheres_fallback(mask_bin)

    try:
        verts, faces, normals, _ = marching_cubes(
            mask_smooth2,
            level=MC_ISO,
            spacing=(Z_UM, DS_XY_UM, DS_XY_UM),   # physical µm — critical
            allow_degenerate=False,
            step_size=1,
        )
    except (ValueError, RuntimeError) as e:
        print(f"    marching_cubes failed ({e}) — sphere fallback")
        return _spheres_fallback(mask_bin)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces,
                           vertex_normals=normals, process=False)
    # Only Laplacian-smooth large meshes — small sparse meshes have degenerate
    # topology that causes the smoother to fly vertices to wrong positions.
    if len(mesh.vertices) >= 2000:
        mesh = trimesh.smoothing.filter_laplacian(mesh, iterations=LAPLACIAN_ITERS)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.fill_holes()
    return mesh


def safe_filename(cell_type: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", cell_type.lower()).strip("_")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Building label map…")
    type_to_id = build_label_map(ANNOTS_CSV)
    for ct in FOCAL_TYPES:
        if ct not in type_to_id:
            raise ValueError(f"Cell type not found in annotations: {ct!r}")
        print(f"  {ct!r:45s} → label {type_to_id[ct]}")

    for group, fish in REP_FISH.items():
        print(f"\n{'='*60}")
        print(f"Group: {group}  (fish {fish})")
        print(f"{'='*60}")

        vol = load_label_vol(fish)
        print(f"  raw shape: {vol.shape}  dtype: {vol.dtype}")

        print(f"  downsampling XY {DS_FACTOR}×…")
        vol_ds = downsample_labels(vol, DS_FACTOR)
        print(f"  downsampled shape: {vol_ds.shape}  "
              f"voxel size: Z={Z_UM} µm, XY={DS_XY_UM:.2f} µm")
        del vol

        for cell_type in FOCAL_TYPES:
            label_id = type_to_id[cell_type]
            n_vox = int((vol_ds == label_id).sum())
            print(f"\n  [{group}] {cell_type!r}  (label {label_id}, {n_vox} voxels after DS)")

            mesh = make_mesh(vol_ds, label_id)
            if mesh is None:
                continue

            fname = f"{group.lower()}_{safe_filename(cell_type)}.ply"
            out_path = os.path.join(OUT_DIR, fname)
            mesh.export(out_path)
            print(f"    → {out_path}  "
                  f"({len(mesh.vertices):,} verts, {len(mesh.faces):,} faces)")

    # ── context tissue (one per group, all non-focal labels merged) ────────────
    focal_ids = set(type_to_id[ct] for ct in FOCAL_TYPES)
    for ctx_group, ctx_fish in REP_FISH.items():
        print(f"\n{'='*60}")
        print(f"Context tissue ({ctx_group}, fish {ctx_fish})")
        print(f"{'='*60}")
        vol_ctx = load_label_vol(ctx_fish)
        vol_ctx_ds = downsample_labels(vol_ctx, DS_FACTOR)
        del vol_ctx
        ctx_mask = (vol_ctx_ds > 0) & ~np.isin(vol_ctx_ds, list(focal_ids))
        n_ctx = int(ctx_mask.sum())
        print(f"  context voxels after DS: {n_ctx:,}")

        ctx_smooth = gaussian_filter(ctx_mask.astype(np.float32), sigma=(0.5, 2.0, 2.0))
        if ctx_smooth.max() >= 0.3:
            try:
                verts, faces, normals, _ = marching_cubes(
                    ctx_smooth, level=0.3,
                    spacing=(Z_UM, DS_XY_UM, DS_XY_UM),
                    allow_degenerate=False, step_size=2,
                )
                ctx_mesh = trimesh.Trimesh(vertices=verts, faces=faces,
                                           vertex_normals=normals, process=False)
                ctx_mesh = trimesh.smoothing.filter_laplacian(ctx_mesh, iterations=3)
                ctx_mesh.update_faces(ctx_mesh.nondegenerate_faces())
                ctx_mesh.update_faces(ctx_mesh.unique_faces())
                ctx_path = os.path.join(OUT_DIR, f"context_{ctx_group.lower()}.ply")
                ctx_mesh.export(ctx_path)
                print(f"  → {ctx_path}  "
                      f"({len(ctx_mesh.vertices):,} verts, {len(ctx_mesh.faces):,} faces)")
            except Exception as e:
                print(f"  context mesh failed: {e}")
        else:
            print("  context mask empty after smoothing — skipped")

    print(f"\nDone. Meshes saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
