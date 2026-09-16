"""
2b_stim_refinement.py

STIM-based residual correction on top of script 2's elastix registration.

Run AFTER 2_rigid_registration.py. Cell centroids are projected into the
same canvas space that script 2 already registered, so STIM only needs to
find the small residual misalignment that elastix missed.

Steps per fish:
1. build     -- apply script 2 forward transform to cell centroids, build STIM N5
2. stim      -- st-align-pairs (captures pairwise transforms to JSON)
                + st-align-global (STIM's own global opt — often unreliable)
3. chain     -- sequential chaining of pairwise transforms from reference outward
                (Option 1: simple, no global optimizer needed)
4. globalopt -- our own least-squares global optimizer over the pairwise graph
                (Option 2: more robust than STIM's GlobalOpt, in our coord space)
5. apply     -- apply per-slice transforms to script 2 registered TIFFs
6. stack     -- 3D stacks

--transform-source controls which transforms step 5 uses: stim|chain|globalopt

Input:  analysis/2_registered/{fish}/transforms.json
        analysis/2_registered/{fish}/transforms/{gnum}.txt  (elastix)
        analysis/2_registered/{fish}/c{ch}/{gnum}.tif       (registered)
        data/{run_folder}/transcripts.parquet + cells.parquet
        analysis/fish_bbox_summary_tagged_*.csv

Output: analysis/stim/{fish}/chain/c{ch}/{gnum}.tif        (or globalopt/)
        analysis/stim/{fish}/chain/rigid_3d_c{ch}.tif
        analysis/stim/{fish}/pairwise_transforms.json
        analysis/stim/{fish}/chain_transforms.json
        analysis/stim/{fish}/globalopt_transforms.json
        analysis/stim/{fish}/stim/                         (N5 container + h5ads)

Usage:
  python 2b_stim_refinement.py --fish 1
  python 2b_stim_refinement.py --from-step chain --transform-source chain --fish 1
  python 2b_stim_refinement.py --from-step globalopt --transform-source globalopt --fish 1
  python 2b_stim_refinement.py --from-step apply --transform-source chain --fish 1
"""

import gc
import glob
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import tifffile
from scipy.ndimage import affine_transform as nd_affine

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# ── paths ─────────────────────────────────────────────────────────────────────
SRC_DIR      = "../analysis/2_registered"   # script 2 output (registered TIFFs)
OUT_DIR      = "../analysis/stim"
DATA_DIR     = "../data"
SUMMARY_GLOB = "../analysis/fish_bbox_summary_tagged_*.csv"
NUM_CHANNELS = 4
DAPI_CHANNEL = 0

STIM_BIN = str(Path(__file__).parent.parent / "stim-bin")

RUN_FOLDERS = {
    "2": "output-XETG00046__0038328__Region_1__20250717__075022",
    "4": "output-XETG00046__0043921__Region_1__20250620__084504",
    "5": "output-XETG00046__0044004__Region_1__20250620__084505",
}

STEPS = ["build", "stim", "chain", "globalopt", "apply", "stack"]

# ── coordinate / imaging ───────────────────────────────────────────────────────
XENIUM_PX_PER_UM = 4.705882   # px / µm
MIN_ANGLE_DEG    = 1.0
MIN_CELLS_PER_SLICE = 20

# ── STIM ──────────────────────────────────────────────────────────────────────
# Scale can be smaller here because slices are already aligned — SIFT only
# needs to detect small residuals, not large coarse offsets.
STIM_SCALE         = 0.5
STIM_NUM_GENES     = 50
STIM_RANGE         = 2
STIM_NUM_THREADS   = 8


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def stim(cmd: List[str], **kwargs) -> None:
    env = os.environ.copy()
    jdk = shutil.which("javac")
    if jdk:
        env["JAVA_HOME"] = str(Path(jdk).parent.parent)
    env["JAVA_TOOL_OPTIONS"] = (
        env.get("JAVA_TOOL_OPTIONS", "") + " -Djava.awt.headless=true"
    ).strip()
    full = [str(Path(STIM_BIN) / cmd[0])] + cmd[1:]
    logging.info("  stim: %s", " ".join(full))
    subprocess.run(full, check=True, env=env, **kwargs)


def fix_h5ad_for_stim(path: str) -> None:
    """Convert newer anndata nullable-string indices to the plain format STIM expects."""
    with h5py.File(path, "r+") as f:
        to_fix = []
        if "var/_index" in f:
            to_fix.append("var/_index")
        obs_index_name = f["obs"].attrs.get("_index", "_index")
        obs_key = f"obs/{obs_index_name}"
        if obs_key in f:
            to_fix.append(obs_key)

        for key in to_fix:
            node = f[key]
            enc = node.attrs.get("encoding-type", "")
            if enc == "string-array" and not isinstance(node, h5py.Group):
                if node.attrs.get("encoding-version") != "0.2.0":
                    node.attrs["encoding-version"] = "0.2.0"
                continue
            if isinstance(node, h5py.Group):
                strings = node["values"][:].tolist() if "values" in node else []
            else:
                raw = node[:]
                strings = [s.decode() if isinstance(s, bytes) else s for s in raw.tolist()]
            del f[key]
            dt = h5py.string_dtype()
            ds = f.create_dataset(key, data=np.array(strings, dtype=object), dtype=dt)
            ds.attrs["encoding-type"] = "string-array"
            ds.attrs["encoding-version"] = "0.2.0"


def load_transforms_json(fish: int) -> Dict:
    path = os.path.join(SRC_DIR, str(fish), "transforms.json")
    with open(path) as f:
        return json.load(f)


def load_elastix_params(fish: int, gnum: int, meta: Dict) -> Optional[Dict]:
    """Read elastix .txt and return parsed transform params, or None."""
    sm = meta["slices"].get(str(gnum), {})
    ef = sm.get("elastix_file")
    if not ef:
        return None
    path = os.path.join(SRC_DIR, str(fish), ef)
    if not os.path.exists(path):
        return None
    params = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("(") and line.endswith(")"):
                parts = line[1:-1].split()
                params[parts[0]] = parts[1:]
    return params


def apply_script2_forward(
    x_px: np.ndarray,
    y_px: np.ndarray,
    sm: Dict,
    canvas_w: float,
    canvas_h: float,
    elastix_params: Optional[Dict],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the same forward transform chain as script 2 to canvas pixel coords.

    Input:  (x_px, y_px) in the padded canvas pixel space, BEFORE DV rotation.
    Output: (x_reg, y_reg) in the registered canvas pixel space (script 2 output).

    Chain: DV rotation (CCW) → elastix forward (moving→fixed).

    Elastix convention: the stored transform T maps fixed→moving (backward
    image mapping). So to map a point from moving→fixed we apply T^{-1},
    which for a rigid transform is: R^T @ (x - c - t) + c.
    """
    xy = np.column_stack([x_px, y_px])

    # DV rotation (CCW, same as script 2's rotate_arr)
    dv = sm.get("dv_angle_deg", 0.0)
    if abs(dv) >= MIN_ANGLE_DEG:
        a = math.radians(dv)
        cx, cy = canvas_w / 2.0, canvas_h / 2.0
        dx, dy = xy[:, 0] - cx, xy[:, 1] - cy
        ca, sa = math.cos(a), math.sin(a)
        # scipy.ndimage.rotate(angle) CCW: new_x = cx + cos*dx - sin*dy? No:
        # scipy rotates image CCW so pixel (r,c) comes from rotating (r,c) CW
        # in the source. For coordinates to follow the image, apply CCW:
        # x' = cx + cos*dx - sin*dy, y' = cy + sin*dx + cos*dy
        # But script 2 uses: cx + cos*dx + sin*dy (DV angle sign absorbs this)
        # Copy exactly from script 2's _map_coords_to_canvas:
        xy = np.column_stack([cx + ca * dx + sa * dy,
                               cy - sa * dx + ca * dy])

    # Elastix: T^{-1} maps moving→fixed (see script 2 _map_coords_to_canvas)
    if elastix_params is not None:
        tp   = [float(v) for v in elastix_params["TransformParameters"]]
        corp = [float(v) for v in elastix_params["CenterOfRotationPoint"]]
        angle, tx_t, ty_t = tp[0], tp[1], tp[2]
        cx, cy = corp[0], corp[1]
        ca, sa = math.cos(angle), math.sin(angle)
        dx = xy[:, 0] - cx - tx_t
        dy = xy[:, 1] - cy - ty_t
        xy = np.column_stack([cx + ca * dx + sa * dy,
                               cy - sa * dx + ca * dy])

    return xy[:, 0], xy[:, 1]


def read_stim_transform(h5ad_path: str) -> Optional[np.ndarray]:
    """Return the 3×3 affine from STIM's model_sift / model_icp, or None."""
    with h5py.File(h5ad_path, "r") as f:
        for key in ("uns/model_sift", "uns/transform", "uns/model_icp"):
            if key in f:
                raw = f[key][:]
                flat = raw.flatten()
                if flat.size == 6:
                    m = flat.reshape(2, 3)
                    return np.vstack([m, [0, 0, 1]])
                if flat.size == 9:
                    return flat.reshape(3, 3)
    return None


def apply_affine(arr: np.ndarray, M3: np.ndarray) -> np.ndarray:
    """Apply a 3×3 forward affine to a 2D float32 image (backward-map internally).

    M3 must be in (row, col) convention — use _corrected_M() to convert STIM
    pairwise matrices from their native (x, y) layout before composing.
    """
    M_inv = np.linalg.inv(M3)
    return nd_affine(arr, M_inv[:2, :2], offset=M_inv[:2, 2],
                     order=1, mode="constant", cval=0.0)


def _corrected_M(M2x3_or_3x3: np.ndarray) -> np.ndarray:
    """Convert a STIM pairwise matrix from (x, y) to (row, col) convention.

    STIM stores transforms as [[a, b, tx], [c, d, ty]] where the two columns of
    the 2×2 part address (x, col) and (y, row) respectively, and the translation
    is (tx=x-shift, ty=y-shift).

    scipy.ndimage.affine_transform (and therefore apply_affine) addresses arrays
    as (row=y, col=x).  To get the right result we need to swap the row-addressing
    and col-addressing parts, which means swapping tx↔ty in the translation vector
    and transposing the 2×2 rotation block:
      M_rc = [[d, c, ty], [b, a, tx], [0, 0, 1]]
    For STIM's pure rotation [[cos, sin, tx], [-sin, cos, ty]] this gives the
    standard CCW rotation matrix [[cos, -sin, ty], [sin, cos, tx]].
    """
    M = np.asarray(M2x3_or_3x3, dtype=float)
    a, b, tx = M[0, 0], M[0, 1], M[0, 2]
    c, d, ty = M[1, 0], M[1, 1], M[1, 2]
    return np.array([
        [d,  c, ty],
        [b,  a, tx],
        [0,  0,  1],
    ])


def get_registered_slices(fish: int, ch: int = DAPI_CHANNEL) -> List[Tuple[int, str]]:
    ch_dir = os.path.join(SRC_DIR, str(fish), f"c{ch}")
    result = []
    for f in glob.glob(os.path.join(ch_dir, "*.tif")):
        stem = os.path.splitext(os.path.basename(f))[0]
        if stem.isdigit():
            result.append((int(stem), f))
    return sorted(result)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — BUILD STIM CONTAINER FROM REGISTERED COORDINATES
# ══════════════════════════════════════════════════════════════════════════════

def step_build(fish_ids: List[int]) -> None:
    summary_csvs = glob.glob(SUMMARY_GLOB)
    if not summary_csvs:
        raise FileNotFoundError(f"No summary CSV at {SUMMARY_GLOB}")
    summary_df = pd.read_csv(summary_csvs[0])

    for fish in fish_ids:
        logging.info("=== Fish %d — building STIM container (Option B) ===", fish)

        meta = load_transforms_json(fish)
        canvas_h = meta["canvas_h"]
        canvas_w = meta["canvas_w"]

        registered_slices = get_registered_slices(fish, DAPI_CHANNEL)
        if len(registered_slices) < 2:
            logging.warning("Fish %d: fewer than 2 registered slices, skipping", fish)
            continue

        all_gnums = [g for g, _ in registered_slices]

        stim_dir = Path(OUT_DIR) / str(fish) / "stim"
        stim_dir.mkdir(parents=True, exist_ok=True)
        container = str(stim_dir / "fish.n5")
        if os.path.exists(container):
            shutil.rmtree(container)

        # Determine which run each slice belongs to
        slice_to_run: Dict[int, str] = {}
        fish_rows = summary_df[summary_df["fish_name"] == fish].dropna(
            subset=["global_slice_num", "run"]
        )
        for _, row in fish_rows.iterrows():
            slice_to_run[int(row["global_slice_num"])] = str(int(row["run"]))

        needed_runs = set(slice_to_run.get(g, "") for g in all_gnums) - {""}
        tx_by_run: Dict[str, pd.DataFrame] = {}
        for run in needed_runs:
            folder = RUN_FOLDERS[run]
            tx = pd.read_parquet(
                os.path.join(DATA_DIR, folder, "transcripts.parquet"),
                columns=["cell_id", "feature_name", "x_location",
                         "y_location", "qv", "is_gene"],
            )
            tx = tx[(tx["qv"] >= 20) & (tx["is_gene"] == True)].copy()
            cells = pd.read_parquet(
                os.path.join(DATA_DIR, folder, "cells.parquet"),
                columns=["cell_id", "x_centroid", "y_centroid"],
            )
            tx = tx.merge(cells[["cell_id", "x_centroid", "y_centroid"]],
                          on="cell_id", how="left")
            tx_by_run[run] = tx
            logging.info("  Run %s: %d transcripts", run, len(tx))

        added = 0
        S = XENIUM_PX_PER_UM

        for gnum in all_gnums:
            run = slice_to_run.get(gnum)
            if run is None:
                logging.warning("  Slice %d: no run mapping, skipping", gnum)
                continue

            sm_rows = summary_df[
                (summary_df["fish_name"] == fish) &
                (summary_df["global_slice_num"] == gnum)
            ]
            if len(sm_rows) == 0:
                logging.warning("  Slice %d: not in summary, skipping", gnum)
                continue
            row = sm_rows.iloc[0]

            # Bbox in µm
            r_min = row["bbox_global_min_row"] / S
            r_max = row["bbox_global_max_row"] / S
            c_min = row["bbox_global_min_col"] / S
            c_max = row["bbox_global_max_col"] / S

            tx_run = tx_by_run[run]
            cid = tx_run["cell_id"]
            assigned = (cid != -1) & (cid != "UNASSIGNED")
            mask = (
                (tx_run["y_location"] >= r_min) &
                (tx_run["y_location"] <= r_max) &
                (tx_run["x_location"] >= c_min) &
                (tx_run["x_location"] <= c_max) &
                assigned
            )
            tx_slice = tx_run[mask].copy()
            if len(tx_slice) == 0:
                logging.warning("  Slice %d: no transcripts, skipping", gnum)
                continue

            counts = (
                tx_slice
                .groupby(["cell_id", "feature_name"])
                .size()
                .reset_index(name="count")
                .pivot(index="cell_id", columns="feature_name", values="count")
                .fillna(0)
                .astype(np.float32)
            )
            centroids = (
                tx_slice[["cell_id", "x_centroid", "y_centroid"]]
                .drop_duplicates("cell_id")
                .set_index("cell_id")
                .reindex(counts.index)
            )
            if len(counts) < MIN_CELLS_PER_SLICE:
                logging.warning("  Slice %d: only %d cells, skipping",
                                gnum, len(counts))
                continue

            # Convert µm centroids → pre-DV canvas pixel space.
            # This matches how script 2 reads the images (load_arr pads to canvas).
            slice_meta = meta["slices"].get(str(gnum), {})
            pad_left = slice_meta.get("pad_left", 0)
            pad_top  = slice_meta.get("pad_top",  0)
            bbox_col_min_px = row["bbox_global_min_col"]  # already in pixels
            bbox_row_min_px = row["bbox_global_min_row"]

            x_px = centroids["x_centroid"].values * S - bbox_col_min_px + pad_left
            y_px = centroids["y_centroid"].values * S - bbox_row_min_px + pad_top

            # Apply the full script 2 forward transform (DV rotation + elastix)
            elastix_params = load_elastix_params(fish, gnum, meta)
            x_reg, y_reg = apply_script2_forward(
                x_px, y_px, slice_meta, canvas_w, canvas_h, elastix_params
            )

            # Build h5ad with registered coordinates
            X_sparse = sp.csr_matrix(counts.values)
            adata = ad.AnnData(
                X=X_sparse,
                obs=pd.DataFrame(index=counts.index.astype(str)),
                var=pd.DataFrame(index=counts.columns.tolist()),
            )
            adata.obsm["spatial"] = np.column_stack(
                [x_reg, y_reg]
            ).astype(np.float32)

            h5ad_path = str(stim_dir / f"slice_{gnum:04d}.h5ad")
            adata.write_h5ad(h5ad_path)
            fix_h5ad_for_stim(h5ad_path)

            stim(
                ["st-add-slice", "--container", container, "--input", h5ad_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            added += 1
            logging.info("  Slice %d: %d cells, %d genes (registered coords)",
                         gnum, len(counts), counts.shape[1])

        logging.info("Fish %d: %d slices added", fish, added)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — STIM ALIGNMENT
# ══════════════════════════════════════════════════════════════════════════════

_PAIR_RE = re.compile(
    r"slice_(\d+)\.h5ad<>slice_(\d+)\.h5ad\s+(\d+)\s+(\d+)\s+"
    r"2d-affine: \(([^)]+)\)"
)


def _parse_pairwise_output(text: str) -> Dict:
    """Parse st-align-pairs stdout into a dict keyed by (a, b) slice numbers."""
    pairs = {}
    for m in _PAIR_RE.finditer(text):
        a, b, inliers, cands = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        vals = [float(v) for v in m.group(5).split(",")]
        # 2d-affine: (a, b, tx, c, d, ty)  — rows of the 2×3 matrix
        M = [[vals[0], vals[1], vals[2]],
             [vals[3], vals[4], vals[5]]]
        key = (min(a, b), max(a, b))
        # Keep only the canonical direction (lower → higher slice)
        if a < b:
            pairs[key] = {"M": M, "inliers": inliers, "candidates": cands, "from": a, "to": b}
        else:
            # Reverse: invert the affine so it goes from lower to higher
            M3 = np.array(M + [[0, 0, 1]], dtype=float)
            Mi = np.linalg.inv(M3)[:2].tolist()
            pairs[key] = {"M": Mi, "inliers": inliers, "candidates": cands, "from": b, "to": a}
    return pairs


def step_stim(fish_ids: List[int]) -> None:
    for fish in fish_ids:
        stim_dir  = Path(OUT_DIR) / str(fish) / "stim"
        container = str(stim_dir / "fish.n5")
        if not os.path.exists(container):
            logging.warning("Fish %d: container missing, skipping", fish)
            continue

        logging.info("=== Fish %d — running STIM alignment ===", fish)

        # Capture st-align-pairs output so we can extract pairwise transforms
        env = os.environ.copy()
        jdk = shutil.which("javac")
        if jdk:
            env["JAVA_HOME"] = str(Path(jdk).parent.parent)
        env["JAVA_TOOL_OPTIONS"] = (
            env.get("JAVA_TOOL_OPTIONS", "") + " -Djava.awt.headless=true"
        ).strip()

        pairs_cmd = [
            str(Path(STIM_BIN) / "st-align-pairs"),
            "--container",         container,
            "--numGenes",          str(STIM_NUM_GENES),
            "--range",             str(STIM_RANGE),
            "--scale",             str(STIM_SCALE),
            "--numThreads",        str(STIM_NUM_THREADS),
            "--minNumInliers",     "10",
            "--minNumInliersGene", "3",
            "--hidePairwiseRendering",
            "--overwrite",
        ]
        logging.info("  stim: %s", " ".join(pairs_cmd))
        result = subprocess.run(
            pairs_cmd, check=True, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True,
        )
        # Log the output
        for line in result.stdout.splitlines():
            if "2d-affine" in line or "Aligning" in line or "ERROR" in line.upper():
                logging.info("  [stim] %s", line.strip())

        # Parse and save pairwise transforms
        pairs = _parse_pairwise_output(result.stdout)
        pairs_path = str(Path(OUT_DIR) / str(fish) / "pairwise_transforms.json")
        # JSON keys must be strings
        pairs_out = {f"{a},{b}": v for (a, b), v in pairs.items()}
        with open(pairs_path, "w") as f:
            json.dump(pairs_out, f, indent=2)
        n_adj = sum(1 for (a, b) in pairs if b - a == 1)
        logging.info("Fish %d: %d pairwise transforms saved (%d adjacent)",
                     fish, len(pairs), n_adj)

        # Also run GlobalOpt (STIM's built-in) — results may be in wrong coord space,
        # but we keep it for comparison via --transform-source stim
        stim([
            "st-align-global",
            "--container",        container,
            "--skipICP",
            "--skipDisplayResults",
        ])

        h5ads = sorted(stim_dir.glob("slice_*.h5ad"))
        n_with = sum(1 for p in h5ads if read_stim_transform(str(p)) is not None)
        logging.info("Fish %d: GlobalOpt done — %d/%d h5ads have model_sift",
                     fish, n_with, len(h5ads))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — SEQUENTIAL CHAINING (Option 1)
# ══════════════════════════════════════════════════════════════════════════════

# After script 2's elastix alignment, adjacent slices should already be close.
# Any pairwise SIFT result with rotation > MAX_PAIR_ROT or translation > MAX_PAIR_T
# is a SIFT failure and is excluded from chaining / global optimization.
MAX_PAIR_ROT_DEG = 20.0   # degrees
MAX_PAIR_T_PX    = 300.0  # pixels


def _load_pairwise(fish: int) -> Tuple[Dict, List[int]]:
    """Load pairwise_transforms.json; return (pairs dict keyed (a,b), sorted gnums).

    Pairs with 0 inliers or physically impossible transforms are excluded.
    """
    path = str(Path(OUT_DIR) / str(fish) / "pairwise_transforms.json")
    with open(path) as f:
        raw = json.load(f)
    pairs = {}
    n_total = n_kept = 0
    for k, v in raw.items():
        a, b = map(int, k.split(","))
        n_total += 1
        if v["inliers"] == 0:
            continue
        M = np.array(v["M"])
        rot = abs(math.degrees(math.atan2(M[0, 1], M[0, 0])))
        t   = math.sqrt(M[0, 2]**2 + M[1, 2]**2)
        if rot > MAX_PAIR_ROT_DEG or t > MAX_PAIR_T_PX:
            logging.debug("  Pair %d<>%d discarded: rot=%.1f° t=%.1fpx", a, b, rot, t)
            continue
        pairs[(a, b)] = v
        n_kept += 1
    logging.info("  Pairwise pairs: %d kept / %d total (outlier threshold %.0f° %.0fpx)",
                 n_kept, n_total, MAX_PAIR_ROT_DEG, MAX_PAIR_T_PX)
    gnums = sorted(set(x for (a, b) in pairs for x in (a, b)))
    return pairs, gnums


def _ref_gnum(fish: int, gnums: List[int]) -> int:
    """Use the same reference as script 2: slice with no elastix transform (middle)."""
    meta = load_transforms_json(fish)
    for g in gnums:
        sm = meta["slices"].get(str(g), {})
        if not sm.get("elastix_file"):
            return g
    # Fallback: middle of available gnums
    return gnums[len(gnums) // 2]


def step_chain(fish_ids: List[int]) -> None:
    """
    Option 1 — sequential chaining of adjacent pairwise transforms.

    STIM logs pairwise transforms with .inverse() applied, so pairwise_transforms.json
    stores T_log(A,B) = inv(T_fwd(A→B)), which maps B's pixel coords to A's pixel coords.
    T_fwd(A→B) maps A's coords to B's coords.

    All matrices are converted from STIM's (x,y) convention to (row,col) convention
    using _corrected_M() before composing, so the resulting T_abs matrices can be
    passed directly to apply_affine().

    Chain rules (where M_rc = _corrected_M(T_log(A,B))):
      Going DOWN from ref (i < ref):
        backward_map(ref→i) = M_rc(i, i+1) @ backward_map(ref→i+1)
        → T_abs[i] = T_abs[i+1] @ inv(M_rc(i, i+1))

      Going UP from ref (i > ref):
        backward_map(ref→i) = inv(M_rc(i-1, i)) @ backward_map(ref→i-1)
        → T_abs[i] = T_abs[i-1] @ M_rc(i-1, i)

    Slices missing a pairwise transform inherit their neighbour's absolute transform
    (i.e., no additional correction applied for that step).
    """
    for fish in fish_ids:
        logging.info("=== Fish %d — sequential chaining ===", fish)
        pairs, gnums = _load_pairwise(fish)
        ref = _ref_gnum(fish, gnums)
        logging.info("  Reference slice: %d", ref)

        # Build absolute transforms in (row, col) convention for apply_affine.
        #
        # STIM logs transforms with .inverse() applied, so pairwise_transforms.json
        # stores T_log(A,B) = inv(T_fwd(A→B)), which maps B's coords to A's coords.
        #
        # Chain rules (derived in comments of step_chain docstring):
        #   Going DOWN from ref: T_abs[i] = T_abs[i+1] @ inv(M_rc(i, i+1))
        #   Going UP from ref:   T_abs[i] = T_abs[i-1] @ M_rc(i-1, i)
        # where M_rc = _corrected_M(M_log) converts (x,y)→(row,col) convention.
        T_abs: Dict[int, np.ndarray] = {ref: np.eye(3)}

        # Go DOWN from ref (toward lower slice numbers)
        for i in sorted([g for g in gnums if g < ref], reverse=True):
            prev = i + 1  # the slice already computed above i
            # nearest already-computed slice above i
            while prev not in T_abs and prev <= ref:
                prev += 1
            T_prev = T_abs.get(prev, np.eye(3))
            pair = pairs.get((i, prev))
            if pair is None:
                # Try to find the nearest pair with i as from/to
                pair = next(
                    (pairs[(a, b)] for (a, b) in sorted(pairs)
                     if a == i and b <= ref and b in T_abs),
                    None,
                )
            if pair is not None:
                M2 = np.array(pair["M"], dtype=float)  # 2×3, T_log(i, prev)
                M_rc = _corrected_M(M2)
                # M_log(i, prev) maps prev→i (backward from ref to i).
                # T_abs[i] is the forward, so: T_abs[i] = T_prev @ inv(M_rc)
                T_abs[i] = T_prev @ np.linalg.inv(M_rc)
            else:
                T_abs[i] = T_prev  # no correction for this step

        # Go UP from ref (toward higher slice numbers)
        for i in sorted([g for g in gnums if g > ref]):
            prev = i - 1
            while prev not in T_abs and prev >= ref:
                prev -= 1
            T_prev = T_abs.get(prev, np.eye(3))
            pair = pairs.get((prev, i))
            if pair is None:
                pair = next(
                    (pairs[(a, b)] for (a, b) in sorted(pairs)
                     if b == i and a >= ref and a in T_abs),
                    None,
                )
            if pair is not None:
                M2 = np.array(pair["M"], dtype=float)  # 2×3, T_log(prev, i)
                M_rc = _corrected_M(M2)
                # M_log(prev, i) maps i→prev (backward from ref toward i);
                # chain going up: T_abs[i] = T_prev @ M_rc
                T_abs[i] = T_prev @ M_rc
            else:
                T_abs[i] = T_prev

        # Save
        out = {str(g): T_abs[g].tolist() for g in sorted(T_abs)}
        path = str(Path(OUT_DIR) / str(fish) / "chain_transforms.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)

        # Report how large corrections are
        rots, mags = [], []
        for g, M in T_abs.items():
            if g == ref:
                continue
            rot = math.degrees(math.atan2(M[0, 1], M[0, 0]))
            tx, ty = M[0, 2], M[1, 2]
            rots.append(rot)
            mags.append(math.sqrt(tx**2 + ty**2))
        import statistics as _s
        if rots:
            logging.info(
                "Fish %d chain: rot mean=%.2f std=%.2f max=%.2f deg | "
                "t mean=%.1f std=%.1f max=%.1f px",
                fish,
                _s.mean(rots), _s.stdev(rots) if len(rots) > 1 else 0, max(abs(r) for r in rots),
                _s.mean(mags), _s.stdev(mags) if len(mags) > 1 else 0, max(mags),
            )
        logging.info("Fish %d: chain transforms saved → %s", fish, path)
    # Return stats for the last fish (or empty if none)
    return {
        'mean_rot': abs(_s.mean(rots)) if rots else 0.0,
        'max_rot':  max(abs(r) for r in rots) if rots else 0.0,
        'mean_t':   _s.mean(mags) if mags else 0.0,
        'max_t':    max(mags) if mags else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — OUR OWN GLOBAL OPTIMIZER (Option 2)
# ══════════════════════════════════════════════════════════════════════════════

def step_globalopt(fish_ids: List[int]) -> None:
    """
    Option 2 — least-squares global optimizer over the pairwise constraint graph.

    For small rotations (valid here since slices are already well-aligned by
    script 2), the rigid transform parameters (theta, tx, ty) decouple and we
    can solve three independent linear least-squares problems:

      For each pairwise (A, B) with weight w = inlier_count:
        w * (theta_B - theta_A) = w * theta_AB
        w * (tx_B   - tx_A)    = w * tx_AB
        w * (ty_B   - ty_A)    = w * ty_AB

    Anchor: the reference slice is fixed to (0, 0, 0).

    This is a standard graph-Laplacian least-squares problem. We solve it with
    scipy.linalg.lstsq. The result is globally consistent and uses our own
    coordinate space — no STIM coordinate frame confusion.
    """
    import scipy.linalg

    for fish in fish_ids:
        logging.info("=== Fish %d — global optimizer ===", fish)
        pairs, gnums = _load_pairwise(fish)
        ref = _ref_gnum(fish, gnums)
        logging.info("  Reference slice: %d  |  %d slices  |  %d pairs",
                     ref, len(gnums), len(pairs))

        idx = {g: i for i, g in enumerate(gnums)}
        N = len(gnums)

        # Collect constraints from pairwise transforms
        rows_theta, rows_tx, rows_ty = [], [], []
        b_theta, b_tx, b_ty = [], [], []

        for (a, b), p in pairs.items():
            if p["inliers"] < 5:
                continue  # skip very weak pairs
            M2 = np.array(p["M"], dtype=float)  # 2×3 in (x,y) STIM convention
            # Rotation angle is the same in both (x,y) and (row,col) conventions.
            theta_AB = math.atan2(M2[0, 1], M2[0, 0])
            # Convert translation to (row, col) convention: swap x/y.
            # STIM (x,y): M[0,2]=tx (col direction), M[1,2]=ty (row direction).
            # (row,col) forward transform has [row_shift, col_shift] = [ty, tx].
            row_shift_AB = M2[1, 2]   # ty → row direction
            col_shift_AB = M2[0, 2]   # tx → col direction
            w = math.sqrt(float(p["inliers"]))  # weight = sqrt(inliers)

            ia, ib = idx[a], idx[b]
            # theta_B - theta_A = theta_AB  → row has +1 at B, -1 at A
            row = [0.0] * N
            row[ib] = w; row[ia] = -w
            rows_theta.append(row); b_theta.append(w * theta_AB)
            rows_tx.append(row[:]); b_tx.append(w * col_shift_AB)
            rows_ty.append(row[:]); b_ty.append(w * row_shift_AB)

        # Anchor: reference slice fixed to zero
        i_ref = idx[ref]
        anchor = [0.0] * N
        anchor[i_ref] = 1e6  # large weight = hard constraint
        rows_theta.append(anchor); b_theta.append(0.0)
        rows_tx.append(anchor[:]); b_tx.append(0.0)
        rows_ty.append(anchor[:]); b_ty.append(0.0)

        A = np.array(rows_theta, dtype=float)
        theta_sol    = scipy.linalg.lstsq(A, b_theta)[0]
        col_shift_sol = scipy.linalg.lstsq(A, b_tx)[0]  # col direction
        row_shift_sol = scipy.linalg.lstsq(A, b_ty)[0]  # row direction

        # Build 3×3 affine matrices in (row, col) convention for apply_affine.
        # Standard CCW rotation in (row,col) space: [[cos,-sin],[sin,cos]].
        # Translation: [row_shift, col_shift] in positions [0,2] and [1,2].
        T_abs: Dict[int, np.ndarray] = {}
        for g in gnums:
            i = idx[g]
            th = theta_sol[i]
            rs, cs = row_shift_sol[i], col_shift_sol[i]
            ca, sa = math.cos(th), math.sin(th)
            T_abs[g] = np.array([
                [ca,  -sa, rs],
                [sa,   ca, cs],
                [0,    0,  1 ],
            ])

        # Save
        out = {str(g): T_abs[g].tolist() for g in sorted(T_abs)}
        path = str(Path(OUT_DIR) / str(fish) / "globalopt_transforms.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)

        # Report
        rots, mags = [], []
        for g in gnums:
            if g == ref:
                continue
            th = theta_sol[idx[g]]
            rs = row_shift_sol[idx[g]]
            cs = col_shift_sol[idx[g]]
            rots.append(math.degrees(th))
            mags.append(math.sqrt(rs**2 + cs**2))
        import statistics as _s
        if rots:
            logging.info(
                "Fish %d globalopt: rot mean=%.2f std=%.2f max=%.2f deg | "
                "t mean=%.1f std=%.1f max=%.1f px",
                fish,
                _s.mean(rots), _s.stdev(rots) if len(rots) > 1 else 0, max(abs(r) for r in rots),
                _s.mean(mags), _s.stdev(mags) if len(mags) > 1 else 0, max(mags),
            )
        logging.info("Fish %d: globalopt transforms saved → %s", fish, path)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — APPLY TRANSFORMS TO SCRIPT 2 REGISTERED TIFFs
# ══════════════════════════════════════════════════════════════════════════════

def _load_transforms_for_apply(
    fish: int, source: str, all_gnums: List[int]
) -> Dict[int, Optional[np.ndarray]]:
    """Load per-slice 3×3 affine matrices from the chosen source."""
    if source == "stim":
        stim_dir = Path(OUT_DIR) / str(fish) / "stim"
        out = {}
        for g in all_gnums:
            h5ad = str(stim_dir / f"slice_{g:04d}.h5ad")
            out[g] = read_stim_transform(h5ad) if os.path.exists(h5ad) else None
        return out
    elif source in ("chain", "globalopt"):
        fname = "chain_transforms.json" if source == "chain" else "globalopt_transforms.json"
        path = str(Path(OUT_DIR) / str(fish) / fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Transform file not found: {path}. "
                                    f"Run --from-step {source} first.")
        with open(path) as f:
            raw = json.load(f)
        return {g: np.array(raw[str(g)]) if str(g) in raw else None for g in all_gnums}
    else:
        raise ValueError(f"Unknown transform source: {source!r}")


def step_apply(fish_ids: List[int], source: str = "chain") -> None:
    for fish in fish_ids:
        logging.info("=== Fish %d — applying transforms (source=%s) ===", fish, source)

        registered_slices = get_registered_slices(fish, DAPI_CHANNEL)
        if not registered_slices:
            logging.warning("Fish %d: no registered slices found in %s", fish, SRC_DIR)
            continue

        all_gnums = [g for g, _ in registered_slices]
        last = all_gnums[-1]

        transforms = _load_transforms_for_apply(fish, source, all_gnums)
        n_with = sum(1 for v in transforms.values() if v is not None)
        logging.info("  %d/%d slices have transforms", n_with, len(all_gnums))

        # Get canvas shape from one registered TIFF
        sample = tifffile.imread(registered_slices[0][1])
        canvas_h, canvas_w = sample.shape[-2], sample.shape[-1]
        zero_frame = np.zeros((canvas_h, canvas_w), dtype=np.float32)

        for ch in (DAPI_CHANNEL,):   # c0 only — for testing; add channels once approach validated
            ch_slices = get_registered_slices(fish, ch)
            ch_paths  = {g: p for g, p in ch_slices}
            # Write to source-specific subdirectory so chain and globalopt
            # outputs can coexist and be compared.
            out_ch_dir = Path(OUT_DIR) / str(fish) / source / f"c{ch}"
            out_ch_dir.mkdir(parents=True, exist_ok=True)

            for gnum in range(1, last + 1):
                out_path = str(out_ch_dir / f"{gnum}.tif")
                if gnum not in ch_paths:
                    tifffile.imwrite(out_path, zero_frame, photometric="minisblack")
                    continue

                arr = tifffile.imread(ch_paths[gnum]).astype(np.float32)
                if arr.ndim != 2:
                    arr = arr[0]

                T = transforms.get(gnum)
                if T is not None:
                    arr = apply_affine(arr, T)

                tifffile.imwrite(out_path, arr, photometric="minisblack")

            logging.info("  c%d: done", ch)

        del zero_frame
        gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — STACK
# ══════════════════════════════════════════════════════════════════════════════

def step_stack(fish_ids: List[int], source: str = "chain") -> None:
    for fish in fish_ids:
        logging.info("=== Fish %d — building 3D stacks (source=%s) ===", fish, source)
        for ch in (DAPI_CHANNEL,):   # c0 only — for testing
            ch_dir = Path(OUT_DIR) / str(fish) / source / f"c{ch}"
            if not ch_dir.is_dir():
                continue
            files = sorted(ch_dir.glob("*.tif"), key=lambda f: int(f.stem))
            if not files:
                continue
            frames = [tifffile.imread(str(f)).astype(np.float32) for f in files]
            vol = np.stack(frames, axis=0)
            out_path = Path(OUT_DIR) / str(fish) / f"rigid_3d_{source}_c{ch}.tif"
            tifffile.imwrite(str(out_path), vol)
            gnums = [int(f.stem) for f in files]
            logging.info("  c%d: shape %s  slices %d–%d  → %s",
                         ch, vol.shape, gnums[0], gnums[-1], out_path)
            del frames, vol
            gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="STIM residual refinement on top of script 2."
    )
    parser.add_argument("--from-step", choices=STEPS, default="build",
                        metavar="STEP",
                        help=f"Resume from this step: {', '.join(STEPS)}")
    parser.add_argument("--fish", type=int, default=None,
                        help="Process only this fish ID. Default: all.")
    parser.add_argument(
        "--transform-source", choices=["stim", "chain", "globalopt"],
        default="chain",
        help="Which per-slice transforms to apply in the apply step. "
             "stim=STIM GlobalOpt (often unreliable), "
             "chain=sequential chaining (Option 1), "
             "globalopt=our least-squares optimizer (Option 2). "
             "Default: chain.",
    )
    args = parser.parse_args()

    if args.fish is not None:
        fish_ids = [args.fish]
    else:
        fish_ids = sorted(
            int(d) for d in os.listdir(os.path.join(SRC_DIR))
            if os.path.isdir(os.path.join(SRC_DIR, d)) and d.isdigit()
        )
    logging.info("Fish to process: %s", fish_ids)
    logging.info("Transform source for apply step: %s", args.transform_source)

    # Thresholds for deciding whether chain corrections look like real residuals.
    # If chain corrections are larger than this, STIM found noise not real signal
    # and globalopt won't help either.
    CHAIN_MAX_ROT_DEG = 10.0   # max absolute rotation across all slices
    CHAIN_MEAN_T_PX   = 50.0   # mean translation magnitude

    from_idx = STEPS.index(args.from_step)
    if from_idx <= STEPS.index("build"):
        step_build(fish_ids)
    if from_idx <= STEPS.index("stim"):
        step_stim(fish_ids)
    chain_stats = None
    if from_idx <= STEPS.index("chain"):
        chain_stats = step_chain(fish_ids)
    if from_idx <= STEPS.index("globalopt"):
        # Load stats from chain if we didn't just compute them
        if chain_stats is None:
            try:
                pairs, gnums = _load_pairwise(fish_ids[0])
                chain_stats = {'max_rot': CHAIN_MAX_ROT_DEG + 1}  # trigger check below
            except Exception:
                chain_stats = {'max_rot': CHAIN_MAX_ROT_DEG + 1}
        if (chain_stats.get('max_rot', 999) <= CHAIN_MAX_ROT_DEG and
                chain_stats.get('mean_t', 999) <= CHAIN_MEAN_T_PX):
            step_globalopt(fish_ids)
        else:
            logging.warning(
                "Skipping globalopt — chain corrections look like noise "
                "(max_rot=%.1f° > %.0f° or mean_t=%.1fpx > %.0fpx). "
                "STIM pairwise transforms are unreliable for this data.",
                chain_stats.get('max_rot', 0), CHAIN_MAX_ROT_DEG,
                chain_stats.get('mean_t', 0), CHAIN_MEAN_T_PX,
            )
    if from_idx <= STEPS.index("apply"):
        step_apply(fish_ids, source=args.transform_source)
    if from_idx <= STEPS.index("stack"):
        step_stack(fish_ids, source=args.transform_source)

    logging.info("Done. Output: %s", OUT_DIR)


if __name__ == "__main__":
    main()
