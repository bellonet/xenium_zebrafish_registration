"""
5_apply_registration.py  —  Apply the full registration pipeline to new data.

Applies the complete 3-layer registration stack to any additional data:

  Layer 1  (script 2)  Per-slice 2D rigid: DV rotation + elastix per slice.
  Layer 2  (script 3)  Per-slice 2D rigid correction on top of layer 1.
  Layer 3  (script 4)  3D rigid/affine cross-fish registration.

IMPORTANT — single-pass resampling:
  Applying transforms sequentially (resample 3 times) compounds interpolation
  errors. This script COMPOSES layers 1 and 2 analytically per slice into a
  single rigid transform, then applies a single 3D backward-mapping through
  the composed transform. Result: one resampling step total, minimum error.

  For POINT DATA (transcripts, centroids): sequential application is exact
  (coordinate math only, no interpolation at any step).

CONFIGURATION — edit this block before running:
"""

# ── USER CONFIGURATION ─────────────────────────────────────────────────────────

FISH           = 2
RUN            = 2            # Xenium run number (2, 4, or 5) — must match FISH
SCRIPT3_EXP    = 'dapi_blend'
SCRIPT4_EXP    = 'dapi_blend_rigid_affine_z10'
Z_SPACING_UM   = 10.0        # confirmed 10 µm z-spacing
REFERENCE_FISH = 1

APPLY_TO_TRANSCRIPTS = True
APPLY_TO_CELLS       = True
APPLY_TO_IMAGE       = False  # set to True + provide path to transform a custom TIFF
CUSTOM_IMAGE_PATH    = ''     # (Z, H, W) float32 TIFF in raw Xenium space
NN_INTERPOLATION     = False  # True for label/seg images (no colour mixing)

OUT_DIR = '../analysis/5_applied'

# Run → Xenium output folder mapping (keep in sync with script 2's RUN_FOLDERS)
RUN_FOLDERS: dict = {
    2: 'output-XETG00046__0038328__Region_1__20250717__075022',
    4: 'output-XETG00046__0043921__Region_1__20250620__084504',
    5: 'output-XETG00046__0044004__Region_1__20250620__084505',
}

# ── END CONFIGURATION ──────────────────────────────────────────────────────────

import glob as _glob, json, logging, math, os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import map_coordinates

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

DATA_DIR = '../data'
IN2_DIR  = '../analysis/2_registered'
IN3_DIR  = '../analysis/3_improved_registration'
IN4_DIR  = '../analysis/4_registered'

XENIUM_PX_UM     = 0.2125
XENIUM_PX_PER_UM = 1.0 / XENIUM_PX_UM

def _get_ds_factor() -> int:
    """Read ds_xy from script-4 transform.json so this script stays in sync."""
    p = os.path.join(IN4_DIR, SCRIPT4_EXP, str(REFERENCE_FISH), 'transform.json')
    if os.path.exists(p):
        with open(p) as fh:
            return int(json.load(fh).get('ds_xy', 1))
    return 1   # default: full resolution

DOWNSAMPLE_XY = _get_ds_factor()
DS_SPACING    = XENIUM_PX_UM * DOWNSAMPLE_XY   # µm/px in script-4 output space


# ══════════════════════════════════════════════════════════════════════════════
# LOAD TRANSFORMS
# ══════════════════════════════════════════════════════════════════════════════

def load_script2_meta(fish: int) -> Tuple[Dict, int, int]:
    with open(os.path.join(IN2_DIR, str(fish), 'transforms.json')) as fh:
        m = json.load(fh)
    return m['slices'], m['canvas_h'], m['canvas_w']


def load_script3_corrections(fish: int, exp: str) -> Dict[int, Optional[Dict]]:
    p = os.path.join(IN3_DIR, exp, str(fish), 'correction_transforms.json')
    if not os.path.exists(p):
        logging.warning(f'Script-3 corrections missing: {p}')
        return {}
    with open(p) as fh:
        data = json.load(fh)
    return {int(k): v for k, v in data.get('corrections', {}).items()}


def load_script4_transform(fish: int, exp: str) -> Dict:
    with open(os.path.join(IN4_DIR, exp, str(fish), 'transform.json')) as fh:
        return json.load(fh)


# ══════════════════════════════════════════════════════════════════════════════
# RIGID TRANSFORM MATH — 2D
# ══════════════════════════════════════════════════════════════════════════════
# A 2D rigid transform: x' = R(angle) * (x - center) + center + translation
# We represent it as (angle, effective_offset) where
#   x' = R * x + offset
#   offset = -R*center + center + translation
#
# Composition:  T2 ∘ T1:  x'' = R2*(R1*x + off1) + off2
#                               = (R2*R1)*x + (R2*off1 + off2)
# This is EXACT — no approximation.

def _R2(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def rigid2d_from_params(angle_deg: float, tx: float, ty: float,
                         center: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (R, offset) for x' = R @ x + offset."""
    a      = math.radians(angle_deg)
    R      = _R2(a)
    c      = np.array([center[0], center[1]], dtype=np.float64)
    offset = -R @ c + c + np.array([tx, ty], dtype=np.float64)
    return R, offset


def rigid2d_compose(R1, off1, R2, off2) -> Tuple[np.ndarray, np.ndarray]:
    """Compose T2 ∘ T1: apply T1 first, then T2.  Exact, no interpolation."""
    return R2 @ R1, R2 @ off1 + off2


def rigid2d_from_euler_file(tf_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse elastix EulerTransform .txt file → (R, offset)."""
    params: Dict = {}
    with open(tf_path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith('(') and line.endswith(')'):
                parts = line[1:-1].split()
                params[parts[0]] = parts[1:]
    tp   = [float(v) for v in params['TransformParameters']]
    corp = [float(v) for v in params['CenterOfRotationPoint']]
    # elastix convention: angle, tx, ty
    return rigid2d_from_params(math.degrees(tp[0]), tp[1], tp[2],
                                (corp[0], corp[1]))


# ══════════════════════════════════════════════════════════════════════════════
# PER-SLICE COMPOSED 2D TRANSFORM (layers 1 + 2)
# ══════════════════════════════════════════════════════════════════════════════

def build_per_slice_composed_transforms(
        fish: int, script3_exp: str,
        bbox_r0: float, bbox_c0: float) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Analytically compose layers 1 and 2 for every slice.

    Returns dict: gnum → (R, offset)  mapping raw pixel (x,y) → canvas pixel (x,y)
    where x' = R @ x + offset.  Exact — no approximation.
    """
    slices_meta, canvas_h, canvas_w = load_script2_meta(fish)
    corrections = load_script3_corrections(fish, script3_exp)
    fish_dir    = os.path.join(IN2_DIR, str(fish))

    composed: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    for gnum_str, sm in slices_meta.items():
        gnum = int(gnum_str)

        # ── Layer 1: bbox shift + DV rotation + elastix rigid ─────────────────
        # Step 1a: shift into canvas
        pad_l = sm.get('pad_left', 0); pad_t = sm.get('pad_top', 0)
        shift_offset = np.array([pad_l - bbox_c0, pad_t - bbox_r0], dtype=np.float64)
        R1, off1 = np.eye(2), shift_offset   # pure translation (no rotation yet)

        # Step 1b: DV rotation around canvas centre (canvas_h/w from global meta)
        dv_deg   = sm.get('dv_angle_deg', 0.0)
        if abs(dv_deg) >= 0.01:
            Rdv, odv = rigid2d_from_params(dv_deg, 0.0, 0.0,
                                            (canvas_w / 2.0, canvas_h / 2.0))
            R1, off1 = rigid2d_compose(R1, off1, Rdv, odv)

        # Step 1c: elastix rigid (per-slice .txt file)
        elastix_file = sm.get('elastix_file')
        if elastix_file:
            tf_path = os.path.join(fish_dir, elastix_file)
            if os.path.exists(tf_path):
                Re, oe = rigid2d_from_euler_file(tf_path)
                R1, off1 = rigid2d_compose(R1, off1, Re, oe)

        # ── Layer 2: script-3 correction ──────────────────────────────────────
        corr = corrections.get(gnum)
        if corr is not None:
            R2, off2 = rigid2d_from_params(
                corr.get('angle_deg', 0.0),
                corr.get('tx', 0.0), corr.get('ty', 0.0),
                tuple(corr.get('center', [canvas_w / 2.0, canvas_h / 2.0]))
            )
            R1, off1 = rigid2d_compose(R1, off1, R2, off2)

        composed[gnum] = (R1, off1)

    return composed


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-PASS IMAGE TRANSFORM
# ══════════════════════════════════════════════════════════════════════════════

def transform_image_single_pass(
        vol_raw: np.ndarray,
        fish: int,
        script3_exp: str = SCRIPT3_EXP,
        script4_exp: str = SCRIPT4_EXP,
        z_spacing_um: float = Z_SPACING_UM,
        bbox_r0: float = 0.0,
        bbox_c0: float = 0.0,
        canvas_h: int = None,
        canvas_w: int = None,
        nn: bool = NN_INTERPOLATION,
) -> np.ndarray:
    """Apply all 3 registration layers in a SINGLE backward-mapping pass.

    How it works
    ------------
    For each output voxel (z_out, y_out, x_out) in the script-4 downsampled
    registered space, we compute — via inverse transforms — where it came from
    in the original raw volume, then sample there once (map_coordinates).

    Inverse order:
      1. Invert layer 3 (3D rigid/affine) → position in script-2/3 canvas DS space
      2. Un-downsample XY → full-resolution canvas position
      3. Invert per-slice composed 2D rigid (layers 1+2) → position in raw volume

    Result: ONE interpolation, minimum error.

    Parameters
    ----------
    vol_raw  : (Z, H, W) float32 in raw Xenium pixel space (before any registration)
    canvas_h/w : script-2 canvas size (loaded automatically if None)

    Returns
    -------
    (Z_out, H_out, W_out) float32 in script-4 downsampled registered space
    """
    # ── Load transforms ───────────────────────────────────────────────────────
    slices_meta, ch, cw = load_script2_meta(fish)
    if canvas_h is None: canvas_h = ch
    if canvas_w is None: canvas_w = cw

    composed_2d = build_per_slice_composed_transforms(fish, script3_exp, bbox_r0, bbox_c0)
    tf4         = load_script4_transform(fish, script4_exp)

    # ── Determine output shape from script-4 reference output ────────────────
    ref_path = os.path.join(IN4_DIR, script4_exp, str(REFERENCE_FISH), 'c0.tif')
    if os.path.exists(ref_path):
        ref_vol   = tifffile.imread(ref_path)
        out_shape = ref_vol.shape   # (Z_out, H_out, W_out)
        del ref_vol
    else:
        out_shape = (vol_raw.shape[0],
                     canvas_h // DOWNSAMPLE_XY,
                     canvas_w // DOWNSAMPLE_XY)
    logging.info(f'  Output shape: {out_shape}')

    # ── Build inverse of layer 3 ──────────────────────────────────────────────
    elastix      = tf4.get('elastix')
    pre_rot      = tf4.get('pre_rotation_deg', 0.0)
    canvas_shift = np.array(tf4.get('canvas_shift', [0.0, 0.0, 0.0]))  # [dZ, dY, dX]
    mov_ds_shape = tf4.get('mov_ds_shape')                              # [Z, H, W] of DS driving vol

    def invert_layer3(coords_zyx: np.ndarray) -> np.ndarray:
        """coords_zyx: (3, N) in output DS space [z,y,x] → (3, N) in pre-rotation DS vol space."""
        c = coords_zyx.astype(np.float64).copy()  # [z, y, x] voxel indices

        # Undo elastix 3D transform
        # ITK physical space convention: phys = (x, y, z) = (col*DS, row*DS, slice*z_um)
        # c[0]=z(slice), c[1]=y(row), c[2]=x(col)  →  phys[0]=x, phys[1]=y, phys[2]=z
        if elastix is not None:
            tp      = elastix['TransformParameters']
            tf_type = elastix.get('Transform', ['EulerTransform'])[0]
            corp    = np.array(elastix.get('CenterOfRotationPoint', [0.0, 0.0, 0.0]))
            # corp is in ITK physical (x, y, z) order in µm

            # voxel [z,y,x] → ITK physical [x,y,z]
            phys = np.stack([c[2] * DS_SPACING,       # phys_x (col → µm)
                              c[1] * DS_SPACING,        # phys_y (row → µm)
                              c[0] * z_spacing_um],     # phys_z (slice → µm)
                             axis=0)                    # (3, N)

            if 'Euler' in tf_type:
                rx, ry, rz = tp[0], tp[1], tp[2]
                tx, ty, tz = tp[3], tp[4], tp[5]
                T = np.array([tx, ty, tz])
                cx_a, sx_a = math.cos(rx), math.sin(rx)
                cy_a, sy_a = math.cos(ry), math.sin(ry)
                cz_a, sz_a = math.cos(rz), math.sin(rz)
                R = np.array([
                    [cy_a*cz_a, cz_a*sx_a*sy_a - cx_a*sz_a, cx_a*cz_a*sy_a + sx_a*sz_a],
                    [cy_a*sz_a, cx_a*cz_a + sx_a*sy_a*sz_a, cx_a*sy_a*sz_a - cz_a*sx_a],
                    [-sy_a,     cy_a*sx_a,                   cx_a*cy_a                  ],
                ])
                # Forward:  phys_out = R @ (phys_in - corp) + corp + T
                # Inverse:  phys_in  = R^T @ (phys_out - corp - T) + corp
                phys_in = R.T @ (phys - corp[:, None] - T[:, None]) + corp[:, None]

            elif 'Affine' in tf_type:
                M = np.array(tp[:9]).reshape(3, 3)
                T = np.array(tp[9:12])
                # Forward:  phys_out = M @ (phys_in - corp) + corp + T
                # Inverse:  phys_in  = M^{-1} @ (phys_out - corp - T) + corp
                phys_in = np.linalg.solve(M, phys - corp[:, None] - T[:, None]) + corp[:, None]

            else:
                phys_in = phys   # unknown transform type — pass through

            # ITK physical [x,y,z] → voxel [z,y,x]
            c = np.stack([phys_in[2] / z_spacing_um,   # z (slice)
                           phys_in[1] / DS_SPACING,      # y (row)
                           phys_in[0] / DS_SPACING],     # x (col)
                          axis=0)

        # Undo canvas shift (shift[0]=dZ, [1]=dY, [2]=dX in DS voxels)
        c[0] -= canvas_shift[0]
        c[1] -= canvas_shift[1]
        c[2] -= canvas_shift[2]

        # Undo pre-rotation (Z-axis rotation applied to DS vol before place_in_canvas)
        # Rotation centre = centre of the DS driving volume (before rotation)
        if abs(pre_rot) >= 0.01:
            if mov_ds_shape is not None:
                cy_r = mov_ds_shape[1] / 2.0   # row centre of original DS vol
                cx_r = mov_ds_shape[2] / 2.0   # col centre of original DS vol
            else:
                # fallback: use canvas centre in DS
                cy_r = out_shape[1] / 2.0
                cx_r = out_shape[2] / 2.0
            a   = math.radians(-pre_rot)         # inverse rotation angle
            cos_a, sin_a = math.cos(a), math.sin(a)
            dy = c[1] - cy_r
            dx = c[2] - cx_r
            c[1] = cy_r + cos_a * dy - sin_a * dx
            c[2] = cx_r + sin_a * dy + cos_a * dx

        return c

    # ── Create output coordinate grid ─────────────────────────────────────────
    Zo, Ho, Wo = out_shape
    z_g, y_g, x_g = np.meshgrid(
        np.arange(Zo, dtype=np.float64),
        np.arange(Ho, dtype=np.float64),
        np.arange(Wo, dtype=np.float64),
        indexing='ij'
    )
    coords = np.stack([z_g.ravel(), y_g.ravel(), x_g.ravel()], axis=0)  # (3, N)

    # ── Apply inverse layer 3 ─────────────────────────────────────────────────
    logging.info('  Applying inverse layer 3 (cross-fish 3D transform) ...')
    coords = invert_layer3(coords)   # still in DS canvas space

    # ── Convert DS → full-resolution canvas space ─────────────────────────────
    coords[1] *= DOWNSAMPLE_XY   # Y
    coords[2] *= DOWNSAMPLE_XY   # X
    # Z stays the same (slice index)

    # ── Apply inverse per-slice composed 2D rigid (layers 1 + 2) ─────────────
    logging.info('  Applying inverse per-slice composed 2D rigid (layers 1+2) ...')
    z_coords = coords[0]
    x_raw    = np.zeros(coords.shape[1], dtype=np.float64)
    y_raw    = np.zeros(coords.shape[1], dtype=np.float64)

    # Group by nearest slice for efficiency.
    # invert_layer3 returns vol_ds indices (0-indexed); add 1 to get gnums (1-indexed)
    # so the lookup into composed_2d (keyed by gnum) works correctly.
    gnum_arr = np.round(z_coords).astype(int) + 1

    for gnum, (R, off) in composed_2d.items():
        mask = (gnum_arr == gnum)
        if not mask.any():
            continue
        xy_canvas = np.stack([coords[2][mask], coords[1][mask]], axis=0)  # (2, N_slice)
        # Inverse: x_raw = R^T @ (x_canvas - off)
        xy_raw         = R.T @ (xy_canvas - off[:, None])
        x_raw[mask]    = xy_raw[0]
        y_raw[mask]    = xy_raw[1]

    # Slices not in composed_2d: pass through unchanged
    all_gnums = set(gnum_arr.tolist())
    missing   = all_gnums - set(composed_2d.keys())
    for gnum in missing:
        mask          = (gnum_arr == gnum)
        x_raw[mask]   = coords[2][mask]
        y_raw[mask]   = coords[1][mask]

    # Final raw-space coordinates: (Z, Y, X) in raw volume voxels.
    # z_coords is already the 0-indexed vol_ds Z (= raw vol Z), so use directly.
    z_raw = z_coords

    raw_coords = np.stack([z_raw, y_raw, x_raw], axis=0)

    # ── Single interpolation step ─────────────────────────────────────────────
    logging.info('  Single interpolation step (map_coordinates) ...')
    order = 0 if nn else 1
    result = map_coordinates(vol_raw.astype(np.float32), raw_coords,
                              order=order, mode='constant', cval=0.0)
    return result.reshape(out_shape).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# POINT DATA TRANSFORM (exact, no interpolation at any step)
# ══════════════════════════════════════════════════════════════════════════════

def transform_points(df: pd.DataFrame,
                     fish: int,
                     x_col: str = 'x_location',
                     y_col: str = 'y_location',
                     z_col: str = 'z_location',
                     script3_exp: str = SCRIPT3_EXP,
                     script4_exp: str = SCRIPT4_EXP,
                     bbox_r0: float = 0.0,
                     bbox_c0: float = 0.0) -> pd.DataFrame:
    """Apply all 3 registration layers to point coordinates.

    For points: every step is pure coordinate math — no interpolation ever.
    Layers 1+2 are analytically composed before applying (one matrix multiply).

    Input columns: x_col, y_col in µm (Xenium space); z_col in µm.
    Added columns: x_reg_px, y_reg_px  (canvas pixels after layers 1+2)
                   x_reg_ds, y_reg_ds  (downsampled canvas after layer 3)
                   z_reg               (registered Z slice index)
    """
    df = df.copy()
    composed_2d = build_per_slice_composed_transforms(fish, script3_exp, bbox_r0, bbox_c0)
    tf4         = load_script4_transform(fish, script4_exp)

    # µm → pixels
    x_px = df[x_col].values * XENIUM_PX_PER_UM
    y_px = df[y_col].values * XENIUM_PX_PER_UM

    # Map z_µm → gnum (1-indexed)
    gnum_arr = np.round(df[z_col].values / Z_SPACING_UM).astype(int) + 1

    x_reg = np.zeros(len(df), dtype=np.float64)
    y_reg = np.zeros(len(df), dtype=np.float64)

    for gnum, (R, off) in composed_2d.items():
        mask = (gnum_arr == gnum)
        if not mask.any():
            continue
        xy_raw    = np.stack([x_px[mask], y_px[mask]], axis=0)
        xy_canvas = R @ xy_raw + off[:, None]
        x_reg[mask] = xy_canvas[0]
        y_reg[mask] = xy_canvas[1]

    df['x_reg_px'] = x_reg
    df['y_reg_px'] = y_reg

    # Layer 3: apply 3D rigid/affine in DS space
    x_ds = x_reg / DOWNSAMPLE_XY   # column DS coords
    y_ds = y_reg / DOWNSAMPLE_XY   # row DS coords
    # gnum_arr is 1-indexed; subtract 1 to get 0-indexed DS vol index,
    # matching how place_in_canvas indexes the volume (vol_ds[0] = gnum 1).
    z_ds = (gnum_arr - 1).astype(np.float64)

    elastix      = tf4.get('elastix')
    canvas_shift = np.array(tf4.get('canvas_shift', [0.0, 0.0, 0.0]))
    pre_rot_deg  = tf4.get('pre_rotation_deg', 0.0)
    mov_ds_shape = tf4.get('mov_ds_shape')

    # Apply pre-rotation (forward: rotate DS coords about DS vol centre before shift)
    if abs(pre_rot_deg) >= 0.01:
        if mov_ds_shape is not None:
            cy_r = mov_ds_shape[1] / 2.0
            cx_r = mov_ds_shape[2] / 2.0
        else:
            cy_r = 0.0; cx_r = 0.0   # no fallback available for points
        a = math.radians(pre_rot_deg)
        cos_a, sin_a = math.cos(a), math.sin(a)
        dy = y_ds - cy_r;  dx = x_ds - cx_r
        x_ds = cx_r + cos_a * dx - sin_a * dy
        y_ds = cy_r + sin_a * dx + cos_a * dy

    # Apply canvas shift
    x_ds += canvas_shift[2]; y_ds += canvas_shift[1]; z_ds += canvas_shift[0]

    if elastix is not None:
        tp      = elastix['TransformParameters']
        tf_type = elastix.get('Transform', ['EulerTransform'])[0]
        corp    = np.array(elastix.get('CenterOfRotationPoint', [0.0, 0.0, 0.0]))
        # ITK physical space: phys[0]=x(col), phys[1]=y(row), phys[2]=z(slice)
        phys = np.stack([x_ds * DS_SPACING,
                          y_ds * DS_SPACING,
                          z_ds * Z_SPACING_UM], axis=0)   # (3, N) — order correct
        if 'Euler' in tf_type:
            rx, ry, rz = tp[0], tp[1], tp[2]
            tx, ty, tz = tp[3], tp[4], tp[5]
            T3 = np.array([tx, ty, tz])
            cx_a, sx_a = math.cos(rx), math.sin(rx)
            cy_a, sy_a = math.cos(ry), math.sin(ry)
            cz_a, sz_a = math.cos(rz), math.sin(rz)
            R3 = np.array([
                [cy_a*cz_a, cz_a*sx_a*sy_a - cx_a*sz_a, cx_a*cz_a*sy_a + sx_a*sz_a],
                [cy_a*sz_a, cx_a*cz_a + sx_a*sy_a*sz_a, cx_a*sy_a*sz_a - cz_a*sx_a],
                [-sy_a,     cy_a*sx_a,                   cx_a*cy_a                  ],
            ])
            # Forward: phys_out = R3 @ (phys_in - corp) + corp + T3
            phys_out = R3 @ (phys - corp[:, None]) + corp[:, None] + T3[:, None]
        elif 'Affine' in tf_type:
            M3 = np.array(tp[:9]).reshape(3, 3)
            T3 = np.array(tp[9:12])
            # Forward: phys_out = M3 @ (phys_in - corp) + corp + T3
            phys_out = M3 @ (phys - corp[:, None]) + corp[:, None] + T3[:, None]
        else:
            phys_out = phys
        x_ds = phys_out[0] / DS_SPACING
        y_ds = phys_out[1] / DS_SPACING
        z_ds = phys_out[2] / Z_SPACING_UM

    df['x_reg_ds'] = x_ds
    df['y_reg_ds'] = y_ds
    df['z_reg']    = z_ds
    return df


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _get_bbox(fish: int) -> Tuple[float, float]:
    """Return (bbox_row0, bbox_col0) for this fish in the configured RUN."""
    csvs = _glob.glob('../analysis/fish_bbox_summary*.csv')
    if not csvs:
        logging.warning('No fish_bbox_summary CSV — bbox offset set to 0.')
        return 0.0, 0.0
    df   = pd.read_csv(csvs[0])
    rows = df[(df['fish_name'] == fish) & (df['run'] == RUN)]
    if len(rows) == 0:
        # fallback: any run for this fish
        rows = df[df['fish_name'] == fish]
    if len(rows) == 0:
        logging.warning(f'No bbox entry for fish={fish} run={RUN}')
        return 0.0, 0.0
    return float(rows['bbox_global_min_row'].min()), float(rows['bbox_global_min_col'].min())


def _find_run_folder() -> Optional[str]:
    """Return path to the Xenium output folder for the configured RUN."""
    folder = RUN_FOLDERS.get(RUN)
    if folder is None:
        logging.error(f'RUN={RUN} not in RUN_FOLDERS — add it to the config block.')
        return None
    rd = os.path.join(DATA_DIR, folder)
    if not os.path.isdir(rd):
        logging.error(f'Run folder not found: {rd}')
        return None
    return rd


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    logging.info(f'Fish {FISH}  |  script3={SCRIPT3_EXP}  |  script4={SCRIPT4_EXP}')
    bbox_r0, bbox_c0 = _get_bbox(FISH)

    if APPLY_TO_TRANSCRIPTS:
        rd = _find_run_folder()
        if rd:
            logging.info(f'Loading transcripts from {rd} ...')
            tx = pd.read_parquet(
                os.path.join(rd, 'transcripts.parquet'),
                columns=['transcript_id', 'cell_id', 'feature_name',
                         'x_location', 'y_location', 'z_location', 'qv', 'is_gene']
            )
            tx = tx[tx['is_gene']].reset_index(drop=True)
            logging.info(f'  {len(tx):,} gene transcripts — transforming coordinates ...')
            tx_reg = transform_points(tx, FISH,
                                       x_col='x_location', y_col='y_location',
                                       z_col='z_location',
                                       bbox_r0=bbox_r0, bbox_c0=bbox_c0)
            out = os.path.join(OUT_DIR, f'transcripts_registered_fish{FISH}.parquet')
            tx_reg.to_parquet(out, index=False)
            logging.info(f'  Saved → {out}')

    if APPLY_TO_CELLS:
        rd = _find_run_folder()
        if rd:
            logging.info('Loading cells.parquet ...')
            cells = pd.read_parquet(os.path.join(rd, 'cells.parquet'),
                                    columns=['cell_id', 'x_centroid', 'y_centroid'])
            cells['z_location'] = 0.0
            cells_reg = transform_points(cells, FISH,
                                          x_col='x_centroid', y_col='y_centroid',
                                          z_col='z_location',
                                          bbox_r0=bbox_r0, bbox_c0=bbox_c0)
            out = os.path.join(OUT_DIR, f'cells_registered_fish{FISH}.parquet')
            cells_reg.to_parquet(out, index=False)
            logging.info(f'  Saved → {out}')

    if APPLY_TO_IMAGE and CUSTOM_IMAGE_PATH:
        logging.info(f'Loading image: {CUSTOM_IMAGE_PATH}')
        vol = tifffile.imread(CUSTOM_IMAGE_PATH).astype(np.float32)
        if vol.ndim == 2:
            vol = vol[np.newaxis]
        logging.info(f'  Shape: {vol.shape} — applying single-pass transform ...')
        vol_reg = transform_image_single_pass(vol, FISH,
                                               bbox_r0=bbox_r0, bbox_c0=bbox_c0,
                                               nn=NN_INTERPOLATION)
        name    = os.path.splitext(os.path.basename(CUSTOM_IMAGE_PATH))[0]
        out     = os.path.join(OUT_DIR, f'{name}_registered_fish{FISH}.tif')
        tifffile.imwrite(out, vol_reg)
        logging.info(f'  Saved → {out}')

    logging.info('Done.')


if __name__ == '__main__':
    main()
