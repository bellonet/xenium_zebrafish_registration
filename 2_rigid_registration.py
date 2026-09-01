"""
2_rigid_registration.py

Rigid 2D registration of individual fish slices (reads directly from script 1 output).

Input:  analysis/1_detection/individual_fish_2d/{fish}/c{ch}/{global_num}_{run}_{tile}.tif
Output: analysis/2_registered/{fish}/c{ch}/{global_num}.tif
        analysis/2_registered/{fish}/rigid_3d_c{ch}.tif
        analysis/2_registered/{fish}/seg_cells/{gnum}.tif   (per-cell RGB, hue from type)
        analysis/2_registered/{fish}/seg_nuclei/{gnum}.tif  (per-nucleus RGB, hue from type)
        analysis/2_registered/{fish}/tissue_map/{gnum}.tif  (uniform flat colour per type)
        analysis/2_registered/{fish}/per_gene/{gene}.tif    (3D gene density stack)

Strategy: DAPI drives registration; same transform applied to all channels.
Propagates outward from the middle reference slice so drift accumulates
over at most n/2 steps.

Two-phase: Phase 1 registers all DAPI and keeps results in memory;
Phase 2 applies saved transforms to remaining channels one slice at a time.

Gaps: zero-filled frames written for every integer up to the last slice
so ImageJ frame N = global_num N.

Usage:
  python 2_rigid_registration.py                              # all steps, all fish
  python 2_rigid_registration.py --fish 1                    # all steps, fish 1
  python 2_rigid_registration.py --steps segmentation stack  # only those two steps
  python 2_rigid_registration.py --steps per_gene --fish 2   # per_gene, fish 2 only
"""

import os, gc, glob, json, re, argparse, logging, math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile
import itk
from scipy.ndimage import rotate as nd_rotate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SLICE_RE = re.compile(r'^(\d+)_')

SRC_DIR      = '../analysis/1_detection/individual_fish_2d'
OUT_DIR      = '../analysis/2_registered'
DATA_DIR     = '../data'
SUMMARY_GLOB = '../analysis/fish_bbox_summary_tagged_*.csv'
NUM_CHANNELS = 4
DAPI_CHANNEL = 0

STEPS = ['register', 'segmentation', 'stack', 'per_gene']
TRANSFORMS_SUBDIR  = 'transforms'   # sub-dir under OUT_DIR/{fish}/ for saved transforms
ANNOT_CSV          = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'leiden10annots.csv')
# Zarr mask indices: 0 = nucleus (DAPI), 1 = cell boundary
ZARR_MASK_IDX = {'cells': '1', 'nuclei': '0'}

# ── elastix parameters ────────────────────────────────────────────────────────
PAD_PX        = 64
N_RESOLUTIONS = 3
MAX_ITER      = 256
NUM_SAMPLES   = 6_000
NUM_THREADS   = 8
PYRAMID_SCHED = ['8', '8', '4', '4', '2', '2']   # 2 entries per resolution level

# ── DV orientation ────────────────────────────────────────────────────────────
DORSAL_GENES  = ['pax6b', 'scrt2', 'pax6a', 'isl1', 'rfx4', 'sox3']
VENTRAL_GENES = ['etv2', 'angpt1', 'foxf1', 'slc4a2a', 'cxcl12b', 'foxa3']
XENIUM_PX_PER_UM = 4.705882
MIN_ANGLE_DEG = 1.0
RUN_FOLDERS   = {
    '2': 'output-XETG00046__0038328__Region_1__20250717__075022',
    '4': 'output-XETG00046__0043921__Region_1__20250620__084504',
    '5': 'output-XETG00046__0044004__Region_1__20250620__084505',
}


# ── helpers ───────────────────────────────────────────────────────────────────

def get_slices(fish: int, ch: int = DAPI_CHANNEL) -> List[Tuple[int, str]]:
    """Sorted (global_num, path) for one fish and channel."""
    files = glob.glob(os.path.join(SRC_DIR, str(fish), f'c{ch}', '*.tif'))
    result = []
    for f in files:
        m = SLICE_RE.match(os.path.basename(f))
        if m:
            result.append((int(m.group(1)), f))
    return sorted(result)


def load_arr(path: str, h: int, w: int, normalize: bool = False) -> np.ndarray:
    """Load TIF → float32 → padded to (h, w). Optionally normalise to [0,1]."""
    arr = tifffile.imread(path).astype(np.float32)
    if arr.ndim != 2:
        arr = arr[0]
    if normalize:
        mx = float(np.max(arr))
        if mx > 0:
            arr /= mx
    ah, aw = arr.shape
    out_h, out_w = max(h, ah), max(w, aw)
    pt = (out_h - ah) // 2
    pb = out_h - ah - pt
    pl = (out_w - aw) // 2
    pr = out_w - aw - pl
    return np.ascontiguousarray(np.pad(arr, ((pt, pb), (pl, pr))))


def _set_itk_props(img: itk.Image) -> None:
    img.SetSpacing((1.0, 1.0))
    img.SetOrigin((0.0, 0.0))


def build_parameter_object() -> itk.ParameterObject:
    po = itk.ParameterObject.New()
    pm = po.GetDefaultParameterMap('rigid', N_RESOLUTIONS)
    pm['Transform']  = ['EulerTransform']
    pm['Metric']     = ['AdvancedMattesMutualInformation']
    pm['Optimizer']  = ['AdaptiveStochasticGradientDescent']
    pm['AutomaticParameterEstimation']           = ['true']
    pm['AutomaticTransformInitialization']       = ['true']
    pm['AutomaticTransformInitializationMethod'] = ['GeometricalCenter']
    pm['NumberOfThreads']            = [str(NUM_THREADS)]
    pm['ImageSampler']               = ['RandomCoordinate']
    pm['NumberOfSpatialSamples']     = [str(NUM_SAMPLES)]
    pm['MaximumNumberOfSamplingAttempts'] = ['200']
    pm['NewSamplesEveryIteration']   = ['true']
    pm['ImagePyramidSchedule']       = PYRAMID_SCHED
    pm['MaximumNumberOfIterations']  = [str(MAX_ITER)] * N_RESOLUTIONS
    pm['BSplineInterpolationOrder']      = ['1']
    pm['FinalBSplineInterpolationOrder'] = ['1']
    pm['ResultImagePixelType']       = ['float']
    po.AddParameterMap(pm)
    return po


def register_pair(fixed_np: np.ndarray, moving_np: np.ndarray,
                  param_obj: itk.ParameterObject) -> Tuple[np.ndarray, object]:
    """Register moving to fixed. Returns (registered_float32, transform_params)."""
    fixed_itk  = itk.GetImageFromArray(fixed_np);  _set_itk_props(fixed_itk)
    moving_itk = itk.GetImageFromArray(moving_np); _set_itk_props(moving_itk)
    result, tparams = itk.elastix_registration_method(
        fixed_itk, moving_itk,
        parameter_object=param_obj,
        log_to_console=False,
        log_to_file=False,
    )
    return itk.GetArrayFromImage(result).astype(np.float32), tparams


def apply_transform_to(moving_np: np.ndarray, tparams: object) -> np.ndarray:
    """Apply elastix transform with linear interpolation (fluorescence channels)."""
    moving_itk = itk.GetImageFromArray(moving_np); _set_itk_props(moving_itk)
    result = itk.transformix_filter(moving_itk, tparams)
    return itk.GetArrayFromImage(result).astype(np.float32)


def apply_transform_nn(moving_np: np.ndarray, tparams: object) -> np.ndarray:
    """Apply elastix transform with nearest-neighbour interpolation (label/seg images)."""
    po = itk.ParameterObject.New()
    for i in range(tparams.GetNumberOfParameterMaps()):
        pm = tparams.GetParameterMap(i)
        pm['ResampleInterpolator'] = ['FinalNearestNeighborInterpolator']
        po.AddParameterMap(pm)
    moving_itk = itk.GetImageFromArray(moving_np.astype(np.float32)); _set_itk_props(moving_itk)
    result = itk.transformix_filter(moving_itk, po)
    return itk.GetArrayFromImage(result).astype(np.float32)


# ── DV helpers ────────────────────────────────────────────────────────────────

def load_dv_transcripts() -> Dict[str, pd.DataFrame]:
    dv_genes = set(DORSAL_GENES + VENTRAL_GENES)
    result = {}
    for run, folder in RUN_FOLDERS.items():
        path = os.path.join(DATA_DIR, folder, 'transcripts.parquet')
        df = pd.read_parquet(path, columns=['feature_name', 'x_location', 'y_location'])
        result[run] = df[df['feature_name'].isin(dv_genes)].reset_index(drop=True)
        logging.info(f"  Run {run}: {len(result[run])} DV transcripts")
    return result


def dv_rotation_angle(fish: int, gnum: int,
                       summary_df: pd.DataFrame,
                       transcripts: Dict[str, pd.DataFrame]) -> Optional[float]:
    """CCW degrees to rotate so dorsal points up. None if insufficient data."""
    rows = summary_df[(summary_df['fish_name'] == fish) &
                      (summary_df['global_slice_num'] == gnum)]
    if len(rows) == 0:
        return None
    row = rows.iloc[0]
    tx = transcripts.get(str(int(row['run'])))
    if tx is None:
        return None
    S = XENIUM_PX_PER_UM
    local = tx[(tx['y_location'] >= row['bbox_global_min_row'] / S) &
               (tx['y_location'] <= row['bbox_global_max_row'] / S) &
               (tx['x_location'] >= row['bbox_global_min_col'] / S) &
               (tx['x_location'] <= row['bbox_global_max_col'] / S)]
    dorsal  = local[local['feature_name'].isin(DORSAL_GENES)]
    ventral = local[local['feature_name'].isin(VENTRAL_GENES)]
    if len(dorsal) < 5 or len(ventral) < 5:
        return None
    dr = float(dorsal['y_location'].mean() - ventral['y_location'].mean())
    dc = float(dorsal['x_location'].mean() - ventral['x_location'].mean())
    angle = 90.0 - math.degrees(math.atan2(-dr, dc))
    return ((angle + 180) % 360) - 180


def rotate_arr(arr: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate arr CCW by angle_deg on its current canvas (no reshape)."""
    if abs(abs(angle_deg) - 180.0) < 0.5:
        return np.rot90(arr, 2).astype(np.float32)
    return nd_rotate(arr.astype(np.float32), angle_deg, reshape=False, cval=0.0, order=1)


def _c0_frame_range(fish: int) -> Tuple[int, int]:
    """Return (first, last) global_num from the DAPI (c0) output dir for one fish.
    Used to ensure segmentation and per_gene stacks have the same frame count as DAPI.
    Falls back to (1, 1) if c0 is missing (register step not yet run).
    """
    files = sorted(
        glob.glob(os.path.join(OUT_DIR, str(fish), 'c0', '*.tif')),
        key=lambda f: int(os.path.splitext(os.path.basename(f))[0]),
    )
    if not files:
        return 1, 1
    return (int(os.path.splitext(os.path.basename(files[0]))[0]),
            int(os.path.splitext(os.path.basename(files[-1]))[0]))


# ── step: register ────────────────────────────────────────────────────────────

def step_register(fish_ids: List[int]) -> None:
    param_obj = build_parameter_object()

    # Load DV data once for all fish
    summary_csvs = glob.glob(SUMMARY_GLOB)
    if not summary_csvs:
        raise FileNotFoundError(f"No summary CSV matching {SUMMARY_GLOB}")
    summary_df = pd.read_csv(summary_csvs[0])
    logging.info("Loading DV transcripts")
    transcripts = load_dv_transcripts()

    for fish in fish_ids:
        logging.info(f"=== Fish {fish} ===")

        dapi_slices = get_slices(fish, DAPI_CHANNEL)
        if len(dapi_slices) < 2:
            logging.warning(f"Fish {fish}: fewer than 2 DAPI slices, skipping")
            continue

        all_gnums  = [g for g, _ in dapi_slices]
        dapi_paths = {g: p for g, p in dapi_slices}
        first, last = all_gnums[0], all_gnums[-1]
        gnum_set = set(all_gnums)
        n_gaps = last - first + 1 - len(all_gnums)
        logging.info(f"  {len(all_gnums)} input slices, range {first}–{last}, {n_gaps} gap(s)")

        # Compute DV angles for all slices
        dv_angles: Dict[int, float] = {}
        for g in all_gnums:
            angle = dv_rotation_angle(fish, g, summary_df, transcripts)
            if angle is not None and abs(angle) >= MIN_ANGLE_DEG:
                dv_angles[g] = angle

        # Canvas: max rotated dimensions + padding
        # (rotation expands bbox, so we account for the DV angle of each slice)
        orig_dims: Dict[int, Tuple[int, int]] = {}   # gnum → (h, w) before rotation/padding
        max_h, max_w = 0, 0
        for g, p in dapi_slices:
            arr = tifffile.imread(p)
            h, w = arr.shape[-2], arr.shape[-1]
            orig_dims[g] = (h, w)
            a = math.radians(abs(dv_angles.get(g, 0.0)))
            rh = int(math.ceil(h * abs(math.cos(a)) + w * abs(math.sin(a))))
            rw = int(math.ceil(h * abs(math.sin(a)) + w * abs(math.cos(a))))
            max_h = max(max_h, rh)
            max_w = max(max_w, rw)
        max_h += 2 * PAD_PX
        max_w += 2 * PAD_PX
        zero_frame = np.zeros((max_h, max_w), dtype=np.float32)
        logging.info(f"  canvas: {max_h}×{max_w}  ({len(dv_angles)} slices with DV rotation)")

        # ── Transform bookkeeping ────────────────────────────────────────────
        tf_dir = os.path.join(OUT_DIR, str(fish), TRANSFORMS_SUBDIR)
        os.makedirs(tf_dir, exist_ok=True)
        # transforms.json accumulates per-slice metadata; written at end of fish.
        # pad_top/left: how load_arr centres the slice on the canvas (matches its formula).
        transforms_meta: Dict = {'canvas_h': max_h, 'canvas_w': max_w, 'slices': {}}

        def _meta_entry(gnum: int, elastix_file: Optional[str]) -> Dict:
            oh, ow = orig_dims[gnum]
            return {
                'orig_h':       oh,
                'orig_w':       ow,
                'pad_top':      (max_h - oh) // 2,
                'pad_left':     (max_w - ow) // 2,
                'dv_angle_deg': dv_angles.get(gnum, 0.0),
                'elastix_file': elastix_file,
            }

        # ── Phase 1: DAPI registration ────────────────────────────────────────
        # registered_dapi[gnum] = aligned float32 array (kept in memory as fixed for neighbours)
        # transforms[gnum]      = elastix transform params (None for reference slice)
        registered_dapi: Dict[int, np.ndarray] = {}
        transforms: Dict[int, Optional[object]] = {}

        ref_gnum = all_gnums[len(all_gnums) // 2]
        logging.info(f"  reference slice: {ref_gnum}")
        ref_arr = load_arr(dapi_paths[ref_gnum], max_h, max_w, normalize=True)
        if ref_gnum in dv_angles:
            ref_arr = rotate_arr(ref_arr, dv_angles[ref_gnum])
        registered_dapi[ref_gnum] = ref_arr
        transforms[ref_gnum] = None  # no elastix correction for reference
        transforms_meta['slices'][str(ref_gnum)] = _meta_entry(ref_gnum, None)

        def find_nearest_registered(target: int) -> int:
            direction = 1 if target < ref_gnum else -1
            k = target + direction
            while first <= k <= last:
                if k in registered_dapi:
                    return k
                k += direction
            return ref_gnum

        # Process outward from ref.
        # Keep a rolling window of registered DAPI frames — once a slice is more
        # than _DAPI_WINDOW steps behind the frontier on its side it can never be
        # used as a fixed image again (find_nearest_registered always returns the
        # closest registered frame, and gaps in this data are at most 2–3 slices).
        _DAPI_WINDOW = 5   # keep this many frames on each side of the frontier
        step = 1
        while (ref_gnum - step) >= first or (ref_gnum + step) <= last:
            for target in (ref_gnum - step, ref_gnum + step):
                if not (first <= target <= last):
                    continue
                if target not in gnum_set:
                    continue   # gap — no input file
                if target in registered_dapi:
                    continue

                preferred = target + 1 if target < ref_gnum else target - 1
                neighbor  = (preferred
                             if preferred in registered_dapi
                             else find_nearest_registered(target))

                logging.info(f"  slice {target:>3d}  ←  neighbor {neighbor:>3d}")
                moving_arr = load_arr(dapi_paths[target], max_h, max_w, normalize=True)
                if target in dv_angles:
                    moving_arr = rotate_arr(moving_arr, dv_angles[target])
                result, tparams = register_pair(
                    registered_dapi[neighbor], moving_arr, param_obj
                )
                del moving_arr
                registered_dapi[target] = result
                transforms[target] = tparams
                # Save elastix transform to disk so script 3 can reuse it.
                tf_path = os.path.join(tf_dir, f'{target}.txt')
                tparams.WriteParameterFile(tf_path)
                transforms_meta['slices'][str(target)] = _meta_entry(
                    target, os.path.join(TRANSFORMS_SUBDIR, f'{target}.txt')
                )

            # Evict frames outside the rolling window on each side.
            # Keep the _DAPI_WINDOW most recent frames near each frontier;
            # evict old frames that have drifted back toward the reference.
            evict_left  = ref_gnum - step + _DAPI_WINDOW   # left-side frames >= this are old
            evict_right = ref_gnum + step - _DAPI_WINDOW   # right-side frames <= this are old
            for k in list(registered_dapi):
                if k == ref_gnum:
                    continue
                if k < ref_gnum and k >= evict_left:
                    del registered_dapi[k]
                elif k > ref_gnum and k <= evict_right:
                    del registered_dapi[k]

            step += 1

        n_reg   = len(transforms)   # transforms has one entry per registered slice
        n_wrote = last - first + 1
        logging.info(f"  DAPI: registered {n_reg} slices, {n_wrote - n_reg} gap(s)")

        # Write transforms.json for this fish.
        meta_path = os.path.join(OUT_DIR, str(fish), 'transforms.json')
        with open(meta_path, 'w') as fh:
            json.dump(transforms_meta, fh, indent=2)
        logging.info(f"  transforms saved → {meta_path}")

        # Free registered DAPI arrays — Phase 2 only needs the transforms dict.
        del registered_dapi; gc.collect()

        # ── Phase 2: apply transforms to all channels with original intensities ─
        # DAPI (ch0) is included here so outputs have raw (non-normalised) values.
        # Frames 1..(first-1) are written as zeros so ImageJ frame N = global_num N.
        for ch in range(NUM_CHANNELS):
            ch_slices = get_slices(fish, ch)
            ch_paths  = {g: p for g, p in ch_slices}
            out_ch_dir = os.path.join(OUT_DIR, str(fish), f'c{ch}')
            os.makedirs(out_ch_dir, exist_ok=True)

            for gnum in range(1, last + 1):
                out_path = os.path.join(out_ch_dir, f'{gnum}.tif')
                if gnum not in ch_paths:
                    tifffile.imwrite(out_path, zero_frame, photometric='minisblack')
                else:
                    arr = load_arr(ch_paths[gnum], max_h, max_w)
                    if gnum in dv_angles:
                        arr = rotate_arr(arr, dv_angles[gnum])
                    if transforms.get(gnum) is not None:
                        arr = apply_transform_to(arr, transforms[gnum])
                    tifffile.imwrite(out_path, arr, photometric='minisblack')

            logging.info(f"  c{ch}: done")

        del transforms, zero_frame; gc.collect()


# ── segmentation helpers ──────────────────────────────────────────────────────

def _cell_type_hues(cell_types: List[str]) -> Dict[str, float]:
    """Assign evenly-spaced hues (0–1) to cell types, sorted alphabetically for stability."""
    n = len(cell_types)
    return {ct: i / n for i, ct in enumerate(sorted(cell_types))}


def _cell_rgb(label_id: int, hue: float) -> Tuple[int, int, int]:
    """Stable per-cell RGB: same hue as its cell type, pseudo-random S and V.
    Cells of the same type share a hue family; individual cells are visually distinct.
    Uses a large-prime hash of label_id for deterministic jitter.
    """
    import colorsys
    h  = label_id * 2654435761 & 0xFFFFFF   # 24-bit pseudo-random hash
    s  = 0.55 + 0.45 * (h & 0xFFF) / 0xFFF
    v  = 0.65 + 0.35 * ((h >> 12) & 0xFFF) / 0xFFF
    r, g, b = colorsys.hsv_to_rgb(hue, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def _open_zarr_masks(run: str):
    """Open cells.zarr (or cells.zarr.zip) for a run."""
    try:
        import zarr
    except ImportError:
        raise ImportError('zarr is required for segmentation step: pip install zarr')
    from pathlib import Path
    folder   = Path(DATA_DIR) / RUN_FOLDERS[run]
    zarr_dir = folder / 'cells.zarr'
    zarr_zip = folder / 'cells.zarr.zip'
    if zarr_dir.exists():
        return zarr.open(str(zarr_dir), mode='r')
    return zarr.open(zarr.storage.ZipStore(str(zarr_zip)), mode='r')


def _build_label_lut(run: str, annotations: Dict[str, str],
                     hues: Dict[str, float]) -> np.ndarray:
    """Return uint8 array shape (N+1, 3): index = zarr label → per-cell RGB.
    Same cell type → same hue family; individual cells vary in S and V.
    Assumes zarr label N → row N-1 (0-indexed) in cells.parquet.
    Unannotated cells get dark grey (60, 60, 60); background (label=0) is black.
    """
    parquet = os.path.join(DATA_DIR, RUN_FOLDERS[run], 'cells.parquet')
    df = pd.read_parquet(parquet, columns=['cell_id'])
    df['cell_id'] = df['cell_id'].astype(str)

    n   = len(df)
    lut = np.full((n + 1, 3), 60, dtype=np.uint8)   # dark grey default
    lut[0] = 0                                        # background = black

    # Tolerate "slideN_" prefix mismatch between parquet and annotation keys
    bare_annot = {k.split('_', 1)[-1] if '_' in k else k: v
                  for k, v in annotations.items()}

    for zarr_id, cell_id in enumerate(df['cell_id'], start=1):
        ct = annotations.get(cell_id) or bare_annot.get(cell_id)
        if ct and ct in hues:
            lut[zarr_id] = _cell_rgb(zarr_id, hues[ct])
    return lut


def _build_nucleus_label_lut(run: str, cell_lut: np.ndarray) -> np.ndarray:
    """Return uint8 array (N_nuclei+1, 3): index = nucleus zarr label → same RGB as its cell.
    Nucleus label N maps to cell label via polygon_sets/0/cell_index[N-1] + 1 (verified
    empirically: cell_index is 0-indexed into polygon_sets/1, cell mask label = index+1).
    """
    z  = _open_zarr_masks(run)
    ci = np.array(z['polygon_sets']['0']['cell_index'])   # shape (N_nuclei,)
    n  = len(ci)
    lut = np.zeros((n + 1, 3), dtype=np.uint8)            # background = black
    cell_labels = ci + 1                                   # 0-indexed → 1-indexed cell mask label
    valid = cell_labels < len(cell_lut)
    lut[1:][valid] = cell_lut[cell_labels[valid]]
    return lut


def _build_type_lut(run: str, annotations: Dict[str, str],
                    hues: Dict[str, float]) -> np.ndarray:
    """Return uint8 array shape (N+1, 3): index = zarr label → uniform type RGB.
    Same cell type → exactly the same flat colour (S=0.85, V=0.90), no per-cell jitter.
    Uses the same hue assignments as _build_label_lut for visual consistency.
    Unannotated cells get dark grey (60, 60, 60); background (label=0) is black.
    """
    import colorsys
    parquet = os.path.join(DATA_DIR, RUN_FOLDERS[run], 'cells.parquet')
    df = pd.read_parquet(parquet, columns=['cell_id'])
    df['cell_id'] = df['cell_id'].astype(str)

    n   = len(df)
    lut = np.full((n + 1, 3), 60, dtype=np.uint8)
    lut[0] = 0

    bare_annot = {k.split('_', 1)[-1] if '_' in k else k: v
                  for k, v in annotations.items()}

    # Pre-compute one fixed RGB per cell type
    type_rgb: Dict[str, Tuple[int, int, int]] = {}
    for ct, hue in hues.items():
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.90)
        type_rgb[ct] = (int(r * 255), int(g * 255), int(b * 255))

    for zarr_id, cell_id in enumerate(df['cell_id'], start=1):
        ct = annotations.get(cell_id) or bare_annot.get(cell_id)
        if ct and ct in type_rgb:
            lut[zarr_id] = type_rgb[ct]
    return lut


def _build_binary_lut(run: str) -> np.ndarray:
    """Return uint8 array shape (N+1,): all non-zero labels → 255 (white); background → 0.
    Used to produce a clean binary tissue-presence mask for registration.
    """
    parquet = os.path.join(DATA_DIR, RUN_FOLDERS[run], 'cells.parquet')
    df = pd.read_parquet(parquet, columns=['cell_id'])
    n   = len(df)
    lut = np.full(n + 1, 255, dtype=np.uint8)
    lut[0] = 0
    return lut


def _build_cell_type_label_lut(run: str, annotations: Dict[str, str],
                                type_to_id: Dict[str, int]) -> np.ndarray:
    """Return uint8 array shape (N+1,): index = zarr label → cell-type integer ID.

    Background (label 0) and unannotated cells map to 0.
    Each annotated cell maps to its leiden10annots category integer ID (1-based,
    alphabetically ordered and stable across all fish).
    This gives a per-pixel cell-type label image usable for registration.
    """
    parquet = os.path.join(DATA_DIR, RUN_FOLDERS[run], 'cells.parquet')
    df = pd.read_parquet(parquet, columns=['cell_id'])
    df['cell_id'] = df['cell_id'].astype(str)

    n   = len(df)
    lut = np.zeros(n + 1, dtype=np.uint8)   # 0 = background / unknown

    bare_annot = {k.split('_', 1)[-1] if '_' in k else k: v
                  for k, v in annotations.items()}

    for zarr_id, cell_id in enumerate(df['cell_id'], start=1):
        ct = annotations.get(cell_id) or bare_annot.get(cell_id)
        if ct and ct in type_to_id:
            lut[zarr_id] = type_to_id[ct]
    return lut


# ── step: segmentation ────────────────────────────────────────────────────────

def step_segmentation(fish_ids: List[int]) -> None:
    """Render segmentation images in registered space.

    For each slice produces four output TIFs:
      seg_cells/{gnum}.tif    — per-cell RGB, hue from cell type (with S/V jitter)
      seg_nuclei/{gnum}.tif   — same as above but from nucleus mask
      tissue_map/{gnum}.tif   — uniform flat colour per cell type, no per-cell jitter

    All three share the same hue assignments per cell type for visual consistency.
    The same DV rotation + elastix rigid transform as fluorescence is applied to
    each so they overlay perfectly with the registered DAPI.
    """
    # Output modes: (subdir, zarr_key ('cells'|'nuclei'), lut_type, n_channels)
    _SEG_OUTPUTS = [
        ('seg_cells',       'cells',  'label',         3),
        ('seg_nuclei',      'nuclei', 'label_nuclei',  3),
        ('tissue_map',      'cells',  'type',          3),
        ('cell_type_label', 'cells',  'cell_type_id',  1),
    ]

    annots = dict(zip(
        pd.read_csv(ANNOT_CSV)['cell_id'].astype(str),
        pd.read_csv(ANNOT_CSV)['leiden10annots'].astype(str)
    ))
    hues = _cell_type_hues(list(set(annots.values())))
    # Stable integer ID per cell type (1-based, alphabetical) — same for all fish
    type_to_id: Dict[str, int] = {ct: i + 1 for i, ct in enumerate(sorted(set(annots.values())))}
    logging.info(f'Segmentation: {len(hues)} cell types, {len(annots):,} annotated cells')
    logging.info(f'Cell-type label IDs (1..{len(type_to_id)}): {list(type_to_id.items())[:5]} ...')

    summary_csvs = glob.glob(SUMMARY_GLOB)
    if not summary_csvs:
        raise FileNotFoundError(f'No summary CSV matching {SUMMARY_GLOB}')
    summary_df = pd.read_csv(summary_csvs[0])
    summary_df['run'] = summary_df['run'].astype(str)

    for fish in fish_ids:
        logging.info(f'=== Fish {fish}: segmentation ===')

        meta_path = os.path.join(OUT_DIR, str(fish), 'transforms.json')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f'transforms.json missing — run --from-step register first.'
            )
        with open(meta_path) as fh:
            meta = json.load(fh)
        canvas_h    = meta['canvas_h']
        canvas_w    = meta['canvas_w']
        slices_meta = meta['slices']

        # Open zarr + build all three LUT types once per run
        zarr_handles: Dict[str, object]             = {}
        all_luts:     Dict[str, Dict[str, np.ndarray]] = {}
        for run in RUN_FOLDERS:
            try:
                zarr_handles[run] = _open_zarr_masks(run)
                cell_lut = _build_label_lut(run, annots, hues)
                all_luts[run] = {
                    'label':         cell_lut,
                    'label_nuclei':  _build_nucleus_label_lut(run, cell_lut),
                    'type':          _build_type_lut(run, annots, hues),
                    'binary':        _build_binary_lut(run),
                    'cell_type_id':  _build_cell_type_label_lut(run, annots, type_to_id),
                }
                logging.info(f'  Run {run}: zarr ready, {len(all_luts[run]["label"])} LUT entries')
            except Exception as exc:
                logging.warning(f'  Run {run}: skipped ({exc})')

        first, last = _c0_frame_range(fish)

        for subdir, zarr_key, lut_type, n_ch in _SEG_OUTPUTS:
            zarr_idx = ZARR_MASK_IDX[zarr_key]
            out_dir  = os.path.join(OUT_DIR, str(fish), subdir)
            os.makedirs(out_dir, exist_ok=True)

            zero_frame = (np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
                          if n_ch == 3
                          else np.zeros((canvas_h, canvas_w), dtype=np.uint8))

            for gnum in range(first, last + 1):
                out_path = os.path.join(out_dir, f'{gnum}.tif')
                sm = slices_meta.get(str(gnum))

                if sm is None:
                    tifffile.imwrite(out_path, zero_frame,
                                     photometric='rgb' if n_ch == 3 else 'minisblack')
                    continue

                rows = summary_df[
                    (summary_df['fish_name'] == fish) &
                    (summary_df['global_slice_num'] == gnum)
                ]
                if len(rows) == 0 or str(int(rows.iloc[0]['run'])) not in zarr_handles:
                    tifffile.imwrite(out_path, zero_frame,
                                     photometric='rgb' if n_ch == 3 else 'minisblack')
                    continue

                row = rows.iloc[0]
                run = str(int(row['run']))
                z   = zarr_handles[run]
                lut = all_luts[run][lut_type]

                # Crop zarr mask to this slice's bbox
                r0, r1 = int(row['bbox_global_min_row']), int(row['bbox_global_max_row'])
                c0, c1 = int(row['bbox_global_min_col']), int(row['bbox_global_max_col'])
                mH, mW = z['masks'][zarr_idx].shape[:2]
                mask = np.array(z['masks'][zarr_idx][max(0,r0):min(mH,r1),
                                                      max(0,c0):min(mW,c1)])

                # Render label → pixel values; place on canvas
                clipped = np.clip(mask, 0, len(lut) - 1)
                crop    = lut[clipped].astype(np.float32)   # (H, W) or (H, W, 3)

                canvas = np.zeros((canvas_h, canvas_w) if n_ch == 1
                                  else (canvas_h, canvas_w, 3), dtype=np.float32)
                ph = min(crop.shape[0], sm['orig_h'])
                pw = min(crop.shape[1], sm['orig_w'])
                canvas[sm['pad_top']:sm['pad_top']+ph,
                       sm['pad_left']:sm['pad_left']+pw] = crop[:ph, :pw]

                # DV rotation — NN (order=0): all seg/label images, no colour mixing
                dv = sm['dv_angle_deg']
                if abs(dv) >= 1.0:
                    if n_ch == 3:
                        canvas = np.stack([
                            nd_rotate(canvas[..., c], dv, reshape=False, cval=0.0, order=0)
                            for c in range(3)
                        ], axis=-1)
                    else:
                        canvas = nd_rotate(canvas, dv, reshape=False, cval=0.0, order=0)

                # Elastix rigid transform — NN: label/seg images must not interpolate
                tf_file = sm.get('elastix_file')
                if tf_file is not None:
                    tf_path = os.path.join(OUT_DIR, str(fish), tf_file)
                    if os.path.exists(tf_path):
                        tp = itk.ParameterObject.New()
                        tp.ReadParameterFile(tf_path)
                        if n_ch == 3:
                            canvas = np.stack([
                                apply_transform_nn(canvas[..., c], tp)
                                for c in range(3)
                            ], axis=-1)
                        else:
                            canvas = apply_transform_nn(canvas, tp)

                out_arr = np.clip(canvas, 0, 255).astype(np.uint8)
                tifffile.imwrite(out_path, out_arr,
                                 photometric='rgb' if n_ch == 3 else 'minisblack')

            logging.info(f'  {subdir}: done')

        del zarr_handles; gc.collect()


# ── step: stack ───────────────────────────────────────────────────────────────

def step_stack(fish_ids: List[int]) -> None:
    for fish in fish_ids:
        logging.info(f"=== Fish {fish}: building 3D stacks ===")

        # Fluorescence channels — float32 single-channel stacks
        for ch in range(NUM_CHANNELS):
            ch_dir = os.path.join(OUT_DIR, str(fish), f'c{ch}')
            if not os.path.isdir(ch_dir):
                logging.warning(f"  c{ch}: output dir missing, skipping")
                continue
            files = sorted(
                glob.glob(os.path.join(ch_dir, '*.tif')),
                key=lambda f: int(os.path.splitext(os.path.basename(f))[0])
            )
            if not files:
                continue
            frames = [tifffile.imread(f).astype(np.float32) for f in files]
            vol = np.stack(frames, axis=0)
            out_path = os.path.join(OUT_DIR, str(fish), f'rigid_3d_c{ch}.tif')
            tifffile.imwrite(out_path, vol)
            gnums = [int(os.path.splitext(os.path.basename(f))[0]) for f in files]
            logging.info(f"  c{ch}: shape {vol.shape}  global_nums {gnums[0]}–{gnums[-1]}  (frame 1 = global_num 1)")
            del frames, vol; gc.collect()

        # Segmentation — RGB stacks (seg_cells, seg_nuclei, tissue_map) → (Z, H, W, 3)
        for seg_dir_name in ('seg_cells', 'seg_nuclei', 'tissue_map'):
            seg_dir = os.path.join(OUT_DIR, str(fish), seg_dir_name)
            if not os.path.isdir(seg_dir):
                continue
            files = sorted(
                glob.glob(os.path.join(seg_dir, '*.tif')),
                key=lambda f: int(os.path.splitext(os.path.basename(f))[0])
            )
            if not files:
                continue
            frames = [tifffile.imread(f) for f in files]   # each (H, W, 3) uint8
            vol = np.stack(frames, axis=0)                  # (Z, H, W, 3)
            out_path = os.path.join(OUT_DIR, str(fish), f'{seg_dir_name}_3d.tif')
            tifffile.imwrite(out_path, vol, photometric='rgb')
            gnums = [int(os.path.splitext(os.path.basename(f))[0]) for f in files]
            logging.info(f"  {seg_dir_name}: shape {vol.shape}  slices {gnums[0]}–{gnums[-1]}")
            del frames, vol; gc.collect()

        # Single-channel uint8 stack: cell_type_label (0=bg, 1–34=cell type)
        for sc_name in ('cell_type_label',):
            sc_dir = os.path.join(OUT_DIR, str(fish), sc_name)
            if not os.path.isdir(sc_dir):
                continue
            files = sorted(
                glob.glob(os.path.join(sc_dir, '*.tif')),
                key=lambda f: int(os.path.splitext(os.path.basename(f))[0])
            )
            if not files:
                continue
            frames = [tifffile.imread(f) for f in files]  # each (H, W) uint8
            vol    = np.stack(frames, axis=0)              # (Z, H, W)
            out_path = os.path.join(OUT_DIR, str(fish), f'{sc_name}_3d.tif')
            tifffile.imwrite(out_path, vol)
            gnums = [int(os.path.splitext(os.path.basename(f))[0]) for f in files]
            logging.info(f"  {sc_name}: shape {vol.shape}  slices {gnums[0]}–{gnums[-1]}")
            del frames, vol; gc.collect()


# ── step: per_gene ────────────────────────────────────────────────────────────
#
# Renders one 3D tif stack per gene in the rigid-registered canvas space.
# These are purely for visual inspection — no additional registration is applied.
# Transcripts are mapped through the same DV rotation + elastix transforms
# saved by step_register, then rendered as 2D Gaussian density images.
#
# Output: analysis/2_registered/{fish}/per_gene/{gene}.tif  (Z, H, W) float32
#
# Constants — tune if needed
_PG_QV_MIN           = 20.0   # minimum transcript quality value
_PG_SIGMA_UM         = 10.0   # Gaussian sigma in µm  (~47 px)
_PG_MIN_TX_SLICE     = 500    # skip slices with fewer transcripts
_PG_SPATIAL_BINS     = 40     # coarse grid for spatial-CV computation
_PG_MIN_COVERAGE_PCT = 5.0    # exclude genes covering < this % of bins
_PG_MAX_COVERAGE_PCT = 45.0   # exclude genes covering > this % of bins
_PG_N_TOP            = 40     # top-N spatially filtered genes to render

# Known zebrafish anatomical markers — always included regardless of ranking
_PG_STRUCTURAL_GENES = [
    'tbxta', 'tbxtb', 'noto',           # notochord
    'sox3',  'sox19a', 'sox19b',        # neural tube
    'myod1', 'mef2ca', 'mef2d', 'myog', # muscle
    'tp63',                              # skin / epithelium
]


def _pg_load_transcripts(run: str) -> pd.DataFrame:
    """Load filtered transcripts for one run (real genes, qv ≥ threshold)."""
    path = os.path.join(DATA_DIR, RUN_FOLDERS[run], 'transcripts.parquet')
    df   = pd.read_parquet(path, columns=['feature_name', 'x_location',
                                           'y_location', 'qv', 'is_gene'])
    return df[(df['is_gene'] == True) & (df['qv'] >= _PG_QV_MIN)].reset_index(drop=True)


def _map_coords_to_canvas(tx: pd.DataFrame,
                          bbox_row_min: float, bbox_col_min: float,
                          sm: Dict, fish_str: str) -> Tuple[np.ndarray, np.ndarray]:
    """Map transcript µm coords → canvas pixels using the DV rotation + elastix transforms.

    Applies the same transform chain as step_register but to point coordinates
    rather than whole images. Used by step_per_gene and script 3 gene-density images.

    Returns (xy (N,2) col/row, valid_mask (N,)).
    """
    S  = XENIUM_PX_PER_UM
    x  = tx['x_location'].values * S - bbox_col_min + sm['pad_left']
    y  = tx['y_location'].values * S - bbox_row_min + sm['pad_top']
    xy = np.column_stack([x, y])

    # DV rotation (CCW)
    if abs(sm.get('dv_angle_deg', 0.0)) >= MIN_ANGLE_DEG:
        a = math.radians(sm['dv_angle_deg'])
        cx, cy   = sm['canvas_w'] / 2.0, sm['canvas_h'] / 2.0
        dx, dy   = xy[:, 0] - cx, xy[:, 1] - cy
        cos_a, sin_a = math.cos(a), math.sin(a)
        xy = np.column_stack([cx + cos_a*dx + sin_a*dy,
                               cy - sin_a*dx + cos_a*dy])

    # Elastix rigid inverse: moving → fixed
    if sm.get('elastix_file'):
        tf_path = os.path.join(OUT_DIR, fish_str, sm['elastix_file'])
        if os.path.exists(tf_path):
            params: Dict = {}
            with open(tf_path) as fh:
                for line in fh:
                    line = line.strip()
                    if line.startswith('(') and line.endswith(')'):
                        parts = line[1:-1].split()
                        params[parts[0]] = parts[1:]
            tp   = [float(v) for v in params['TransformParameters']]
            corp = [float(v) for v in params['CenterOfRotationPoint']]
            angle, tx_t, ty_t = tp[0], tp[1], tp[2]
            cx, cy = corp[0], corp[1]
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            dx = xy[:, 0] - cx - tx_t
            dy = xy[:, 1] - cy - ty_t
            xy = np.column_stack([cx + cos_a*dx + sin_a*dy,
                                   cy - sin_a*dx + cos_a*dy])

    valid = ((xy[:, 0] >= 0) & (xy[:, 0] < sm['canvas_w']) &
             (xy[:, 1] >= 0) & (xy[:, 1] < sm['canvas_h']))
    return xy, valid


def _pg_select_genes(all_tx_canvas: list, canvas_h: int, canvas_w: int,
                     panel_genes: set) -> List[str]:
    """Select top-N spatially filtered genes + structural genes present in panel."""
    B = _PG_SPATIAL_BINS
    gene_cv:  Dict[str, list] = {}
    gene_cov: Dict[str, list] = {}

    for xy, genes in all_tx_canvas:
        if len(xy) < _PG_MIN_TX_SLICE:
            continue
        col_b = np.clip((xy[:, 0] / canvas_w * B).astype(int), 0, B-1)
        row_b = np.clip((xy[:, 1] / canvas_h * B).astype(int), 0, B-1)
        for gene in np.unique(genes):
            mask = genes == gene
            if mask.sum() < 5:
                continue
            counts = np.zeros((B, B), dtype=np.float32)
            np.add.at(counts, (row_b[mask], col_b[mask]), 1)
            n   = counts.sum()
            p   = counts / n
            cv  = float(p.var())
            cov = float((counts > 0).sum()) / (B * B) * 100.0
            gene_cv.setdefault(gene, []).append(cv)
            gene_cov.setdefault(gene, []).append(cov)

    n_slices   = len(all_tx_canvas)
    min_slices = max(2, n_slices // 4)
    rows = []
    for g in gene_cv:
        if len(gene_cv[g]) < min_slices:
            continue
        cov = float(np.mean(gene_cov[g]))
        if not (_PG_MIN_COVERAGE_PCT <= cov <= _PG_MAX_COVERAGE_PCT):
            continue
        rows.append({'gene': g, 'cv': float(np.mean(gene_cv[g])), 'cov': cov})

    top = sorted(rows, key=lambda r: r['cv'], reverse=True)[:_PG_N_TOP]
    top_genes  = [r['gene'] for r in top]
    structural = [g for g in _PG_STRUCTURAL_GENES if g in panel_genes]
    genes = list(dict.fromkeys(top_genes + structural))   # dedup, order preserved
    logging.info(f'  per_gene: {len(top_genes)} top-spatial + '
                 f'{len(structural)} structural = {len(genes)} genes to render')
    return genes


def step_per_gene(fish_ids: List[int], sigma_um: Optional[float] = None) -> None:
    """Render per-gene transcript density stacks in rigid-registered space.
    Output: analysis/2_registered/{fish}/per_gene/{gene}.tif  (94 frames to match DAPI).
    sigma_um: Gaussian sigma in µm (default: _PG_SIGMA_UM = 10.0; try 3–5 for sharper images).
    """
    import scipy.ndimage as _snd

    sigma_um   = sigma_um if sigma_um is not None else _PG_SIGMA_UM
    sigma_px   = sigma_um * XENIUM_PX_PER_UM
    logging.info(f'  per_gene sigma: {sigma_um} µm = {sigma_px:.1f} px')
    summary_df = pd.read_csv(glob.glob(SUMMARY_GLOB)[0])
    summary_df['run'] = summary_df['run'].astype(int).astype(str)

    needed_runs = summary_df[summary_df['fish_name'].isin(fish_ids)]['run'].unique()
    run_tx: Dict[str, pd.DataFrame] = {}
    for run in needed_runs:
        run = str(int(float(run)))
        if run not in RUN_FOLDERS:
            continue
        logging.info(f'  Loading run {run} transcripts...')
        run_tx[run] = _pg_load_transcripts(run)
        logging.info(f'  Run {run}: {len(run_tx[run]):,} transcripts')

    for fish in fish_ids:
        logging.info(f'=== Fish {fish}: per_gene ===')
        out_dir = os.path.join(OUT_DIR, str(fish), 'per_gene')
        os.makedirs(out_dir, exist_ok=True)

        # Load transforms.json saved by step_register
        meta_path = os.path.join(OUT_DIR, str(fish), 'transforms.json')
        if not os.path.exists(meta_path):
            logging.error(f'  {meta_path} missing — run register step first')
            continue
        with open(meta_path) as fh:
            meta = json.load(fh)
        canvas_h    = meta['canvas_h']
        canvas_w    = meta['canvas_w']
        slices_meta = meta['slices']
        for sm in slices_meta.values():
            sm['canvas_h'] = canvas_h
            sm['canvas_w'] = canvas_w

        # Full range from DAPI c0 output (so z-indices match DAPI stack exactly)
        full_first, full_last = _c0_frame_range(fish)
        logging.info(f'  DAPI range: {full_first}–{full_last} ({full_last - full_first + 1} frames)')

        fish_rows = summary_df[summary_df['fish_name'] == fish]

        # ── First pass: map all transcripts, collect for gene selection ──────
        all_tx_canvas: list = []
        slice_tx_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        all_genes: set = set()

        for gnum_str, sm in slices_meta.items():
            gnum = int(gnum_str)
            rows = fish_rows[fish_rows['global_slice_num'] == gnum]
            if len(rows) == 0:
                continue
            row = rows.iloc[0]
            run = str(int(row['run']))
            if run not in run_tx:
                continue
            S    = XENIUM_PX_PER_UM
            r0, r1 = row['bbox_global_min_row'], row['bbox_global_max_row']
            c0, c1 = row['bbox_global_min_col'], row['bbox_global_max_col']
            tx   = run_tx[run]
            mask = ((tx['y_location'] * S >= r0) & (tx['y_location'] * S <= r1) &
                    (tx['x_location'] * S >= c0) & (tx['x_location'] * S <= c1))
            sub  = tx[mask]
            if len(sub) < _PG_MIN_TX_SLICE:
                continue
            xy, valid = _map_coords_to_canvas(sub, r0, c0, sm, str(fish))
            xy    = xy[valid]
            genes = sub['feature_name'].values[valid]
            slice_tx_cache[gnum] = (xy, genes)
            all_tx_canvas.append((xy, genes))
            all_genes.update(np.unique(genes))

        logging.info(f'  {len(slice_tx_cache)} slices with data  '
                     f'(mean {np.mean([len(v[1]) for v in slice_tx_cache.values()]):.0f} tx/slice)  '
                     f'{len(all_genes)} genes in panel')

        # ── Gene selection ───────────────────────────────────────────────────
        genes_to_render = _pg_select_genes(all_tx_canvas, canvas_h, canvas_w, all_genes)

        with open(os.path.join(out_dir, 'genes.txt'), 'w') as fh:
            fh.write('\n'.join(genes_to_render))

        # ── Second pass: render per-gene stacks ──────────────────────────────
        zero = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        for gene in genes_to_render:
            frames = []
            for gnum in range(full_first, full_last + 1):
                if gnum not in slice_tx_cache:
                    frames.append(zero)
                    continue
                xy, gs = slice_tx_cache[gnum]
                mask   = gs == gene
                if not mask.any():
                    frames.append(zero)
                    continue
                img = np.zeros((canvas_h, canvas_w), dtype=np.float32)
                cols = xy[mask, 0].astype(np.int32)
                rows_idx = xy[mask, 1].astype(np.int32)
                np.add.at(img, (rows_idx, cols), 1)
                _snd.gaussian_filter(img, sigma=sigma_px, output=img, mode='constant')
                mx = img.max()
                if mx > 0:
                    img /= mx
                frames.append(img)
            vol = np.stack(frames, axis=0)
            tifffile.imwrite(os.path.join(out_dir, f'{gene}.tif'), vol)
            logging.info(f'  saved {gene}.tif  {vol.shape}')
            del frames, vol; gc.collect()

        logging.info(f'  per_gene → {out_dir}')
        del slice_tx_cache, all_tx_canvas; gc.collect()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Rigid 2D registration of preprocessed fish slices.'
    )
    parser.add_argument(
        '--steps', nargs='+', choices=STEPS, default=STEPS, metavar='STEP',
        help=f'Steps to run (space-separated). Choices: {", ".join(STEPS)}. '
             f'Default: all steps.'
    )
    parser.add_argument(
        '--fish', type=int, default=None,
        help='Process only this fish ID (1–6). Default: all fish.'
    )
    parser.add_argument(
        '--sigma-um', type=float, default=None,
        help='Gaussian sigma for per_gene step in µm (default: 10.0). '
             'Try 3–5 for sharper features.'
    )
    args = parser.parse_args()

    if args.fish is not None:
        fish_ids = [args.fish]
    else:
        fish_ids = sorted(
            int(d) for d in os.listdir(SRC_DIR)
            if os.path.isdir(os.path.join(SRC_DIR, d)) and d.isdigit()
        )
    logging.info(f"Fish to process: {fish_ids}")
    logging.info(f"Steps: {args.steps}")

    run = set(args.steps)
    if 'register'     in run: step_register(fish_ids)
    if 'segmentation' in run: step_segmentation(fish_ids)
    if 'stack'        in run: step_stack(fish_ids)
    if 'per_gene'     in run: step_per_gene(fish_ids, sigma_um=args.sigma_um)

    logging.info(f"Done. Output: {OUT_DIR}")


if __name__ == '__main__':
    main()
