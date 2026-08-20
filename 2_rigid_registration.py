"""
2_rigid_registration.py

Rigid 2D registration of individual fish slices (reads directly from script 1 output).

Input:  analysis/individual_fish_2d/{fish}/c{ch}/{global_num}_{run}_{tile}.tif
Output: analysis/2_registered/{fish}/c{ch}/{global_num}.tif
        analysis/2_registered/{fish}/rigid_3d_c{ch}.tif

Strategy: DAPI drives registration; same transform applied to all channels.
Propagates outward from the middle reference slice so drift accumulates
over at most n/2 steps.

Two-phase: Phase 1 registers all DAPI and keeps results in memory;
Phase 2 applies saved transforms to remaining channels one slice at a time.

Gaps: zero-filled frames written for every integer up to the last slice
so ImageJ frame N = global_num N.

Usage:
  python 2_rigid_registration.py
  python 2_rigid_registration.py --fish 1
  python 2_rigid_registration.py --from-step stack
"""

import os, gc, glob, re, argparse, logging, math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile
import itk
from scipy.ndimage import rotate as nd_rotate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SLICE_RE = re.compile(r'^(\d+)_')

SRC_DIR      = '../analysis/individual_fish_2d'
OUT_DIR      = '../analysis/2_registered'
DATA_DIR     = '../data'
SUMMARY_GLOB = '../analysis/fish_bbox_summary_tagged_*.csv'
NUM_CHANNELS = 4
DAPI_CHANNEL = 0

STEPS = ['register', 'stack']

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
    moving_itk = itk.GetImageFromArray(moving_np); _set_itk_props(moving_itk)
    result = itk.transformix_filter(moving_itk, tparams)
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
        max_h, max_w = 0, 0
        for g, p in dapi_slices:
            arr = tifffile.imread(p)
            h, w = arr.shape[-2], arr.shape[-1]
            a = math.radians(abs(dv_angles.get(g, 0.0)))
            rh = int(math.ceil(h * abs(math.cos(a)) + w * abs(math.sin(a))))
            rw = int(math.ceil(h * abs(math.sin(a)) + w * abs(math.cos(a))))
            max_h = max(max_h, rh)
            max_w = max(max_w, rw)
        max_h += 2 * PAD_PX
        max_w += 2 * PAD_PX
        zero_frame = np.zeros((max_h, max_w), dtype=np.float32)
        logging.info(f"  canvas: {max_h}×{max_w}  ({len(dv_angles)} slices with DV rotation)")

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

        def find_nearest_registered(target: int) -> int:
            direction = 1 if target < ref_gnum else -1
            k = target + direction
            while first <= k <= last:
                if k in registered_dapi:
                    return k
                k += direction
            return ref_gnum

        # Process outward from ref
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

            step += 1

        n_reg   = len(registered_dapi)
        n_wrote = last - first + 1
        logging.info(f"  DAPI: registered {n_reg} slices, {n_wrote - n_reg} gap(s)")

        # ── Phase 2: apply transforms to all channels with original intensities ─
        # DAPI (ch0) is included here so outputs have raw (non-normalised) values;
        # registered_dapi holds normalised versions used only as fixed-image chain.
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

        del registered_dapi, transforms, zero_frame; gc.collect()


# ── step: stack ───────────────────────────────────────────────────────────────

def step_stack(fish_ids: List[int]) -> None:
    for fish in fish_ids:
        logging.info(f"=== Fish {fish}: building 3D stacks ===")
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


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Rigid 2D registration of preprocessed fish slices.'
    )
    parser.add_argument(
        '--from-step', choices=STEPS, default='register', metavar='STEP',
        help=f'Resume from this step. Choices: {", ".join(STEPS)}'
    )
    parser.add_argument(
        '--fish', type=int, default=None,
        help='Process only this fish ID (1–6). Default: all fish.'
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

    from_idx = STEPS.index(args.from_step)
    if from_idx <= STEPS.index('register'):
        step_register(fish_ids)
    if from_idx <= STEPS.index('stack'):
        step_stack(fish_ids)

    logging.info(f"Done. Output: {OUT_DIR}")


if __name__ == '__main__':
    main()
