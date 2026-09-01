"""
4_cross_fish_registration.py

3D cross-fish registration: registers every fish's volume to a reference fish,
exploring all combinations of driving image, registration stages, and Z-spacing.

Experiment matrix (12 total):
  driving      : dapi | dapi_blend | cell_type_map
  stages       : rigid | rigid_affine
  z_spacing_um : 10

Registration is performed at full resolution (no downsampling).

Fish groups:
  WT:     1, 2, 3
  Mutant: 4, 5, 6

Input
-----
  analysis/3_improved_registration/dapi_blend/{fish}/c0_3d.tif   best per-fish DAPI (script 3)
  analysis/2_registered/{fish}/rigid_3d_c{ch}.tif     fluorescence channels 1-3
  analysis/2_registered/{fish}/seg_cells_3d.tif
  analysis/2_registered/{fish}/seg_nuclei_3d.tif
  analysis/2_registered/{fish}/cell_type_label_3d.tif

Output
------
  analysis/4_registered/{experiment}/{fish}/c{ch}.tif       float32 3D (full resolution)
  analysis/4_registered/{experiment}/{fish}/seg_cells.tif      uint8 (3D or RGB), NN interp
  analysis/4_registered/{experiment}/{fish}/seg_nuclei.tif     uint8, NN interp
  analysis/4_registered/{experiment}/{fish}/cell_type_label.tif  uint8 0–34, NN interp
  analysis/4_registered/{experiment}/{fish}/transform.json
  analysis/4_registered/evaluation.csv           summary table
  analysis/4_registered/evaluation_per_fish.csv  per-fish breakdown

Steps: register | evaluate

Usage
-----
  python 4_cross_fish_registration.py
  python 4_cross_fish_registration.py --steps register
  python 4_cross_fish_registration.py --steps evaluate
  python 4_cross_fish_registration.py --driving dapi --stages rigid
  python 4_cross_fish_registration.py --reference 2
"""

import csv, gc, json, argparse, logging, math, os
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
import tifffile
import itk
from scipy.ndimage import zoom, shift as nd_shift, rotate as nd_rotate

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# ── paths ──────────────────────────────────────────────────────────────────────
IN2_DIR = '../analysis/2_registered'
IN3_DIR = '../analysis/3_improved_registration'
OUT_DIR = '../analysis/4_registered'

# ── fish groups ────────────────────────────────────────────────────────────────
WT_FISH     = [1, 2, 3]
MUTANT_FISH = [4, 5, 6]

# ── imaging constants ──────────────────────────────────────────────────────────
XENIUM_PX_UM  = 0.2125
DOWNSAMPLE_XY = 1
DS_SPACING    = XENIUM_PX_UM * DOWNSAMPLE_XY   # 0.2125 µm/px (full resolution)
NUM_CHANNELS  = 4
CANVAS_PAD    = 8

# ── experiment axes ────────────────────────────────────────────────────────────
DRIVING_OPTIONS   = ['dapi', 'dapi_blend', 'cell_type_map']
STAGES_OPTIONS    = ['rigid', 'rigid_affine']
Z_SPACING_OPTIONS = [10]   # confirmed 10 µm z-spacing
STEPS             = ['register', 'evaluate']

# ── elastix ────────────────────────────────────────────────────────────────────
N_RES_RIGID     = 4
N_RES_AFFINE    = 3
MAX_ITER_RIGID  = 500
MAX_ITER_AFFINE = 300
NUM_SAMPLES     = 5_000
PYRAMID_RIGID   = ['8','8','2', '4','4','1', '2','2','1', '1','1','1']
PYRAMID_AFFINE  = ['4','4','1', '2','2','1', '1','1','1']


# ── evaluation outputs covered ────────────────────────────────────────────────
EVAL_CHANNELS = list(range(NUM_CHANNELS))            # c0..c3
EVAL_SEGS     = ['seg_cells', 'seg_nuclei']          # binary NCC after grayscale collapse


# ══════════════════════════════════════════════════════════════════════════════
# NAMES
# ══════════════════════════════════════════════════════════════════════════════

def exp_name(driving: str, stages: str, z_um: int) -> str:
    return f'{driving}_{stages}_z{z_um}'


def all_experiments(driving_opts, stages_opts, z_opts):
    return [exp_name(d, s, z) for d, s, z in product(driving_opts, stages_opts, z_opts)]


# ══════════════════════════════════════════════════════════════════════════════
# LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_dapi_vol(fish: int) -> np.ndarray:
    p = os.path.join(IN3_DIR, 'dapi_blend', str(fish), 'c0_3d.tif')
    if not os.path.exists(p):
        p = os.path.join(IN2_DIR, str(fish), 'rigid_3d_c0.tif')
    return tifffile.imread(p).astype(np.float32)


def load_tissue_map_gray_vol(fish: int) -> np.ndarray:
    """Cell-type grayscale volume derived from tissue_map RGB (Z, H, W, 3).
    tissue_map assigns each cell type a unique evenly-spaced HSV hue (same flat
    colour per type). RGB→grayscale gives a distinct value per cell type.
    Used as driving image for cell_type_map experiments (with MI metric) or
    blended with DAPI for dapi_blend experiments (with NCC metric)."""
    vol = tifffile.imread(os.path.join(IN2_DIR, str(fish), 'tissue_map_3d.tif'))
    if vol.ndim == 4:
        gray = vol.mean(axis=-1).astype(np.float32)
    else:
        gray = vol.astype(np.float32)
    mx = gray.max()
    return gray / mx if mx > 0 else gray


def _norm(v: np.ndarray) -> np.ndarray:
    mx = float(v.max())
    return v / mx if mx > 0 else v.copy()


def load_driving_vol(fish: int, driving: str) -> np.ndarray:
    if driving == 'dapi':
        return _norm(load_dapi_vol(fish))
    if driving == 'cell_type_map':
        # Cell-type label map (not binary) — each pixel = leiden10annots integer ID
        return _norm(load_tissue_map_gray_vol(fish))
    if driving == 'dapi_blend':
        dapi  = _norm(load_dapi_vol(fish))
        ctype = load_tissue_map_gray_vol(fish)
        if ctype.shape != dapi.shape:
            padded = np.zeros_like(dapi)
            sz = tuple(min(ctype.shape[i], dapi.shape[i]) for i in range(3))
            padded[:sz[0], :sz[1], :sz[2]] = ctype[:sz[0], :sz[1], :sz[2]]
            ctype = padded
        return _norm(0.5 * dapi + 0.5 * ctype)
    raise ValueError(f'Unknown driving: {driving!r}')


def load_channel_vol(fish: int, ch: int) -> Optional[np.ndarray]:
    if ch == 0:
        return load_dapi_vol(fish)
    p = os.path.join(IN2_DIR, str(fish), f'rigid_3d_c{ch}.tif')
    return tifffile.imread(p).astype(np.float32) if os.path.exists(p) else None


def load_seg_vol(fish: int, name: str) -> Optional[np.ndarray]:
    p = os.path.join(IN2_DIR, str(fish), f'{name}_3d.tif')
    return tifffile.imread(p) if os.path.exists(p) else None


# ══════════════════════════════════════════════════════════════════════════════
# CANVAS + DOWNSAMPLING
# ══════════════════════════════════════════════════════════════════════════════

def _ds_xy(vol: np.ndarray, factor: int, nn: bool = False) -> np.ndarray:
    """Downsample XY by factor.  nn=True uses nearest-neighbour (labels/segs)."""
    if factor == 1:
        return vol
    order = 0 if nn else 1
    return zoom(vol, (1.0, 1.0 / factor, 1.0 / factor), order=order)


def determine_canvas(fish_ids: List[int]) -> Tuple[int, int, int]:
    max_z = max_h = max_w = 0
    for fish in fish_ids:
        vol = load_dapi_vol(fish)
        z, h, w = vol.shape; del vol
        max_z = max(max_z, z)
        max_h = max(max_h, math.ceil(h / DOWNSAMPLE_XY))
        max_w = max(max_w, math.ceil(w / DOWNSAMPLE_XY))
    canvas = (max_z + 2 * CANVAS_PAD, max_h + 2 * CANVAS_PAD, max_w + 2 * CANVAS_PAD)
    logging.info(f'Common canvas (ds): {canvas}')
    return canvas


def place_in_canvas(vol_ds: np.ndarray,
                    canvas_shape: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    out = np.zeros(canvas_shape, dtype=np.float32)
    dz  = min(vol_ds.shape[0], canvas_shape[0])
    dy  = min(vol_ds.shape[1], canvas_shape[1])
    dx  = min(vol_ds.shape[2], canvas_shape[2])
    out[:dz, :dy, :dx] = vol_ds[:dz, :dy, :dx]
    thresh = 0.01 * out.max() if out.max() > 0 else 0.01
    mask   = out > thresh
    if not mask.any():
        return out, np.zeros(3)
    com = np.array([m.mean() for m in np.where(mask)])
    ctr = np.array(canvas_shape, dtype=float) / 2.0
    sv  = ctr - com
    return nd_shift(out, sv, order=1, cval=0.0), sv


def apply_canvas_tf(vol_ds: np.ndarray, canvas_shape: Tuple[int, int, int],
                    pre_rot_deg: float, shift_vec: np.ndarray,
                    nn: bool = False) -> np.ndarray:
    """Place vol_ds into canvas and apply shift.  nn=True for label/seg images."""
    order   = 0 if nn else 1
    vol_rot = _rotate_z(vol_ds, pre_rot_deg, nn=nn)
    out     = np.zeros(canvas_shape, dtype=vol_ds.dtype if nn else np.float32)
    dz = min(vol_rot.shape[0], canvas_shape[0])
    dy = min(vol_rot.shape[1], canvas_shape[1])
    dx = min(vol_rot.shape[2], canvas_shape[2])
    out[:dz, :dy, :dx] = vol_rot[:dz, :dy, :dx]
    return nd_shift(out, shift_vec, order=order, cval=0.0)


def _rotate_z(vol: np.ndarray, deg: float, nn: bool = False) -> np.ndarray:
    deg = float(deg) % 360.0
    if deg == 0.0:
        return vol
    if deg % 90.0 == 0.0:
        return np.rot90(vol, int(deg // 90) % 4, axes=(1, 2))
    order = 0 if nn else 1
    return nd_rotate(vol.astype(np.float32), deg, axes=(1, 2),
                     reshape=False, order=order, cval=0.0)


def _ncc_vols(a: np.ndarray, b: np.ndarray) -> float:
    mask = (a > 0.01) | (b > 0.01)
    if mask.sum() < 100:
        return 0.0
    am = a[mask].astype(np.float64); bm = b[mask].astype(np.float64)
    am -= am.mean(); bm -= bm.mean()
    d = math.sqrt((am ** 2).sum() * (bm ** 2).sum())
    return float((am * bm).sum() / d) if d > 1e-9 else 0.0



# ══════════════════════════════════════════════════════════════════════════════
# ITK + ELASTIX
# ══════════════════════════════════════════════════════════════════════════════

def _to_itk(vol: np.ndarray, z_um: float) -> itk.Image:
    img = itk.GetImageFromArray(vol.astype(np.float32))
    img.SetSpacing((DS_SPACING, DS_SPACING, float(z_um)))
    img.SetOrigin((0.0, 0.0, 0.0))
    return img


def _build_rigid_params(use_mi: bool = False) -> itk.ParameterObject:
    """use_mi=True for label/cell-type driving (Mutual Information);
    False for fluorescence driving (Normalised Cross-Correlation)."""
    po = itk.ParameterObject.New()
    pm = po.GetDefaultParameterMap('rigid', N_RES_RIGID)
    pm['Transform']                              = ['EulerTransform']
    if use_mi:
        pm['Metric']                             = ['AdvancedMattesMutualInformation']
        pm['NumberOfHistogramBins']              = ['64']
    else:
        pm['Metric']                             = ['AdvancedNormalizedCorrelation']
    pm['Optimizer']                              = ['AdaptiveStochasticGradientDescent']
    pm['AutomaticParameterEstimation']           = ['true']
    pm['AutomaticTransformInitialization']       = ['true']
    pm['AutomaticTransformInitializationMethod'] = ['CenterOfGravity']
    pm['ImageSampler']                           = ['RandomCoordinate']
    pm['NumberOfSpatialSamples']                 = [str(NUM_SAMPLES)]
    pm['MaximumNumberOfSamplingAttempts']        = ['200']
    pm['NewSamplesEveryIteration']               = ['true']
    pm['ImagePyramidSchedule']                   = PYRAMID_RIGID
    pm['MaximumNumberOfIterations']              = [str(MAX_ITER_RIGID)] * N_RES_RIGID
    pm['BSplineInterpolationOrder']              = ['1']
    pm['FinalBSplineInterpolationOrder']         = ['1']
    pm['ResultImagePixelType']                   = ['float']
    po.AddParameterMap(pm)
    return po


def _build_affine_params(use_mi: bool = False) -> itk.ParameterObject:
    """use_mi=True for label/cell-type driving (Mutual Information);
    False for fluorescence driving (Normalised Cross-Correlation)."""
    po = itk.ParameterObject.New()
    pm = po.GetDefaultParameterMap('affine', N_RES_AFFINE)
    pm['Transform']                              = ['AffineTransform']
    if use_mi:
        pm['Metric']                             = ['AdvancedMattesMutualInformation']
        pm['NumberOfHistogramBins']              = ['64']
    else:
        pm['Metric']                             = ['AdvancedNormalizedCorrelation']
    pm['Optimizer']                              = ['AdaptiveStochasticGradientDescent']
    pm['AutomaticParameterEstimation']           = ['true']
    pm['AutomaticTransformInitialization']       = ['false']
    pm['ImageSampler']                           = ['RandomCoordinate']
    pm['NumberOfSpatialSamples']                 = [str(NUM_SAMPLES)]
    pm['MaximumNumberOfSamplingAttempts']        = ['200']
    pm['NewSamplesEveryIteration']               = ['true']
    pm['ImagePyramidSchedule']                   = PYRAMID_AFFINE
    pm['MaximumNumberOfIterations']              = [str(MAX_ITER_AFFINE)] * N_RES_AFFINE
    pm['BSplineInterpolationOrder']              = ['1']
    pm['FinalBSplineInterpolationOrder']         = ['1']
    pm['ResultImagePixelType']                   = ['float']
    po.AddParameterMap(pm)
    return po


def _register(fixed: itk.Image, moving: itk.Image,
               params: itk.ParameterObject, init_tp=None):
    kw = dict(parameter_object=params, log_to_console=False, log_to_file=False)
    if init_tp is not None:
        kw['initial_transform_parameter_object'] = init_tp
    result, tp = itk.elastix_registration_method(fixed, moving, **kw)
    return itk.GetArrayFromImage(result).astype(np.float32), tp


def _apply_tp(canvas: np.ndarray, tp, z_um: float, nn: bool = False) -> np.ndarray:
    if nn:
        po = itk.ParameterObject.New()
        for i in range(tp.GetNumberOfParameterMaps()):
            pm = tp.GetParameterMap(i)
            pm['ResampleInterpolator'] = ['FinalNearestNeighborInterpolator']
            po.AddParameterMap(pm)
        tp_use = po
    else:
        tp_use = tp
    img = itk.GetImageFromArray(canvas.astype(np.float32))
    img.SetSpacing((DS_SPACING, DS_SPACING, float(z_um)))
    img.SetOrigin((0.0, 0.0, 0.0))
    return itk.GetArrayFromImage(itk.transformix_filter(img, tp_use)).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _save_outputs_for_fish(fish: int, ref_fish: int, name: str,
                            canvas_shape: Tuple[int, int, int],
                            pre_rot_deg: float, shift_vec: np.ndarray,
                            tp, z_um: int,
                            mov_ds_shape: Optional[Tuple] = None) -> None:
    out_dir = os.path.join(OUT_DIR, name, str(fish))
    os.makedirs(out_dir, exist_ok=True)
    is_ref  = (tp is None)

    # fluorescence channels
    for ch in range(NUM_CHANNELS):
        vol = load_channel_vol(fish, ch)
        if vol is None:
            logging.warning(f'    c{ch}: missing, skipping'); continue
        ds     = _ds_xy(vol, DOWNSAMPLE_XY); del vol
        canvas = apply_canvas_tf(ds, canvas_shape, pre_rot_deg, shift_vec); del ds
        out    = np.clip(canvas if is_ref else _apply_tp(canvas, tp, z_um), 0, None)
        tifffile.imwrite(os.path.join(out_dir, f'c{ch}.tif'), out.astype(np.float32))
        del canvas, out; gc.collect()
    logging.info(f'    fluorescence channels saved')

    # seg stacks — nearest-neighbour throughout (no interpolation of label boundaries)
    for seg_name in ('seg_cells', 'seg_nuclei'):
        vol = load_seg_vol(fish, seg_name)
        if vol is None:
            logging.warning(f'    {seg_name}: missing, skipping'); continue
        if vol.ndim == 4:   # (Z, H, W, 3) RGB seg
            out_chs = []
            for c in range(vol.shape[-1]):
                ch_ds  = _ds_xy(vol[..., c], DOWNSAMPLE_XY, nn=True)
                ch_can = apply_canvas_tf(ch_ds, canvas_shape, pre_rot_deg, shift_vec, nn=True)
                ch_out = ch_can if is_ref else _apply_tp(ch_can, tp, z_um, nn=True)
                out_chs.append(np.clip(ch_out, 0, 255).astype(np.uint8))
                del ch_ds, ch_can, ch_out
            result = np.stack(out_chs, axis=-1)
            tifffile.imwrite(os.path.join(out_dir, f'{seg_name}.tif'),
                             result, photometric='rgb')
        else:
            ds     = _ds_xy(vol, DOWNSAMPLE_XY, nn=True)
            canvas = apply_canvas_tf(ds, canvas_shape, pre_rot_deg, shift_vec, nn=True)
            result = canvas if is_ref else _apply_tp(canvas, tp, z_um, nn=True)
            tifffile.imwrite(os.path.join(out_dir, f'{seg_name}.tif'),
                             np.clip(result, 0, 255).astype(np.uint8))
            del ds, canvas
        del vol, result; gc.collect()
    logging.info(f'    seg stacks saved')

    # cell-type label map (uint8, 0=bg, 1–34=cell type) — NN throughout
    ctlabel = load_seg_vol(fish, 'cell_type_label')
    if ctlabel is not None:
        ds     = _ds_xy(ctlabel, DOWNSAMPLE_XY, nn=True); del ctlabel
        canvas = apply_canvas_tf(ds, canvas_shape, pre_rot_deg, shift_vec, nn=True); del ds
        result = canvas if is_ref else _apply_tp(canvas, tp, z_um, nn=True)
        tifffile.imwrite(os.path.join(out_dir, 'cell_type_label.tif'),
                         result.astype(np.uint8))
        del canvas, result; gc.collect()
    else:
        logging.warning(f'    cell_type_label_3d.tif missing for fish {fish}')

    # transform metadata
    tf_info: Dict = {
        'experiment': name, 'fish': fish, 'reference_fish': ref_fish,
        'pre_rotation_deg': float(pre_rot_deg),
        'canvas_shift': shift_vec.tolist() if hasattr(shift_vec, 'tolist') else list(shift_vec),
        'z_spacing_um': z_um, 'ds_xy': DOWNSAMPLE_XY,
        'identity': is_ref,
    }
    if mov_ds_shape is not None:
        tf_info['mov_ds_shape'] = list(mov_ds_shape)   # [Z, H, W] of DS driving vol before rotation
    if not is_ref:
        try:
            pm = tp.GetParameterMap(tp.GetNumberOfParameterMaps() - 1)
            try:
                corp = [float(v) for v in pm['CenterOfRotationPoint']]
            except (KeyError, RuntimeError):
                corp = [0.0, 0.0, 0.0]
            tf_info['elastix'] = {
                'Transform':              list(pm['Transform']),
                'TransformParameters':    [float(v) for v in pm['TransformParameters']],
                'CenterOfRotationPoint':  corp,   # ITK physical (x, y, z) in µm
            }
        except Exception:
            pass
    with open(os.path.join(out_dir, 'transform.json'), 'w') as fh:
        json.dump(tf_info, fh, indent=2)
    logging.info(f'    transform.json saved → {out_dir}')


# ══════════════════════════════════════════════════════════════════════════════
# STEP: REGISTER
# ══════════════════════════════════════════════════════════════════════════════

def step_register(fish_ids: List[int], ref_fish: int,
                  driving_opts: List[str], stages_opts: List[str],
                  z_spacing_opts: List[int]) -> None:
    logging.info('Determining common canvas ...')
    canvas_shape = determine_canvas(fish_ids)

    for driving, stages, z_um in product(driving_opts, stages_opts, z_spacing_opts):
        # cell_type_map driving uses discrete labels → Mutual Information metric.
        # dapi / dapi_blend are continuous images → NCC is correct.
        use_mi        = (driving == 'cell_type_map')
        rigid_params  = _build_rigid_params(use_mi=use_mi)
        affine_params = _build_affine_params(use_mi=use_mi)
        name = exp_name(driving, stages, z_um)
        logging.info(f'\n{"="*60}\nExperiment: {name}\n{"="*60}')

        logging.info(f'  Reference fish {ref_fish} ({driving}) ...')
        ref_drv_ds              = _ds_xy(load_driving_vol(ref_fish, driving), DOWNSAMPLE_XY)
        ref_drv_canvas, ref_shv = place_in_canvas(ref_drv_ds, canvas_shape)
        del ref_drv_ds
        ref_itk = _to_itk(ref_drv_canvas, z_um)
        _save_outputs_for_fish(ref_fish, ref_fish, name, canvas_shape,
                                0.0, ref_shv, None, z_um)

        for fish in fish_ids:
            if fish == ref_fish:
                continue
            logging.info(f'  Fish {fish} ...')
            mov_drv_ds = _ds_xy(load_driving_vol(fish, driving), DOWNSAMPLE_XY)
            mov_ds_shape = tuple(mov_drv_ds.shape)   # (Z, H, W) saved for script-5 inversion

            mov_drv_canvas, mov_shv = place_in_canvas(mov_drv_ds, canvas_shape)
            best_rot = 0.0
            del mov_drv_ds

            mov_itk = _to_itk(mov_drv_canvas, z_um); del mov_drv_canvas

            logging.info(f'    Rigid ...')
            _, rigid_tp = _register(ref_itk, mov_itk, rigid_params)

            if stages == 'rigid_affine':
                logging.info(f'    Affine ...')
                _, final_tp = _register(ref_itk, mov_itk, affine_params, init_tp=rigid_tp)
            else:
                final_tp = rigid_tp

            del mov_itk; gc.collect()
            _save_outputs_for_fish(fish, ref_fish, name, canvas_shape,
                                    best_rot, mov_shv, final_tp, z_um,
                                    mov_ds_shape=mov_ds_shape)
            gc.collect()

        del ref_drv_canvas, ref_itk; gc.collect()
        logging.info(f'  {name}: complete')


# ══════════════════════════════════════════════════════════════════════════════
# STEP: EVALUATE
# ══════════════════════════════════════════════════════════════════════════════

def _ncc_slices(a: np.ndarray, b: np.ndarray, min_fill: float = 0.02) -> float:
    """Mean per-Z-slice NCC."""
    scores = []
    for i in range(min(a.shape[0], b.shape[0])):
        ai = a[i].ravel().astype(np.float64)
        bi = b[i].ravel().astype(np.float64)
        if (ai > 0).mean() < min_fill or (bi > 0).mean() < min_fill:
            continue
        sa, sb = ai.std(), bi.std()
        if sa < 1e-6 or sb < 1e-6:
            continue
        scores.append(float(((ai - ai.mean()) * (bi - bi.mean())).mean() / (sa * sb)))
    return float(np.mean(scores)) if scores else float('nan')


def _dice_slices(a: np.ndarray, b: np.ndarray, thresh: float = 0.5) -> float:
    """Mean per-Z-slice Dice between binarised volumes."""
    scores = []
    for i in range(min(a.shape[0], b.shape[0])):
        ai = (a[i] > thresh).ravel()
        bi = (b[i] > thresh).ravel()
        denom = ai.sum() + bi.sum()
        if denom == 0:
            continue
        scores.append(float(2 * (ai & bi).sum() / denom))
    return float(np.mean(scores)) if scores else float('nan')


def _seg_to_gray(vol: np.ndarray) -> np.ndarray:
    """Collapse RGB seg stack to single-channel float by max projection over colour axis."""
    if vol.ndim == 4:   # (Z, H, W, 3)
        return vol.max(axis=-1).astype(np.float32)
    return vol.astype(np.float32)


def _load_output(name: str, fish: int, fname: str) -> Optional[np.ndarray]:
    p = os.path.join(OUT_DIR, name, str(fish), fname)
    return tifffile.imread(p) if os.path.exists(p) else None


def _per_fish_metrics(name: str, fish: int, ref_fish: int) -> Dict[str, float]:
    """All evaluation metrics for one fish vs reference in one experiment."""
    metrics: Dict[str, float] = {}

    # fluorescence channels c0-c3: NCC
    for ch in EVAL_CHANNELS:
        fname = f'c{ch}.tif'
        ref = _load_output(name, ref_fish, fname)
        mov = _load_output(name, fish,     fname)
        if ref is None or mov is None:
            metrics[f'ncc_c{ch}'] = float('nan')
        else:
            metrics[f'ncc_c{ch}'] = _ncc_slices(mov.astype(np.float32),
                                                  ref.astype(np.float32))
        del ref, mov

    # seg stacks: NCC + Dice on grayscale-collapsed image
    for seg_name in EVAL_SEGS:
        ref_raw = _load_output(name, ref_fish, f'{seg_name}.tif')
        mov_raw = _load_output(name, fish,     f'{seg_name}.tif')
        if ref_raw is None or mov_raw is None:
            metrics[f'ncc_{seg_name}']  = float('nan')
            metrics[f'dice_{seg_name}'] = float('nan')
        else:
            ref_g = _seg_to_gray(ref_raw); mov_g = _seg_to_gray(mov_raw)
            metrics[f'ncc_{seg_name}']  = _ncc_slices(mov_g, ref_g)
            metrics[f'dice_{seg_name}'] = _dice_slices(mov_g, ref_g)
        del ref_raw, mov_raw

    # cell-type label map: NCC + Dice (multi-label, NN-interpolated output)
    ref_m = _load_output(name, ref_fish, 'cell_type_label.tif')
    mov_m = _load_output(name, fish,     'cell_type_label.tif')
    if ref_m is None or mov_m is None:
        metrics['ncc_tissue']  = float('nan')
        metrics['dice_tissue'] = float('nan')
    else:
        rf = ref_m.astype(np.float32); mf = mov_m.astype(np.float32)
        metrics['ncc_tissue']  = _ncc_slices(mf, rf)
        metrics['dice_tissue'] = _dice_slices(mf, rf)
    del ref_m, mov_m

    gc.collect()
    return metrics


# All metric keys in a fixed order
def _metric_keys() -> List[str]:
    keys = [f'ncc_c{ch}' for ch in EVAL_CHANNELS]
    for sn in EVAL_SEGS:
        keys += [f'ncc_{sn}', f'dice_{sn}']
    keys += ['ncc_tissue', 'dice_tissue']
    return keys


def step_evaluate(fish_ids: List[int], ref_fish: int,
                  experiments: List[str]) -> None:
    mkeys = _metric_keys()

    # per-fish rows (for per-fish CSV)
    pf_rows = []
    # summary rows (for summary CSV + printed table)
    sum_rows = []

    for name in experiments:
        logging.info(f'  Evaluating {name} ...')

        # check reference exists
        if not os.path.exists(os.path.join(OUT_DIR, name, str(ref_fish), 'c0.tif')):
            logging.warning(f'    reference outputs missing, skipping'); continue

        fish_metrics: Dict[int, Dict[str, float]] = {}
        for fish in fish_ids:
            if fish == ref_fish:
                continue
            if not os.path.exists(os.path.join(OUT_DIR, name, str(fish), 'c0.tif')):
                logging.warning(f'    fish {fish}: outputs missing, skipping'); continue
            fm = _per_fish_metrics(name, fish, ref_fish)
            fish_metrics[fish] = fm
            pf_rows.append({'experiment': name, 'fish': fish, **fm})
            logging.info(f'    fish {fish}: ' +
                         '  '.join(f'{k}={fm[k]:.3f}' for k in mkeys if not math.isnan(fm.get(k, float('nan')))))

        if not fish_metrics:
            continue

        # aggregate: mean per group
        wt_fish     = [f for f in fish_ids if f in WT_FISH     and f != ref_fish and f in fish_metrics]
        mutant_fish = [f for f in fish_ids if f in MUTANT_FISH and f != ref_fish and f in fish_metrics]
        all_fish    = list(fish_metrics.keys())

        def _grp_mean(fish_list, key):
            vals = [fish_metrics[f][key] for f in fish_list if not math.isnan(fish_metrics[f].get(key, float('nan')))]
            return float(np.mean(vals)) if vals else float('nan')

        srow = {'experiment': name}
        for k in mkeys:
            srow[f'all_{k}']    = _grp_mean(all_fish,    k)
            srow[f'wt_{k}']     = _grp_mean(wt_fish,     k)
            srow[f'mutant_{k}'] = _grp_mean(mutant_fish, k)
        sum_rows.append(srow)

    # ── Save CSVs ──────────────────────────────────────────────────────────────
    if pf_rows:
        pf_path = os.path.join(OUT_DIR, 'evaluation_per_fish.csv')
        with open(pf_path, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(pf_rows[0].keys()))
            w.writeheader(); w.writerows(pf_rows)
        logging.info(f'Per-fish CSV saved → {pf_path}')

    if sum_rows:
        sum_path = os.path.join(OUT_DIR, 'evaluation.csv')
        with open(sum_path, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(sum_rows[0].keys()))
            w.writeheader(); w.writerows(sum_rows)
        logging.info(f'Summary CSV saved → {sum_path}')

    # ── Print summary table ────────────────────────────────────────────────────
    if not sum_rows:
        logging.error('No evaluation results.'); return

    def _f(v):
        return f'{v:.4f}' if (v is not None and not math.isnan(v)) else ' N/A '

    # Find best per column for annotation
    def _best(col):
        vals = [r[col] for r in sum_rows if col in r and not math.isnan(r[col])]
        return max(vals) if vals else None

    # Print one table per metric group: channels, segs, mask
    metric_groups = [
        ('DAPI NCC (c0)',        ['ncc_c0']),
        ('Channel NCC',          [f'ncc_c{c}' for c in EVAL_CHANNELS]),
        ('Seg NCC + Dice',       [f'{t}_{s}' for s in EVAL_SEGS for t in ['ncc', 'dice']]),
        ('Tissue NCC + Dice',    ['ncc_tissue', 'dice_tissue']),
    ]

    print(f'\n{"═"*90}')
    print(f'  Cross-fish 3D registration evaluation  (reference fish {ref_fish})')
    print(f'  Metrics are mean over all non-reference fish (separate WT / mutant rows below)')
    print(f'{"═"*90}')

    for group_title, group_keys in metric_groups:
        col_labels = [f'all_{k}' for k in group_keys]
        bests      = {cl: _best(cl) for cl in col_labels}
        hdr = f'  {"experiment":<36}  ' + '  '.join(f'{k:<10}' for k in group_keys)
        print(f'\n  ── {group_title} ──')
        print(hdr)
        print('  ' + '-' * (len(hdr) - 2))
        for r in sum_rows:
            vals = '  '.join(
                f'{_f(r.get(cl, float("nan")))}{" ◀" if r.get(cl) == bests[cl] else "  "}'
                for cl in col_labels
            )
            print(f'  {r["experiment"]:<36}  {vals}')

    print(f'\n{"═"*90}')
    print(f'  ◀ = best across experiments for that metric')
    print(f'  NCC: higher = better alignment   Dice: higher = better overlap')
    print(f'  Channels: c0=DAPI c1=other fluorescence (independent of driving image)')
    print(f'{"═"*90}\n')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description='3D cross-fish registration experiments (script 4).'
    )
    parser.add_argument('--steps', nargs='+', choices=STEPS, default=STEPS,
                        help='Steps to run (default: all)')
    parser.add_argument('--driving', nargs='+', choices=DRIVING_OPTIONS,
                        default=DRIVING_OPTIONS, metavar='DRV')
    parser.add_argument('--stages', nargs='+', choices=STAGES_OPTIONS,
                        default=STAGES_OPTIONS, metavar='STG')
    parser.add_argument('--z-spacing', nargs='+', type=int, choices=Z_SPACING_OPTIONS,
                        default=Z_SPACING_OPTIONS, dest='z_spacing', metavar='UM')
    parser.add_argument('--fish', nargs='+', type=int, default=None)
    parser.add_argument('--reference', type=int, default=1)
    args = parser.parse_args()

    fish_ids = sorted(args.fish if args.fish else WT_FISH + MUTANT_FISH)
    steps    = set(args.steps)
    exps     = all_experiments(args.driving, args.stages, args.z_spacing)

    os.makedirs(OUT_DIR, exist_ok=True)
    logging.info(f'Fish: {fish_ids}   reference: {args.reference}')
    logging.info(f'Steps: {args.steps}')
    logging.info(f'Experiments ({len(exps)}): {exps}')

    if 'register' in steps:
        step_register(fish_ids, args.reference, args.driving, args.stages, args.z_spacing)

    if 'evaluate' in steps:
        step_evaluate(fish_ids, args.reference, exps)

    logging.info('Done.')


if __name__ == '__main__':
    main()
