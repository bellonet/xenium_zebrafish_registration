"""
3_registration_experiments.py

Second-pass rigid registration experiments, starting from script 2 rigid output.
Tries different registration images to refine the existing DAPI-driven alignment,
then evaluates each experiment quantitatively.

Experiments
-----------
  tissue_mask    binary cell-presence mask (any cell = 1)
  tp63_only      tp63 gene density at sigma=3 µm  (skin-boundary ring/arc)
  gene_composite tp63 + myod1 + tbxta + sox3 at sigma=3 µm
  dapi_blend     0.5 × DAPI + 0.5 × tissue_mask
  multi_metric   0.6 × tissue_mask + 0.4 × gene_composite

Steps
-----
  register   Run second-pass registration for each experiment and save outputs.
  evaluate   Evaluate all experiments quantitatively and save a CSV per fish.

Input:  analysis/2_registered/{fish}/
Output: analysis/3_registered/{experiment}/{fish}/c0/{gnum}.tif
                                               c0_3d.tif
                                               correction_transforms.json
                                               transforms/{gnum}.txt
        analysis/3_registered/evaluation_fish{N}.csv

Usage
-----
  python 3_registration_experiments.py --fish 1
  python 3_registration_experiments.py --fish 1 --experiments dapi_blend multi_metric
  python 3_registration_experiments.py --steps evaluate --fish 1
  python 3_registration_experiments.py --steps register evaluate --fish 1 --experiments dapi_blend
  python 3_registration_experiments.py          # all steps, all experiments, all fish
"""

import csv, gc, glob, json, math, argparse, logging, os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile
import itk
import scipy.ndimage as _snd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ── paths ──────────────────────────────────────────────────────────────────────
IN2_DIR      = '../analysis/2_registered'
OUT_DIR      = '../analysis/3_registered'
DATA_DIR     = '../data'
SUMMARY_GLOB = '../analysis/fish_bbox_summary_tagged_*.csv'

# ── shared constants (must match script 2) ─────────────────────────────────────
XENIUM_PX_PER_UM = 4.705882
MIN_ANGLE_DEG    = 1.0
RUN_FOLDERS = {
    '2': 'output-XETG00046__0038328__Region_1__20250717__075022',
    '4': 'output-XETG00046__0043921__Region_1__20250620__084504',
    '5': 'output-XETG00046__0044004__Region_1__20250620__084505',
}

STEPS       = ['register', 'evaluate']
EXPERIMENTS = ['tissue_mask', 'tp63_only', 'gene_composite', 'dapi_blend', 'multi_metric']

# Structural genes used in composite / tp63_only
COMPOSITE_GENES    = ['tp63', 'myod1', 'tbxta', 'sox3']
COMPOSITE_SIGMA_UM = 3.0
QV_MIN             = 20.0

# ── elastix params ─────────────────────────────────────────────────────────────
N_RESOLUTIONS = 3
MAX_ITER      = 200
NUM_SAMPLES   = 6_000
NUM_THREADS   = 8
PYRAMID_SCHED = ['8', '8', '4', '4', '2', '2']


# ══════════════════════════════════════════════════════════════════════════════
# STEP: REGISTER
# ══════════════════════════════════════════════════════════════════════════════

# ── elastix helpers ────────────────────────────────────────────────────────────

def _set_props(img: itk.Image) -> None:
    img.SetSpacing((1.0, 1.0))
    img.SetOrigin((0.0, 0.0))


def build_param_obj() -> itk.ParameterObject:
    """NCC-based rigid parameter map for fine-tuning already-aligned images."""
    po = itk.ParameterObject.New()
    pm = po.GetDefaultParameterMap('rigid', N_RESOLUTIONS)
    pm['Transform']                              = ['EulerTransform']
    pm['Metric']                                 = ['AdvancedNormalizedCorrelation']
    pm['Optimizer']                              = ['AdaptiveStochasticGradientDescent']
    pm['AutomaticParameterEstimation']           = ['true']
    pm['AutomaticTransformInitialization']       = ['true']
    pm['AutomaticTransformInitializationMethod'] = ['GeometricalCenter']
    pm['NumberOfThreads']                        = [str(NUM_THREADS)]
    pm['ImageSampler']                           = ['RandomCoordinate']
    pm['NumberOfSpatialSamples']                 = [str(NUM_SAMPLES)]
    pm['MaximumNumberOfSamplingAttempts']        = ['200']
    pm['NewSamplesEveryIteration']               = ['true']
    pm['ImagePyramidSchedule']                   = PYRAMID_SCHED
    pm['MaximumNumberOfIterations']              = [str(MAX_ITER)] * N_RESOLUTIONS
    pm['BSplineInterpolationOrder']              = ['1']
    pm['FinalBSplineInterpolationOrder']         = ['1']
    pm['ResultImagePixelType']                   = ['float']
    po.AddParameterMap(pm)
    return po


def register_pair(fixed: np.ndarray, moving: np.ndarray,
                  pm: itk.ParameterObject) -> Tuple[np.ndarray, object]:
    f = itk.GetImageFromArray(fixed.astype(np.float32));  _set_props(f)
    m = itk.GetImageFromArray(moving.astype(np.float32)); _set_props(m)
    result, tp = itk.elastix_registration_method(
        f, m, parameter_object=pm, log_to_console=False, log_to_file=False
    )
    return itk.GetArrayFromImage(result).astype(np.float32), tp


def apply_transform_to(img: np.ndarray, tp: object) -> np.ndarray:
    i = itk.GetImageFromArray(img.astype(np.float32)); _set_props(i)
    return itk.GetArrayFromImage(itk.transformix_filter(i, tp)).astype(np.float32)


# ── image loading ──────────────────────────────────────────────────────────────

def load_dapi(fish: int, gnum: int) -> Optional[np.ndarray]:
    p = os.path.join(IN2_DIR, str(fish), 'c0', f'{gnum}.tif')
    return tifffile.imread(p).astype(np.float32) if os.path.exists(p) else None


def load_tissue_mask(fish: int, gnum: int) -> Optional[np.ndarray]:
    p = os.path.join(IN2_DIR, str(fish), 'tissue_mask', f'{gnum}.tif')
    return (tifffile.imread(p) > 0).astype(np.float32) if os.path.exists(p) else None


# ── gene composite rendering ───────────────────────────────────────────────────

def _load_filtered_transcripts(run: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, RUN_FOLDERS[run], 'transcripts.parquet')
    df = pd.read_parquet(
        path, columns=['feature_name', 'x_location', 'y_location', 'qv', 'is_gene']
    )
    return df[(df['is_gene'] == True) & (df['qv'] >= QV_MIN) &
              df['feature_name'].isin(set(COMPOSITE_GENES))].reset_index(drop=True)


def _map_to_canvas(tx: pd.DataFrame, r0: float, c0: float,
                   sm: Dict, fish_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    S  = XENIUM_PX_PER_UM
    x  = tx['x_location'].values * S - c0 + sm['pad_left']
    y  = tx['y_location'].values * S - r0 + sm['pad_top']
    xy = np.column_stack([x, y])

    if abs(sm.get('dv_angle_deg', 0.0)) >= MIN_ANGLE_DEG:
        a            = math.radians(sm['dv_angle_deg'])
        cx, cy       = sm['canvas_w'] / 2.0, sm['canvas_h'] / 2.0
        dx, dy       = xy[:, 0] - cx, xy[:, 1] - cy
        cos_a, sin_a = math.cos(a), math.sin(a)
        xy = np.column_stack([cx + cos_a*dx + sin_a*dy,
                               cy - sin_a*dx + cos_a*dy])

    if sm.get('elastix_file'):
        tf_path = os.path.join(fish_dir, sm['elastix_file'])
        if os.path.exists(tf_path):
            params: Dict = {}
            with open(tf_path) as fh:
                for line in fh:
                    line = line.strip()
                    if line.startswith('(') and line.endswith(')'):
                        parts = line[1:-1].split()
                        params[parts[0]] = parts[1:]
            tp_v              = [float(v) for v in params['TransformParameters']]
            corp              = [float(v) for v in params['CenterOfRotationPoint']]
            angle, tx_t, ty_t = tp_v[0], tp_v[1], tp_v[2]
            cx_e, cy_e        = corp[0], corp[1]
            cos_a, sin_a      = math.cos(angle), math.sin(angle)
            dx = xy[:, 0] - cx_e - tx_t
            dy = xy[:, 1] - cy_e - ty_t
            xy = np.column_stack([cx_e + cos_a*dx + sin_a*dy,
                                   cy_e - sin_a*dx + cos_a*dy])

    valid = ((xy[:, 0] >= 0) & (xy[:, 0] < sm['canvas_w']) &
             (xy[:, 1] >= 0) & (xy[:, 1] < sm['canvas_h']))
    return xy, valid


def build_gene_cache(fish: int, summary_df: pd.DataFrame,
                     run_tx: Dict[str, pd.DataFrame],
                     slices_meta: Dict) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    fish_dir  = os.path.join(IN2_DIR, str(fish))
    fish_rows = summary_df[summary_df['fish_name'] == fish]
    cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for gnum_str, sm in slices_meta.items():
        gnum = int(gnum_str)
        rows = fish_rows[fish_rows['global_slice_num'] == gnum]
        if len(rows) == 0:
            continue
        row = rows.iloc[0]
        run = str(int(row['run']))
        if run not in run_tx:
            continue
        S  = XENIUM_PX_PER_UM
        r0 = row['bbox_global_min_row']; c0 = row['bbox_global_min_col']
        r1 = row['bbox_global_max_row']; c1 = row['bbox_global_max_col']
        tx = run_tx[run]
        sub = tx[(tx['y_location'] * S >= r0) & (tx['y_location'] * S <= r1) &
                 (tx['x_location'] * S >= c0) & (tx['x_location'] * S <= c1)]
        if len(sub) < 5:
            continue
        xy, valid = _map_to_canvas(sub, r0, c0, sm, fish_dir)
        if valid.any():
            cache[gnum] = (xy[valid], sub['feature_name'].values[valid])
    return cache


def render_gene_composite(gnum: int, genes: List[str], sigma_um: float,
                           cache: Dict, canvas_h: int, canvas_w: int) -> Optional[np.ndarray]:
    if gnum not in cache:
        return None
    xy, gs    = cache[gnum]
    sigma_px  = sigma_um * XENIUM_PX_PER_UM
    composite = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    any_gene  = False
    for gene in genes:
        mask = gs == gene
        if not mask.any():
            continue
        img  = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        cols = np.clip(xy[mask, 0].astype(np.int32), 0, canvas_w - 1)
        rows = np.clip(xy[mask, 1].astype(np.int32), 0, canvas_h - 1)
        np.add.at(img, (rows, cols), 1)
        _snd.gaussian_filter(img, sigma=sigma_px, output=img, mode='constant')
        mx = img.max()
        if mx > 0:
            img /= mx; composite += img; any_gene = True
    if not any_gene:
        return None
    mx = composite.max()
    return composite / mx if mx > 0 else None


def get_reg_image(experiment: str, fish: int, gnum: int,
                  gene_cache: Optional[Dict],
                  canvas_h: int, canvas_w: int) -> Optional[np.ndarray]:
    if experiment == 'tissue_mask':
        return load_tissue_mask(fish, gnum)
    elif experiment == 'tp63_only':
        return (None if gene_cache is None else
                render_gene_composite(gnum, ['tp63'], COMPOSITE_SIGMA_UM,
                                      gene_cache, canvas_h, canvas_w))
    elif experiment == 'gene_composite':
        return (None if gene_cache is None else
                render_gene_composite(gnum, COMPOSITE_GENES, COMPOSITE_SIGMA_UM,
                                      gene_cache, canvas_h, canvas_w))
    elif experiment == 'dapi_blend':
        dapi = load_dapi(fish, gnum)
        mask = load_tissue_mask(fish, gnum)
        if dapi is None or mask is None:
            return None
        mx = dapi.max()
        return 0.5 * (dapi / mx if mx > 0 else dapi) + 0.5 * mask
    elif experiment == 'multi_metric':
        if gene_cache is None:
            return None
        mask = load_tissue_mask(fish, gnum)
        comp = render_gene_composite(gnum, COMPOSITE_GENES, COMPOSITE_SIGMA_UM,
                                     gene_cache, canvas_h, canvas_w)
        if mask is None or comp is None:
            return None
        return 0.6 * mask + 0.4 * comp
    return None


# ── second-pass registration ───────────────────────────────────────────────────

def _extract_tp_params(tp: object) -> Optional[Dict]:
    try:
        pm     = tp.GetParameterMap(0)
        vals   = [float(v) for v in pm['TransformParameters']]
        center = [float(v) for v in pm['CenterOfRotationPoint']]
        return {'angle_deg': math.degrees(vals[0]), 'tx': vals[1], 'ty': vals[2],
                'center': center}
    except Exception:
        return None


def run_second_pass(fish: int, gnums: List[int],
                    reg_imgs: Dict[int, np.ndarray],
                    dapi_slices: Dict[int, np.ndarray],
                    canvas_h: int, canvas_w: int,
                    tf_dir: Optional[str] = None,
                    ) -> Tuple[Dict[int, np.ndarray], Dict[int, Optional[Dict]]]:
    """Propagate second-pass corrections outward from middle reference.
    Returns (corrected DAPI dict, correction_params dict).
    Each correction is the DELTA on top of script-2's transform.
    """
    pm   = build_param_obj()
    zero = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    valid_gnums = sorted(g for g in gnums if reg_imgs.get(g) is not None
                         and reg_imgs[g].any())
    if not valid_gnums:
        logging.warning('  No valid registration images — returning uncorrected DAPI')
        return dict(dapi_slices), {}

    ref_gnum = valid_gnums[len(valid_gnums) // 2]
    logging.info(f'  Reference slice: {ref_gnum}')

    corrected:         Dict[int, np.ndarray]     = {}
    correction_params: Dict[int, Optional[Dict]] = {}
    corrected[ref_gnum]         = dapi_slices.get(ref_gnum, zero)
    correction_params[ref_gnum] = None

    for direction in (-1, 1):
        seq = (sorted([g for g in gnums if g < ref_gnum], reverse=True)
               if direction == -1
               else sorted([g for g in gnums if g > ref_gnum]))
        prev_gnum = ref_gnum

        for i, gnum in enumerate(seq):
            fixed_img  = reg_imgs.get(prev_gnum)
            moving_img = reg_imgs.get(gnum)

            if fixed_img is None or moving_img is None or not fixed_img.any():
                corrected[gnum]         = dapi_slices.get(gnum, zero)
                correction_params[gnum] = None
                prev_gnum = gnum
                continue

            try:
                _, tp = register_pair(fixed_img, moving_img, pm)
                corrected[gnum]         = apply_transform_to(dapi_slices.get(gnum, zero), tp)
                correction_params[gnum] = _extract_tp_params(tp)
                if tf_dir is not None:
                    try:
                        tp.WriteParameterFile(os.path.join(tf_dir, f'{gnum}.txt'))
                    except Exception:
                        pass
                if (i + 1) % 10 == 0:
                    logging.info(f'    ... {i+1}/{len(seq)} slices done')
            except Exception as exc:
                logging.warning(f'    gnum {gnum}: failed ({exc}), using uncorrected')
                corrected[gnum]         = dapi_slices.get(gnum, zero)
                correction_params[gnum] = None

            prev_gnum = gnum

    for gnum in gnums:
        if gnum not in corrected:
            corrected[gnum]         = dapi_slices.get(gnum, zero)
        if gnum not in correction_params:
            correction_params[gnum] = None

    return corrected, correction_params


def _run_one_experiment(experiment: str, fish: int,
                        summary_df: pd.DataFrame,
                        run_tx: Optional[Dict[str, pd.DataFrame]]) -> None:
    needs_genes  = experiment in ('tp63_only', 'gene_composite', 'multi_metric')
    out_fish_dir = os.path.join(OUT_DIR, experiment, str(fish))
    out_slices_dir = os.path.join(out_fish_dir, 'c0')
    tf_dir         = os.path.join(out_fish_dir, 'transforms')
    os.makedirs(out_slices_dir, exist_ok=True)
    os.makedirs(tf_dir, exist_ok=True)

    meta_path = os.path.join(IN2_DIR, str(fish), 'transforms.json')
    if not os.path.exists(meta_path):
        logging.error(f'  transforms.json missing — run script 2 first'); return
    with open(meta_path) as fh:
        meta = json.load(fh)
    canvas_h    = meta['canvas_h']
    canvas_w    = meta['canvas_w']
    slices_meta = meta['slices']
    for sm in slices_meta.values():
        sm['canvas_h'] = canvas_h; sm['canvas_w'] = canvas_w
    gnums = sorted(int(k) for k in slices_meta)

    gene_cache = None
    if needs_genes and run_tx is not None:
        logging.info(f'  Building gene cache...')
        gene_cache = build_gene_cache(fish, summary_df, run_tx, slices_meta)
        logging.info(f'  Gene cache: {len(gene_cache)} slices with data')

    reg_imgs: Dict[int, np.ndarray]    = {}
    dapi_slices: Dict[int, np.ndarray] = {}
    n_none = 0
    for gnum in gnums:
        img = get_reg_image(experiment, fish, gnum, gene_cache, canvas_h, canvas_w)
        if img is not None:
            reg_imgs[gnum] = img
        else:
            n_none += 1
        dapi = load_dapi(fish, gnum)
        if dapi is not None:
            dapi_slices[gnum] = dapi
    logging.info(f'  {len(reg_imgs)} reg images ({n_none} empty), {len(dapi_slices)} DAPI slices')

    corrected, correction_params = run_second_pass(
        fish, gnums, reg_imgs, dapi_slices, canvas_h, canvas_w, tf_dir=tf_dir
    )

    # Save correction transforms JSON
    tf_json_path = os.path.join(out_fish_dir, 'correction_transforms.json')
    with open(tf_json_path, 'w') as fh:
        json.dump({
            'experiment': experiment, 'fish': fish,
            'canvas_h': canvas_h, 'canvas_w': canvas_w,
            'note': 'Corrections are DELTAS on top of script-2 rigid transforms.',
            'corrections': {str(k): v for k, v in correction_params.items()},
        }, fh, indent=2)

    # Log correction magnitudes
    actual = [v for v in correction_params.values() if v is not None]
    if actual:
        angles = np.array([v['angle_deg'] for v in actual])
        txs    = np.array([v['tx']        for v in actual])
        tys    = np.array([v['ty']        for v in actual])
        logging.info(
            f'  Corrections ({len(actual)} slices): '
            f'angle mean|Δ|={np.mean(np.abs(angles)):.3f}° max={np.max(np.abs(angles)):.3f}°  '
            f'tx mean|Δ|={np.mean(np.abs(txs)):.1f}px  ty mean|Δ|={np.mean(np.abs(tys)):.1f}px'
        )

    # Write per-slice DAPI (frame N = global_num N, matching script 2)
    last = max(dapi_slices) if dapi_slices else gnums[-1]
    zero = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    for gnum in range(1, last + 1):
        tifffile.imwrite(os.path.join(out_slices_dir, f'{gnum}.tif'),
                         corrected.get(gnum, zero), photometric='minisblack')

    # Build 3D DAPI stack
    files = sorted(glob.glob(os.path.join(out_slices_dir, '*.tif')),
                   key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))
    if files:
        frames = [tifffile.imread(f).astype(np.float32) for f in files]
        vol    = np.stack(frames, axis=0)
        stack_path = os.path.join(out_fish_dir, 'c0_3d.tif')
        tifffile.imwrite(stack_path, vol)
        gnums_out = [int(os.path.splitext(os.path.basename(f))[0]) for f in files]
        logging.info(f'  Stack {vol.shape}  slices {gnums_out[0]}–{gnums_out[-1]}  → {stack_path}')
        del frames, vol

    del reg_imgs, dapi_slices, corrected, correction_params
    if gene_cache is not None:
        del gene_cache
    gc.collect()


def step_register(fish_ids: List[int], experiments: List[str],
                  summary_df: pd.DataFrame) -> None:
    needs_genes = any(e in experiments for e in ('tp63_only', 'gene_composite', 'multi_metric'))
    run_tx: Optional[Dict[str, pd.DataFrame]] = None
    if needs_genes:
        needed_runs = summary_df[summary_df['fish_name'].isin(fish_ids)]['run'].unique()
        run_tx = {}
        for run in needed_runs:
            run = str(int(float(run)))
            if run not in RUN_FOLDERS:
                continue
            logging.info(f'Loading run {run} transcripts...')
            run_tx[run] = _load_filtered_transcripts(run)
            logging.info(f'  {len(run_tx[run]):,} transcripts')

    for experiment in experiments:
        logging.info(f'\n{"="*60}\nExperiment: {experiment}\n{"="*60}')
        for fish in fish_ids:
            logging.info(f'=== fish {fish} ===')
            _run_one_experiment(experiment, fish, summary_df, run_tx)


# ══════════════════════════════════════════════════════════════════════════════
# STEP: EVALUATE
# ══════════════════════════════════════════════════════════════════════════════

def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    sa, sb = a.std(), b.std()
    if sa < 1e-6 or sb < 1e-6:
        return np.nan
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def _load_slices(slices_dir: str) -> Optional[np.ndarray]:
    files = sorted(glob.glob(os.path.join(slices_dir, '*.tif')),
                   key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))
    if not files:
        return None
    return np.stack([tifffile.imread(f).astype(np.float32) for f in files], axis=0)


def _mean_ncc_adj(vol: np.ndarray, min_nonzero: float = 0.02) -> float:
    scores = []
    for i in range(len(vol) - 1):
        a, b = vol[i], vol[i + 1]
        if (a > 0).mean() < min_nonzero or (b > 0).mean() < min_nonzero:
            continue
        v = _ncc(a, b)
        if not np.isnan(v):
            scores.append(v)
    return float(np.mean(scores)) if scores else np.nan


def _z_sharpness(vol: np.ndarray) -> float:
    proj = vol.max(axis=0)
    return float(np.sqrt(_snd.sobel(proj, axis=1)**2 + _snd.sobel(proj, axis=0)**2).mean())


def _apply_correction(arr: np.ndarray, tf_txt: str) -> np.ndarray:
    tp = itk.ParameterObject.New()
    tp.ReadParameterFile(tf_txt)
    img = itk.GetImageFromArray(arr.astype(np.float32))
    img.SetSpacing((1.0, 1.0)); img.SetOrigin((0.0, 0.0))
    return itk.GetArrayFromImage(itk.transformix_filter(img, tp)).astype(np.float32)


def _load_corrected_tissue(tf_dir: str, baseline_tissue_dir: str) -> Optional[np.ndarray]:
    """Load script-2 tissue_mask slices and apply this experiment's corrections."""
    files = sorted(glob.glob(os.path.join(baseline_tissue_dir, '*.tif')),
                   key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))
    if not files:
        return None
    frames = []
    for f in files:
        gnum = int(os.path.splitext(os.path.basename(f))[0])
        arr  = tifffile.imread(f).astype(np.float32)
        tf   = os.path.join(tf_dir, f'{gnum}.txt')
        if os.path.exists(tf):
            try:
                arr = _apply_correction(arr, tf)
            except Exception as e:
                logging.debug(f'    gnum {gnum}: transform failed ({e})')
        frames.append(arr)
    return np.stack(frames, axis=0)


def _tf_smoothness(tf_json_path: str) -> Tuple[float, float]:
    with open(tf_json_path) as fh:
        data = json.load(fh)
    corr  = data['corrections']
    gnums = sorted(int(k) for k in corr)
    txs   = [0.0 if corr[str(g)] is None else corr[str(g)]['tx']        for g in gnums]
    tys   = [0.0 if corr[str(g)] is None else corr[str(g)]['ty']        for g in gnums]
    angs  = [0.0 if corr[str(g)] is None else corr[str(g)]['angle_deg'] for g in gnums]
    dtrans = np.sqrt(np.diff(txs)**2 + np.diff(tys)**2)
    return float(dtrans.mean()), float(np.abs(np.diff(angs)).mean())


def _eval_one(name: str, dapi_dir: str, tissue_dir: Optional[str],
              tf_json: Optional[str]) -> Dict:
    r: Dict = {'experiment': name}
    vol = _load_slices(dapi_dir)
    if vol is not None:
        r['ncc_dapi']     = _mean_ncc_adj(vol)
        r['z_sharp_dapi'] = _z_sharpness(vol)
        del vol
    else:
        r['ncc_dapi'] = r['z_sharp_dapi'] = None
        logging.warning(f'  {name}: DAPI slices missing')
    if tissue_dir is not None:
        tmask = _load_slices(tissue_dir)
        r['ncc_tissue'] = _mean_ncc_adj(tmask) if tmask is not None else None
        if tmask is not None:
            del tmask
    else:
        r['ncc_tissue'] = None
    if tf_json and os.path.exists(tf_json):
        r['smooth_px'], r['smooth_deg'] = _tf_smoothness(tf_json)
    else:
        r['smooth_px'] = r['smooth_deg'] = None
    return r


def step_evaluate(fish_ids: List[int], experiments: List[str]) -> None:
    for fish in fish_ids:
        logging.info(f'=== Evaluating fish {fish} ===')
        rows = []

        # Baseline
        logging.info('  Baseline (script 2 rigid)...')
        rows.append(_eval_one(
            name       = 'script2_baseline',
            dapi_dir   = os.path.join(IN2_DIR, str(fish), 'c0'),
            tissue_dir = os.path.join(IN2_DIR, str(fish), 'tissue_mask'),
            tf_json    = None,
        ))

        # Experiments — apply correction transforms to tissue_mask for fair ncc_tissue
        baseline_tissue_dir = os.path.join(IN2_DIR, str(fish), 'tissue_mask')
        for exp in experiments:
            logging.info(f'  {exp}...')
            tf_dir  = os.path.join(OUT_DIR, exp, str(fish), 'transforms')
            tf_json = os.path.join(OUT_DIR, exp, str(fish), 'correction_transforms.json')
            r = _eval_one(
                name       = exp,
                dapi_dir   = os.path.join(OUT_DIR, exp, str(fish), 'c0'),
                tissue_dir = None,
                tf_json    = tf_json,
            )
            if os.path.isdir(tf_dir):
                logging.info(f'    applying correction transforms to tissue_mask...')
                tmask = _load_corrected_tissue(tf_dir, baseline_tissue_dir)
                r['ncc_tissue'] = _mean_ncc_adj(tmask) if tmask is not None else None
                if tmask is not None:
                    del tmask
            else:
                r['ncc_tissue'] = None
                logging.warning(f'  {exp}: transforms dir missing')
            rows.append(r)

        # Print table
        def _f(v, fmt):
            return format(v, fmt) if (v is not None and
                                       not (isinstance(v, float) and np.isnan(v))) else 'N/A'
        metrics = ['ncc_dapi', 'ncc_tissue', 'smooth_px', 'z_sharp_dapi']
        best = {}
        for m in metrics:
            vals = [r[m] for r in rows if r.get(m) is not None and not np.isnan(r[m])]
            if vals:
                best[m] = min(vals) if m == 'smooth_px' else max(vals)

        W = 90
        print(f'\n{"═"*W}')
        print(f'  Registration experiment evaluation — fish {fish}')
        print(f'{"═"*W}')
        print(f'  {"experiment":<22}  {"ncc_dapi":>9}  {"ncc_tissue":>11}  {"smooth_px":>10}  {"z_sharp_dapi":>13}')
        print(f'  {"-"*22}  {"-"*9}  {"-"*11}  {"-"*10}  {"-"*13}')
        for r in rows:
            def mark(m):
                v = r.get(m)
                return ' ◀' if (v is not None and not np.isnan(v) and v == best.get(m)) else ''
            print(
                f'  {r["experiment"]:<22}  '
                f'{_f(r["ncc_dapi"],    ".4f"):>9}{mark("ncc_dapi"):<2}  '
                f'{_f(r["ncc_tissue"],  ".4f"):>11}{mark("ncc_tissue"):<2}  '
                f'{_f(r["smooth_px"],   ".2f"):>10}{mark("smooth_px"):<2}  '
                f'{_f(r["z_sharp_dapi"],".2f"):>13}{mark("z_sharp_dapi")}'
            )
        print(f'\n  ◀ = best  |  smooth_px: lower=better; rest: higher=better')
        print(f'  ncc_dapi     adjacent-slice NCC on DAPI  (biased toward script-2 baseline)')
        print(f'  ncc_tissue   adjacent-slice NCC on tissue_mask after applying corrections')
        print(f'  smooth_px    mean |Δtranslation| px between consecutive transforms')
        print(f'  z_sharp_dapi mean Sobel gradient of DAPI Z max-projection')
        print(f'{"═"*W}\n')

        # Save CSV
        csv_path = os.path.join(OUT_DIR, f'evaluation_fish{fish}.csv')
        fields = ['experiment', 'ncc_dapi', 'ncc_tissue', 'smooth_px', 'smooth_deg', 'z_sharp_dapi']
        with open(csv_path, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
            w.writeheader(); w.writerows(rows)
        logging.info(f'  Saved → {csv_path}')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Second-pass registration experiments + evaluation (script 3).'
    )
    parser.add_argument(
        '--steps', nargs='+', choices=STEPS, default=STEPS, metavar='STEP',
        help=f'Steps to run. Choices: {", ".join(STEPS)}. Default: all.'
    )
    parser.add_argument(
        '--experiments', nargs='+', choices=EXPERIMENTS, default=EXPERIMENTS,
        metavar='EXP',
        help=f'Experiments to include. Choices: {", ".join(EXPERIMENTS)}. Default: all.'
    )
    parser.add_argument(
        '--fish', type=int, default=None,
        help='Process only this fish ID. Default: all fish found in script-2 output.'
    )
    args = parser.parse_args()

    fish_ids = ([args.fish] if args.fish is not None else
                sorted(int(d) for d in os.listdir(IN2_DIR)
                       if os.path.isdir(os.path.join(IN2_DIR, d)) and d.isdigit()))

    logging.info(f'Steps: {args.steps}')
    logging.info(f'Experiments: {args.experiments}')
    logging.info(f'Fish: {fish_ids}')

    steps = set(args.steps)

    if 'register' in steps:
        summary_df = pd.read_csv(glob.glob(SUMMARY_GLOB)[0])
        summary_df['run'] = summary_df['run'].astype(int).astype(str)
        step_register(fish_ids, args.experiments, summary_df)

    if 'evaluate' in steps:
        step_evaluate(fish_ids, args.experiments)

    logging.info(f'Done. Output: {OUT_DIR}')


if __name__ == '__main__':
    main()
