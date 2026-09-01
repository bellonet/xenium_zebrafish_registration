"""
Zebrafish 2D slice detection and extraction pipeline.

Steps: tiles → bboxes → crop → tag → rescue → save_dapi → unify

Usage:
  python 1_crop_and_tag_2d_slices.py [--from-step STEP]
"""
import os
import re
import gc
import math
import logging
import warnings
import argparse
from typing import Tuple, List, Dict
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import label as ndimage_label, find_objects, rotate as nd_rotate
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings('ignore', message='.*OME series cannot read multi-file pyramids.*')

# ── Configuration ──────────────────────────────────────────────────────────────

BASE_PATH = '../data'
INPUT_FOLDERS = [
    'output-XETG00046__0038328__Region_1__20250717__075022',
    'output-XETG00046__0043921__Region_1__20250620__084504',
    'output-XETG00046__0044004__Region_1__20250620__084505',
]
MORPHOLOGY_SUBPATH = 'morphology_focus/morphology_focus_0000.ome.tif'
DAPI_CHANNEL = 0
NUM_CHANNELS = 4

ANALYSIS_DIR     = '../analysis'
DETECTION_SUBDIR = '1_detection'
INITIAL_TILES_DIR = '_initial_tiles_tmp'
TILES_DIR         = 'tiles'
BBOX_IMAGES_DIR   = 'zfish_bboxs'
ROTATED_TILES_DIR = 'rotated_tiles'
RESCUED_VIS_DIR   = 'zfish_bboxs_rescued'
INDIVIDUAL_FISH_DIR = 'individual_fish_2d'

DOWNSAMPLE_FACTOR = 8
MIN_BOX_AREA      = 1000
BACKGROUND_INTENSITY_THRESHOLD = 0
MAX_HEIGHT        = 14000
MAX_WIDTH         = 14000
MIN_MAX_INTENSITY = 500
MIN_VERTICAL_OVERLAP = 100

BBOX_MARGIN        = 5

XENIUM_PX_PER_UM   = 4.705882
CELL_EPS_UM        = 30
MIN_CELLS_PER_FISH = 15
MIN_FISH_UM        = 22
MAX_FISH_UM        = 250
NUCLEUS_PAD_UM     = 7
FRAG_PROXIMITY_UM  = 50

IGNORE_TILES: dict[str, list[int]] = {
    '0038328': [15],
}

# Bad slices to skip during unify: fish_name (1-based) → global_slice_nums
DELETE_SLICES: Dict[int, List[int]] = {
    1: [46, 95],
    2: [40, 92, 93, 94],
    4: [40, 61, 95],
    5: [40, 74],
    6: [55, 93],
}

CROP_MARGIN  = 5
ANGLE_MIN    = -25
ANGLE_MAX    = 25
ANGLE_STEP   = 5
Y_RANGE_SINGLE_ROW_FRAC = 0.18
FISH_CROP_MARGIN = 10

STEPS = ['tiles', 'bboxes', 'crop', 'tag', 'rescue', 'save_dapi', 'unify']

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ── Path helpers ───────────────────────────────────────────────────────────────

def det(output_base: str, *parts) -> str:
    return os.path.join(output_base, *parts)


# ── Utilities ──────────────────────────────────────────────────────────────────

def minimal_unique_suffixes(names: List[str]) -> List[str]:
    if len(names) <= 1:
        return [n[-3:] if len(n) >= 3 else n for n in names]
    names = [n.rstrip('/').rstrip('\\') for n in names]
    for length in range(1, max(len(n) for n in names) + 1):
        suffixes = [n[-length:] if len(n) >= length else n for n in names]
        if len(set(suffixes)) == len(names):
            return suffixes
    return names


def get_channel_files(base_file_path: str) -> List[str]:
    base_path = Path(base_file_path)
    stem = base_path.stem.replace('.ome', '')
    match = re.match(r'(.+)_\d+$', stem)
    if match:
        stem = match.group(1)
    channel_files = sorted(base_path.parent.glob(f"{stem}_*.ome.tif"))
    if len(channel_files) > 1:
        logging.info(f"Multi-file OME-TIFF: {len(channel_files)} files")
        return [str(f) for f in channel_files]
    logging.info("Single-file mode")
    return [str(base_file_path)]


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(int(v), hi))


def _tile_idx_from_name(fname: str) -> int:
    digits = re.sub(r"\D", "", os.path.splitext(fname)[0])
    if not digits:
        raise ValueError(f"Could not parse tile_idx from filename: {fname}")
    return int(digits)


def simple_downsample(img: np.ndarray, factor: int) -> np.ndarray:
    h, w = img.shape[:2]
    h_ds, w_ds = h // factor, w // factor
    img_crop = img[:h_ds * factor, :w_ds * factor]
    if img_crop.ndim == 2:
        return img_crop.reshape(h_ds, factor, w_ds, factor).mean(axis=(1, 3)).astype(img.dtype)
    return img_crop.reshape(h_ds, factor, w_ds, factor, -1).mean(axis=(1, 3)).astype(img.dtype)


# ── Step 1: Extract tiles ──────────────────────────────────────────────────────

def vertical_intervals_connected(a_min, a_max, b_min, b_max, min_overlap=MIN_VERTICAL_OVERLAP):
    return min(a_max, b_max) - max(a_min, b_min) >= min_overlap


def group_rows_by_vertical_connectivity(boxes, min_overlap=MIN_VERTICAL_OVERLAP):
    n = len(boxes)
    if n <= 1:
        return boxes
    adj = [[] for _ in range(n)]
    for i in range(n):
        _, a_min, _, a_max, _, _ = boxes[i]
        for j in range(i + 1, n):
            _, b_min, _, b_max, _, _ = boxes[j]
            if vertical_intervals_connected(a_min, a_max, b_min, b_max, min_overlap):
                adj[i].append(j); adj[j].append(i)
    comp = [-1] * n; cid = 0
    for i in range(n):
        if comp[i] != -1:
            continue
        q = deque([i]); comp[i] = cid
        while q:
            u = q.popleft()
            for v in adj[u]:
                if comp[v] == -1:
                    comp[v] = cid; q.append(v)
        cid += 1
    groups: dict = {}
    for idx, c in enumerate(comp):
        groups.setdefault(c, []).append(boxes[idx])
    rows = []
    for blist in groups.values():
        blist_sorted = sorted(blist, key=lambda b: b[2])
        median_y = float(np.median([0.5 * (b[1] + b[3]) for b in blist_sorted]))
        rows.append((median_y, blist_sorted))
    rows.sort(key=lambda x: -x[0])
    return [b for _, row in rows for b in row]


def find_bounding_boxes(img: np.ndarray, downsample_factor: int = 1) -> List[Tuple]:
    img_ds = simple_downsample(img, downsample_factor) if downsample_factor > 1 else img
    labeled, _ = ndimage_label(img_ds > BACKGROUND_INTENSITY_THRESHOLD)
    boxes = []
    for idx, slc in enumerate(find_objects(labeled), start=1):
        if slc is None:
            continue
        min_row = slc[0].start * downsample_factor
        max_row = (slc[0].stop - 1) * downsample_factor
        min_col = slc[1].start * downsample_factor
        max_col = (slc[1].stop - 1) * downsample_factor
        area = (max_row - min_row + 1) * (max_col - min_col + 1)
        if area >= MIN_BOX_AREA:
            boxes.append((idx, min_row, min_col, max_row, max_col, area))
    return boxes


def extract_tiles_dapi(input_path: str, output_base: str, dapi_channel: int = DAPI_CHANNEL) -> pd.DataFrame:
    logging.info(f"Step 1: Extracting tiles from DAPI (downsampled {DOWNSAMPLE_FACTOR}x)")
    channel_files = get_channel_files(input_path)
    with tifffile.TiffFile(channel_files[dapi_channel]) as tif:
        img = tif.pages[0].asarray() if len(tif.pages) > 0 else tif.asarray()
    logging.info(f"DAPI shape: {img.shape}")

    orig = find_bounding_boxes(img, downsample_factor=DOWNSAMPLE_FACTOR)
    logging.info(f"Found {len(orig)} initial boxes")
    boxes = []
    for oid, r0, c0, r1, c1, area in orig:
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if h > MAX_HEIGHT:
            mid = (r0 + r1) // 2
            boxes += [(oid, r0, c0, mid, c1, area // 2), (oid, mid + 1, c0, r1, c1, area // 2)]
        elif w > MAX_WIDTH:
            mid = (c0 + c1) // 2
            boxes += [(oid, r0, c0, r1, mid, area // 2), (oid, r0, mid + 1, r1, c1, area // 2)]
        else:
            boxes.append((oid, r0, c0, r1, c1, area))
    logging.info(f"After splitting: {len(boxes)} tiles")

    tiles_dir = det(output_base, INITIAL_TILES_DIR, f'ch{dapi_channel}')
    os.makedirs(tiles_dir, exist_ok=True)
    filename = os.path.basename(input_path)
    all_boxes = []
    for seq_idx, (_, min_row, min_col, max_row, max_col, area) in enumerate(boxes, start=1):
        crop = img[min_row:max_row + 1, min_col:max_col + 1]
        min_val, max_val = float(np.min(crop)), float(np.max(crop))
        if max_val < MIN_MAX_INTENSITY:
            logging.info(f"Discarding box (max_intensity={max_val:.1f} < {MIN_MAX_INTENSITY})")
            continue
        tifffile.imwrite(os.path.join(tiles_dir, f'{seq_idx}.tif'), crop)
        all_boxes.append({
            'filename': filename, 'channel': dapi_channel, 'crop_idx': seq_idx,
            'min_row': min_row, 'min_col': min_col, 'max_row': max_row, 'max_col': max_col,
            'area': area, 'min_intensity': min_val, 'max_intensity': max_val,
        })

    del img; gc.collect()
    df = pd.DataFrame(all_boxes)
    df.to_csv(det(output_base, f'{INITIAL_TILES_DIR}.csv'), index=False)
    logging.info(f"Saved {len(all_boxes)} tile metadata")
    return df


# ── Step 2: Detect bboxes in tiles ────────────────────────────────────────────

def detect_bboxes_in_tiles(output_base: str, input_folder: str,
                           dapi_channel: int = DAPI_CHANNEL) -> pd.DataFrame:
    """Detect fish bboxes per tile by clustering Xenium cell centroids."""
    logging.info("Step 2: Detecting bboxes in tiles (cell-centroid clustering)")

    parquet_path = os.path.join(input_folder, 'cells.parquet')
    logging.info(f"  Loading centroids: {parquet_path}")
    coords = pd.read_parquet(parquet_path, columns=['x_centroid', 'y_centroid']).to_numpy()
    n = len(coords)
    logging.info(f"  {n} cells")

    logging.info(f"  Building graph (eps={CELL_EPS_UM} µm)…")
    tree = cKDTree(coords)
    pairs = list(tree.query_pairs(r=CELL_EPS_UM))
    if pairs:
        ri, ci = zip(*pairs)
        mat = csr_matrix((np.ones(len(ri)), (ri, ci)), shape=(n, n))
        _, labels = connected_components(mat + mat.T, directed=False)
    else:
        labels = np.arange(n)
    logging.info(f"  {int(labels.max()) + 1} raw connected components")

    S      = XENIUM_PX_PER_UM
    PAD_PX = int(NUCLEUS_PAD_UM * S)
    fish_clusters = []
    for c in range(int(labels.max()) + 1):
        m = labels == c
        if m.sum() < MIN_CELLS_PER_FISH:
            continue
        xs, ys = coords[m, 0], coords[m, 1]
        h_um = ys.max() - ys.min()
        w_um = xs.max() - xs.min()
        if h_um > MAX_FISH_UM or w_um > MAX_FISH_UM:
            continue
        if h_um < MIN_FISH_UM and w_um < MIN_FISH_UM:
            continue
        fish_clusters.append(dict(
            n_cells=int(m.sum()),
            g_min_r=max(0, int(ys.min() * S) - PAD_PX),
            g_min_c=max(0, int(xs.min() * S) - PAD_PX),
            g_max_r=int(ys.max() * S) + PAD_PX,
            g_max_c=int(xs.max() * S) + PAD_PX,
        ))
    logging.info(f"  {len(fish_clusters)} fish clusters after size filter")

    # Drop fragment clusters: when two overlap/are close, keep the larger one
    PROX_PX = FRAG_PROXIMITY_UM * S
    keep = [True] * len(fish_clusters)
    for i in range(len(fish_clusters)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(fish_clusters)):
            if not keep[j]:
                continue
            a, b = fish_clusters[i], fish_clusters[j]
            dy = max(0, max(a['g_min_r'], b['g_min_r']) - min(a['g_max_r'], b['g_max_r']))
            dx = max(0, max(a['g_min_c'], b['g_min_c']) - min(a['g_max_c'], b['g_max_c']))
            if math.sqrt(dy*dy + dx*dx) < PROX_PX:
                if a['n_cells'] >= b['n_cells']:
                    keep[j] = False
                else:
                    keep[i] = False; break
    fish_clusters = [fc for fc, k in zip(fish_clusters, keep) if k]
    logging.info(f"  {len(fish_clusters)} fish clusters after fragment removal")

    tiles_df  = pd.read_csv(det(output_base, f'{INITIAL_TILES_DIR}.csv'))
    tile_info = tiles_df.set_index('crop_idx')[['min_row', 'min_col', 'max_row', 'max_col']]

    tiles_dir    = det(output_base, INITIAL_TILES_DIR, f'ch{dapi_channel}')
    bbox_img_dir = det(output_base, BBOX_IMAGES_DIR)
    os.makedirs(bbox_img_dir, exist_ok=True)

    image_files = sorted(
        [f for f in os.listdir(tiles_dir) if f.lower().endswith(('.tif', '.tiff'))],
        key=lambda f: int(re.sub(r'\D', '', os.path.splitext(f)[0]))
    )

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib import patheffects

    all_detections = []
    for image_name in image_files:
        tile_idx = _tile_idx_from_name(image_name)
        if tile_idx not in tile_info.index:
            logging.warning(f"  Tile {tile_idx} not in {INITIAL_TILES_DIR}.csv — skipping")
            continue
        ti     = tile_info.loc[tile_idx]
        orig_r = int(ti['min_row']); orig_c = int(ti['min_col'])
        end_r  = int(ti['max_row']); end_c  = int(ti['max_col'])

        tile_fish = [
            fc for fc in fish_clusters
            if fc['g_max_r'] >= orig_r and fc['g_min_r'] <= end_r and
               fc['g_max_c'] >= orig_c and fc['g_min_c'] <= end_c
        ]
        logging.info(f"  {image_name}: {len(tile_fish)} fish clusters")
        image = tifffile.imread(os.path.join(tiles_dir, image_name))

        local_coords = [
            (fc['g_min_r'] - orig_r, fc['g_min_c'] - orig_c,
             fc['g_max_r'] - orig_r, fc['g_max_c'] - orig_c)
            for fc in tile_fish
        ]
        for label_idx, (fc, (loc_min_r, loc_min_c, loc_max_r, loc_max_c)) in \
                enumerate(zip(tile_fish, local_coords), start=1):
            all_detections.append({
                'image_name': image_name, 'tile_idx': tile_idx, 'label': label_idx,
                'area': fc['n_cells'],
                'min_row': loc_min_r, 'min_col': loc_min_c,
                'max_row': loc_max_r, 'max_col': loc_max_c,
            })

        fig, ax = plt.subplots(1)
        ax.imshow(image, cmap='gray')
        for lbl, (loc_min_r, loc_min_c, loc_max_r, loc_max_c) in enumerate(local_coords, start=1):
            s_r = max(0, loc_min_r - BBOX_MARGIN)
            s_c = max(0, loc_min_c - BBOX_MARGIN)
            ax.add_patch(Rectangle(
                (s_c, s_r),
                (loc_max_c + BBOX_MARGIN) - s_c,
                (loc_max_r + BBOX_MARGIN) - s_r,
                linewidth=1, edgecolor='r', facecolor='none'))
            txt = ax.text(s_c, s_r, str(lbl), fontsize=8, color='r', va='bottom', ha='left')
            txt.set_path_effects([
                patheffects.Stroke(linewidth=2, foreground='black'),
                patheffects.Normal()])
        ax.set_axis_off()
        plt.savefig(os.path.join(bbox_img_dir, f"{os.path.splitext(image_name)[0]}.png"),
                    bbox_inches='tight')
        plt.close(fig)

    df = pd.DataFrame(all_detections)
    df.to_csv(det(output_base, 'zfish_bboxs.csv'), index=False)
    logging.info(f"Detected {len(all_detections)} bboxes")
    return df


# ── Step 3: Crop tiles ─────────────────────────────────────────────────────────

def crop_tiles(tiles_df: pd.DataFrame, bbox_df: pd.DataFrame,
               output_base: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logging.info("Step 3: Cropping tiles (DAPI only, full resolution)")
    bbox_df["tile_idx"]  = bbox_df["tile_idx"].astype(int)
    tiles_df["crop_idx"] = tiles_df["crop_idx"].astype(int)

    tile_key = tiles_df.set_index("crop_idx")[
        ["filename", "channel", "min_row", "min_col", "max_row", "max_col"]
    ].to_dict("index")
    groups    = {k: g for k, g in bbox_df.groupby("tile_idx")}
    tiles_dir = det(output_base, INITIAL_TILES_DIR, 'ch0')
    files     = sorted(
        [f for f in os.listdir(tiles_dir) if f.lower().endswith(('.tif', '.tiff'))],
        key=_tile_idx_from_name
    )

    # Phase 1: crop all tiles, collect metadata
    tmp_crops = []
    for fname in files:
        tile_idx = _tile_idx_from_name(fname)
        g = groups.get(tile_idx)
        if g is None or g.empty:
            continue
        img = tifffile.imread(os.path.join(tiles_dir, fname))
        H, W = img.shape[:2]
        u_min_r = int(g["min_row"].min()); u_min_c = int(g["min_col"].min())
        u_max_r = int(g["max_row"].max()); u_max_c = int(g["max_col"].max())
        s_r = _clamp(u_min_r - CROP_MARGIN, 0, H - 1); s_c = _clamp(u_min_c - CROP_MARGIN, 0, W - 1)
        e_r = _clamp(u_max_r + CROP_MARGIN, 0, H - 1); e_c = _clamp(u_max_c + CROP_MARGIN, 0, W - 1)
        crop = img[s_r:e_r + 1, s_c:e_c + 1]
        crop_h, crop_w = int(crop.shape[0]), int(crop.shape[1])

        if tile_idx not in tile_key:
            raise KeyError(f"Tile {tile_idx} not found in initial_tiles.csv")
        tinfo = tile_key[tile_idx]
        tile_min_r_g, tile_min_c_g = int(tinfo["min_row"]), int(tinfo["min_col"])
        crop_min_r_g = tile_min_r_g + s_r; crop_min_c_g = tile_min_c_g + s_c
        crop_max_r_g = tile_min_r_g + e_r; crop_max_c_g = tile_min_c_g + e_c

        tile_row = {
            "filename": tinfo["filename"], "channel": int(tinfo["channel"]),
            "min_row": crop_min_r_g, "min_col": crop_min_c_g,
            "max_row": crop_max_r_g, "max_col": crop_max_c_g,
            "area": (crop_max_r_g - crop_min_r_g + 1) * (crop_max_c_g - crop_min_c_g + 1),
        }
        bbox_rows = []
        for _, r in g.iterrows():
            rmin, cmin, rmax, cmax = int(r["min_row"]), int(r["min_col"]), int(r["max_row"]), int(r["max_col"])
            bbox_rows.append({
                "image_name": r["image_name"], "label": int(r["label"]), "area": int(r["area"]),
                "min_row": rmin, "min_col": cmin, "max_row": rmax, "max_col": cmax,
                "crop_origin_row": s_r, "crop_origin_col": s_c, "crop_h": crop_h, "crop_w": crop_w,
                "bbox_local_min_row": rmin - s_r, "bbox_local_min_col": cmin - s_c,
                "bbox_local_max_row": rmax - s_r, "bbox_local_max_col": cmax - s_c,
                "bbox_global_min_row": tile_min_r_g + rmin, "bbox_global_min_col": tile_min_c_g + cmin,
                "bbox_global_max_row": tile_min_r_g + rmax, "bbox_global_max_col": tile_min_c_g + cmax,
            })
        tmp_crops.append((tile_idx, crop, tile_row, bbox_rows))

    # Phase 2: sort by position (bottom-left to top-right)
    sortable = [
        (i, tr["min_row"], tr["min_col"], tr["max_row"], tr["max_col"], tr["area"])
        for i, (_, _, tr, _) in enumerate(tmp_crops)
    ]
    sorted_order = [box[0] for box in group_rows_by_vertical_connectivity(sortable, min_overlap=MIN_VERTICAL_OVERLAP)]
    logging.info(f"Tile ordering: {[tmp_crops[i][0] for i in sorted_order]} -> {list(range(1, len(sorted_order)+1))}")

    # Phase 3: save with final ordering
    final_tiles_dir = det(output_base, TILES_DIR, 'ch0')
    os.makedirs(final_tiles_dir, exist_ok=True)
    out_tiles_rows, out_bbox_rows = [], []

    for new_tile_idx, orig_list_idx in enumerate(sorted_order, start=1):
        _, crop, tile_row, bbox_rows = tmp_crops[orig_list_idx]
        tifffile.imwrite(os.path.join(final_tiles_dir, f"{new_tile_idx}.tif"), crop)
        tile_row["crop_idx"] = new_tile_idx
        out_tiles_rows.append(tile_row)
        for br in bbox_rows:
            br["tile_idx"] = new_tile_idx; br["cropped_name"] = f"{new_tile_idx}.tif"
            out_bbox_rows.append(br)

    import shutil
    for path in [det(output_base, INITIAL_TILES_DIR), det(output_base, f'{INITIAL_TILES_DIR}.csv')]:
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)

    tiles_out_df = pd.DataFrame(out_tiles_rows)
    bbox_out_df  = pd.DataFrame(out_bbox_rows)
    tiles_out_df.to_csv(det(output_base, "tiles.csv"), index=False)
    bbox_out_df.to_csv(det(output_base, "tiles_bboxs.csv"), index=False)
    logging.info(f"Cropped & ordered {len(out_tiles_rows)} tiles")
    return tiles_out_df, bbox_out_df


# ── Step 4: Tag bboxes by position ────────────────────────────────────────────

def rotation_padding(h: int, w: int, angle_deg: float) -> Tuple[int, int, int, int]:
    a = math.radians(abs(angle_deg))
    pad_w = max(0, int(math.ceil((abs(w * math.cos(a)) + abs(h * math.sin(a)) - w) / 2)))
    pad_h = max(0, int(math.ceil((abs(w * math.sin(a)) + abs(h * math.cos(a)) - h) / 2)))
    return pad_h, pad_h, pad_w, pad_w


def rotate_points(points_rc: np.ndarray, angle_deg: float, center_rc: Tuple[float, float]) -> np.ndarray:
    cy, cx = center_rc
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    pts = points_rc.astype(np.float64).copy()
    x = pts[:, 1] - cx; y = -(pts[:, 0] - cy)
    pts[:, 1] = x * c - y * s + cx
    pts[:, 0] = -(x * s + y * c) + cy
    return pts


def kmeans_1d_two_clusters(y: np.ndarray, iters: int = 20) -> Tuple[np.ndarray, float, float]:
    y = y.astype(np.float64)
    c0, c1 = np.percentile(y, [30, 70])
    lab = np.zeros(len(y), dtype=int)
    for _ in range(iters):
        d0, d1 = np.abs(y - c0), np.abs(y - c1)
        lab = (d1 < d0).astype(int)
        if np.all(lab == 0) or np.all(lab == 1):
            order = np.argsort(y); lab = np.zeros_like(y, dtype=int)
            lab[order[len(y) // 2:]] = 1
        c0_new, c1_new = y[lab == 0].mean(), y[lab == 1].mean()
        if abs(c0_new - c0) < 1e-6 and abs(c1_new - c1) < 1e-6:
            break
        c0, c1 = c0_new, c1_new
    return lab, float(c0), float(c1)


def score_two_rows(rot_centers: np.ndarray) -> Tuple[float, np.ndarray]:
    y = rot_centers[:, 0]
    lab, c0, c1 = kmeans_1d_two_clusters(y)
    v0 = float(np.var(y[lab == 0])) if np.any(lab == 0) else 1e9
    v1 = float(np.var(y[lab == 1])) if np.any(lab == 1) else 1e9
    sep = abs(c0 - c1) + 1e-6
    return float(v0 + v1 + 2000.0 / (sep * sep)), lab


def decide_prefer_two_rows(n: int, y_range: float, tile_h: int) -> bool:
    if n >= 4:
        return True
    if n == 3:
        return (y_range / max(1.0, tile_h)) > Y_RANGE_SINGLE_ROW_FRAC
    return False


def to_uint8(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((img - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def rotated_bbox_from_local_bbox(min_r, min_c, max_r, max_c, pad_t, pad_l, angle_deg, center_rc):
    corners = np.array([
        [min_r + pad_t, min_c + pad_l], [min_r + pad_t, max_c + pad_l],
        [max_r + pad_t, min_c + pad_l], [max_r + pad_t, max_c + pad_l],
    ], dtype=np.float64)
    rot = rotate_points(corners, angle_deg, center_rc)
    return float(rot[:, 0].min()), float(rot[:, 1].min()), float(rot[:, 0].max()), float(rot[:, 1].max())


def x_match(a0, a1, b0, b1, tol_px=50.0) -> bool:
    if abs(0.5 * (a0 + a1) - 0.5 * (b0 + b1)) <= tol_px:
        return True
    return max(0.0, min(a1, b1) - max(a0, b0)) > 0.0


def infer_ids_by_column_overlap(rot_boxes, row_labels, fish_ids, tagged):
    n = len(fish_ids)
    tag_source = ["strict" if tagged[i] else "untagged" for i in range(n)]
    fish_ids = list(fish_ids); tagged_final = list(tagged)
    centers_y = np.array([(r0 + r1) / 2.0 for (r0, c0, r1, c1) in rot_boxes], dtype=float)
    y0 = centers_y[row_labels == 0].mean() if np.any(row_labels == 0) else 1e9
    y1 = centers_y[row_labels == 1].mean() if np.any(row_labels == 1) else 1e9
    top_lab = 0 if y0 < y1 else 1
    tagged_by_col = {0: [], 1: [], 2: []}
    for i in range(n):
        if not tagged_final[i]:
            continue
        col = int(fish_ids[i]) % 3
        r0, c0, r1, c1 = rot_boxes[i]
        tagged_by_col[col].append((i, c0, c1))
    for i in range(n):
        if tagged_final[i]:
            continue
        row_base = 0 if row_labels[i] == top_lab else 3
        r0, c0, r1, c1 = rot_boxes[i]
        candidate_cols = {col for col, items in tagged_by_col.items()
                          for (j, tc0, tc1) in items if x_match(c0, c1, tc0, tc1, tol_px=500.0)}
        if len(candidate_cols) == 1:
            col = list(candidate_cols)[0]
            fish_ids[i] = row_base + col
            tagged_final[i] = True
            tag_source[i] = "inferred"
    return fish_ids, tagged_final, tag_source


def assign_only_unambiguous(rot_centers, row_labels):
    n = len(rot_centers)
    fish_id = [-1] * n; tagged = [False] * n
    if n == 0:
        return fish_id, tagged
    centers_y = rot_centers[:, 0]
    y0 = centers_y[row_labels == 0].mean() if np.any(row_labels == 0) else 1e9
    y1 = centers_y[row_labels == 1].mean() if np.any(row_labels == 1) else 1e9
    top_lab = 0 if y0 < y1 else 1
    bot_lab = 1 - top_lab
    top_idx = [i for i in range(n) if row_labels[i] == top_lab]
    bot_idx = [i for i in range(n) if row_labels[i] == bot_lab]

    def label_row(indices, base):
        if len(indices) != 3:
            return
        for col, (_, i) in enumerate(sorted((rot_centers[i, 1], i) for i in indices)):
            fish_id[i] = base + col; tagged[i] = True

    if n == 6:
        label_row(top_idx, 0); label_row(bot_idx, 3)
    elif n in (4, 5):
        if len(top_idx) == 3:
            label_row(top_idx, 0)
        if len(bot_idx) == 3:
            label_row(bot_idx, 3)
    return fish_id, tagged


def load_font(size: int = 60):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans-Bold.ttf"]:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def tag_bboxs_by_position(bbox_df: pd.DataFrame, output_base: str,
                           dapi_channel: int = DAPI_CHANNEL,
                           run_suffix: str = '') -> Tuple[pd.DataFrame, pd.DataFrame]:
    logging.info("Step 4: Tagging bboxes by position (full resolution)")
    ignored = set(IGNORE_TILES.get(run_suffix, []))
    if ignored:
        before = len(bbox_df)
        bbox_df = bbox_df[~bbox_df['tile_idx'].isin(ignored)].copy()
        logging.info(f"  Ignoring tiles {sorted(ignored)} ({before - len(bbox_df)} rows dropped)")

    cropped_tiles_dir = det(output_base, TILES_DIR, f'ch{dapi_channel}')
    rotated_img_dir   = det(output_base, ROTATED_TILES_DIR)
    os.makedirs(rotated_img_dir, exist_ok=True)
    bbox_df["tile_idx"] = bbox_df["tile_idx"].astype(int)

    tile_rotation_rows, tagged_rows, untagged_rows = [], [], []
    font = load_font(150)

    for tile_idx, g in bbox_df.groupby("tile_idx"):
        tile_path = os.path.join(cropped_tiles_dir, f"{tile_idx}.tif")
        if not os.path.isfile(tile_path):
            logging.warning(f"Missing tile {tile_idx}"); continue
        img = tifffile.imread(tile_path)
        if img.ndim != 2:
            img = img[0]
        H, W = img.shape; n = len(g)
        centers = np.stack([
            (g["bbox_local_min_row"].to_numpy() + g["bbox_local_max_row"].to_numpy()) / 2.0,
            (g["bbox_local_min_col"].to_numpy() + g["bbox_local_max_col"].to_numpy()) / 2.0,
        ], axis=1)
        y_range    = float(centers[:, 0].max() - centers[:, 0].min()) if n > 0 else 0.0
        prefer_two = decide_prefer_two_rows(n, y_range, H)
        angles     = list(range(ANGLE_MIN, ANGLE_MAX + 1, ANGLE_STEP)) if n >= 4 else [0]
        center_rc  = (H / 2.0, W / 2.0)
        best_angle, best_score = 0.0, float("inf")
        best_labels = np.zeros(n, dtype=int); best_rot_centers = centers.copy()

        for ang in angles:
            rot_c = rotate_points(centers, ang, center_rc)
            if prefer_two and n >= 2:
                s, lab = score_two_rows(rot_c)
            else:
                s, lab = (float(np.var(rot_c[:, 0])) if n > 1 else 0.0), np.zeros(n, dtype=int)
            if s < best_score:
                best_score = s; best_angle = float(ang)
                best_labels = lab; best_rot_centers = rot_c

        pad_t, pad_b, pad_l, pad_r = rotation_padding(H, W, best_angle)
        img_pad  = np.pad(img, ((pad_t, pad_b), (pad_l, pad_r)), mode="constant", constant_values=0)
        Hp, Wp   = img_pad.shape; center_pad = (Hp / 2.0, Wp / 2.0)
        img_rot  = nd_rotate(img_pad, best_angle, reshape=False, order=1, mode="constant", cval=0.0)
        img_u8   = to_uint8(img_rot)
        fish_ids, tagged = assign_only_unambiguous(best_rot_centers, best_labels)
        rot_boxes = [
            rotated_bbox_from_local_bbox(
                row["bbox_local_min_row"], row["bbox_local_min_col"],
                row["bbox_local_max_row"], row["bbox_local_max_col"],
                pad_t, pad_l, best_angle, center_pad
            )
            for _, row in g.reset_index(drop=True).iterrows()
        ]
        fish_ids, tagged, tag_source = infer_ids_by_column_overlap(rot_boxes, best_labels, fish_ids, tagged)

        tile_rotation_rows.append({
            "tile_idx": tile_idx, "n_detections": n, "best_angle_deg": best_angle,
            "score": best_score, "pad_top": pad_t, "pad_bottom": pad_b, "pad_left": pad_l,
            "pad_right": pad_r, "prefer_two_rows": int(prefer_two), "n_tagged": int(np.sum(tagged)),
        })

        gg = g.copy().reset_index(drop=True)
        gg["tile_best_angle_deg"] = best_angle
        gg["tile_pad_top"] = pad_t; gg["tile_pad_left"] = pad_l
        gg["tile_pad_bottom"] = pad_b; gg["tile_pad_right"] = pad_r
        gg["fish_id"] = fish_ids; gg["is_tagged"] = tagged; gg["tag_source"] = tag_source
        gg["bbox_rot_min_row"] = [r[0] for r in rot_boxes]; gg["bbox_rot_min_col"] = [r[1] for r in rot_boxes]
        gg["bbox_rot_max_row"] = [r[2] for r in rot_boxes]; gg["bbox_rot_max_col"] = [r[3] for r in rot_boxes]
        tagged_rows.append(gg[gg["is_tagged"] == True])
        untagged_rows.append(gg[gg["is_tagged"] == False])

        im = Image.fromarray(img_u8, mode="L").convert("RGB"); draw = ImageDraw.Draw(im)
        for i, row in gg.iterrows():
            rmin, cmin, rmax, cmax = rot_boxes[i]
            rmin = max(0, min(rmin, Hp-1)); rmax = max(0, min(rmax, Hp-1))
            cmin = max(0, min(cmin, Wp-1)); cmax = max(0, min(cmax, Wp-1))
            draw.rectangle([cmin, rmin, cmax, rmax], outline=(255, 255, 255), width=2)
            if bool(row["is_tagged"]) and int(row["fish_id"]) >= 0:
                txt = str(int(row["fish_id"]) + 1); tx, ty = float(cmin) + 4, float(rmin) + 4
                l, t, r, b = draw.textbbox((0, 0), txt, font=font)
                draw.rectangle([tx-2, ty-2, tx+(r-l)+2, ty+(b-t)+2], fill=(0, 0, 0))
                draw.text((tx, ty), txt, fill=(255, 255, 255), font=font)
            else:
                x1, y1, x2, y2 = float(cmin)+4, float(rmin)+4, float(cmax)-4, float(rmax)-4
                draw.line([x1, y1, x2, y2], fill=(255, 0, 0), width=3)
                draw.line([x1, y2, x2, y1], fill=(255, 0, 0), width=3)
        im.save(os.path.join(rotated_img_dir, f"{tile_idx}.png"))
        logging.info(f"tile {tile_idx}: n={n}, angle={best_angle:+.0f}°, tagged={int(np.sum(tagged))}")

    rot_df = pd.DataFrame(tile_rotation_rows).sort_values("tile_idx")
    rot_df.to_csv(det(output_base, "tile_rotation.csv"), index=False)
    tagged_df   = pd.concat(tagged_rows,   ignore_index=True) if tagged_rows   else pd.DataFrame()
    untagged_df = pd.concat(untagged_rows, ignore_index=True) if untagged_rows else pd.DataFrame()
    tagged_df.to_csv(det(output_base, "zfish_bboxs_tagged.csv"), index=False)
    untagged_df.to_csv(det(output_base, "zfish_bboxs_untagged.csv"), index=False)
    logging.info(f"Tagged {len(tagged_df)} rows, untagged {len(untagged_df)} rows")
    return tagged_df, untagged_df


# ── Step 4B: Rescue untagged fish by NCC matching to neighbours ───────────────

def _load_tile_dapi(tile_idx: int, tiles_dir: str) -> 'np.ndarray | None':
    path = os.path.join(tiles_dir, f"{tile_idx}.tif")
    if not os.path.isfile(path):
        return None
    img = tifffile.imread(path)
    return (img[0] if img.ndim != 2 else img).astype(np.float32)


def _extract_fish_patch(tile_img, row, tile_origin_row, tile_origin_col):
    H, W = tile_img.shape
    r0 = max(0, int(row["bbox_global_min_row"]) - tile_origin_row)
    c0 = max(0, int(row["bbox_global_min_col"]) - tile_origin_col)
    r1 = min(H, int(row["bbox_global_max_row"]) - tile_origin_row)
    c1 = min(W, int(row["bbox_global_max_col"]) - tile_origin_col)
    if r1 <= r0 or c1 <= c0:
        return np.zeros((4, 4), dtype=np.float32)
    return tile_img[r0:r1, c0:c1].copy()


def _center_on_center_ncc(patch_a, patch_b) -> float:
    if patch_a.size == 0 or patch_b.size == 0:
        return 0.0
    canvas_h = max(patch_a.shape[0], patch_b.shape[0])
    canvas_w = max(patch_a.shape[1], patch_b.shape[1])

    def place_centered(patch, ch, cw):
        canvas = np.zeros((ch, cw), dtype=np.float32)
        r0 = (ch - patch.shape[0]) // 2; c0 = (cw - patch.shape[1]) // 2
        canvas[r0:r0+patch.shape[0], c0:c0+patch.shape[1]] = patch
        return canvas

    ca = place_centered(patch_a, canvas_h, canvas_w)
    cb = place_centered(patch_b, canvas_h, canvas_w)
    mask = (ca > 0) & (cb > 0)
    if int(mask.sum()) < 16:
        return 0.0
    a_vals = ca[mask].astype(np.float64); b_vals = cb[mask].astype(np.float64)
    a_std, b_std = a_vals.std(), b_vals.std()
    if a_std < 1e-6 or b_std < 1e-6:
        return 0.0
    return float(np.mean((a_vals - a_vals.mean()) * (b_vals - b_vals.mean())) / (a_std * b_std))


def rescue_untagged_fish(tagged_df: pd.DataFrame, untagged_df: pd.DataFrame,
                          output_base: str,
                          tiles_cropped_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (tagged_final_df, still_untagged_df)."""
    from scipy.optimize import linear_sum_assignment

    if untagged_df is None or len(untagged_df) == 0:
        logging.info("Step 4B: No untagged fish — nothing to rescue.")
        out = tagged_df.copy() if tagged_df is not None else pd.DataFrame()
        if "tag_source" not in out.columns:
            out["tag_source"] = "original"
        out.to_csv(det(output_base, "zfish_bboxs_tagged_final.csv"), index=False)
        pd.DataFrame().to_csv(det(output_base, "zfish_bboxs_still_untagged.csv"), index=False)
        return out, pd.DataFrame()

    logging.info(f"Step 4B: Rescuing {len(untagged_df)} untagged rows "
                 f"across {untagged_df['tile_idx'].nunique()} tiles")

    tile_origin: dict[int, tuple] = {
        int(row["tile_idx"]): (int(row["min_row"]), int(row["min_col"]))
        for _, row in tiles_cropped_df.iterrows()
    }
    tiles_dir = det(output_base, TILES_DIR, f'ch{DAPI_CHANNEL}')

    tagged_by_tile: dict[int, dict[int, pd.Series]] = {}
    if tagged_df is not None and len(tagged_df) > 0:
        for _, row in tagged_df.iterrows():
            tagged_by_tile.setdefault(int(row["tile_idx"]), {})[int(row["fish_id"])] = row

    untagged_by_tile = {int(tidx): grp.reset_index(drop=True)
                        for tidx, grp in untagged_df.groupby("tile_idx")}
    remaining = set(untagged_by_tile.keys())
    rescued_rows: list[pd.Series] = []

    def n_tagged(tidx): return len(tagged_by_tile.get(tidx, {}))
    def neighbour_score(tidx):
        prev = max((t for t in tagged_by_tile if t < tidx), default=None)
        nxt  = min((t for t in tagged_by_tile if t > tidx), default=None)
        return (n_tagged(prev) if prev is not None else 0) + (n_tagged(nxt) if nxt is not None else 0)

    for _pass in range(20):
        if not remaining:
            break
        made_progress = False
        for tile_idx in sorted(remaining, key=neighbour_score, reverse=True):
            grp = untagged_by_tile[tile_idx]
            prev_t = max((t for t in tagged_by_tile if t < tile_idx), default=None)
            next_t = min((t for t in tagged_by_tile if t > tile_idx), default=None)
            neighbour_tiles = [t for t in [prev_t, next_t] if t is not None]
            if not neighbour_tiles:
                continue

            img_u  = _load_tile_dapi(tile_idx, tiles_dir)
            orig_u = tile_origin.get(tile_idx, (0, 0))
            already_tagged_ids = set(tagged_by_tile.get(tile_idx, {}).keys())
            free_ids    = sorted(set(range(6)) - already_tagged_ids)
            n_untagged  = len(grp)
            if not free_ids or n_untagged == 0:
                remaining.discard(tile_idx); continue

            existing    = tagged_by_tile.get(tile_idx, {})
            u_cy_list   = [(float(r["bbox_global_min_row"]) + float(r["bbox_global_max_row"])) / 2.0
                           for _, r in grp.iterrows()]
            u_cx_list   = [(float(r["bbox_global_min_col"]) + float(r["bbox_global_max_col"])) / 2.0
                           for _, r in grp.iterrows()]
            upper_ys    = [(float(r["bbox_global_min_row"]) + float(r["bbox_global_max_row"])) / 2.0
                           for fid2, r in existing.items() if fid2 < 3]
            lower_ys    = [(float(r["bbox_global_min_row"]) + float(r["bbox_global_max_row"])) / 2.0
                           for fid2, r in existing.items() if fid2 >= 3]

            u_cy_arr    = np.array(u_cy_list)
            u_is_two_rows = False
            if not (upper_ys or lower_ys) and len(u_cy_list) >= 3:
                u_heights = np.array([float(r["bbox_global_max_row"]) - float(r["bbox_global_min_row"])
                                      for _, r in grp.iterrows()])
                sep_ratio = float(u_cy_arr.max() - u_cy_arr.min()) / max(float(u_heights.mean()), 1.0)
                u_is_two_rows = sep_ratio > 0.5
                logging.info(f"  tile {tile_idx}: sep_ratio={sep_ratio:.2f} -> "
                             f"{'two rows' if u_is_two_rows else 'single row'}")
            elif upper_ys and lower_ys:
                u_is_two_rows = True

            effective_free_ids = list(free_ids)
            if not u_is_two_rows and not (upper_ys or lower_ys):
                u_cx_mean  = float(np.mean(u_cx_list))
                nb_upper_xs, nb_lower_xs = [], []
                for nb_tile in neighbour_tiles:
                    for fid2, nb_r in tagged_by_tile.get(nb_tile, {}).items():
                        cx = (float(nb_r["bbox_global_min_col"]) + float(nb_r["bbox_global_max_col"])) / 2.0
                        (nb_upper_xs if fid2 < 3 else nb_lower_xs).append(cx)
                if nb_upper_xs and nb_lower_xs:
                    if abs(u_cx_mean - float(np.mean(nb_upper_xs))) < abs(u_cx_mean - float(np.mean(nb_lower_xs))):
                        effective_free_ids = [f for f in free_ids if f < 3]
                    else:
                        effective_free_ids = [f for f in free_ids if f >= 3]
                elif nb_upper_xs:
                    effective_free_ids = [f for f in free_ids if f < 3]
                elif nb_lower_xs:
                    effective_free_ids = [f for f in free_ids if f >= 3]
                logging.info(f"  tile {tile_idx}: single-row, restricting free_ids to {effective_free_ids}")

            if upper_ys and lower_ys:
                mid_y = (np.mean(upper_ys) + np.mean(lower_ys)) / 2.0
                u_is_upper_list = [cy < mid_y for cy in u_cy_list]
            elif upper_ys:
                u_is_upper_list = [False] * len(u_cy_list)
            elif lower_ys:
                u_is_upper_list = [True] * len(u_cy_list)
            elif u_is_two_rows:
                med_y = float(np.median(u_cy_arr))
                u_is_upper_list = [cy < med_y for cy in u_cy_list]
                if all(u_is_upper_list) or not any(u_is_upper_list):
                    u_is_upper_list = [None] * len(u_cy_list)
            else:
                u_is_upper_list = [None] * len(u_cy_list)

            ncc_mat = np.zeros((n_untagged, len(effective_free_ids)), dtype=np.float64)
            for i, (_, u_row) in enumerate(grp.iterrows()):
                u_patch   = (_extract_fish_patch(img_u, u_row, orig_u[0], orig_u[1])
                             if img_u is not None else np.zeros((4, 4), dtype=np.float32))
                u_is_upper = u_is_upper_list[i]
                for j, fid in enumerate(effective_free_ids):
                    if u_is_upper is not None and u_is_upper != (fid < 3):
                        ncc_mat[i, j] = -1.0; continue
                    scores = []
                    for nb_tile in neighbour_tiles:
                        nb_row = tagged_by_tile.get(nb_tile, {}).get(fid)
                        if nb_row is None: continue
                        nb_img  = _load_tile_dapi(nb_tile, tiles_dir)
                        nb_orig = tile_origin.get(nb_tile, (0, 0))
                        if nb_img is None: continue
                        scores.append(_center_on_center_ncc(u_patch,
                                       _extract_fish_patch(nb_img, nb_row, nb_orig[0], nb_orig[1])))
                    ncc_mat[i, j] = float(np.mean(scores)) if scores else 0.0

            logging.info(f"  tile {tile_idx}: NCC matrix\n"
                         f"    free_ids={effective_free_ids}\n    ncc=\n{np.round(ncc_mat, 3)}")

            row_ind, col_ind = linear_sum_assignment(-ncc_mat)
            assignment = [(row_ind[k], col_ind[k]) for k in range(len(row_ind))]

            # Within each row, reassign by x-order to fix column ordering
            for row_base in (0, 3):
                row_assigns = [(ri, ci) for ri, ci in assignment
                               if effective_free_ids[ci] in range(row_base, row_base + 3)]
                if len(row_assigns) < 2:
                    continue
                row_assigns_by_x = sorted(row_assigns, key=lambda rc: u_cx_list[rc[0]])
                fids_sorted = sorted([effective_free_ids[ci] for _, ci in row_assigns])
                for rank, (ri, _ci) in enumerate(row_assigns_by_x):
                    correct_fid = fids_sorted[rank]
                    for k, (r2, c2) in enumerate(assignment):
                        if r2 == ri:
                            assignment[k] = (ri, effective_free_ids.index(correct_fid)); break

            any_rescued = False
            for ri, ci in assignment:
                fid   = effective_free_ids[ci]
                u_row = grp.iloc[ri].copy()
                u_row["fish_id"] = fid; u_row["is_tagged"] = True; u_row["tag_source"] = "rescued"
                rescued_rows.append(u_row)
                tagged_by_tile.setdefault(tile_idx, {})[fid] = u_row
                any_rescued = True
                logging.info(f"    untagged#{ri} -> fish_id={fid}  NCC={ncc_mat[ri, ci]:.3f}")

            remaining.discard(tile_idx)
            if any_rescued:
                made_progress = True

        if not made_progress:
            logging.warning(f"Step 4B: No progress in pass {_pass+1} — stopping"); break

    if remaining:
        logging.warning(f"Step 4B: Could not rescue tiles: {sorted(remaining)}")

    still_untagged_rows = [row for tile_idx in remaining
                           for _, row in untagged_by_tile[tile_idx].iterrows()]
    still_untagged_df = pd.DataFrame(still_untagged_rows)

    parts = []
    if tagged_df is not None and len(tagged_df) > 0:
        td = tagged_df.copy()
        if "tag_source" not in td.columns:
            td["tag_source"] = "original"
        parts.append(td)
    if rescued_rows:
        parts.append(pd.DataFrame(rescued_rows))

    tagged_final = (pd.concat(parts, ignore_index=True)
                    .sort_values(["tile_idx", "fish_id"]).reset_index(drop=True)
                    if parts else pd.DataFrame())
    tagged_final.to_csv(det(output_base, "zfish_bboxs_tagged_final.csv"), index=False)
    still_untagged_df.to_csv(det(output_base, "zfish_bboxs_still_untagged.csv"), index=False)
    logging.info(f"Step 4B: {len(tagged_df) if tagged_df is not None else 0} original "
                 f"+ {len(rescued_rows)} rescued = {len(tagged_final)} total, "
                 f"{len(still_untagged_df)} still untagged")

    # Visualise: green=original, blue=rescued
    vis_dir = det(output_base, RESCUED_VIS_DIR)
    os.makedirs(vis_dir, exist_ok=True)
    font = load_font(150)
    for tile_idx, grp in tagged_final.groupby("tile_idx"):
        tile_idx = int(tile_idx)
        img = _load_tile_dapi(tile_idx, tiles_dir)
        if img is None: continue
        im = Image.fromarray(to_uint8(img)).convert("RGB"); draw = ImageDraw.Draw(im)
        orig = tile_origin.get(tile_idx, (0, 0))
        for _, row in grp.iterrows():
            fid = int(row["fish_id"])
            r0 = int(row["bbox_global_min_row"]) - orig[0]; c0 = int(row["bbox_global_min_col"]) - orig[1]
            r1 = int(row["bbox_global_max_row"]) - orig[0]; c1 = int(row["bbox_global_max_col"]) - orig[1]
            src   = str(row.get("tag_source", "original"))
            color = ((0, 255, 0) if src == "original" else (0, 150, 255) if src == "rescued" else (255, 0, 0))
            draw.rectangle([c0, r0, c1, r1], outline=color, width=4)
            txt = str(fid + 1)
            l, t, r, b = draw.textbbox((0, 0), txt, font=font)
            draw.rectangle([c0-2, r0-2, c0+(r-l)+6, r0+(b-t)+6], fill=(0, 0, 0))
            draw.text((c0+2, r0+2), txt, fill=color, font=font)
        im.save(os.path.join(vis_dir, f"{tile_idx}.png"))

    return tagged_final, still_untagged_df


# ── Step 5: Save DAPI fish crops per run (debug) ──────────────────────────────

def save_fish_dapi_crops(tagged_df: pd.DataFrame, output_base: str, input_path: str):
    logging.info("Step 5 (save_dapi): Saving DAPI fish crops")
    if tagged_df is None or len(tagged_df) == 0:
        logging.warning("No tagged fish to save"); return

    tagged_df["tile_idx"] = tagged_df["tile_idx"].astype(int)
    tagged_df["fish_id"]  = tagged_df["fish_id"].astype(int)

    with tifffile.TiffFile(get_channel_files(input_path)[DAPI_CHANNEL]) as tif:
        img_full = tif.pages[0].asarray() if len(tif.pages) > 0 else tif.asarray()

    written = 0
    for _, row in tagged_df.iterrows():
        fish = int(row["fish_id"]) + 1
        r0 = max(0, int(row["bbox_global_min_row"]) - FISH_CROP_MARGIN)
        c0 = max(0, int(row["bbox_global_min_col"]) - FISH_CROP_MARGIN)
        r1 = min(img_full.shape[0] - 1, int(row["bbox_global_max_row"]) + FISH_CROP_MARGIN)
        c1 = min(img_full.shape[1] - 1, int(row["bbox_global_max_col"]) + FISH_CROP_MARGIN)
        if r1 <= r0 or c1 <= c0:
            continue
        out_dir = os.path.join(output_base, INDIVIDUAL_FISH_DIR, str(fish), 'c0')
        os.makedirs(out_dir, exist_ok=True)
        tifffile.imwrite(os.path.join(out_dir, f"{int(row['tile_idx'])}.tif"),
                         img_full[r0:r1 + 1, c0:c1 + 1], photometric="minisblack")
        written += 1

    del img_full; gc.collect()
    logging.info(f"Saved {written} DAPI crops -> {os.path.join(output_base, INDIVIDUAL_FISH_DIR)}")


# ── Step 6: Unify individual fish across runs ─────────────────────────────────

def _fish_dapi_patch_at_tile(det_base: str, tagged_df: pd.DataFrame,
                              fish_id: int, tile_idx: int) -> 'np.ndarray | None':
    row = tagged_df[(tagged_df['fish_id'] == fish_id) & (tagged_df['tile_idx'] == tile_idx)]
    if len(row) == 0:
        return None
    row = row.iloc[0]
    tile_img = _load_tile_dapi(tile_idx, os.path.join(det_base, TILES_DIR, f'ch{DAPI_CHANNEL}'))
    if tile_img is None:
        return None
    r0, c0 = int(row['bbox_local_min_row']), int(row['bbox_local_min_col'])
    r1, c1 = int(row['bbox_local_max_row']), int(row['bbox_local_max_col'])
    return tile_img[r0:r1, c0:c1].astype(np.float32) if r1 > r0 and c1 > c0 else None


def determine_run_order(suffixes: List[str]) -> List[str]:
    """Order runs by z-position using NCC between adjacent run endpoints."""
    from itertools import permutations

    n = len(suffixes)
    if n == 1:
        return list(suffixes)

    det_bases     = [os.path.join(ANALYSIS_DIR, DETECTION_SUBDIR, s) for s in suffixes]
    tagged_finals = []
    for det_base in det_bases:
        df = pd.read_csv(det(det_base, 'zfish_bboxs_tagged_final.csv'))
        df['fish_id']  = df['fish_id'].astype(int)
        df['tile_idx'] = df['tile_idx'].astype(int)
        tagged_finals.append(df)

    ncc_mat = np.zeros((n, n), dtype=np.float64)
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            scores = []
            for fish_id in range(6):
                fa = tagged_finals[a][tagged_finals[a]['fish_id'] == fish_id]
                fb = tagged_finals[b][tagged_finals[b]['fish_id'] == fish_id]
                if len(fa) == 0 or len(fb) == 0:
                    continue
                pa = _fish_dapi_patch_at_tile(det_bases[a], tagged_finals[a],
                                              fish_id, int(fa['tile_idx'].max()))
                pb = _fish_dapi_patch_at_tile(det_bases[b], tagged_finals[b],
                                              fish_id, int(fb['tile_idx'].min()))
                if pa is not None and pb is not None:
                    scores.append(_center_on_center_ncc(pa, pb))
            ncc_mat[a, b] = float(np.mean(scores)) if scores else 0.0
            logging.info(f"  NCC(last {suffixes[a]} -> first {suffixes[b]}) = "
                         f"{ncc_mat[a, b]:.3f} over {len(scores)} fish")

    best_perm, best_score = list(range(n)), -np.inf
    for perm in permutations(range(n)):
        score = sum(ncc_mat[perm[i], perm[i + 1]] for i in range(n - 1))
        if score > best_score:
            best_score = score; best_perm = list(perm)

    ordered = [suffixes[i] for i in best_perm]
    logging.info(f"Run order: {' -> '.join(ordered)}  (boundary NCC sum={best_score:.3f})")
    return ordered


def unify_individual_fish(suffixes: List[str], input_paths: List[str],
                           num_channels: int = NUM_CHANNELS) -> pd.DataFrame:
    """Save all-channel fish crops with global slice numbering across runs, skipping DELETE_SLICES."""
    logging.info("Step 6 (unify): Building unified individual_fish_2d across runs")

    import shutil
    unified_dir = os.path.join(ANALYSIS_DIR, DETECTION_SUBDIR, INDIVIDUAL_FISH_DIR)
    if os.path.exists(unified_dir):
        shutil.rmtree(unified_dir)
        logging.info(f"Cleared {unified_dir}")

    ordered = determine_run_order(suffixes)
    suffix_to_input = dict(zip(suffixes, input_paths))

    tagged_finals, max_tiles = {}, {}
    for s in ordered:
        det_base = os.path.join(ANALYSIS_DIR, DETECTION_SUBDIR, s)
        df = pd.read_csv(det(det_base, 'zfish_bboxs_tagged_final.csv'))
        df['fish_id']  = df['fish_id'].astype(int)
        df['tile_idx'] = df['tile_idx'].astype(int)
        tagged_finals[s] = df
        max_tiles[s] = int(df['tile_idx'].max())

    offsets, cum = {}, 0
    for s in ordered:
        offsets[s] = cum
        cum += max_tiles[s]

    mapping_rows = []

    for s in ordered:
        df     = tagged_finals[s]
        offset = offsets[s]
        channel_files = get_channel_files(suffix_to_input[s])
        logging.info(f"  Run {s} (offset={offset}, max_tile={max_tiles[s]})")

        for ch in range(num_channels):
            with tifffile.TiffFile(channel_files[ch]) as tif:
                img_full = tif.pages[0].asarray() if len(tif.pages) > 0 else tif.asarray()

            for _, row in df.iterrows():
                fish       = int(row['fish_id']) + 1
                tile_idx   = int(row['tile_idx'])
                global_num = offset + tile_idx

                if fish in DELETE_SLICES and global_num in DELETE_SLICES[fish]:
                    continue

                r0 = max(0, int(row['bbox_global_min_row']) - FISH_CROP_MARGIN)
                c0 = max(0, int(row['bbox_global_min_col']) - FISH_CROP_MARGIN)
                r1 = min(img_full.shape[0] - 1, int(row['bbox_global_max_row']) + FISH_CROP_MARGIN)
                c1 = min(img_full.shape[1] - 1, int(row['bbox_global_max_col']) + FISH_CROP_MARGIN)
                if r1 <= r0 or c1 <= c0:
                    continue

                out_dir = os.path.join(unified_dir, str(fish), f'c{ch}')
                os.makedirs(out_dir, exist_ok=True)
                tifffile.imwrite(os.path.join(out_dir, f'{global_num}_{s}_{tile_idx}.tif'),
                                 img_full[r0:r1 + 1, c0:c1 + 1], photometric='minisblack')

                if ch == 0:
                    mapping_rows.append({'run': s, 'fish_id': int(row['fish_id']),
                                         'tile_idx': tile_idx, 'global_slice_num': global_num})

            del img_full; gc.collect()

    logging.info(f"Unified output -> {unified_dir}")
    return pd.DataFrame(mapping_rows).drop_duplicates()


# ── Summary CSVs across all runs ──────────────────────────────────────────────

KEEP_COLUMNS = [
    'run', 'tile_idx', 'global_slice_num', 'fish_name',
    'bbox_global_min_row', 'bbox_global_min_col',
    'bbox_global_max_row', 'bbox_global_max_col', 'area',
]


def generate_summary_csvs(suffixes: List[str], slice_mapping: pd.DataFrame = None) -> None:
    tagged_frames, untagged_frames = [], []
    found_tagged, found_untagged = [], []

    for suffix in suffixes:
        output_base  = os.path.join(ANALYSIS_DIR, DETECTION_SUBDIR, suffix)
        tagged_csv   = det(output_base, "zfish_bboxs_tagged_final.csv")
        untagged_csv = det(output_base, "zfish_bboxs_still_untagged.csv")

        if os.path.exists(tagged_csv):
            df = pd.read_csv(tagged_csv)
            df.insert(0, 'run', suffix)
            df.insert(1, 'fish_name', df['fish_id'] + 1)
            tagged_frames.append(df); found_tagged.append(suffix)
            logging.info(f"  Tagged {suffix}/: {len(df)} rows")
        else:
            logging.info(f"  SKIP {suffix}/ — tagged CSV not found")

        if os.path.exists(untagged_csv):
            try:
                df = pd.read_csv(untagged_csv)
                if len(df) > 0:
                    df.insert(0, 'run', suffix)
                    untagged_frames.append(df); found_untagged.append(suffix)
            except pd.errors.EmptyDataError:
                pass

    if tagged_frames:
        combined = pd.concat(tagged_frames, ignore_index=True)
        if slice_mapping is not None and not slice_mapping.empty:
            combined = combined.merge(
                slice_mapping[['run', 'fish_id', 'tile_idx', 'global_slice_num']],
                on=['run', 'fish_id', 'tile_idx'], how='left',
            )
        available_cols = [c for c in KEEP_COLUMNS if c in combined.columns]
        combined = combined[available_cols]
        out_path = os.path.join(ANALYSIS_DIR, f'fish_bbox_summary_tagged_{"_".join(found_tagged)}.csv')
        combined.to_csv(out_path, index=False)
        logging.info(f"Tagged summary: {len(combined)} rows -> {out_path}")
    else:
        logging.warning("No tagged CSVs found for summary")

    if untagged_frames:
        combined = pd.concat(untagged_frames, ignore_index=True)
        out_path = os.path.join(ANALYSIS_DIR, f'fish_bbox_summary_untagged_{"_".join(found_untagged)}.csv')
        combined.to_csv(out_path, index=False)
        logging.info(f"Untagged summary: {len(combined)} rows -> {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Zebrafish 2D slice detection and extraction pipeline.')
    parser.add_argument('--from-step', choices=STEPS, default='tiles', metavar='STEP',
                        help=f'Resume from this step. Choices: {", ".join(STEPS)}')
    args     = parser.parse_args()
    from_idx = STEPS.index(args.from_step)

    logging.info(f"Starting from step: {args.from_step}")
    suffixes = minimal_unique_suffixes(INPUT_FOLDERS)
    for folder, suffix in zip(INPUT_FOLDERS, suffixes):
        logging.info(f"  {folder} -> {suffix}")
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    input_paths = []
    for folder, suffix in zip(INPUT_FOLDERS, suffixes):
        input_path  = os.path.join(BASE_PATH, folder, MORPHOLOGY_SUBPATH)
        output_base = os.path.join(ANALYSIS_DIR, DETECTION_SUBDIR, suffix)
        run_suffix  = folder.split('__')[1] if '__' in folder else folder
        input_paths.append(input_path)

        logging.info(f"Processing: {folder}")
        os.makedirs(output_base, exist_ok=True)

        tiles_df = tiles_cropped_df = bbox_df = bbox_cropped_df = None
        tagged_df = untagged_df = tagged_final_df = None

        if from_idx in (STEPS.index('bboxes'), STEPS.index('crop')):
            tiles_df = pd.read_csv(det(output_base, f'{INITIAL_TILES_DIR}.csv'))
        if from_idx == STEPS.index('crop'):
            bbox_df = pd.read_csv(det(output_base, 'zfish_bboxs.csv'))
        if from_idx in (STEPS.index('tag'), STEPS.index('rescue')):
            tiles_cropped_df = pd.read_csv(det(output_base, 'tiles.csv'))
        if from_idx == STEPS.index('tag'):
            bbox_cropped_df = pd.read_csv(det(output_base, 'tiles_bboxs.csv'))
        if from_idx == STEPS.index('rescue'):
            tagged_df   = pd.read_csv(det(output_base, 'zfish_bboxs_tagged.csv'))
            untagged_df = pd.read_csv(det(output_base, 'zfish_bboxs_untagged.csv'))
        if from_idx == STEPS.index('save_dapi'):
            tagged_final_df = pd.read_csv(det(output_base, 'zfish_bboxs_tagged_final.csv'))

        if from_idx <= STEPS.index('tiles'):
            tiles_df = extract_tiles_dapi(input_path=input_path, output_base=output_base)

        if from_idx <= STEPS.index('bboxes'):
            bbox_df = detect_bboxes_in_tiles(output_base=output_base,
                                              input_folder=os.path.join(BASE_PATH, folder))

        if from_idx <= STEPS.index('crop'):
            tiles_cropped_df, bbox_cropped_df = crop_tiles(
                tiles_df=tiles_df, bbox_df=bbox_df, output_base=output_base)

        if from_idx <= STEPS.index('tag'):
            tagged_df, untagged_df = tag_bboxs_by_position(
                bbox_df=bbox_cropped_df, output_base=output_base, run_suffix=run_suffix)

        if from_idx <= STEPS.index('rescue'):
            tiles_for_rescue = tiles_cropped_df.rename(columns={"crop_idx": "tile_idx"})
            tagged_final_df, _ = rescue_untagged_fish(
                tagged_df=tagged_df, untagged_df=untagged_df,
                output_base=output_base, tiles_cropped_df=tiles_for_rescue)

        if from_idx <= STEPS.index('save_dapi'):
            save_fish_dapi_crops(tagged_df=tagged_final_df,
                                 output_base=output_base, input_path=input_path)

        logging.info(f"Done: {folder}")
        del tiles_df, bbox_df, tiles_cropped_df, bbox_cropped_df, tagged_df, untagged_df, tagged_final_df
        gc.collect()

    slice_mapping = None
    if from_idx <= STEPS.index('unify'):
        logging.info("Step: unify — building unified individual_fish_2d")
        slice_mapping = unify_individual_fish(suffixes, input_paths)

    logging.info("Generating cross-run summary CSVs")
    generate_summary_csvs(suffixes, slice_mapping)
    logging.info(f"All done. Results in: {ANALYSIS_DIR}/")


if __name__ == "__main__":
    main()
