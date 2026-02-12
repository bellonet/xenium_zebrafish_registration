"""
Results in individual 2D slices sorted to each of the 6 fish.
Works at full resolution throughout (pyramid didn't work).
Processes DAPI for detection, extracts all channels at the end.
"""
import os
import re
import gc
import math
import logging
import warnings
from typing import Tuple, List, Dict
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import label as ndimage_label, find_objects, binary_dilation, rotate as nd_rotate
from skimage.filters import threshold_otsu
from skimage.measure import regionprops, label as measure_label
from PIL import Image, ImageDraw, ImageFont

# Suppress tifffile multi-file pyramid warning
warnings.filterwarnings('ignore', message='.*OME series cannot read multi-file pyramids.*')

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Input - list of output folder names (relative or absolute)
BASE_PATH = '..'
INPUT_FOLDERS = [
    'output-XETG00046__0043921__Region_1__20250620__084504',
    'output-XETG00046__0044004__Region_1__20250620__084505',
    'output-XETG00046__0056729__Region_1__20250507__103905',
]
MORPHOLOGY_SUBPATH = 'morphology_focus/morphology_focus_0000.ome.tif'
DAPI_CHANNEL = 0  # Channel used for detection/analysis
NUM_CHANNELS = 4  # Total channels to process

# Output
ANALYSIS_DIR = '../analysis'
INITIAL_TILES_DIR = '_initial_tiles_tmp'  # Temporary, pre-ordering
TILES_DIR = 'tiles'  # Final tiles (cropped + ordered)
INDIVIDUAL_FISH_DIR = 'individual_fish'
BBOX_IMAGES_DIR = 'zfish_bboxs'
ROTATED_TILES_DIR = 'rotated_tiles'

# Processing parameters
DOWNSAMPLE_FACTOR = 8  # Downsample by 8x for detection (same as pyramid level 3)
MIN_BOX_AREA = 1000  # Min area at FULL resolution
BACKGROUND_INTENSITY_THRESHOLD = 0
MAX_HEIGHT = 14000
MAX_WIDTH = 14000
MIN_MAX_INTENSITY = 500
MIN_VERTICAL_OVERLAP = 100


# Bbox detection
DILATION_ITER = 30
BBOX_MARGIN = 5
MIN_LABEL_SIZE = 7000  # At full resolution (tiles are small enough)
MIN_BBOX_AREA = 100000   # Min bbox area (full res) - smaller ones get merged into nearby large boxes
MERGE_DISTANCE = 1000   # Max gap (px) between a small bbox and a large one to merge

# Cropping (script 3)
CROP_MARGIN = 5

# Rotation/tagging (script 4)
ANGLE_MIN = -25
ANGLE_MAX = 25
ANGLE_STEP = 5
Y_RANGE_SINGLE_ROW_FRAC = 0.18

# Individual fish (script 5)
FISH_CROP_MARGIN = 10

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def minimal_unique_suffixes(names: List[str]) -> List[str]:
    if len(names) <= 1:
        return [n[-3:] if len(n) >= 3 else n for n in names]

    # Strip trailing slashes
    names = [n.rstrip('/').rstrip('\\') for n in names]

    for length in range(1, max(len(n) for n in names) + 1):
        suffixes = [n[-length:] if len(n) >= length else n for n in names]
        if len(set(suffixes)) == len(names):
            return suffixes

    # Fallback: shouldn't happen unless there are duplicates
    return names

def get_channel_files(base_file_path: str) -> List[str]:
    """Get all channel files for multi-file OME-TIFF."""
    base_path = Path(base_file_path)
    stem = base_path.stem.replace('.ome', '')
    match = re.match(r'(.+)_\d+$', stem)
    if match:
        stem = match.group(1)

    parent = base_path.parent
    pattern = f"{stem}_*.ome.tif"
    channel_files = sorted(parent.glob(pattern))

    if len(channel_files) > 1:
        logging.info(f"Multi-file OME-TIFF detected: {len(channel_files)} files")
        return [str(f) for f in channel_files]
    else:
        logging.info("Single-file mode")
        return [str(base_file_path)]


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(int(v), hi))


def _tile_idx_from_name(fname: str) -> int:
    base = os.path.splitext(fname)[0]
    digits = re.sub(r"\D", "", base)
    if digits == "":
        raise ValueError(f"Could not parse tile_idx from filename: {fname}")
    return int(digits)


def simple_downsample(img: np.ndarray, factor: int) -> np.ndarray:
    """Simple average downsampling."""
    h, w = img.shape[:2]
    h_ds = h // factor
    w_ds = w // factor

    # Crop to multiple of factor
    img_crop = img[:h_ds * factor, :w_ds * factor]

    # Average downsample
    if img_crop.ndim == 2:
        img_ds = img_crop.reshape(h_ds, factor, w_ds, factor).mean(axis=(1, 3))
    else:
        img_ds = img_crop.reshape(h_ds, factor, w_ds, factor, -1).mean(axis=(1, 3))

    return img_ds.astype(img.dtype)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: EXTRACT TILES
# ══════════════════════════════════════════════════════════════════════════════

def vertical_intervals_connected(a_min, a_max, b_min, b_max, min_overlap=MIN_VERTICAL_OVERLAP):
    # compute signed overlap (can be negative if gap)
    overlap = min(a_max, b_max) - max(a_min, b_min)
    return overlap >= min_overlap


def group_rows_by_vertical_connectivity(boxes, min_overlap=MIN_VERTICAL_OVERLAP):
    """
    boxes: list of tuples (id, min_row, min_col, max_row, max_col, area)
    Returns: boxes sorted row-by-row (bottom->top) and left->right within rows.
    """
    n = len(boxes)
    if n <= 1:
        return boxes

    # build adjacency
    adj = [[] for _ in range(n)]
    for i in range(n):
        _, a_min, _, a_max, _, _ = boxes[i]
        for j in range(i + 1, n):
            _, b_min, _, b_max, _, _ = boxes[j]
            if vertical_intervals_connected(a_min, a_max, b_min, b_max, min_overlap):
                adj[i].append(j)
                adj[j].append(i)

    # connected components
    comp = [-1] * n
    cid = 0
    for i in range(n):
        if comp[i] != -1:
            continue
        q = deque([i]); comp[i] = cid
        while q:
            u = q.popleft()
            for v in adj[u]:
                if comp[v] == -1:
                    comp[v] = cid
                    q.append(v)
        cid += 1

    # group and compute median center_y for row ordering
    groups = {}
    for idx, c in enumerate(comp):
        groups.setdefault(c, []).append(boxes[idx])

    rows = []
    for c, blist in groups.items():
        blist_sorted = sorted(blist, key=lambda b: b[2])  # min_col => left->right
        centers_y = [0.5 * (b[1] + b[3]) for b in blist_sorted]
        median_y = float(np.median(centers_y))
        rows.append((median_y, blist_sorted))

    # bottom->top (larger center_y first)
    rows.sort(key=lambda x: -x[0])
    sorted_boxes = [b for _, row in rows for b in row]
    return sorted_boxes


def find_bounding_boxes(img: np.ndarray, downsample_factor: int = 1) -> List[Tuple]:
    """
    Find bounding boxes in image.
    If downsample_factor > 1, downsamples for detection and scales coordinates back.
    """
    if downsample_factor > 1:
        logging.info(f"Downsampling by {downsample_factor}x for detection")
        img_ds = simple_downsample(img, downsample_factor)
        logging.info(f"Downsampled from {img.shape} to {img_ds.shape}")
    else:
        img_ds = img

    mask = img_ds > BACKGROUND_INTENSITY_THRESHOLD
    labeled, num = ndimage_label(mask)
    slices = find_objects(labeled)
    boxes = []

    for idx, slc in enumerate(slices, start=1):
        if slc is None:
            continue

        # Coordinates in downsampled space
        min_row_ds = slc[0].start
        max_row_ds = slc[0].stop - 1
        min_col_ds = slc[1].start
        max_col_ds = slc[1].stop - 1
        area_ds = (max_row_ds - min_row_ds + 1) * (max_col_ds - min_col_ds + 1)

        # Scale back to full resolution
        min_row = int(min_row_ds * downsample_factor)
        max_row = int(max_row_ds * downsample_factor)
        min_col = int(min_col_ds * downsample_factor)
        max_col = int(max_col_ds * downsample_factor)
        area = (max_row - min_row + 1) * (max_col - min_col + 1)

        if area < MIN_BOX_AREA:
            continue

        boxes.append((idx, min_row, min_col, max_row, max_col, area))

    return boxes


def extract_tiles_dapi(
        input_path: str = None,
        output_base: str = None,
        dapi_channel: int = None
) -> pd.DataFrame:
    """Extract tiles from DAPI channel only."""
    if input_path is None:
        raise ValueError("input_path must be provided")
    if output_base is None:
        raise ValueError("output_base must be provided")
    if dapi_channel is None:
        dapi_channel = DAPI_CHANNEL

    logging.info(f"Step 1: Extracting tiles from DAPI (downsampled {DOWNSAMPLE_FACTOR}x for detection)")

    filename = os.path.basename(input_path)
    channel_files = get_channel_files(input_path)

    # Read DAPI channel
    dapi_file = channel_files[dapi_channel]
    logging.info(f"Reading DAPI from {Path(dapi_file).name}")

    with tifffile.TiffFile(dapi_file) as tif:
        # For multi-file OME, each file should have one page
        # But tif.asarray() might read metadata incorrectly
        # Read the first page directly
        if len(tif.pages) > 0:
            img = tif.pages[0].asarray()
        else:
            img = tif.asarray()

    logging.info(f"DAPI image shape: {img.shape}")

    # Find boxes (downsampled for speed, coordinates scaled to full res)
    orig = find_bounding_boxes(img, downsample_factor=DOWNSAMPLE_FACTOR)
    logging.info(f"Found {len(orig)} initial boxes")

    # Split oversized boxes
    boxes = []
    for oid, r0, c0, r1, c1, area in orig:
        h = r1 - r0 + 1
        w = c1 - c0 + 1

        if h > MAX_HEIGHT:
            mid = (r0 + r1) // 2
            boxes.append((oid, r0, c0, mid, c1, area // 2))
            boxes.append((oid, mid + 1, c0, r1, c1, area // 2))
        elif w > MAX_WIDTH:
            mid = (c0 + c1) // 2
            boxes.append((oid, r0, c0, r1, mid, area // 2))
            boxes.append((oid, r0, mid + 1, r1, c1, area // 2))
        else:
            boxes.append((oid, r0, c0, r1, c1, area))

    logging.info(f"After splitting: {len(boxes)} tiles")

    # NOTE: Does NOT sort boxes here. Ordering is deferred to after cropping (Step 3).

    # Extract tiles from DAPI
    tiles_dir = os.path.join(output_base, INITIAL_TILES_DIR, f'ch{dapi_channel}')
    os.makedirs(tiles_dir, exist_ok=True)

    all_boxes = []
    seq_idx = 1
    for enumerated_idx, (old_idx, min_row, min_col, max_row, max_col, area) in enumerate(boxes, start=1):
        crop = img[min_row:max_row + 1, min_col:max_col + 1]

        # compute intensities and discard if too dark
        min_val = float(np.min(crop))
        max_val = float(np.max(crop))
        if max_val < MIN_MAX_INTENSITY:
            logging.info(f"Discarding original box {enumerated_idx} (max_intensity={max_val:.1f} < {MIN_MAX_INTENSITY})")
            continue

        out_path = os.path.join(tiles_dir, f'{seq_idx}.tif')
        tifffile.imwrite(out_path, crop)

        all_boxes.append({
            'filename': filename,
            'channel': dapi_channel,
            'crop_idx': seq_idx,
            'min_row': min_row,
            'min_col': min_col,
            'max_row': max_row,
            'max_col': max_col,
            'area': area,
            'min_intensity': min_val,
            'max_intensity': max_val,
        })

        seq_idx += 1

    # Free full-res DAPI image
    del img
    gc.collect()

    output_csv_path = os.path.join(output_base, f'{INITIAL_TILES_DIR}.csv')
    df = pd.DataFrame(all_boxes)
    df.to_csv(output_csv_path, index=False)
    logging.info(f"Saved {len(all_boxes)} tile metadata → {output_csv_path}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: DETECT BBOXES IN TILES
# ══════════════════════════════════════════════════════════════════════════════

def process_tile_for_bboxes(tile: np.ndarray, dilation_iter: int) -> List[Dict]:
    """Apply Otsu thresholding and detect fish bboxes."""
    thr = threshold_otsu(tile)
    mask = binary_dilation(tile > thr, iterations=dilation_iter)
    lbl = measure_label(mask)  # Use skimage.measure.label
    detections = []

    for rp in regionprops(lbl):
        if rp.area < MIN_LABEL_SIZE:
            continue
        min_r, min_c, max_r, max_c = rp.bbox

        detections.append({
            'label': rp.label,
            'bbox': (min_r, min_c, max_r, max_c),
            'area': rp.area
        })

    return detections


def _bbox_gap(a, b):
    """Min gap between two bboxes (0 if overlapping)."""
    a_min_r, a_min_c, a_max_r, a_max_c = a
    b_min_r, b_min_c, b_max_r, b_max_c = b
    dy = max(0, max(a_min_r, b_min_r) - min(a_max_r, b_max_r))
    dx = max(0, max(a_min_c, b_min_c) - min(a_max_c, b_max_c))
    return math.sqrt(dy * dy + dx * dx)


def merge_small_bboxes(detections: List[Dict],
                       min_area: int = MIN_BBOX_AREA,
                       max_dist: float = MERGE_DISTANCE) -> List[Dict]:
    """Merge small bboxes into nearby large bboxes.

    A bbox whose area (width*height) < min_area is merged into the closest
    large bbox within max_dist pixels.  A large bbox can absorb multiple
    small ones.  Small bboxes with no large neighbour are kept as-is.
    """
    if not detections:
        return detections

    # Classify into large / small
    large = []
    small = []
    for det in detections:
        min_r, min_c, max_r, max_c = det['bbox']
        bbox_area = (max_r - min_r) * (max_c - min_c)
        if bbox_area >= min_area:
            large.append(det)
        else:
            small.append(det)

    if not small:
        return detections

    # For each small bbox, find closest large bbox and merge if within distance
    unmerged = []
    for s in small:
        best_idx = -1
        best_dist = float('inf')
        for i, lg in enumerate(large):
            d = _bbox_gap(s['bbox'], lg['bbox'])
            if d < best_dist:
                best_dist = d
                best_idx = i
        if best_idx >= 0 and best_dist <= max_dist:
            # Expand the large bbox to include the small one
            lg = large[best_idx]
            s_min_r, s_min_c, s_max_r, s_max_c = s['bbox']
            l_min_r, l_min_c, l_max_r, l_max_c = lg['bbox']
            new_bbox = (
                min(l_min_r, s_min_r),
                min(l_min_c, s_min_c),
                max(l_max_r, s_max_r),
                max(l_max_c, s_max_c),
            )
            large[best_idx] = {
                'label': lg['label'],
                'bbox': new_bbox,
                'area': lg['area'] + s['area'],
            }
            logging.debug(f"Merged small bbox (area={s['bbox']}) into label {lg['label']}")
        else:
            unmerged.append(s)
            logging.debug(f"Kept small bbox with no nearby large neighbour (dist={best_dist:.0f})")

    return large + unmerged


def detect_bboxes_in_tiles(
        tiles_df: pd.DataFrame = None,
        tiles_base: str = None,
        output_base: str = None,
        dapi_channel: int = None
) -> pd.DataFrame:
    """Detect fish bboxes in tiles at full resolution."""
    if tiles_base is None:
        raise ValueError("tiles_base must be provided")
    if output_base is None:
        raise ValueError("output_base must be provided")
    if dapi_channel is None:
        dapi_channel = DAPI_CHANNEL

    logging.info(f"Step 2: Detecting bboxes in tiles (full resolution)")

    tiles_dir = os.path.join(tiles_base, INITIAL_TILES_DIR, f'ch{dapi_channel}')
    bbox_img_dir = os.path.join(output_base, BBOX_IMAGES_DIR)
    os.makedirs(bbox_img_dir, exist_ok=True)

    csv_path = os.path.join(output_base, 'zfish_bboxs.csv')
    all_detections = []

    # Get tile files
    image_files = [
        f for f in os.listdir(tiles_dir)
        if os.path.isfile(os.path.join(tiles_dir, f)) and f.lower().endswith(('.tif', '.tiff'))
    ]
    image_files.sort(key=lambda f: int(re.sub(r'\D', '', os.path.splitext(f)[0])))

    for image_name in image_files:
        input_path = os.path.join(tiles_dir, image_name)
        tile_idx = _tile_idx_from_name(image_name)

        # Read tile at FULL resolution
        image = tifffile.imread(input_path)

        # Detect at full resolution (no downsampling)
        detections = process_tile_for_bboxes(image, DILATION_ITER)

        # Merge small bboxes into nearby large ones
        detections = merge_small_bboxes(detections)

        for det in detections:
            min_r, min_c, max_r, max_c = det['bbox']
            all_detections.append({
                'image_name': image_name,
                'tile_idx': tile_idx,
                'label': det['label'],
                'area': det['area'],
                'min_row': min_r,
                'min_col': min_c,
                'max_row': max_r,
                'max_col': max_c
            })

        # Draw visualization at full resolution
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from matplotlib import patheffects

        fig, ax = plt.subplots(1)
        ax.imshow(image, cmap='gray')

        for det in detections:
            min_r, min_c, max_r, max_c = det['bbox']
            start_row = max(0, min_r - BBOX_MARGIN)
            start_col = max(0, min_c - BBOX_MARGIN)
            height = (max_r + BBOX_MARGIN) - start_row
            width = (max_c + BBOX_MARGIN) - start_col

            ax.add_patch(Rectangle(
                (start_col, start_row), width, height,
                linewidth=1, edgecolor='r', facecolor='none'
            ))

            txt = ax.text(start_col, start_row, str(det['label']),
                          fontsize=8, color='r', va='bottom', ha='left')
            txt.set_path_effects([
                patheffects.Stroke(linewidth=2, foreground='black'),
                patheffects.Normal()
            ])

        ax.set_axis_off()
        output_path = os.path.join(bbox_img_dir, f"{os.path.splitext(image_name)[0]}.png")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)

    # Save CSV
    df = pd.DataFrame(all_detections)
    df.to_csv(csv_path, index=False)
    logging.info(f"Detected {len(all_detections)} bboxes → {csv_path}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: CROP TILES
# ══════════════════════════════════════════════════════════════════════════════

def crop_tiles(
        tiles_df: pd.DataFrame = None,
        bbox_df: pd.DataFrame = None,
        tiles_base: str = None,
        output_base: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Crop tiles around detected bboxes (DAPI only), then sort bottom-left to top-right."""
    if tiles_base is None:
        raise ValueError("tiles_base must be provided")
    if output_base is None:
        raise ValueError("output_base must be provided")

    logging.info("Step 3: Cropping tiles (DAPI only, full resolution)")

    bbox_df["tile_idx"] = bbox_df["tile_idx"].astype(int)
    tiles_df["crop_idx"] = tiles_df["crop_idx"].astype(int)

    tile_key = tiles_df.set_index("crop_idx")[
        ["filename", "channel", "min_row", "min_col", "max_row", "max_col"]
    ].to_dict("index")

    groups = {k: g for k, g in bbox_df.groupby("tile_idx")}

    tiles_dir = os.path.join(tiles_base, INITIAL_TILES_DIR, 'ch0')

    files = [
        f for f in os.listdir(tiles_dir)
        if os.path.isfile(os.path.join(tiles_dir, f)) and f.lower().endswith(('.tif', '.tiff'))
    ]
    files.sort(key=_tile_idx_from_name)

    # --- Phase 1: Crop all tiles, collect metadata with TEMPORARY indices ---
    tmp_crops = []  # list of (tmp_tile_idx, crop_array, tile_row_dict, bbox_row_dicts, crop_global_coords)

    for fname in files:
        tile_idx = _tile_idx_from_name(fname)
        g = groups.get(tile_idx, None)

        if g is None or g.empty:
            continue

        path = os.path.join(tiles_dir, fname)
        img = tifffile.imread(path)
        H, W = img.shape[:2]

        # Union bbox in TILE frame
        u_min_r = int(g["min_row"].min())
        u_min_c = int(g["min_col"].min())
        u_max_r = int(g["max_row"].max())
        u_max_c = int(g["max_col"].max())

        # Expand + clamp
        s_r = _clamp(u_min_r - CROP_MARGIN, 0, H - 1)
        s_c = _clamp(u_min_c - CROP_MARGIN, 0, W - 1)
        e_r = _clamp(u_max_r + CROP_MARGIN, 0, H - 1)
        e_c = _clamp(u_max_c + CROP_MARGIN, 0, W - 1)

        crop = img[s_r:e_r + 1, s_c:e_c + 1]
        crop_h, crop_w = int(crop.shape[0]), int(crop.shape[1])

        if tile_idx not in tile_key:
            raise KeyError(f"Tile {tile_idx} not found in initial_tiles.csv")

        tinfo = tile_key[tile_idx]
        tile_min_r_g = int(tinfo["min_row"])
        tile_min_c_g = int(tinfo["min_col"])

        crop_min_r_g = tile_min_r_g + s_r
        crop_min_c_g = tile_min_c_g + s_c
        crop_max_r_g = tile_min_r_g + e_r
        crop_max_c_g = tile_min_c_g + e_c

        tile_row = {
            "filename": tinfo["filename"],
            "channel": int(tinfo["channel"]),
            "min_row": crop_min_r_g,
            "min_col": crop_min_c_g,
            "max_row": crop_max_r_g,
            "max_col": crop_max_c_g,
            "area": (crop_max_r_g - crop_min_r_g + 1) * (crop_max_c_g - crop_min_c_g + 1),
        }

        bbox_rows = []
        for _, r in g.iterrows():
            rmin, cmin = int(r["min_row"]), int(r["min_col"])
            rmax, cmax = int(r["max_row"]), int(r["max_col"])

            loc_min_r = rmin - s_r
            loc_min_c = cmin - s_c
            loc_max_r = rmax - s_r
            loc_max_c = cmax - s_c

            g_min_r = tile_min_r_g + rmin
            g_min_c = tile_min_c_g + cmin
            g_max_r = tile_min_r_g + rmax
            g_max_c = tile_min_c_g + cmax

            bbox_rows.append({
                "image_name": r["image_name"],
                "label": int(r["label"]),
                "area": int(r["area"]),
                "min_row": rmin,
                "min_col": cmin,
                "max_row": rmax,
                "max_col": cmax,
                "crop_origin_row": s_r,
                "crop_origin_col": s_c,
                "crop_h": crop_h,
                "crop_w": crop_w,
                "bbox_local_min_row": int(loc_min_r),
                "bbox_local_min_col": int(loc_min_c),
                "bbox_local_max_row": int(loc_max_r),
                "bbox_local_max_col": int(loc_max_c),
                "bbox_global_min_row": int(g_min_r),
                "bbox_global_min_col": int(g_min_c),
                "bbox_global_max_row": int(g_max_r),
                "bbox_global_max_col": int(g_max_c),
            })

        tmp_crops.append((tile_idx, crop, tile_row, bbox_rows))

    # --- Phase 2: Sort cropped tiles by position (bottom-left to top-right) ---
    # Build boxes in the format expected by group_rows_by_vertical_connectivity:
    #   (id, min_row, min_col, max_row, max_col, area)
    sortable_boxes = []
    for i, (tmp_idx, crop, tile_row, bbox_rows) in enumerate(tmp_crops):
        sortable_boxes.append((
            i,  # use list index as id
            tile_row["min_row"],
            tile_row["min_col"],
            tile_row["max_row"],
            tile_row["max_col"],
            tile_row["area"],
        ))

    sorted_boxes = group_rows_by_vertical_connectivity(sortable_boxes, min_overlap=MIN_VERTICAL_OVERLAP)

    # Map from original list index to new sequential tile_idx (1-based)
    sorted_order = [box[0] for box in sorted_boxes]  # box[0] is the original list index

    logging.info(f"Tile ordering after crop: {[tmp_crops[i][0] for i in sorted_order]} -> {list(range(1, len(sorted_order) + 1))}")

    # --- Phase 3: Save with final ordering ---
    final_tiles_dir = os.path.join(output_base, TILES_DIR, 'ch0')
    os.makedirs(final_tiles_dir, exist_ok=True)

    out_tiles_rows = []
    out_bbox_rows = []

    for new_idx_0, orig_list_idx in enumerate(sorted_order):
        new_tile_idx = new_idx_0 + 1  # 1-based
        tmp_idx, crop, tile_row, bbox_rows = tmp_crops[orig_list_idx]

        out_name = f"{new_tile_idx}.tif"
        tifffile.imwrite(os.path.join(final_tiles_dir, out_name), crop)

        tile_row["crop_idx"] = new_tile_idx
        out_tiles_rows.append(tile_row)

        for br in bbox_rows:
            br["tile_idx"] = new_tile_idx
            br["cropped_name"] = out_name
            out_bbox_rows.append(br)

    tiles_out_csv = os.path.join(output_base, "tiles.csv")
    bbox_out_csv = os.path.join(output_base, "tiles_bboxs.csv")

    tiles_out_df = pd.DataFrame(out_tiles_rows)
    bbox_out_df = pd.DataFrame(out_bbox_rows)

    tiles_out_df.to_csv(tiles_out_csv, index=False)
    bbox_out_df.to_csv(bbox_out_csv, index=False)

    logging.info(f"Cropped & ordered {len(out_tiles_rows)} tiles")
    logging.info(f"Wrote {tiles_out_csv}")
    logging.info(f"Wrote {bbox_out_csv}")

    # Clean up temporary initial tiles
    import shutil
    tmp_dir = os.path.join(output_base, INITIAL_TILES_DIR)
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
        logging.info(f"Removed temporary directory: {tmp_dir}")
    tmp_csv = os.path.join(output_base, f'{INITIAL_TILES_DIR}.csv')
    if os.path.isfile(tmp_csv):
        os.remove(tmp_csv)
        logging.info(f"Removed temporary CSV: {tmp_csv}")

    return tiles_out_df, bbox_out_df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: TAG BBOXES BY POSITION
# ══════════════════════════════════════════════════════════════════════════════

def rotation_padding(h: int, w: int, angle_deg: float) -> Tuple[int, int, int, int]:
    a = math.radians(abs(angle_deg))
    new_w = abs(w * math.cos(a)) + abs(h * math.sin(a))
    new_h = abs(w * math.sin(a)) + abs(h * math.cos(a))
    pad_w = max(0, int(math.ceil((new_w - w) / 2)))
    pad_h = max(0, int(math.ceil((new_h - h) / 2)))
    return pad_h, pad_h, pad_w, pad_w


def rotate_points(points_rc: np.ndarray, angle_deg: float, center_rc: Tuple[float, float]) -> np.ndarray:
    cy, cx = center_rc
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)

    pts = points_rc.astype(np.float64).copy()
    x = pts[:, 1] - cx
    y = -(pts[:, 0] - cy)

    xr = x * c - y * s
    yr = x * s + y * c

    pts[:, 1] = xr + cx
    pts[:, 0] = (-yr) + cy
    return pts


def kmeans_1d_two_clusters(y: np.ndarray, iters: int = 20) -> Tuple[np.ndarray, float, float]:
    y = y.astype(np.float64)
    c0, c1 = np.percentile(y, [30, 70])
    lab = np.zeros(len(y), dtype=int)
    for _ in range(iters):
        d0 = np.abs(y - c0)
        d1 = np.abs(y - c1)
        lab = (d1 < d0).astype(int)
        if np.all(lab == 0) or np.all(lab == 1):
            order = np.argsort(y)
            lab = np.zeros_like(y, dtype=int)
            lab[order[len(y) // 2:]] = 1
        c0_new = y[lab == 0].mean()
        c1_new = y[lab == 1].mean()
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
    score = (v0 + v1) + (2000.0 / (sep * sep))
    return float(score), lab


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
    img = np.clip((img - lo) / (hi - lo), 0, 1)
    return (img * 255.0).astype(np.uint8)


def rotated_bbox_from_local_bbox(
        min_r: float, min_c: float, max_r: float, max_c: float,
        pad_t: int, pad_l: int,
        angle_deg: float, center_rc: Tuple[float, float],
) -> Tuple[float, float, float, float]:
    corners = np.array([
        [min_r + pad_t, min_c + pad_l],
        [min_r + pad_t, max_c + pad_l],
        [max_r + pad_t, min_c + pad_l],
        [max_r + pad_t, max_c + pad_l],
    ], dtype=np.float64)
    rot = rotate_points(corners, angle_deg, center_rc)
    rmin = float(rot[:, 0].min())
    cmin = float(rot[:, 1].min())
    rmax = float(rot[:, 0].max())
    cmax = float(rot[:, 1].max())
    return rmin, cmin, rmax, cmax


def x_match(a0: float, a1: float, b0: float, b1: float, tol_px: float = 50.0) -> bool:
    ac = 0.5 * (a0 + a1)
    bc = 0.5 * (b0 + b1)
    if abs(ac - bc) <= tol_px:
        return True
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    return inter > 0.0


def infer_ids_by_column_overlap(
        rot_boxes: List[Tuple[float, float, float, float]],
        row_labels: np.ndarray,
        fish_ids: List[int],
        tagged: List[bool],
) -> Tuple[List[int], List[bool], List[str]]:
    n = len(fish_ids)
    tag_source = ["strict" if tagged[i] else "untagged" for i in range(n)]
    fish_ids = list(fish_ids)
    tagged_final = list(tagged)

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

        candidate_cols = set()
        for col, items in tagged_by_col.items():
            for (j, tc0, tc1) in items:
                if x_match(c0, c1, tc0, tc1, tol_px=500.0):
                    candidate_cols.add(col)

        if len(candidate_cols) == 1:
            col = list(candidate_cols)[0]
            fish_ids[i] = row_base + col
            tagged_final[i] = True
            tag_source[i] = "inferred"

    return fish_ids, tagged_final, tag_source


def assign_only_unambiguous(
        rot_centers: np.ndarray,
        row_labels: np.ndarray,
) -> Tuple[List[int], List[bool]]:
    n = len(rot_centers)
    fish_id = [-1] * n
    tagged = [False] * n

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
        xs = [(rot_centers[i, 1], i) for i in indices]
        xs.sort()
        for col, (_, i) in enumerate(xs):
            fish_id[i] = base + col
            tagged[i] = True

    if n == 6:
        label_row(top_idx, 0)
        label_row(bot_idx, 3)
        return fish_id, tagged

    if n in (4, 5):
        if len(top_idx) == 3:
            label_row(top_idx, 0)
        if len(bot_idx) == 3:
            label_row(bot_idx, 3)
        return fish_id, tagged

    return fish_id, tagged


def load_font(size: int = 60):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def tag_bboxs_by_position(
        bbox_df: pd.DataFrame = None,
        cropped_tiles_base: str = None,
        output_base: str = None,
        dapi_channel: int = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Find best rotation and tag fish by position."""
    if output_base is None:
        raise ValueError("output_base must be provided")
    if cropped_tiles_base is None:
        cropped_tiles_base = output_base
    if dapi_channel is None:
        dapi_channel = DAPI_CHANNEL

    logging.info("Step 4: Tagging bboxes by position (full resolution)")

    cropped_tiles_dir = os.path.join(cropped_tiles_base, TILES_DIR, f'ch{dapi_channel}')
    rotated_img_dir = os.path.join(output_base, ROTATED_TILES_DIR)
    os.makedirs(rotated_img_dir, exist_ok=True)

    need_cols = {"tile_idx", "bbox_local_min_row", "bbox_local_min_col",
                 "bbox_local_max_row", "bbox_local_max_col"}
    if not need_cols.issubset(bbox_df.columns):
        raise ValueError(f"Missing columns: {need_cols - set(bbox_df.columns)}")

    bbox_df["tile_idx"] = bbox_df["tile_idx"].astype(int)

    tile_rotation_rows: List[Dict] = []
    tagged_rows: List[pd.DataFrame] = []
    untagged_rows: List[pd.DataFrame] = []

    font = load_font(150)

    for tile_idx, g in bbox_df.groupby("tile_idx"):
        tile_path = os.path.join(cropped_tiles_dir, f"{tile_idx}.tif")
        if not os.path.isfile(tile_path):
            logging.warning(f"Missing tile {tile_idx}")
            continue

        img = tifffile.imread(tile_path)
        if img.ndim != 2:
            img = img[0]
        H, W = img.shape
        n = len(g)

        # bbox centers
        centers = np.stack([
            (g["bbox_local_min_row"].to_numpy() + g["bbox_local_max_row"].to_numpy()) / 2.0,
            (g["bbox_local_min_col"].to_numpy() + g["bbox_local_max_col"].to_numpy()) / 2.0,
        ], axis=1)

        y_range = float(centers[:, 0].max() - centers[:, 0].min()) if n > 0 else 0.0
        prefer_two = decide_prefer_two_rows(n, y_range, H)

        angles = list(range(ANGLE_MIN, ANGLE_MAX + 1, ANGLE_STEP)) if n >= 4 else [0]
        center_rc = (H / 2.0, W / 2.0)

        best_angle = 0.0
        best_score = float("inf")
        best_labels = np.zeros(n, dtype=int)
        best_rot_centers = centers.copy()

        if n == 0:
            angles = [0]

        for ang in angles:
            rot_c = rotate_points(centers, ang, center_rc)
            if prefer_two and n >= 2:
                s, lab = score_two_rows(rot_c)
            else:
                s, lab = float(np.var(rot_c[:, 0])) if n > 1 else 0.0, np.zeros(n, dtype=int)

            if s < best_score:
                best_score = s
                best_angle = float(ang)
                best_labels = lab
                best_rot_centers = rot_c

        # padding and rotate
        pad_t, pad_b, pad_l, pad_r = rotation_padding(H, W, best_angle)
        img_pad = np.pad(img, ((pad_t, pad_b), (pad_l, pad_r)), mode="constant", constant_values=0)
        Hp, Wp = img_pad.shape
        center_pad = (Hp / 2.0, Wp / 2.0)

        img_rot = nd_rotate(img_pad, best_angle, reshape=False, order=1, mode="constant", cval=0.0)
        img_u8 = to_uint8(img_rot)

        fish_ids, tagged = assign_only_unambiguous(best_rot_centers, best_labels)

        # compute rotated boxes
        rot_boxes = []
        for _, row in g.reset_index(drop=True).iterrows():
            rmin, cmin, rmax, cmax = rotated_bbox_from_local_bbox(
                row["bbox_local_min_row"], row["bbox_local_min_col"],
                row["bbox_local_max_row"], row["bbox_local_max_col"],
                pad_t, pad_l, best_angle, center_pad
            )
            rot_boxes.append((rmin, cmin, rmax, cmax))

        # infer extra IDs
        fish_ids, tagged, tag_source = infer_ids_by_column_overlap(
            rot_boxes=rot_boxes,
            row_labels=best_labels,
            fish_ids=fish_ids,
            tagged=tagged,
        )

        tile_rotation_rows.append({
            "tile_idx": tile_idx,
            "n_detections": n,
            "best_angle_deg": best_angle,
            "score": best_score,
            "pad_top": pad_t,
            "pad_bottom": pad_b,
            "pad_left": pad_l,
            "pad_right": pad_r,
            "prefer_two_rows": int(prefer_two),
            "n_tagged": int(np.sum(tagged)),
        })

        # build output dataframe
        gg = g.copy().reset_index(drop=True)
        gg["tile_best_angle_deg"] = best_angle
        gg["tile_pad_top"] = pad_t
        gg["tile_pad_left"] = pad_l
        gg["tile_pad_bottom"] = pad_b
        gg["tile_pad_right"] = pad_r
        gg["fish_id"] = fish_ids
        gg["is_tagged"] = tagged
        gg["tag_source"] = tag_source

        gg["bbox_rot_min_row"] = [r[0] for r in rot_boxes]
        gg["bbox_rot_min_col"] = [r[1] for r in rot_boxes]
        gg["bbox_rot_max_row"] = [r[2] for r in rot_boxes]
        gg["bbox_rot_max_col"] = [r[3] for r in rot_boxes]

        tagged_rows.append(gg[gg["is_tagged"] == True])
        untagged_rows.append(gg[gg["is_tagged"] == False])

        # draw visualization
        im = Image.fromarray(img_u8, mode="L").convert("RGB")
        draw = ImageDraw.Draw(im)

        for i, row in gg.iterrows():
            rmin, cmin, rmax, cmax = rot_boxes[i]
            rmin = max(0, min(rmin, Hp - 1))
            rmax = max(0, min(rmax, Hp - 1))
            cmin = max(0, min(cmin, Wp - 1))
            cmax = max(0, min(cmax, Wp - 1))

            draw.rectangle([cmin, rmin, cmax, rmax], outline=(255, 255, 255), width=2)

            if bool(row["is_tagged"]) and int(row["fish_id"]) >= 0:
                txt = str(int(row["fish_id"]) + 1)
                tx = float(cmin) + 4
                ty = float(rmin) + 4

                l, t, r, b = draw.textbbox((0, 0), txt, font=font)
                tw, th = (r - l), (b - t)

                draw.rectangle([tx - 2, ty - 2, tx + tw + 2, ty + th + 2], fill=(0, 0, 0))
                draw.text((tx, ty), txt, fill=(255, 255, 255), font=font)
            else:
                x1, y1 = float(cmin) + 4, float(rmin) + 4
                x2, y2 = float(cmax) - 4, float(rmax) - 4
                draw.line([x1, y1, x2, y2], fill=(255, 0, 0), width=3)
                draw.line([x1, y2, x2, y1], fill=(255, 0, 0), width=3)

        out_png = os.path.join(rotated_img_dir, f"{tile_idx}.png")
        im.save(out_png)

        logging.info(f"tile {tile_idx}: n={n}, angle={best_angle:+.0f}°, tagged={int(np.sum(tagged))}")

    # write CSVs
    rot_csv = os.path.join(output_base, "tile_rotation.csv")
    tagged_csv = os.path.join(output_base, "zfish_bboxs_tagged.csv")
    untagged_csv = os.path.join(output_base, "zfish_bboxs_untagged.csv")

    rot_df = pd.DataFrame(tile_rotation_rows).sort_values("tile_idx")
    rot_df.to_csv(rot_csv, index=False)

    tagged_df = pd.concat(tagged_rows, axis=0, ignore_index=True) if tagged_rows else pd.DataFrame()
    untagged_df = pd.concat(untagged_rows, axis=0, ignore_index=True) if untagged_rows else pd.DataFrame()

    tagged_df.to_csv(tagged_csv, index=False)
    untagged_df.to_csv(untagged_csv, index=False)

    logging.info(f"Wrote {tagged_csv} ({len(tagged_df)} rows)")
    logging.info(f"Wrote {untagged_csv} ({len(untagged_df)} rows)")

    return tagged_df, untagged_df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: SAVE INDIVIDUAL FISH - ALL CHANNELS
# ══════════════════════════════════════════════════════════════════════════════

def clamp_int(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(int(round(v)), hi)))


def save_individual_fish_multi_channel(
        tagged_df: pd.DataFrame = None,
        output_base: str = None,
        input_path: str = None,
        num_channels: int = None
):
    """Save individual fish from all channels using global bbox coordinates."""
    if output_base is None:
        raise ValueError("output_base must be provided")
    if input_path is None:
        raise ValueError("input_path must be provided")
    if num_channels is None:
        num_channels = NUM_CHANNELS

    logging.info("Step 5: Saving individual fish (all channels, full resolution)")

    if tagged_df is None or len(tagged_df) == 0:
        logging.warning("No tagged fish to save")
        return

    req = {"tile_idx", "fish_id", "bbox_global_min_row", "bbox_global_min_col",
           "bbox_global_max_row", "bbox_global_max_col"}
    missing = req - set(tagged_df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    tagged_df["tile_idx"] = tagged_df["tile_idx"].astype(int)
    tagged_df["fish_id"] = tagged_df["fish_id"].astype(int)

    # Get channel files
    channel_files = get_channel_files(input_path)

    written = 0
    skipped = 0

    # Process each channel
    for ch in range(num_channels):
        logging.info(f"Extracting individual fish from channel {ch}")
        channel_file = channel_files[ch]

        # Read full resolution image
        with tifffile.TiffFile(channel_file) as tif:
            if len(tif.pages) > 0:
                img_full = tif.pages[0].asarray()
            else:
                img_full = tif.asarray()

        for _, row in tagged_df.iterrows():
            fish = int(row["fish_id"]) + 1  # 1..6

            # Use global bbox coordinates
            rmin = int(row["bbox_global_min_row"])
            cmin = int(row["bbox_global_min_col"])
            rmax = int(row["bbox_global_max_row"])
            cmax = int(row["bbox_global_max_col"])

            # Add margin
            r0 = max(0, rmin - FISH_CROP_MARGIN)
            c0 = max(0, cmin - FISH_CROP_MARGIN)
            r1 = min(img_full.shape[0] - 1, rmax + FISH_CROP_MARGIN)
            c1 = min(img_full.shape[1] - 1, cmax + FISH_CROP_MARGIN)

            if r1 <= r0 or c1 <= c0:
                if ch == 0:
                    skipped += 1
                continue

            crop = img_full[r0:r1 + 1, c0:c1 + 1]

            # Folder structure: ch#/fish#/2d/
            out_dir = os.path.join(output_base, INDIVIDUAL_FISH_DIR, f'ch{ch}', str(fish), "2d")
            os.makedirs(out_dir, exist_ok=True)
            tile_idx = int(row["tile_idx"])
            out_path = os.path.join(out_dir, f"{tile_idx}.tif")

            tifffile.imwrite(out_path, crop, photometric="minisblack")
            if ch == 0:
                written += 1

        logging.info(f"Channel {ch}: wrote {written} crops")

        # Free full-res image before loading next channel
        del img_full
        gc.collect()

    logging.info(f"Done. Written={written}, skipped={skipped}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Run the complete pipeline for all input folders."""
    logging.info("=" * 80)
    logging.info("ZEBRAFISH PROCESSING PIPELINE - MULTI-FOLDER")
    logging.info("=" * 80)

    # Compute minimal unique suffixes for directory names
    suffixes = minimal_unique_suffixes(INPUT_FOLDERS)
    logging.info(f"Input folders ({len(INPUT_FOLDERS)}):")
    for folder, suffix in zip(INPUT_FOLDERS, suffixes):
        logging.info(f"  {folder} -> {suffix}")

    # Create top-level analysis directory
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    for folder, suffix in zip(INPUT_FOLDERS, suffixes):
        input_path = os.path.join(BASE_PATH, folder, MORPHOLOGY_SUBPATH)
        output_base = os.path.join(ANALYSIS_DIR, suffix)

        logging.info("=" * 80)
        logging.info(f"PROCESSING: {folder}")
        logging.info(f"  Input:  {input_path}")
        logging.info(f"  Output: {output_base}")
        logging.info(f"  Downsampling by {DOWNSAMPLE_FACTOR}x for detection")
        logging.info("=" * 80)

        # Create output directory
        os.makedirs(output_base, exist_ok=True)

        # Step 1: Extract tiles (DAPI only)
        tiles_df = extract_tiles_dapi(input_path=input_path, output_base=output_base)

        # Step 2: Detect bboxes (DAPI only)
        bbox_df = detect_bboxes_in_tiles(tiles_df=tiles_df, tiles_base=output_base, output_base=output_base)

        # Step 3: Crop tiles (DAPI only)
        tiles_cropped_df, bbox_cropped_df = crop_tiles(
            tiles_df=tiles_df,
            bbox_df=bbox_df,
            tiles_base=output_base,
            output_base=output_base,
        )

        # Step 4: Tag bboxes by position
        tagged_df, untagged_df = tag_bboxs_by_position(bbox_df=bbox_cropped_df, output_base=output_base)

        # Step 5: Save individual fish (all channels)
        save_individual_fish_multi_channel(tagged_df=tagged_df, output_base=output_base, input_path=input_path)

        logging.info(f"DONE: {folder} -> {output_base}")
        logging.info(f"Individual fish saved to: {os.path.join(output_base, INDIVIDUAL_FISH_DIR)}")

        # Free memory before processing next folder
        del tiles_df, bbox_df, tiles_cropped_df, bbox_cropped_df, tagged_df, untagged_df
        gc.collect()

    logging.info("=" * 80)
    logging.info("ALL FOLDERS COMPLETE")
    logging.info(f"Results in: {ANALYSIS_DIR}/")
    for suffix in suffixes:
        logging.info(f"  {suffix}/")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()