import argparse
from pathlib import Path
import yaml
import numpy as np
import itk
import tifffile as tif
from scipy.ndimage import zoom
import functools
from typing import Any
import shutil
import re
import pandas as pd


print = functools.partial(print, flush=True)

SLICE_RE = re.compile(r"^(?P<slice>\d+)(?:_.*)?\.tif{1,2}$", re.IGNORECASE)
FINAL_METRIC_RE = re.compile(r"Final metric value\s*=\s*([-\d.eE+]+)")

def parse_slice_index(p: Path) -> int:
    m = SLICE_RE.match(p.name)
    if not m:
        raise ValueError(f"Filename does not start with an integer slice index: {p.name}")
    return int(m.group("slice"))

def is_input_slice(p: Path) -> bool:
    name = p.name.lower()
    if not (name.endswith(".tif") or name.endswith(".tiff")):
        return False
    if "_fixed" in name or "moving_" in name or "registered" in name:
        return False
    return True

def set_itk_image_properties(image, spacing=(1.0, 1.0), origin=(0.0, 0.0)):
    image.SetSpacing(spacing)
    image.SetOrigin(origin)

def load_slice(path: Path, downsample_factor: float, normalize: bool = True) -> np.ndarray:
    arr = tif.imread(str(path)).astype(np.float32)
    if arr.ndim != 2:
        arr = arr[0]
    if downsample_factor != 1:
        arr = zoom(arr, 1.0 / downsample_factor, order=1)
    if normalize:
        mx = float(np.max(arr))
        if mx > 0:
            arr /= mx
    return np.ascontiguousarray(arr)

def pad_to(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    h, w = arr.shape
    if h > out_h or w > out_w:
        out_h = max(out_h, h)
        out_w = max(out_w, w)
    pad_top = (out_h - h) // 2
    pad_bottom = out_h - h - pad_top
    pad_left = (out_w - w) // 2
    pad_right = out_w - w - pad_left
    return np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=0)

def _iters_list(max_iterations: int, n_resolutions: int):
    return [str(int(max_iterations)) for _ in range(int(n_resolutions))]

def build_parameter_object(cfg: dict) -> Any:
    n_resolutions = int(cfg.get("n_resolutions", 3))
    max_iterations = int(cfg.get("max_iterations", 500))
    num_spatial_samples = int(cfg.get("num_spatial_samples", 20000))
    metric_default = str(cfg.get("metric", "AdvancedNormalizedCorrelation"))
    metric_rigid = str(cfg.get("rigid_metric", metric_default))

    param_obj = itk.ParameterObject.New()
    pm = param_obj.GetDefaultParameterMap("rigid", n_resolutions)
    pm["Transform"] = ["EulerTransform"]
    pm["Metric"] = [metric_rigid]
    pm["Optimizer"] = ["AdaptiveStochasticGradientDescent"]
    pm["AutomaticParameterEstimation"] = ["true"]
    pm["AutomaticTransformInitialization"] = ["true"]
    pm["AutomaticTransformInitializationMethod"] = [str(cfg.get("rigid_init", "CenterOfGravity"))]

    pm["ImageSampler"] = ["RandomCoordinate"]
    pm["NumberOfSpatialSamples"] = [str(num_spatial_samples)]
    pm["MaximumNumberOfSamplingAttempts"] = [str(int(cfg.get("max_sampling_attempts", 200)))]
    pm["NewSamplesEveryIteration"] = [str(cfg.get("new_samples_every_iteration", True)).lower()]

    if "image_pyramid_schedule" in cfg:
        pm["ImagePyramidSchedule"] = [str(x) for x in cfg["image_pyramid_schedule"]]
    else:
        pm["ImagePyramidSchedule"] = ["16","16","8","8","4","4","2","2"]

    pm["MaximumNumberOfIterations"] = _iters_list(max_iterations, n_resolutions)
    pm["BSplineInterpolationOrder"] = [str(int(cfg.get("bspline_interpolation_order", 1)))]
    pm["FinalBSplineInterpolationOrder"] = [str(int(cfg.get("final_bspline_interpolation_order", 1)))]
    pm["ResultImagePixelType"] = [str(cfg.get("result_pixel_type", "float"))]

    param_obj.AddParameterMap(pm)
    return param_obj


def rotate90_np(arr: np.ndarray, angle_deg: int) -> np.ndarray:
    k = (angle_deg // 90) % 4
    return np.ascontiguousarray(np.rot90(arr, k)) if k else np.ascontiguousarray(arr)


def apply_transform(moving_np: np.ndarray, transform_params) -> np.ndarray:
    """Apply a pre-computed elastix transform to an image using transformix."""
    moving_itk = itk.GetImageFromArray(moving_np)
    set_itk_image_properties(moving_itk)
    result = itk.transformix_filter(moving_itk, transform_params)
    return itk.GetArrayFromImage(result).astype(np.float32)


def run_elastix_best_rotation(
    fixed_np: np.ndarray,
    moving_np: np.ndarray,
    parameter_object,
    angles=(0, 90, 180, 270),
    metric_goal: str = "min",
    scratch_dir: Path | None = None,
):
    """
    Try rotations of MOVING (0/90/180/270), pick best by last 'Final metric value' from elastix.log.
    Returns: (best_result_np, best_angle, best_metric, transform_parameters)
    """
    metric_goal = metric_goal.lower()
    if metric_goal not in ("min", "max"):
        raise ValueError('metric_goal must be "min" or "max"')

    if scratch_dir is None:
        raise ValueError("scratch_dir must be provided")

    scratch_dir.mkdir(parents=True, exist_ok=True)

    best = None  # {angle, metric}

    for ang in angles:
        run_dir = scratch_dir / f"try_rot_{ang}"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        moving_try = rotate90_np(moving_np, int(ang))

        fixed_itk = itk.GetImageFromArray(fixed_np)
        moving_itk = itk.GetImageFromArray(moving_try)
        set_itk_image_properties(fixed_itk)
        set_itk_image_properties(moving_itk)

        itk.elastix_registration_method(
            fixed_itk,
            moving_itk,
            parameter_object=parameter_object,
            log_to_console=False,
            log_to_file=True,
            output_directory=str(run_dir),
        )

        log_path = run_dir / "elastix.log"
        txt = log_path.read_text(errors="ignore") if log_path.exists() else ""
        vals = FINAL_METRIC_RE.findall(txt)
        metric = float(vals[-1]) if vals else None

        print(f"  trying rotation: {ang}  final_metric={metric}")

        if metric is None:
            continue

        if best is None:
            best = {"angle": int(ang), "metric": metric}
        else:
            better = (metric < best["metric"]) if metric_goal == "min" else (metric > best["metric"])
            if better:
                best = {"angle": int(ang), "metric": metric}

    if best is None:
        raise RuntimeError("No valid metric found in elastix logs for any rotation")

    # final run with best angle
    final_dir = scratch_dir / "final"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)

    best_ang = best["angle"]
    moving_best = rotate90_np(moving_np, best_ang)

    fixed_itk = itk.GetImageFromArray(fixed_np)
    moving_itk = itk.GetImageFromArray(moving_best)
    set_itk_image_properties(fixed_itk)
    set_itk_image_properties(moving_itk)

    result_image, result_transform_params = itk.elastix_registration_method(
        fixed_itk,
        moving_itk,
        parameter_object=parameter_object,
        log_to_console=False,
        log_to_file=True,
        output_directory=str(final_dir),
    )

    result_np = itk.GetArrayFromImage(result_image).astype(np.float32)

    # cleanup scratch
    shutil.rmtree(scratch_dir, ignore_errors=True)

    return result_np, best_ang, best["metric"], result_transform_params


def longest_run(indices: list[int], max_gap: int = 1) -> list[int]:
    if not indices:
        return []
    best_start = 0
    best_len = 1
    cur_start = 0
    for i in range(1, len(indices)):
        if (indices[i] - indices[i-1]) > max_gap:
            cur_len = i - cur_start
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
            cur_start = i
    cur_len = len(indices) - cur_start
    if cur_len > best_len:
        best_len = cur_len
        best_start = cur_start
    return indices[best_start:best_start + best_len]

def out_name(moving_slice: int, fixed_slice: int) -> str:
    return f"{moving_slice}_{fixed_slice}fixed.tif"


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACT MISSING CHANNEL SLICES FROM FULL IMAGE
# ══════════════════════════════════════════════════════════════════════════════

def get_channel_files(base_file_path: str):
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
        return [str(f) for f in channel_files]
    else:
        return [str(base_file_path)]


def extract_channel_slices(tagged_csv: str, input_image: str, channel: int,
                           output_root: str, fish_crop_margin: int = 10):
    """
    Extract individual fish slices for a given channel from the full image,
    using the tagged CSV coordinates.
    """
    df = pd.read_csv(tagged_csv)
    req = {"tile_idx", "fish_id", "bbox_global_min_row", "bbox_global_min_col",
           "bbox_global_max_row", "bbox_global_max_col"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in tagged CSV: {missing}")

    df["tile_idx"] = df["tile_idx"].astype(int)
    df["fish_id"] = df["fish_id"].astype(int)

    channel_files = get_channel_files(input_image)
    if channel >= len(channel_files):
        raise ValueError(f"Channel {channel} not found (only {len(channel_files)} files)")

    print(f"Reading channel {channel} from {Path(channel_files[channel]).name}")
    with tif.TiffFile(channel_files[channel], _multifile=False) as tf:
        img_full = tf.pages[0].asarray()
    if img_full.ndim > 2:
        img_full = img_full[0]

    written = 0
    for _, row in df.iterrows():
        fish = int(row["fish_id"]) + 1
        tile_idx = int(row["tile_idx"])

        rmin = int(row["bbox_global_min_row"])
        cmin = int(row["bbox_global_min_col"])
        rmax = int(row["bbox_global_max_row"])
        cmax = int(row["bbox_global_max_col"])

        r0 = max(0, rmin - fish_crop_margin)
        c0 = max(0, cmin - fish_crop_margin)
        r1 = min(img_full.shape[0] - 1, rmax + fish_crop_margin)
        c1 = min(img_full.shape[1] - 1, cmax + fish_crop_margin)

        if r1 <= r0 or c1 <= c0:
            continue

        crop = img_full[r0:r1 + 1, c0:c1 + 1]

        out_dir = Path(output_root) / str(fish) / "2d"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{tile_idx}.tif"
        tif.imwrite(str(out_path), crop, photometric="minisblack")
        written += 1

    print(f"Extracted {written} slices for channel {channel}")
    del img_full


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Rigid 2D registration per fish (multi-channel)")
    ap.add_argument("--config", required=True, help="Path to config.yml")
    ap.add_argument("--root", default="analysis/individual_fish",
                    help="Root individual_fish directory (contains ch0/, ch1/, ...)")
    ap.add_argument("--fish", type=int, default=None, help="If set, process only this fish id (1..6)")
    ap.add_argument("--reg-channel", type=int, default=0,
                    help="Channel to compute registration on (default: 0 = DAPI)")
    ap.add_argument("--channels", type=str, default="0,1",
                    help="Comma-separated channel indices to process (default: 0,1)")
    ap.add_argument("--tagged-csv", type=str, default=None,
                    help="Path to zfish_bboxs_tagged.csv (needed to extract missing channels)")
    ap.add_argument("--input-image", type=str, default=None,
                    help="Path to base OME-TIFF (needed to extract missing channels)")
    ap.add_argument("--fish-crop-margin", type=int, default=10,
                    help="Margin around fish bbox when extracting (default: 10)")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    downsample_factor = float(cfg.get("downsample_factor", 1))
    normalize = bool(cfg.get("normalize", True))
    pad_px = int(cfg.get("pad_px", 96))
    max_gap = int(cfg.get("max_tile_gap", 1))

    parameter_object = build_parameter_object(cfg)

    channels = [int(c.strip()) for c in args.channels.split(",")]
    reg_ch = args.reg_channel
    extra_channels = [c for c in channels if c != reg_ch]

    root = Path(args.root)
    reg_root = root / f"ch{reg_ch}"

    # ── Extract missing channel slices if needed ──
    for ch in extra_channels:
        ch_root = root / f"ch{ch}"
        if ch_root.exists() and any(ch_root.rglob("*.tif")):
            print(f"Channel {ch} slices already exist at {ch_root}")
        else:
            if not args.tagged_csv or not args.input_image:
                raise ValueError(
                    f"Channel {ch} slices not found at {ch_root}. "
                    f"Provide --tagged-csv and --input-image to extract them."
                )
            print(f"Extracting channel {ch} slices ...")
            extract_channel_slices(
                tagged_csv=args.tagged_csv,
                input_image=args.input_image,
                channel=ch,
                output_root=str(ch_root),
                fish_crop_margin=args.fish_crop_margin,
            )

    # ── Discover fish IDs ──
    fish_ids = [args.fish] if args.fish is not None else sorted(
        [int(p.name) for p in reg_root.iterdir() if p.is_dir() and p.name.isdigit()]
    )

    for fish in fish_ids:
        print(f"\n{'='*60}")
        print(f"Processing fish {fish}")
        print(f"{'='*60}")

        # ── Paths ──
        reg_fish_dir = reg_root / str(fish)
        reg_in_dir = reg_fish_dir / "2d"
        if not reg_in_dir.exists():
            print(f"[fish {fish}] skip (no 2d dir for ch{reg_ch}): {reg_in_dir}")
            continue

        # Build per-channel dirs
        ch_dirs = {}  # ch -> {in_dir, out_dir, fish_dir}
        for ch in channels:
            ch_fish_dir = root / f"ch{ch}" / str(fish)
            ch_in = ch_fish_dir / "2d"
            ch_out = ch_fish_dir / "2d_rigid"
            ch_out.mkdir(parents=True, exist_ok=True)
            ch_dirs[ch] = {"fish_dir": ch_fish_dir, "in_dir": ch_in, "out_dir": ch_out}

        # ── Discover slices from registration channel ──
        files = sorted(
            [p for p in reg_in_dir.iterdir() if is_input_slice(p)],
            key=parse_slice_index
        )
        if not files:
            print(f"[fish {fish}] skip (no input .tif found)")
            continue

        all_slices_sorted = sorted([parse_slice_index(p) for p in files])
        kept_slices = longest_run(all_slices_sorted, max_gap=max_gap)
        kept_set = set(kept_slices)
        kept_files = [p for p in files if parse_slice_index(p) in kept_set]
        kept_files.sort(key=parse_slice_index)

        if len(kept_files) < 2:
            print(f"[fish {fish}] only {len(kept_files)} slice(s) in longest run; skip")
            continue

        ref_i = len(kept_files) // 2
        ref_slice = parse_slice_index(kept_files[ref_i])

        print(f"[fish {fish}] total={len(files)} slices, longest_run={len(kept_slices)} "
              f"(gap>{max_gap} breaks), ref={ref_slice}")
        print(f"[fish {fish}] kept_slices = {kept_slices}")

        # ── Load all channels for kept slices ──
        # loaded[ch][slice_idx] = np.ndarray (downsampled, normalized, NOT yet padded)
        loaded_raw = {ch: {} for ch in channels}
        shapes = []

        for p in kept_files:
            sidx = parse_slice_index(p)
            # Registration channel
            arr = load_slice(p, downsample_factor, normalize=normalize)
            loaded_raw[reg_ch][sidx] = arr
            shapes.append(arr.shape)

            # Extra channels
            for ch in extra_channels:
                ch_path = ch_dirs[ch]["in_dir"] / p.name
                if ch_path.exists():
                    arr_ch = load_slice(ch_path, downsample_factor, normalize=normalize)
                    loaded_raw[ch][sidx] = arr_ch
                    shapes.append(arr_ch.shape)
                else:
                    print(f"  WARNING: missing ch{ch} slice {p.name} for fish {fish}")

        max_h = max(h for h, w in shapes) + 2 * pad_px
        max_w = max(w for h, w in shapes) + 2 * pad_px

        # Pad all to common canvas
        loaded = {ch: {} for ch in channels}
        for ch in channels:
            for sidx, arr in loaded_raw[ch].items():
                loaded[ch][sidx] = pad_to(arr, max_h, max_w)

        # ── Registration (outward from ref) ──
        # registered[ch][slice_idx] = np.ndarray
        registered = {ch: {} for ch in channels}

        # Reference slice: no transform needed, just copy
        for ch in channels:
            if ref_slice in loaded[ch]:
                registered[ch][ref_slice] = loaded[ch][ref_slice]

        # Save reference slice
        for ch in channels:
            if ref_slice in registered[ch]:
                ref_out = ch_dirs[ch]["out_dir"] / f"{ref_slice}_fixed.tif"
                tif.imwrite(str(ref_out), registered[ch][ref_slice].astype(np.float32))

        by_slice = {parse_slice_index(p): p for p in kept_files}
        mn, mx = min(kept_slices), max(kept_slices)

        def find_nearest_registered(ts: int) -> int:
            step_dir = 1 if ts < ref_slice else -1
            k = ts + step_dir
            while mn <= k <= mx:
                if k in registered[reg_ch]:
                    return k
                k += step_dir
            return ref_slice

        step = 1
        while (ref_slice - step) >= mn or (ref_slice + step) <= mx:
            for target_slice in (ref_slice - step, ref_slice + step):
                if target_slice < mn or target_slice > mx:
                    continue
                if target_slice not in loaded[reg_ch]:
                    continue
                if target_slice in registered[reg_ch]:
                    continue

                neighbor = target_slice + 1 if (target_slice < ref_slice) else target_slice - 1
                if neighbor not in registered[reg_ch]:
                    neighbor = find_nearest_registered(target_slice)

                fixed_np = registered[reg_ch][neighbor]
                moving_np = loaded[reg_ch][target_slice]

                print(f"\nregistering slice {target_slice} -> {neighbor} ...")

                angles = cfg.get("try_rotations_deg", [0, 90, 180, 270])
                goal = str(cfg.get("metric_goal", "min"))

                scratch = ch_dirs[reg_ch]["out_dir"] / "_tmp_logs" / f"{target_slice}_to_{neighbor}"
                result_np, best_ang, best_metric, transform_params = run_elastix_best_rotation(
                    fixed_np, moving_np, parameter_object,
                    angles=angles,
                    metric_goal=goal,
                    scratch_dir=scratch,
                )
                print(f"  chosen rotation: {best_ang}  metric={best_metric}")

                # Save registration channel result
                registered[reg_ch][target_slice] = result_np
                out_path = ch_dirs[reg_ch]["out_dir"] / out_name(target_slice, neighbor)
                tif.imwrite(str(out_path), result_np)

                # Apply same transform to extra channels
                for ch in extra_channels:
                    if target_slice not in loaded[ch]:
                        continue
                    moving_ch = rotate90_np(loaded[ch][target_slice], best_ang)
                    result_ch = apply_transform(moving_ch, transform_params)
                    registered[ch][target_slice] = result_ch

                    out_path_ch = ch_dirs[ch]["out_dir"] / out_name(target_slice, neighbor)
                    tif.imwrite(str(out_path_ch), result_ch)

                print(f"  saved slice {target_slice} for {len(channels)} channel(s)")

            step += 1

        # ── Stack into 3D volumes ──
        stack_slices = sorted([s for s in kept_slices if s in registered[reg_ch]])

        for ch in channels:
            slices_for_ch = [s for s in stack_slices if s in registered[ch]]
            if not slices_for_ch:
                continue
            vol = np.stack([registered[ch][s] for s in slices_for_ch], axis=0).astype(np.float32)
            out_3d = ch_dirs[ch]["fish_dir"] / "rigid_3d.tif"
            tif.imwrite(str(out_3d), vol)
            print(f"[fish {fish}] ch{ch}: wrote {out_3d}  shape={vol.shape}  "
                  f"slices={slices_for_ch[0]}..{slices_for_ch[-1]}")

    print("\nDone.")


if __name__ == "__main__":
    main()