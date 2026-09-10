"""
5_trim_yolk.py
==============
Identify and remove the yolk-sac region from registered zebrafish volumes.

Background
----------
In posterior z-slices (~z 40+) the yolk sac breaks down differently per
individual fish, causing high inter-fish disagreement that is noise, not
biology.  Simply thresholding on all-6-fish agreement would incorrectly
discard biologically meaningful regions where WT and mutant genuinely differ
(e.g. blood present in WT, absent in mutant).

Strategy: compute agreement *within* each group separately.
  - A voxel is valid in WT     if ≥ MIN_AGREEMENT of the 3 WT fish have any
                                  non-background label there.
  - A voxel is valid in Mutant if ≥ MIN_AGREEMENT of the 3 Mutant fish have
                                  any non-background label there.
  - Final consensus mask = union: valid if it passes in *either* group.

This preserves voxels where WT all agree (blood present) even when mutant all
agree it is absent — both are internally consistent anatomy.  The yolk fails
within both groups and is excluded.

A hard Z cutoff is also derived from the per-z within-group agreement curve:
the first z at which the *mean* within-group agreement (averaged over both
groups) drops permanently below Z_AGREEMENT_THRESHOLD.  All z-slices at or
after this cutoff are excluded entirely before the voxel-wise mask is applied.

Outputs  (analysis/5_consensus/)
--------
  consensus_mask.tif          — (Z_trim, H, W) uint8 binary mask; 1 = keep
  cutoff_info.json            — chosen z cutoff and parameters used
  qc_agreement_curve.png      — per-z within-group agreement, cutoff marked

The mask can be applied to cell_type_label.tif and channel TIFFs before
quantitative analysis or 3D visualisation.

Usage
-----
  python3 5_trim_yolk.py
  python3 5_trim_yolk.py --experiment dapi_blend_rigid_affine_z10
  python3 5_trim_yolk.py --min-agreement 0.67 --z-threshold 0.5
"""

import argparse
import json
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── configuration ──────────────────────────────────────────────────────────────

EXPERIMENT   = "dapi_blend_rigid_affine_z10"   # best experiment from script 4
IN_BASE      = "../analysis/4_registered"
OUT_DIR      = "../analysis/5_consensus"

WT_FISH      = [4, 5, 6]
MUT_FISH     = [1, 2, 3]
ALL_FISH     = WT_FISH + MUT_FISH

# Fraction of fish within a group that must have any label (non-zero) for a
# voxel to be counted as "agreed upon" by that group.  With 3 fish per group,
# 0.67 means at least 2/3; 1.0 means all 3.
MIN_AGREEMENT = 2 / 3

# Mean within-group agreement (0–1) below which a z-slice is considered part
# of the yolk / unreliable tail.  The cutoff is the first z at which this
# drops permanently (stays below threshold for the rest of the stack).
Z_AGREEMENT_THRESHOLD = 0.5

# ── helpers ────────────────────────────────────────────────────────────────────

def load_label_vol(fish: int, experiment: str, in_base: str) -> np.ndarray:
    path = os.path.join(in_base, experiment, str(fish), "cell_type_label.tif")
    log.info("  loading fish %d: %s", fish, path)
    return tifffile.imread(path)


def tissue_mask(vol: np.ndarray) -> np.ndarray:
    """Binary mask: True where any cell-type label is present (non-background)."""
    return vol > 0


def within_group_agreement_per_z(masks: list[np.ndarray]) -> np.ndarray:
    """Per-z fraction of voxels (that are tissue in *any* fish of the group)
    where at least MIN_AGREEMENT fish of the group agree there is tissue.

    Returns array of shape (Z,) with values in [0, 1].
    NaN for z-slices where no fish has any tissue.
    """
    n = len(masks)
    threshold_count = MIN_AGREEMENT * n   # e.g. 2.0 for n=3, MIN_AGREEMENT=2/3
    # Stack into (n, Z, H, W)
    stack = np.stack(masks, axis=0).astype(np.uint8)   # 0 or 1
    counts = stack.sum(axis=0)                          # (Z, H, W): how many fish agree
    n_z = counts.shape[0]
    agreement = np.full(n_z, np.nan)
    for z in range(n_z):
        any_tissue = counts[z] > 0                      # voxels tissue in at least 1 fish
        n_any = int(any_tissue.sum())
        if n_any == 0:
            continue
        n_agreed = int((counts[z] >= threshold_count).sum())
        agreement[z] = n_agreed / n_any
    return agreement


def find_z_cutoff(wt_agree: np.ndarray, mut_agree: np.ndarray,
                  threshold: float) -> int:
    """Return the first z at which the mean agreement drops permanently below
    threshold.  'Permanently' = all remaining z-slices are also below threshold
    (ignoring NaN slices).  Returns len(array) if it never drops (keep all).
    """
    mean_agree = np.where(
        np.isnan(wt_agree) & np.isnan(mut_agree),
        np.nan,
        np.nanmean(np.stack([wt_agree, mut_agree], axis=1), axis=1),
    )
    n = len(mean_agree)
    # Walk from the end; find last z that is below threshold
    cutoff = n   # default: keep everything
    for z in range(n - 1, -1, -1):
        if np.isnan(mean_agree[z]):
            continue
        if mean_agree[z] >= threshold:
            break
        cutoff = z
    return cutoff


def build_consensus_mask(wt_masks: list[np.ndarray],
                         mut_masks: list[np.ndarray],
                         z_cutoff: int) -> np.ndarray:
    """Build (Z_trim, H, W) uint8 consensus mask.

    For each group, a voxel is valid if ≥ MIN_AGREEMENT fish have tissue there.
    Final mask = union of both group masks, truncated at z_cutoff.
    """
    wt_stack  = np.stack(wt_masks,  axis=0).astype(np.uint8)
    mut_stack = np.stack(mut_masks, axis=0).astype(np.uint8)
    threshold_count = MIN_AGREEMENT * len(wt_masks)
    wt_valid  = wt_stack.sum(axis=0)  >= threshold_count   # (Z, H, W) bool
    mut_valid = mut_stack.sum(axis=0) >= threshold_count
    union     = (wt_valid | mut_valid)[:z_cutoff]           # truncate in Z
    return union.astype(np.uint8)


def plot_agreement_curve(wt_agree: np.ndarray, mut_agree: np.ndarray,
                         z_cutoff: int, out_path: str) -> None:
    mean_agree = np.nanmean(np.stack([wt_agree, mut_agree], axis=1), axis=1)
    zs = np.arange(len(wt_agree))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(zs, wt_agree,   color="#4C9BE8", lw=2, label="Within WT (fish 4–6)")
    ax.plot(zs, mut_agree,  color="#E8714C", lw=2, label="Within Mutant (fish 1–3)")
    ax.plot(zs, mean_agree, color="#2C3E50", lw=1.5, ls="--", label="Mean")
    ax.axhline(Z_AGREEMENT_THRESHOLD, color="gray", ls=":", lw=1,
               label=f"Threshold ({Z_AGREEMENT_THRESHOLD:.2f})")
    ax.axvline(z_cutoff, color="red", lw=2, label=f"Z cutoff = {z_cutoff}")
    ax.fill_betweenx([0, 1], z_cutoff, len(zs),
                     color="red", alpha=0.08, label="Excluded (yolk)")
    ax.set_xlabel("Z slice index")
    ax.set_ylabel("Within-group agreement (fraction of tissue voxels)")
    ax.set_title("Per-z within-group cell-type agreement\n"
                 "(fraction of tissue voxels where ≥2/3 fish in group agree)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("QC plot saved: %s", out_path)


# ── main ───────────────────────────────────────────────────────────────────────

def main(experiment: str, in_base: str, out_dir: str,
         min_agreement: float, z_threshold: float) -> None:
    global MIN_AGREEMENT, Z_AGREEMENT_THRESHOLD
    MIN_AGREEMENT          = min_agreement
    Z_AGREEMENT_THRESHOLD  = z_threshold

    os.makedirs(out_dir, exist_ok=True)

    # 1. Load tissue masks for all fish
    log.info("Loading WT fish label volumes (%s)…", WT_FISH)
    wt_vols  = [tissue_mask(load_label_vol(f, experiment, in_base)) for f in WT_FISH]
    log.info("Loading Mutant fish label volumes (%s)…", MUT_FISH)
    mut_vols = [tissue_mask(load_label_vol(f, experiment, in_base)) for f in MUT_FISH]

    # 2. Per-z agreement curves within each group
    log.info("Computing per-z within-group agreement…")
    wt_agree  = within_group_agreement_per_z(wt_vols)
    mut_agree = within_group_agreement_per_z(mut_vols)

    # 3. Z cutoff
    z_cutoff = find_z_cutoff(wt_agree, mut_agree, z_threshold)
    log.info("Z cutoff: %d  (volume shape Z axis = %d)", z_cutoff, len(wt_agree))

    # 4. Consensus mask
    log.info("Building consensus mask (union of within-group agreement)…")
    mask = build_consensus_mask(wt_vols, mut_vols, z_cutoff)
    mask_path = os.path.join(out_dir, "consensus_mask.tif")
    tifffile.imwrite(mask_path, mask, compression="zlib")
    log.info("Consensus mask saved: %s  shape=%s  kept=%d/%d voxels",
             mask_path, mask.shape, int(mask.sum()), mask.size)

    # 5. QC plot
    plot_agreement_curve(wt_agree, mut_agree, z_cutoff,
                         os.path.join(out_dir, "qc_agreement_curve.png"))

    # 6. Apply mask to all fish — save masked cell_type_label.tif per fish
    log.info("Applying consensus mask to all fish…")
    for fish in ALL_FISH:
        fish_in_dir  = os.path.join(in_base, experiment, str(fish))
        fish_out_dir = os.path.join(out_dir, str(fish))
        os.makedirs(fish_out_dir, exist_ok=True)

        label_path = os.path.join(fish_in_dir, "cell_type_label.tif")
        log.info("  fish %d: %s", fish, label_path)
        label_vol = tifffile.imread(label_path)               # (Z, H, W) uint8
        label_trimmed = label_vol[:z_cutoff]                  # drop tail z-slices
        label_masked  = label_trimmed * mask                  # zero out yolk voxels
        out_path = os.path.join(fish_out_dir, "cell_type_label.tif")
        tifffile.imwrite(out_path, label_masked, compression="zlib")
        log.info("    saved: %s", out_path)

    # 7. Metadata
    info = {
        "experiment":            experiment,
        "z_cutoff":              z_cutoff,
        "total_z_slices":        int(len(wt_agree)),
        "min_agreement":         min_agreement,
        "z_agreement_threshold": z_threshold,
        "wt_fish":               WT_FISH,
        "mut_fish":              MUT_FISH,
        "mask_shape":            list(mask.shape),
        "voxels_kept":           int(mask.sum()),
        "voxels_total":          int(mask.size),
        "fraction_kept":         float(mask.sum()) / mask.size,
    }
    info_path = os.path.join(out_dir, "cutoff_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    log.info("Cutoff info saved: %s", info_path)
    log.info("Done.  Fraction of volume kept: %.1f%%", 100 * info["fraction_kept"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment",     default=EXPERIMENT,
                        help=f"Experiment folder under {IN_BASE}/ (default: %(default)s)")
    parser.add_argument("--in-base",        default=IN_BASE)
    parser.add_argument("--out-dir",        default=OUT_DIR)
    parser.add_argument("--min-agreement",  type=float, default=MIN_AGREEMENT,
                        help="Fraction of fish in group that must agree (default: %(default).2f)")
    parser.add_argument("--z-threshold",    type=float, default=Z_AGREEMENT_THRESHOLD,
                        help="Mean within-group agreement below which z is cut (default: %(default).2f)")
    args = parser.parse_args()
    main(args.experiment, args.in_base, args.out_dir,
         args.min_agreement, args.z_threshold)
