#!/usr/bin/env python3
"""
analyze_wt_vs_mutant.py
=======================
Quantitative spatial analysis of WT vs mutant zebrafish cell-type volumes
using cell_type_label.tif volumes registered to a common space.

WT:      fish 4, 5, 6
Mutant:  fish 1, 2, 3

Cell types of primary interest:
  Decreased in mutant  — blood, hematopoietic cell, cranial vasculature
  Control (unchanged)  — spinal cord
  Increased in mutant  — pronephric distal early tubule

Resolution:
  XY pixel size : 0.2125 µm  (Xenium native, ds_xy=1)
  Z slice spacing: 10.0 µm   (47:1 anisotropy ratio Z:XY)
  Voxel volume  : 0.2125 × 0.2125 × 10 = 0.4516 µm³

Metrics:
  Volume  — sum of voxels × physical voxel volume → µm³
  Dice    — slice-wise (per Z-slice, then mean over slices with signal in
            either fish). Each anatomical level contributes equally,
            independent of Z thickness. Avoids 47:1 anisotropy bias.
  Stats   — effect sizes (log₂FC, Cohen's d on volumes).
            No p-values: with n=3 vs 3, the minimum two-sided Mann-Whitney
            p is 0.10 — significance thresholds are meaningless.

Output: visualization/report.html  (self-contained, all plots embedded)

Run from the repo root or visualization/ dir.  Uses ../scripts/.venv.
"""

import base64
import io
import os
import warnings
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import tifffile
from scipy import stats

# ─── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT    = os.path.dirname(SCRIPT_DIR)
ANNOTS_CSV   = os.path.join(REPO_ROOT, "leiden10annots.csv")
REG_BASE     = os.path.join(os.path.dirname(REPO_ROOT), "analysis",
                             "5_consensus")
OUT_HTML     = os.path.join(SCRIPT_DIR, "report.html")

# ─── Groups ───────────────────────────────────────────────────────────────────

WT_FISH      = [4, 5, 6]
MUT_FISH     = [1, 2, 3]
ALL_FISH     = WT_FISH + MUT_FISH

FOCUS_TYPES  = [
    "blood",
    "hematopoietic cell",
    "cranial vasculature",
    "spinal cord",
    "pronephric distal early tubule",
]

# ─── Physical resolution ─────────────────────────────────────────────────────

XY_UM        = 0.2125          # µm per pixel (Xenium native, ds_xy=1)
Z_UM         = 10.0            # µm per Z slice
VOXEL_UM3    = XY_UM * XY_UM * Z_UM   # 0.4516 µm³ per voxel
ANISOTROPY   = Z_UM / XY_UM   # ~47 — Z is 47× coarser than XY

# ─── Aesthetics ───────────────────────────────────────────────────────────────

WT_COLOR     = "#4C9BE8"   # blue
MUT_COLOR    = "#E8714C"   # orange
WT_LABEL     = "WT (fish 4–6)"
MUT_LABEL    = "Mutant (fish 1–3)"

FOCUS_COLORS = {
    "blood":                       "#C0392B",
    "hematopoietic cell":          "#8E44AD",
    "cranial vasculature":         "#E67E22",
    "spinal cord":                 "#27AE60",
    "pronephric distal early tubule": "#2980B9",
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.size":        11,
})

# ─────────────────────────────────────────────────────────────────────────────
# 1. LABEL MAPPING
# ─────────────────────────────────────────────────────────────────────────────

def build_label_map(annots_csv: str):
    """Return (type_to_id, id_to_type) from leiden10annots.csv.
    IDs are 1-based alphabetical — matches 2_rigid_registration.py."""
    df = pd.read_csv(annots_csv)
    cell_types = sorted(set(df["leiden10annots"].astype(str).tolist()))
    type_to_id = {ct: i + 1 for i, ct in enumerate(cell_types)}
    id_to_type = {v: k for k, v in type_to_id.items()}
    return type_to_id, id_to_type


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_label_volumes(fish_ids, reg_base=REG_BASE):
    """Return dict fish -> (Z, H, W) uint8 label volume."""
    vols = {}
    for fish in fish_ids:
        path = os.path.join(reg_base, str(fish), "cell_type_label.tif")
        print(f"  Loading fish {fish}: {path}")
        vols[fish] = tifffile.imread(path)
    return vols


# ─────────────────────────────────────────────────────────────────────────────
# 3. METRICS
# ─────────────────────────────────────────────────────────────────────────────

def physical_volume_um3(vol, label_id):
    """Physical volume in µm³: voxel count × (XY_UM² × Z_UM)."""
    return float(np.sum(vol == label_id)) * VOXEL_UM3


def dice_slicewise(vol_a, vol_b, label_id):
    """Slice-wise mean Dice — avoids 47:1 Z:XY anisotropy bias.

    For each Z-slice that has signal in at least one fish, compute the 2D
    Dice between the two fish on that slice.  Average over all such slices.
    This gives each anatomical level equal weight regardless of how thick it
    is in physical space (i.e. Z resolution does not inflate/deflate the
    metric relative to XY resolution).

    Returns NaN if no slice has signal in either fish.
    """
    a = (vol_a == label_id)
    b = (vol_b == label_id)
    slice_dice = []
    for z in range(a.shape[0]):
        az, bz = a[z], b[z]
        denom = int(az.sum()) + int(bz.sum())
        if denom == 0:
            continue   # skip empty slices — no contribution
        slice_dice.append(2.0 * float((az & bz).sum()) / denom)
    return float(np.mean(slice_dice)) if slice_dice else np.nan


def pairwise_dice(vols, label_id, fish_ids):
    """Symmetric N×N slice-wise Dice matrix."""
    n = len(fish_ids)
    mat = np.full((n, n), np.nan)
    for i, fi in enumerate(fish_ids):
        for j, fj in enumerate(fish_ids):
            if i == j:
                mat[i, j] = 1.0
            elif i < j:
                d = dice_slicewise(vols[fi], vols[fj], label_id)
                mat[i, j] = d
                mat[j, i] = d
    return mat


def group_dice_stats(vols, label_id, wt=WT_FISH, mut=MUT_FISH):
    """Within-WT, within-mutant, and between-group slice-wise Dice values."""
    def pair_dice(fish_list):
        return [dice_slicewise(vols[a], vols[b], label_id)
                for a, b in combinations(fish_list, 2)]

    within_wt  = pair_dice(wt)
    within_mut = pair_dice(mut)
    between    = [dice_slicewise(vols[a], vols[b], label_id)
                  for a in wt for b in mut]
    return within_wt, within_mut, between


def compute_all_dice(vols, type_to_id):
    """Compute within-WT, within-Mutant, between-group slice-wise Dice for ALL
    cell types in one pass — call once and share the result across plot functions.

    Returns dict: cell_type -> (within_wt, within_mut, between) lists of floats.
    """
    cache = {}
    for ct, lid in type_to_id.items():
        cache[ct] = group_dice_stats(vols, lid)
    return cache


def cohens_d(a, b):
    """Cohen's d effect size (pooled SD)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled_sd = np.sqrt(((na - 1) * np.std(a, ddof=1)**2 +
                         (nb - 1) * np.std(b, ddof=1)**2) / (na + nb - 2))
    if pooled_sd == 0:
        return np.nan
    return (np.mean(a) - np.mean(b)) / pooled_sd


def volume_df(vols, type_to_id, id_to_type, wt=WT_FISH, mut=MUT_FISH):
    """Long-form DataFrame: fish, group, cell_type, vol_um3.

    Uses np.bincount for a single pass per fish (34× faster than calling
    np.sum(vol == label_id) per label separately).
    """
    n_labels = max(id_to_type.keys()) + 1   # labels are 0..N
    rows = []
    for fish in wt + mut:
        group = "WT" if fish in wt else "Mutant"
        counts = np.bincount(vols[fish].ravel(), minlength=n_labels)
        for label_id, ct in id_to_type.items():
            v = float(counts[label_id]) * VOXEL_UM3
            rows.append(dict(fish=fish, group=group, cell_type=ct,
                             vol_um3=v, label_id=label_id))
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 4. SPATIAL MAPS
# ─────────────────────────────────────────────────────────────────────────────

def group_avg_mask(vols, label_id, fish_list):
    """Average binary mask (0–1 float): fraction of fish with that label."""
    stack = np.stack([vols[f] == label_id for f in fish_list], axis=0).astype(np.float32)
    return stack.mean(axis=0)


def diff_map(wt_avg, mut_avg):
    """Signed difference map: mutant_avg − wt_avg (range −1 to +1)."""
    return mut_avg.astype(np.float64) - wt_avg.astype(np.float64)



# ─────────────────────────────────────────────────────────────────────────────
# 5. PLOTTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def fig_to_b64(fig, dpi=130):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def img_tag(b64, caption="", width="100%"):
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return f'<figure style="margin:1.5em 0">{cap}<img src="data:image/png;base64,{b64}" style="width:{width};max-width:1200px" /></figure>'


# ─────────────────────────────────────────────────────────────────────────────
# 6. FIGURE GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def plot_volume_overview(vol_df):
    """Bar chart: log2 fold-change mutant/WT for every cell type, sorted."""
    grp = vol_df.groupby(["cell_type", "group"])["vol_um3"].mean().unstack()
    grp = grp.fillna(0)
    # avoid div-by-zero
    wt_mean  = grp.get("WT",     pd.Series(dtype=float))
    mut_mean = grp.get("Mutant", pd.Series(dtype=float))
    log2fc = np.log2((mut_mean + 1) / (wt_mean + 1))
    log2fc = log2fc.sort_values()

    focus = set(FOCUS_TYPES)
    colors = [
        (FOCUS_COLORS.get(ct, "#C0392B") if ct in focus and log2fc[ct] < 0 else
         FOCUS_COLORS.get(ct, "#2980B9") if ct in focus else
         MUT_COLOR if log2fc[ct] > 0 else WT_COLOR)
        for ct in log2fc.index
    ]

    fig, ax = plt.subplots(figsize=(10, 9))
    bars = ax.barh(range(len(log2fc)), log2fc.values, color=colors, edgecolor="none",
                   height=0.7)
    ax.set_yticks(range(len(log2fc)))
    ax.set_yticklabels(log2fc.index, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("log₂ fold-change (Mutant / WT)")
    ax.set_title("Volume fold-change: Mutant vs WT\n(all cell types, sorted by log₂FC)")

    # Mark focus cell types
    for i, ct in enumerate(log2fc.index):
        if ct in focus:
            ax.get_yticklabels()[i].set_fontweight("bold")
            ax.get_yticklabels()[i].set_color(FOCUS_COLORS.get(ct, "black"))

    # legend
    patches = [
        mpatches.Patch(color=MUT_COLOR, label="Increased in mutant"),
        mpatches.Patch(color=WT_COLOR,  label="Decreased in mutant"),
    ]
    for ct, col in FOCUS_COLORS.items():
        patches.append(mpatches.Patch(color=col, label=f"★ {ct}"))
    ax.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=8, framealpha=0.8, borderaxespad=0)
    fig.tight_layout()
    return fig


def plot_volume_per_fish(vol_df, cell_types=FOCUS_TYPES):
    """Strip + bar plots: physical volume (µm³) per fish, per focal cell type.

    Effect size shown as Cohen's d (pooled SD).  No p-values — with n=3 vs 3
    the minimum achievable two-sided Mann-Whitney p is 0.10, making
    significance thresholds meaningless.
    """
    n = len(cell_types)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 7), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, ct in zip(axes, cell_types):
        sub = vol_df[vol_df["cell_type"] == ct].copy()
        wt_vals  = sub[sub["group"] == "WT"]["vol_um3"].values
        mut_vals = sub[sub["group"] == "Mutant"]["vol_um3"].values

        # bar for mean
        ax.bar([0], [wt_vals.mean()],  color=WT_COLOR,  width=0.5, alpha=0.6, zorder=1)
        ax.bar([1], [mut_vals.mean()], color=MUT_COLOR, width=0.5, alpha=0.6, zorder=1)

        # error bars (std)
        ax.errorbar([0], [wt_vals.mean()],  yerr=wt_vals.std(),  fmt="none",
                    color="black", capsize=5, linewidth=1.5, zorder=2)
        ax.errorbar([1], [mut_vals.mean()], yerr=mut_vals.std(), fmt="none",
                    color="black", capsize=5, linewidth=1.5, zorder=2)

        # individual points
        np.random.seed(42)
        jitter = np.random.uniform(-0.08, 0.08, size=len(wt_vals))
        ax.scatter(np.zeros(len(wt_vals))  + jitter, wt_vals,  color=WT_COLOR,
                   s=60, zorder=3, edgecolors="white", linewidths=0.5)
        jitter = np.random.uniform(-0.08, 0.08, size=len(mut_vals))
        ax.scatter(np.ones(len(mut_vals)) + jitter, mut_vals, color=MUT_COLOR,
                   s=60, zorder=3, edgecolors="white", linewidths=0.5)

        # Cohen's d effect size (no p-value — n=3 makes them meaningless)
        d = cohens_d(wt_vals, mut_vals)
        fc = np.log2((mut_vals.mean() + 1) / (wt_vals.mean() + 1))
        ax.annotate(f"log₂FC={fc:+.1f}\nd={d:.1f}" if not np.isnan(d) else f"log₂FC={fc:+.1f}",
                    xy=(0.5, 0.93), xycoords="axes fraction",
                    ha="center", fontsize=11, color="#333")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["WT\n(4–6)", "Mutant\n(1–3)"], fontsize=13)
        ax.set_title(ct, fontsize=13, wrap=True)
        ax.set_ylabel("Volume (µm³)" if ax == axes[0] else "", fontsize=13)
        ax.set_xlim(-0.5, 1.5)
        color = FOCUS_COLORS.get(ct, "black")
        ax.title.set_color(color)

    fig.suptitle("Physical volume per fish — focal cell types\n"
                 "(effect sizes: log₂FC and Cohen's d; no p-values, n=3)",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    return fig


def plot_dice_heatmaps_grid(vols, type_to_id, cell_types=FOCUS_TYPES, ncols=3):
    """All focal-type 6×6 Dice heatmaps in one figure, ncols per row."""
    n = len(cell_types)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.5 * ncols, 3.8 * nrows))
    axes_flat = np.array(axes).reshape(-1)

    tick_labels = [f"Fish {f}\n({'WT' if f in WT_FISH else 'Mut'})" for f in ALL_FISH]

    for idx, ct in enumerate(cell_types):
        ax = axes_flat[idx]
        label_id = type_to_id[ct]
        mat = pairwise_dice(vols, label_id, ALL_FISH)

        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(ALL_FISH)))
        ax.set_yticks(range(len(ALL_FISH)))
        ax.set_xticklabels(tick_labels, fontsize=7)
        ax.set_yticklabels(tick_labels, fontsize=7)

        # Group rectangles
        for (r0, c0, sz, col) in [(0, 0, 3, WT_COLOR), (3, 3, 3, MUT_COLOR)]:
            rect = plt.Rectangle((c0 - 0.5, r0 - 0.5), sz, sz,
                                  fill=False, edgecolor=col, linewidth=2.5, linestyle="--")
            ax.add_patch(rect)

        # Cell annotations
        for i in range(len(ALL_FISH)):
            for j in range(len(ALL_FISH)):
                val = mat[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=6.5, color="white" if val < 0.6 else "black")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Dice")
        ax.set_title(ct, fontsize=9,
                     color=FOCUS_COLORS.get(ct, "black"), fontweight="bold")

    # Hide unused subplots
    for idx in range(len(cell_types), len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle(
        "Pairwise slice-wise Dice — focal cell types\n"
        "(dashed boxes: WT=blue, Mutant=orange)",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def plot_dice_comparison(vols, type_to_id, cell_types=FOCUS_TYPES):
    """Grouped strip plot: within-WT / within-mutant / between slice-wise Dice."""
    fig, axes = plt.subplots(1, len(cell_types), figsize=(5 * len(cell_types), 7), sharey=True)
    if len(cell_types) == 1:
        axes = [axes]

    for ax, ct in zip(axes, cell_types):
        label_id = type_to_id[ct]
        wt_d, mut_d, btw_d = group_dice_stats(vols, label_id)
        group_data = [wt_d, mut_d, btw_d]
        group_cols = [WT_COLOR, MUT_COLOR, "gray"]

        for xi, (vals, col) in enumerate(zip(group_data, group_cols)):
            vals = [v for v in vals if not np.isnan(v)]
            if not vals:
                continue
            ax.bar([xi], [np.mean(vals)], color=col, alpha=0.5, width=0.5)
            ax.errorbar([xi], [np.mean(vals)], yerr=np.std(vals),
                        fmt="none", color="black", capsize=5)
            jitter = np.random.uniform(-0.1, 0.1, len(vals))
            ax.scatter(np.full(len(vals), xi) + jitter, vals, color=col,
                       s=60, zorder=3, edgecolors="white")

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["within\nWT", "within\nMut", "between"], fontsize=13)
        ax.set_title(ct, fontsize=13)
        ax.set_ylim(-0.05, 1.05)
        color = FOCUS_COLORS.get(ct, "black")
        ax.title.set_color(color)

    axes[0].set_ylabel("Slice-wise mean Dice", fontsize=13)
    fig.suptitle("Slice-wise Dice: within- vs between-group\n"
                 "(per Z-slice Dice, averaged — corrects for 47:1 Z:XY anisotropy)",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    return fig


def plot_spatial_diff(vols, type_to_id, cell_types=FOCUS_TYPES):
    """3-projection difference maps with physically correct aspect ratios.

    Projections (max-abs signed):
      col 0 — Z-projection → (Y, X) plane: both axes = XY_UM → aspect = 1
      col 1 — Y-projection → (Z, X) plane: row axis = Z_UM, col = XY_UM
              → imshow aspect = Z_UM / XY_UM ≈ 47 (Z is stretched)
      col 2 — X-projection → (Z, Y) plane: same aspect as col 1

    'aspect' in imshow is height/width per data unit, so to make Z axis
    physically correct we pass aspect = Z_UM / XY_UM.
    """
    # Physical aspect ratios for each projection
    # imshow: rows = first index, cols = second index
    # col 0: rows=Y (XY_UM), cols=X (XY_UM)  → aspect = 1
    # col 1: rows=Z (Z_UM),  cols=X (XY_UM)  → aspect = Z_UM/XY_UM
    # col 2: rows=Z (Z_UM),  cols=Y (XY_UM)  → aspect = Z_UM/XY_UM
    proj_aspects = [XY_UM / XY_UM, Z_UM / XY_UM, Z_UM / XY_UM]
    view_labels  = [
        f"XY plane\n(Z max-proj, 1:1)",
        f"XZ plane\n(Y max-proj, Z scaled ×{ANISOTROPY:.0f})",
        f"YZ plane\n(X max-proj, Z scaled ×{ANISOTROPY:.0f})",
    ]

    n_types = len(cell_types)
    n_views = 3
    fig, axes = plt.subplots(n_types, n_views,
                              figsize=(5 * n_views, 3.5 * n_types))
    if n_types == 1:
        axes = axes[np.newaxis, :]

    for row, ct in enumerate(cell_types):
        label_id = type_to_id[ct]
        wt_avg   = group_avg_mask(vols, label_id, WT_FISH)
        mut_avg  = group_avg_mask(vols, label_id, MUT_FISH)
        diff     = diff_map(wt_avg, mut_avg)   # (Z, Y, X)

        # Max-abs signed projections
        # col0: project along axis0 (Z) → shape (Y, X)
        # col1: project along axis1 (Y) → shape (Z, X)
        # col2: project along axis2 (X) → shape (Z, Y)
        projs_signed = []
        for ax_idx in range(3):
            pos = diff.max(axis=ax_idx)
            neg = diff.min(axis=ax_idx)
            signed = np.where(np.abs(pos) >= np.abs(neg), pos, neg)
            projs_signed.append(signed)

        vmax = max(np.nanmax(np.abs(p)) for p in projs_signed)
        if vmax < 1e-6:
            vmax = 1e-6
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        for col, (proj, vlbl, asp) in enumerate(
                zip(projs_signed, view_labels, proj_aspects)):
            ax = axes[row, col]
            im = ax.imshow(proj, cmap="RdBu_r", norm=norm,
                           aspect=asp,          # ← physically correct
                           interpolation="nearest")
            if col == n_views - 1:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                             label="mut−WT\nfreq")
            if row == 0:
                ax.set_title(vlbl, fontsize=8)
            if col == 0:
                ax.set_ylabel(ct, fontsize=9, color=FOCUS_COLORS.get(ct, "black"),
                              fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        "Spatial difference maps: Mutant − WT\n"
        "(blue = depleted in mutant, red = enriched; aspect ratios are physically correct)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    return fig


def plot_wt_mut_overlay(vols, type_to_id, cell_types=FOCUS_TYPES):
    """Side-by-side WT vs Mutant mean frequency maps — XY plane (Z max-proj).
    Z max-projection → (Y, X) plane; both axes are XY_UM → aspect = 1.
    """
    n = len(cell_types)
    fig, axes = plt.subplots(n, 2, figsize=(10, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, ct in enumerate(cell_types):
        label_id = type_to_id[ct]
        wt_avg   = group_avg_mask(vols, label_id, WT_FISH)
        mut_avg  = group_avg_mask(vols, label_id, MUT_FISH)

        wt_proj  = wt_avg.max(axis=0)   # (Y, X) — both XY_UM → aspect=1
        mut_proj = mut_avg.max(axis=0)
        vmax     = max(wt_proj.max(), mut_proj.max(), 1e-6)

        for col, (proj, grp, col_color) in enumerate(zip(
                [wt_proj, mut_proj], ["WT (4–6)", "Mutant (1–3)"],
                [WT_COLOR, MUT_COLOR])):
            ax = axes[row, col]
            im = ax.imshow(proj, cmap="hot", vmin=0, vmax=vmax,
                           interpolation="nearest", aspect=1)   # ← 1:1 XY
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="freq")
            ax.set_title(f"{grp}: {ct}", fontsize=9, color=col_color,
                         fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        "Average occupancy maps — XY plane (Z max-projection)\n"
        "(fraction of fish with cell type at each voxel; 1:1 physical aspect)",
        fontsize=12)
    fig.tight_layout()
    return fig


def plot_volume_stats_table(vol_df, type_to_id):
    """Summary statistics table — volumes in µm³, effect sizes, no p-values.

    Columns: cell type | WT mean±SD (µm³) | Mut mean±SD (µm³) | log₂FC | Cohen's d
    No p-values: minimum two-sided Mann-Whitney p with n=3 vs 3 is 0.10.
    """
    rows = []
    for ct in sorted(type_to_id.keys()):
        sub = vol_df[vol_df["cell_type"] == ct]
        wt  = sub[sub["group"] == "WT"]["vol_um3"].values
        mut = sub[sub["group"] == "Mutant"]["vol_um3"].values
        fc  = np.log2((mut.mean() + 1) / (wt.mean() + 1))
        d   = cohens_d(wt, mut)   # positive = WT > Mutant

        def fmt(v):
            if v >= 1e6:
                return f"{v/1e6:.2f}M"
            if v >= 1e3:
                return f"{v/1e3:.1f}k"
            return f"{v:.0f}"

        rows.append({
            "Cell type":        ct,
            "WT µm³ (mean±SD)": f"{fmt(wt.mean())} ± {fmt(wt.std())}",
            "Mut µm³ (mean±SD)":f"{fmt(mut.mean())} ± {fmt(mut.std())}",
            "log₂FC":           f"{fc:+.2f}",
            "Cohen's d":        f"{d:.1f}" if not np.isnan(d) else "—",
            "★":                "★" if ct in FOCUS_TYPES else "",
        })
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(16, 0.52 * len(rows) + 0.8))
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.4)

    # Header
    for j in range(len(df.columns)):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight focus rows, colour log₂FC column
    for i, row_d in enumerate(rows, start=1):
        ct  = row_d["Cell type"]
        fc  = float(row_d["log₂FC"])
        col = FOCUS_COLORS.get(ct)
        for j in range(len(df.columns)):
            cell = tbl[i, j]
            if col:
                cell.set_facecolor(col + "18")
            if j == 3:   # log₂FC column
                if fc > 0.5:
                    cell.set_facecolor("#E8F8F5")
                    cell.set_text_props(color="#1E8449", fontweight="bold")
                elif fc < -0.5:
                    cell.set_facecolor("#FDEDEC")
                    cell.set_text_props(color="#C0392B", fontweight="bold")

    # Title is rendered as an HTML heading above the figure; no matplotlib title needed
    fig.tight_layout(pad=0.1)
    return fig


def plot_all_dice_summary(dice_cache, type_to_id):
    """For every cell type: plot within-WT, within-Mut, between-group mean Dice.
    Accepts pre-computed dice_cache dict from compute_all_dice().
    """
    def safe_mean(x):
        vals = [v for v in x if not np.isnan(v)]
        return float(np.mean(vals)) if vals else np.nan

    records = []
    for ct in type_to_id:
        wt_d, mut_d, btw_d = dice_cache[ct]
        wt_m, mut_m, btw_m = safe_mean(wt_d), safe_mean(mut_d), safe_mean(btw_d)
        within_max = float(np.nanmax([wt_m, mut_m]))   # NaN-safe max
        records.append({
            "cell_type":  ct,
            "within_wt":  wt_m,
            "within_mut": mut_m,
            "between":    btw_m,
            "delta":      btw_m - within_max,
        })
    df = pd.DataFrame(records).dropna().sort_values("delta")

    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(df))
    ax.barh(y,        df["between"],   height=0.25,
            color="gray",    alpha=0.7, label="between groups", align="center")
    ax.barh(y - 0.28, df["within_wt"],  height=0.25,
            color=WT_COLOR,  alpha=0.7, label="within WT",      align="center")
    ax.barh(y + 0.28, df["within_mut"], height=0.25,
            color=MUT_COLOR, alpha=0.7, label="within Mutant",  align="center")

    ax.set_yticks(y)
    ax.set_yticklabels(df["cell_type"].tolist(), fontsize=8)
    for i, ct in enumerate(df["cell_type"]):
        if ct in FOCUS_TYPES:
            ax.get_yticklabels()[i].set_fontweight("bold")
            ax.get_yticklabels()[i].set_color(FOCUS_COLORS.get(ct, "black"))

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean Dice coefficient")
    ax.set_title("Dice reproducibility: within-group vs between-group\n"
                 "(sorted by between − max(within), ↑ = more disrupted)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return fig


def plot_fc_vs_dice_scatter(dice_cache, vol_df, type_to_id):
    """Scatter: log2FC (x) vs between-group Dice loss (y) for all cell types.
    Accepts pre-computed dice_cache dict from compute_all_dice().
    """
    def safe_mean(x):
        v = [v for v in x if not np.isnan(v)]
        return float(np.mean(v)) if v else np.nan

    grp = vol_df.groupby(["cell_type", "group"])["vol_um3"].mean().unstack().fillna(0)
    wt_mean  = grp.get("WT",     pd.Series(dtype=float))
    mut_mean = grp.get("Mutant", pd.Series(dtype=float))
    log2fc = np.log2((mut_mean + 1) / (wt_mean + 1))

    fig, ax = plt.subplots(figsize=(9, 7))
    for ct in type_to_id:
        wt_d, mut_d, btw_d = dice_cache[ct]
        wt_wm  = safe_mean(wt_d)
        mut_wm = safe_mean(mut_d)
        btw    = safe_mean(btw_d)
        if np.isnan(btw) or ct not in log2fc.index:
            continue
        within_max = float(np.nanmax([wt_wm, mut_wm]))   # NaN-safe
        dice_loss  = within_max - btw if not np.isnan(within_max) else np.nan

        col   = FOCUS_COLORS.get(ct, "slategray")
        size  = 100 if ct in FOCUS_TYPES else 40
        alpha = 1.0 if ct in FOCUS_TYPES else 0.55

        ax.scatter(log2fc.get(ct, np.nan), dice_loss,
                   color=col, s=size, alpha=alpha, zorder=3,
                   edgecolors="black" if ct in FOCUS_TYPES else "none",
                   linewidths=0.8)
        if ct in FOCUS_TYPES:
            ax.annotate(ct, (log2fc.get(ct, np.nan), dice_loss),
                        textcoords="offset points", xytext=(6, 3), fontsize=8,
                        color=col, fontweight="bold")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("log₂FC (Mutant / WT)")
    ax.set_ylabel("Dice loss: max(within) − between\n(positive = less consistent across groups)")
    ax.set_title("Volume change vs spatial reproducibility\n"
                 "— cell types in the upper-left/right quadrants are most affected")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7. HTML REPORT
# ─────────────────────────────────────────────────────────────────────────────

HTML_HEAD = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WT vs Mutant Zebrafish Cell-Type Analysis</title>
<style>
  body  {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
           max-width: 1300px; margin: 0 auto; padding: 2em 1.5em;
           color: #222; background: #fafafa; }}
  h1   {{ color: #1a1a2e; border-bottom: 3px solid #4C9BE8; padding-bottom:.4em }}
  h2   {{ color: #16213e; margin-top: 2.5em; border-left: 4px solid #E8714C;
           padding-left: .6em }}
  h3   {{ color: #0f3460; }}
  p, li{{ line-height: 1.7 }}
  .meta{{ background:#eef2ff; border-radius:8px; padding:1em 1.5em; margin:1em 0 }}
  .focus-table {{ border-collapse:collapse; width:100%; margin:1em 0 }}
  .focus-table td, .focus-table th {{
    border:1px solid #ddd; padding:.5em .8em; }}
  .focus-table th {{ background:#2C3E50; color:white }}
  .focus-table tr:nth-child(even) {{ background:#f8f8f8 }}
  figure {{ background: white; border-radius: 8px; padding: 1em;
            box-shadow: 0 1px 6px rgba(0,0,0,.1); }}
  figcaption {{ font-size:.9em; color:#555; margin-bottom:.5em; font-style:italic }}
  hr   {{ border:none; border-top:1px solid #ddd; margin:2em 0 }}
</style>
</head>
<body>
"""

HTML_TAIL = """\
<hr>
<p style="font-size:.8em;color:#888">
  Generated by <code>visualization/analyze_wt_vs_mutant.py</code> ·
  {date}
</p>
</body></html>
"""

def build_html(sections):
    import datetime
    parts = [HTML_HEAD]
    for s in sections:
        parts.append(s)
    parts.append(HTML_TAIL.format(date=datetime.date.today().isoformat()))
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)

    print("=== WT vs Mutant Zebrafish Cell-Type Analysis ===")
    print(f"  Registration: {REG_BASE}")
    print(f"  WT fish:      {WT_FISH}")
    print(f"  Mutant fish:  {MUT_FISH}")

    # 1. Label map
    print("\n[1/6] Building label map ...")
    type_to_id, id_to_type = build_label_map(ANNOTS_CSV)

    # 2. Load volumes
    print("\n[2/6] Loading label volumes ...")
    vols = load_label_volumes(ALL_FISH)

    # 3. Volume dataframe (one bincount pass per fish)
    print("\n[3/6] Computing voxel volumes ...")
    vol_df = volume_df(vols, type_to_id, id_to_type)
    print(f"  Volume table: {len(vol_df)} rows")

    # 3b. Pre-compute Dice for all cell types once (used by two plot functions)
    print("  Pre-computing Dice for all 34 cell types ...")
    dice_cache = compute_all_dice(vols, type_to_id)
    print(f"  Done — {len(dice_cache)} cell types cached.")

    # 4. Build plots
    print("\n[4/6] Generating figures ...")

    sections = []

    # ── Intro ─────────────────────────────────────────────────────────────────
    sections.append("""
<h1>WT vs Mutant Zebrafish: Cell-Type Spatial Analysis</h1>
<div class="meta">
<b>Registration:</b> dapi_blend_rigid_affine_z10 (yolk-trimmed) &nbsp;|&nbsp;
<b>WT:</b> fish 4, 5, 6 &nbsp;|&nbsp;
<b>Mutant:</b> fish 1, 2, 3<br>
<b>Volume:</b> 111 × 1239 × 1236 voxels (same registered space for all fish)<br>
<b>Labels:</b> 34 cell types, 1-based alphabetically assigned integer IDs
(from <code>leiden10annots.csv</code>)
</div>
<h2>Focal cell types</h2>
<table class="focus-table">
<tr><th>Cell type</th><th>Expected change</th><th>Label ID</th></tr>
<tr><td style="color:#C0392B;font-weight:bold">blood</td>
    <td>Decreased / absent in mutant</td><td>""" + str(type_to_id["blood"]) + """</td></tr>
<tr><td style="color:#8E44AD;font-weight:bold">hematopoietic cell</td>
    <td>Decreased in mutant</td><td>""" + str(type_to_id["hematopoietic cell"]) + """</td></tr>
<tr><td style="color:#E67E22;font-weight:bold">cranial vasculature</td>
    <td>Decreased in mutant</td><td>""" + str(type_to_id["cranial vasculature"]) + """</td></tr>
<tr><td style="color:#27AE60;font-weight:bold">spinal cord</td>
    <td>Control — relatively unchanged</td><td>""" + str(type_to_id["spinal cord"]) + """</td></tr>
<tr><td style="color:#2980B9;font-weight:bold">pronephric distal early tubule</td>
    <td>Increased in mutant</td><td>""" + str(type_to_id["pronephric distal early tubule"]) + """</td></tr>
</table>
""")

    # ── Data-consistency check ────────────────────────────────────────────────
    # Compare observed vs expected direction for focus cell types
    grp_check = vol_df.groupby(["cell_type", "group"])["vol_um3"].mean().unstack().fillna(0)
    expected_dir = {
        "blood":                       "decrease",
        "hematopoietic cell":          "decrease",
        "cranial vasculature":         "decrease",
        "spinal cord":                 "unchanged",
        "pronephric distal early tubule": "increase",
    }
    discrepancy_rows = []
    for ct, exp in expected_dir.items():
        if ct not in grp_check.index:
            continue
        wt_v  = grp_check.loc[ct, "WT"]
        mut_v = grp_check.loc[ct, "Mutant"]
        fc    = np.log2((mut_v + 1) / (wt_v + 1))
        obs   = "increase" if fc > 0.3 else ("decrease" if fc < -0.3 else "unchanged")
        match = (exp == obs) or (exp == "unchanged" and obs == "unchanged")
        discrepancy_rows.append(dict(cell_type=ct, expected=exp, observed=obs,
                                     log2fc=fc, match=match))
    any_mismatch = any(not r["match"] for r in discrepancy_rows)
    alert_html = ""
    if any_mismatch:
        rows_html = ""
        for r in discrepancy_rows:
            col = "#1E8449" if r["match"] else "#C0392B"
            icon = "✓" if r["match"] else "✗"
            rows_html += (
                f"<tr style='color:{col}'>"
                f"<td><b>{r['cell_type']}</b></td>"
                f"<td>{r['expected']}</td>"
                f"<td>{r['observed']} (log₂FC {r['log2fc']:+.1f})</td>"
                f"<td style='font-size:1.2em'>{icon}</td></tr>"
            )
        alert_html = f"""
<div style="background:#FFF3CD;border:2px solid #E67E22;border-radius:8px;
            padding:1.2em 1.5em;margin:1.5em 0">
<h3 style="color:#E67E22;margin-top:0">⚠️  Unexpected Direction in Focal Cell Types</h3>
<p>One or more focal cell types show volume changes in the <b>opposite</b> direction
to biological expectation (WT = fish 4–6, Mutant = fish 1–3).
Spinal cord (control) should be relatively unchanged.</p>
<table style="border-collapse:collapse;width:100%;margin-top:.8em">
<tr style="background:#2C3E50;color:white">
<th style="padding:.4em .8em">Cell type</th>
<th>Expected</th>
<th>Observed (Mutant vs WT)</th>
<th>Match?</th></tr>
{rows_html}
</table>
</div>"""
    sections.append(alert_html)

    # ── Section 1: Overview fold-change ───────────────────────────────────────
    sections.append("<h2>1 · Global volume changes: all cell types</h2>")
    sections.append("<p>log₂ fold-change of mean voxel volume (Mutant / WT), "
                    "averaged across fish within each group. Focal cell types are "
                    "bold and coloured. Bars are sorted from most decreased (left) "
                    "to most increased (right).</p>")
    print("  fig: volume overview ...")
    fig = plot_volume_overview(vol_df)
    sections.append(img_tag(fig_to_b64(fig),
        "Log₂ fold-change of voxel volume — all 34 cell types. "
        "Blue = depleted in mutant, orange = enriched. Bold labels = focal cell types."))

    # ── Section 2: Focal cell type volumes ────────────────────────────────────
    sections.append("<h2>2 · Focal cell-type volumes — individual fish</h2>")
    sections.append("<p>Physical volume in µm³ (voxel count × 0.4516 µm³/voxel). "
                    "Bar = group mean; error bar = ±1 SD; dots = individual fish. "
                    "Effect sizes shown on each plot: <b>log₂FC</b> (fold-change) and "
                    "<b>Cohen's d</b> (standardised mean difference). "
                    "No p-values: with n=3 per group the minimum achievable two-sided "
                    "Mann–Whitney p is 0.10, so significance thresholds are meaningless.</p>")
    print("  fig: volume per fish ...")
    fig = plot_volume_per_fish(vol_df)
    sections.append(img_tag(fig_to_b64(fig),
        "Physical volume (µm³) per fish for each focal cell type. Individual fish points overlay "
        "the group mean bar."))

    # ── Section 3: Dice heatmaps ───────────────────────────────────────────────
    sections.append("<h2>3 · Pairwise Dice coefficients — focal cell types</h2>")
    sections.append("<p>Dice similarity between every pair of fish for each focal cell type. "
                    "WT–WT and Mut–Mut pairs are outlined; values near 1 indicate high "
                    "spatial overlap. Disrupted cell types typically show low between-group "
                    "Dice compared with within-group Dice.</p>")
    print("  fig: dice heatmaps grid ...")
    fig = plot_dice_heatmaps_grid(vols, type_to_id, ncols=3)
    sections.append(img_tag(fig_to_b64(fig, dpi=110),
        "6×6 pairwise Dice for all focal cell types. "
        "Dashed boxes outline WT–WT (blue) and Mutant–Mutant (orange) pairs."))

    # ── Section 4: Dice within vs between ──────────────────────────────────────
    sections.append("<h2>4 · Within- vs between-group Dice comparison</h2>")
    sections.append("<p>Comparing within-WT (3 pairs), within-Mutant (3 pairs), "
                    "and between-group (9 pairs) slice-wise Dice for each focal cell type. "
                    "For a control cell type (spinal cord) all three values should be similar; "
                    "for disrupted types, between-group Dice drops below within-group. "
                    "No p-values (small sample sizes make them uninformative).</p>")
    print("  fig: dice within vs between ...")
    fig = plot_dice_comparison(vols, type_to_id)
    sections.append(img_tag(fig_to_b64(fig),
        "Within-WT, within-Mutant, and between-group Dice for focal cell types. "
        "A drop in between-group Dice (relative to within-group) indicates spatial "
        "reorganisation in the mutant."))

    # ── Section 5: All-cell-type Dice summary ──────────────────────────────────
    sections.append("<h2>5 · Dice reproducibility — all cell types</h2>")
    sections.append("<p>For every cell type: mean within-WT, within-Mutant, and "
                    "between-group Dice, sorted by how much the between-group Dice "
                    "drops below the within-group maximum. The bottom of this list "
                    "contains the most genotype-disrupted cell types.</p>")
    print("  fig: all-cell Dice summary ...")
    fig = plot_all_dice_summary(dice_cache, type_to_id)
    sections.append(img_tag(fig_to_b64(fig),
        "Dice reproducibility for all 34 cell types. Cell types with a large "
        "gap between within- and between-group Dice are spatially most disrupted."))

    # ── Section 6: log2FC vs Dice-loss scatter ─────────────────────────────────
    sections.append("<h2>6 · Volume change vs spatial disruption (scatter)</h2>")
    sections.append("<p>Each point = one cell type. X-axis: volume fold-change in "
                    "mutant. Y-axis: Dice loss (within-group Dice − between-group Dice). "
                    "Cell types in the top-left (lost volume, disrupted) or top-right "
                    "(gained volume, disrupted) are most changed. The bottom centre "
                    "contains cell types that are spatially conserved.</p>")
    print("  fig: FC vs dice-loss scatter ...")
    fig = plot_fc_vs_dice_scatter(dice_cache, vol_df, type_to_id)
    sections.append(img_tag(fig_to_b64(fig),
        "Volume fold-change vs spatial Dice loss for all cell types. Labelled points "
        "are focal cell types."))

    # ── Section 7: Spatial difference maps ────────────────────────────────────
    sections.append("<h2>7 · Spatial difference maps — Mutant minus WT</h2>")
    sections.append("<p>For each focal cell type, each voxel was first averaged "
                    "across WT fish and across mutant fish (giving a 0–1 occupancy "
                    "frequency). The difference (Mutant − WT) is shown in three "
                    "orthogonal max-intensity projections. "
                    "<b>Blue</b> = region present in WT but absent/reduced in mutant; "
                    "<b>Red</b> = region present in mutant but absent/reduced in WT. "
                    "Row = cell type, Column = projection axis.</p>")
    print("  fig: spatial difference maps ...")
    fig = plot_spatial_diff(vols, type_to_id)
    sections.append(img_tag(fig_to_b64(fig, dpi=120),
        "3-projection spatial difference maps (Mutant − WT). "
        "Focal cell types × 3 orthogonal views."))

    # ── Section 8: WT vs Mutant frequency overlays ────────────────────────────
    sections.append("<h2>8 · Average occupancy maps — WT vs Mutant side-by-side</h2>")
    sections.append("<p>Each map shows the fraction of fish (0–1) that have a given "
                    "cell type at each voxel (Z max-projection). Comparing WT and "
                    "mutant side-by-side shows where cell types shift, expand, "
                    "or collapse.</p>")
    print("  fig: WT vs Mutant frequency overlays ...")
    fig = plot_wt_mut_overlay(vols, type_to_id)
    sections.append(img_tag(fig_to_b64(fig, dpi=120),
        "Voxel occupancy frequency maps (XY plane, 1:1 aspect) for each focal cell type, "
        "WT (left) vs Mutant (right). Colour scale is shared per row."))

    # ── Section 9: Statistics table ───────────────────────────────────────────
    sections.append("<h2>9 · Summary statistics table — all cell types</h2>")
    sections.append("<p>Physical volume mean ± SD (µm³), log₂ fold-change, "
                    "and Cohen's d effect size for every cell type. "
                    "Focal cell types have coloured backgrounds. "
                    "No p-values: with n=3 per group the minimum achievable two-sided "
                    "Mann–Whitney p is 0.10.</p>")
    print("  fig: summary statistics table ...")
    fig = plot_volume_stats_table(vol_df, type_to_id)
    sections.append(img_tag(fig_to_b64(fig, dpi=100),
        "Full statistics table across all 34 cell types."))

    # 5. Write HTML
    print(f"\n[5/6] Writing report → {OUT_HTML}")
    html = build_html(sections)
    with open(OUT_HTML, "w") as fh:
        fh.write(html)

    # 6. Also save volume CSV
    csv_path = os.path.join(SCRIPT_DIR, "volumes.csv")
    vol_df.to_csv(csv_path, index=False)
    print(f"[6/6] Volume table saved → {csv_path}")

    print("\n=== Done ===")
    print(f"  Open: {OUT_HTML}")


if __name__ == "__main__":
    main()
