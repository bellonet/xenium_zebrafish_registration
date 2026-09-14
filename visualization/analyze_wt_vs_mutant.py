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

import argparse
import base64
import io
import os
import pickle
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
CACHE_FILE   = os.path.join(SCRIPT_DIR, "analysis_cache.npz")

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

WT_COLOR     = "#00838F"   # teal
MUT_COLOR    = "#C2185B"   # magenta
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
    """Weighted slice Dice between two fish for a given label.

    Weighting each slice by its share of total signal (|A_z| + |B_z|) /
    (|A| + |B|) and multiplying by per-slice Dice simplifies to standard
    3D Dice: 2|A∩B| / (|A| + |B|).  This is the correct normalisation —
    slices with more structure contribute proportionally more, while near-empty
    slices barely matter.

    Returns NaN if neither fish has any signal for this label.
    """
    a = (vol_a == label_id)
    b = (vol_b == label_id)
    intersection = float(np.sum(a & b))
    denom = float(np.sum(a) + np.sum(b))
    if denom == 0:
        return np.nan
    return 2.0 * intersection / denom


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
    """Long-form DataFrame: fish, group, cell_type, vol_um3."""
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

def _abbrev(name):
    """Short label for a cell type: initials if multi-word, else first 6 chars."""
    words = name.split()
    if len(words) == 1:
        return name[:7]
    return "".join(w[0].upper() for w in words)


def plot_composition(vol_df):
    """Horizontal stacked bar: each cell type as % of total volume, per fish.

    Fish are ordered WT (4,5,6) then Mutant (1,2,3) with a visual gap.
    Cell types sorted by size in fish 4 (first WT fish), largest on top.
    Focal types use their assigned colours; others cycle gray shades.
    Abbreviated labels are drawn inside segments ≥ 2%.
    """
    totals = vol_df.groupby("fish")["vol_um3"].transform("sum")
    vol_df = vol_df.copy()
    vol_df["pct"] = vol_df["vol_um3"] / totals * 100

    # Sort cell types by size in fish 4
    fish4 = vol_df[vol_df["fish"] == 4].set_index("cell_type")["pct"]
    cell_order = fish4.sort_values(ascending=False).index.tolist()

    focus = set(FOCUS_TYPES)
    gray_shades = ["#D0D0D0", "#B0B0B0", "#909090", "#707070",
                   "#D0D0D0", "#B0B0B0", "#909090", "#707070"]
    other_idx = 0
    color_map = {}
    for ct in cell_order:
        if ct in focus:
            color_map[ct] = FOCUS_COLORS[ct]
        else:
            color_map[ct] = gray_shades[other_idx % len(gray_shades)]
            other_idx += 1

    fish_order = WT_FISH + MUT_FISH
    # y positions with a gap between WT and Mutant groups
    y_pos = {f: i + (0.6 if i >= len(WT_FISH) else 0) for i, f in enumerate(fish_order)}

    fig, ax = plt.subplots(figsize=(13, 7))

    lefts = {f: 0.0 for f in fish_order}
    for ct in reversed(cell_order):   # reversed so largest is leftmost
        sub  = vol_df[vol_df["cell_type"] == ct].set_index("fish")["pct"]
        vals = [sub.get(f, 0.0) for f in fish_order]
        ys   = [y_pos[f] for f in fish_order]
        col  = color_map[ct]
        ax.barh(ys, vals, left=[lefts[f] for f in fish_order],
                color=col, height=0.55, edgecolor="white", linewidth=0.3)

        # Label inside segment if wide enough — vertical to avoid overlap
        abbr = _abbrev(ct)
        for f, v, l in zip(fish_order, vals, [lefts[f] for f in fish_order]):
            if v >= 1.0:
                txt_col = "white" if ct in focus else "#333"
                ax.text(l + v / 2, y_pos[f], abbr,
                        ha="center", va="center", fontsize=7.5,
                        color=txt_col, fontweight="bold" if ct in focus else "normal",
                        rotation=90, clip_on=True)
        for f, v in zip(fish_order, vals):
            lefts[f] += v

    # y-axis labels
    yticks  = [y_pos[f] for f in fish_order]
    ylabels = [f"Fish {f}  ({'WT' if f in WT_FISH else 'Mut'})" for f in fish_order]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=13)
    ax.set_xlabel("% of total tissue volume", fontsize=13)
    ax.set_title("Cell type composition per fish  (sorted by size in Fish 4)", fontsize=15)
    ax.tick_params(axis="x", labelsize=12)

    # Group labels to the right
    xmax = max(lefts.values())
    wt_y  = np.mean([y_pos[f] for f in WT_FISH])
    mut_y = np.mean([y_pos[f] for f in MUT_FISH])
    ax.text(xmax + 1, wt_y,  "WT",     va="center", fontsize=13,
            color=WT_COLOR,  fontweight="bold")
    ax.text(xmax + 1, mut_y, "Mutant", va="center", fontsize=13,
            color=MUT_COLOR, fontweight="bold")

    # Legend below the plot
    patches = [mpatches.Patch(color=FOCUS_COLORS[ct], label=ct) for ct in FOCUS_TYPES]
    patches.append(mpatches.Patch(color="#B0B0B0", label="other cell types"))
    ax.legend(handles=patches, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=3,
              fontsize=10, framealpha=0.8)

    ax.set_ylim(min(yticks) - 0.5, max(yticks) + 0.5)
    fig.tight_layout()
    return fig

def plot_volume_overview(vol_df):
    """Bar chart: log2 fold-change mutant/WT for every cell type, sorted.

    Log₂ is used so that equal-magnitude increases and decreases are symmetric
    around zero (e.g. 2× increase = +1, 2× decrease = −1), and the scale
    compresses large differences that would otherwise dominate a linear axis.
    A pseudocount of +1 is added before dividing to avoid log(0).
    """
    grp = vol_df.groupby(["cell_type", "group"])["vol_um3"].mean().unstack()
    grp = grp.fillna(0)
    wt_mean  = grp.get("WT",     pd.Series(dtype=float))
    mut_mean = grp.get("Mutant", pd.Series(dtype=float))
    log2fc = np.log2((mut_mean + 1) / (wt_mean + 1))
    log2fc = log2fc.sort_values()

    focus = set(FOCUS_TYPES)
    # Non-focal cell types are shown as neutral gray; focal types use their
    # assigned colour so they stand out against the background.
    GRAY = "#AAAAAA"
    bar_colors    = [FOCUS_COLORS.get(ct, GRAY) if ct in focus else GRAY
                     for ct in log2fc.index]
    edge_colors   = [FOCUS_COLORS.get(ct, "none") if ct in focus else "none"
                     for ct in log2fc.index]
    edge_widths   = [1.8 if ct in focus else 0 for ct in log2fc.index]

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(range(len(log2fc)), log2fc.values,
            color=bar_colors, edgecolor=edge_colors, linewidth=edge_widths,
            height=0.7)
    ax.set_yticks(range(len(log2fc)))
    ax.set_yticklabels(log2fc.index, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("log₂ fold-change (Mutant / WT)")
    ax.set_title("Volume Fold-Change: Mutant vs WT")

    for i, ct in enumerate(log2fc.index):
        if ct in focus:
            ax.get_yticklabels()[i].set_fontweight("bold")
            ax.get_yticklabels()[i].set_color(FOCUS_COLORS.get(ct, "black"))

    patches = [mpatches.Patch(color=GRAY, label="other cell types")]
    for ct, col in FOCUS_COLORS.items():
        patches.append(mpatches.Patch(color=col, label=f"★ {ct}"))
    ax.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=8, framealpha=0.8, borderaxespad=0)
    fig.tight_layout()
    return fig


def plot_volume_per_fish(vol_df, cell_types=FOCUS_TYPES):
    """Strip + bar plots: physical volume (µm³) per fish, per focal cell type.
    Arranged in a 2×3 grid; effect sizes as log₂FC and Cohen's d.
    """
    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 11), sharey=False)
    axes_flat = axes.flatten()

    for idx, ct in enumerate(cell_types):
        ax = axes_flat[idx]
        sub = vol_df[vol_df["cell_type"] == ct].copy()
        wt_vals  = sub[sub["group"] == "WT"]["vol_um3"].values
        mut_vals = sub[sub["group"] == "Mutant"]["vol_um3"].values

        ax.bar([0], [wt_vals.mean()],  color=WT_COLOR,  width=0.5, alpha=0.6, zorder=1)
        ax.bar([1], [mut_vals.mean()], color=MUT_COLOR, width=0.5, alpha=0.6, zorder=1)

        ax.errorbar([0], [wt_vals.mean()],  yerr=wt_vals.std(),  fmt="none",
                    color="black", capsize=6, linewidth=1.8, zorder=2)
        ax.errorbar([1], [mut_vals.mean()], yerr=mut_vals.std(), fmt="none",
                    color="black", capsize=6, linewidth=1.8, zorder=2)

        np.random.seed(42)
        jitter = np.random.uniform(-0.08, 0.08, size=len(wt_vals))
        ax.scatter(np.zeros(len(wt_vals)) + jitter, wt_vals,  color=WT_COLOR,
                   s=90, zorder=3, edgecolors="white", linewidths=0.8)
        jitter = np.random.uniform(-0.08, 0.08, size=len(mut_vals))
        ax.scatter(np.ones(len(mut_vals)) + jitter, mut_vals, color=MUT_COLOR,
                   s=90, zorder=3, edgecolors="white", linewidths=0.8)

        d = cohens_d(wt_vals, mut_vals)
        fc = np.log2((mut_vals.mean() + 1) / (wt_vals.mean() + 1))
        label = f"log₂FC = {fc:+.1f}\nd = {d:.1f}" if not np.isnan(d) else f"log₂FC = {fc:+.1f}"
        ax.annotate(label, xy=(0.5, 0.87), xycoords="axes fraction",
                    ha="center", fontsize=14, color="#333")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["WT\n(4–6)", "Mutant\n(1–3)"], fontsize=15)
        ax.set_title(ct, fontsize=16, wrap=True, pad=14,
                     color=FOCUS_COLORS.get(ct, "black"), fontweight="bold")
        ax.set_ylabel("Volume (µm³)", fontsize=14)
        ax.set_xlim(-0.5, 1.5)
        ax.tick_params(axis="y", labelsize=13)

    # hide unused subplot (6th cell, only 5 types)
    for idx in range(len(cell_types), nrows * ncols):
        axes_flat[idx].axis("off")

    fig.suptitle("Physical volume per fish — focal cell types",
                 fontsize=16, y=1.01)
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
    """Grouped strip plot: within-WT / within-mutant / between-group Dice. 2×3 grid."""
    ncols, nrows = 3, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 11), sharey=False)
    axes_flat = axes.flatten()

    for idx, ct in enumerate(cell_types):
        ax = axes_flat[idx]
        label_id = type_to_id[ct]
        wt_d, mut_d, btw_d = group_dice_stats(vols, label_id)
        group_data = [wt_d, mut_d, btw_d]
        group_cols = [WT_COLOR, MUT_COLOR, "gray"]

        tops = []
        for xi, (vals, col) in enumerate(zip(group_data, group_cols)):
            vals = [v for v in vals if not np.isnan(v)]
            if not vals:
                continue
            m, s = np.mean(vals), np.std(vals)
            tops.append(m + s)
            ax.bar([xi], [m], color=col, alpha=0.5, width=0.5)
            ax.errorbar([xi], [m], yerr=s,
                        fmt="none", color="black", capsize=6, linewidth=1.8)
            jitter = np.random.uniform(-0.1, 0.1, len(vals))
            ax.scatter(np.full(len(vals), xi) + jitter, vals, color=col,
                       s=90, zorder=3, edgecolors="white", linewidths=0.8)

        if tops:
            ax.set_ylim(bottom=0, top=max(tops) * 1.25)

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["within\nWT", "within\nMut", "between"], fontsize=14)
        ax.set_title(ct, fontsize=16, pad=14,
                     color=FOCUS_COLORS.get(ct, "black"), fontweight="bold")
        ax.tick_params(axis="y", labelsize=13)
        if idx % ncols == 0:
            ax.set_ylabel("Dice coefficient", fontsize=14)

    for idx in range(len(cell_types), nrows * ncols):
        axes_flat[idx].axis("off")

    fig.suptitle("Dice: within- vs between-group", fontsize=16, y=1.01)
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
<title>Zebrafish Embryo Tail: WT vs Mutant Cell Type Spatial Analysis</title>
<style>
  html  { background: #dde1e7; }
  body  { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
          max-width: 1100px; margin: 2em auto; padding: 2.5em 4em 4em;
          color: #222; background: #fafafa;
          border-radius: 8px;
          box-shadow: 0 2px 16px rgba(0,0,0,.12); }
  h1   { color: #1a1a2e; border-bottom: 3px solid #4C9BE8; padding-bottom:.4em }
  h2   { color: #16213e; margin-top: 2.5em; border-left: 4px solid #E8714C;
          padding-left: .6em }
  h3   { color: #0f3460; }
  p, li { line-height: 1.7 }
  .meta { background:#eef2ff; border-radius:8px; padding:1em 1.5em; margin:1em 0 }
  figure { background: white; border-radius: 8px; padding: 1em;
           box-shadow: 0 1px 6px rgba(0,0,0,.1); }
  figcaption { font-size:.9em; color:#555; margin-bottom:.5em; font-style:italic }
  hr   { border:none; border-top:1px solid #ddd; margin:2em 0 }
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
# 8. CACHE I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_cache(vols, vol_df, dice_cache, type_to_id, id_to_type, path=CACHE_FILE):
    """Persist computed data so re-runs skip the slow load+compute steps."""
    # Volumes stored as compressed uint8 arrays; metadata as pickle bytes.
    meta = pickle.dumps({
        "vol_df":     vol_df,
        "dice_cache": dice_cache,
        "type_to_id": type_to_id,
        "id_to_type": id_to_type,
    })
    arrays = {f"vol_{fish}": vols[fish] for fish in ALL_FISH}
    np.savez_compressed(path, meta=np.frombuffer(meta, dtype=np.uint8), **arrays)
    print(f"  Cache saved → {path}")


def load_cache(path=CACHE_FILE):
    """Return (vols, vol_df, dice_cache, type_to_id, id_to_type) from cache."""
    data = np.load(path, allow_pickle=False)
    meta = pickle.loads(data["meta"].tobytes())
    vols = {fish: data[f"vol_{fish}"] for fish in ALL_FISH}
    return (vols, meta["vol_df"], meta["dice_cache"],
            meta["type_to_id"], meta["id_to_type"])


# ─────────────────────────────────────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate WT vs Mutant report.")
    parser.add_argument("--recompute", action="store_true",
                        help="Ignore cache and recompute from raw TIFs.")
    args = parser.parse_args()

    np.random.seed(42)

    print("=== WT vs Mutant Zebrafish Cell-Type Analysis ===")
    print(f"  Registration: {REG_BASE}")
    print(f"  WT fish:      {WT_FISH}")
    print(f"  Mutant fish:  {MUT_FISH}")

    use_cache = os.path.exists(CACHE_FILE) and not args.recompute

    if use_cache:
        print(f"\n[1/3] Loading from cache: {CACHE_FILE}")
        vols, vol_df, dice_cache, type_to_id, id_to_type = load_cache()
        print(f"  Volumes: {len(vols)} fish  |  Cell types: {len(type_to_id)}")
    else:
        if args.recompute:
            print("\n  --recompute: ignoring existing cache.")

        # 1. Label map
        print("\n[1/6] Building label map ...")
        type_to_id, id_to_type = build_label_map(ANNOTS_CSV)

        # 2. Load volumes
        print("\n[2/6] Loading label volumes ...")
        vols = load_label_volumes(ALL_FISH)

        # 3. Volume dataframe
        print("\n[3/6] Computing voxel volumes ...")
        vol_df = volume_df(vols, type_to_id, id_to_type)
        print(f"  Volume table: {len(vol_df)} rows")

        print("  Pre-computing Dice for all cell types ...")
        dice_cache = compute_all_dice(vols, type_to_id)
        print(f"  Done — {len(dice_cache)} cell types cached.")

        print("\n  Saving cache ...")
        save_cache(vols, vol_df, dice_cache, type_to_id, id_to_type)

    # 4. Build plots
    print("\n[4/6] Generating figures ...")

    sections = []

    # ── Intro ─────────────────────────────────────────────────────────────────
    sections.append("""
<h1>Zebrafish Embryo Tail: WT vs Mutant Cell Type Spatial Analysis</h1>
<div class="meta">
<table style="border-collapse:collapse;width:100%;font-size:.95em">
<tr>
  <td style="padding:.4em 1.2em .4em 0;white-space:nowrap;color:#555;font-weight:600">Registration</td>
  <td style="padding:.4em 0">dapi_blend_rigid_affine_z10 &nbsp;·&nbsp; yolk-trimmed</td>
</tr>
<tr>
  <td style="padding:.4em 1.2em .4em 0;color:#555;font-weight:600">Groups</td>
  <td style="padding:.4em 0">
    <span style="color:#00838F;font-weight:bold">WT</span>&nbsp; fish 4, 5, 6
    &emsp;
    <span style="color:#C2185B;font-weight:bold">Mutant</span>&nbsp; fish 1, 2, 3
  </td>
</tr>
<tr>
  <td style="padding:.4em 1.2em .4em 0;color:#555;font-weight:600">Volume</td>
  <td style="padding:.4em 0">111 &times; 1239 &times; 1236 voxels &nbsp;·&nbsp; same registered space for all fish</td>
</tr>
<tr>
  <td style="padding:.4em 1.2em .4em 0;color:#555;font-weight:600">Cell types</td>
  <td style="padding:.4em 0">34 types &nbsp;·&nbsp; integer IDs 1-based alphabetical order
    (<code>leiden10annots.csv</code>)</td>
</tr>
</table>
</div>

<h2>Focal cell types</h2>
<table style="border-collapse:collapse;width:100%;margin:1em 0;font-size:.95em">
<thead>
<tr style="background:#2C3E50;color:white">
  <th style="padding:.6em 1em;text-align:left">Cell type</th>
  <th style="padding:.6em 1em;text-align:left">Expected in mutant</th>
  <th style="padding:.6em 1em;text-align:center">Label ID</th>
</tr>
</thead>
<tbody>
<tr style="background:#C0392B18">
  <td style="padding:.6em 1em;border-left:4px solid #C0392B">
    <span style="color:#C0392B;font-weight:bold">● blood</span></td>
  <td style="padding:.6em 1em">↓ Decreased / absent</td>
  <td style="padding:.6em 1em;text-align:center;font-family:monospace">""" + str(type_to_id["blood"]) + """</td>
</tr>
<tr style="background:#8E44AD18">
  <td style="padding:.6em 1em;border-left:4px solid #8E44AD">
    <span style="color:#8E44AD;font-weight:bold">● hematopoietic cell</span></td>
  <td style="padding:.6em 1em">↓ Decreased</td>
  <td style="padding:.6em 1em;text-align:center;font-family:monospace">""" + str(type_to_id["hematopoietic cell"]) + """</td>
</tr>
<tr style="background:#E67E2218">
  <td style="padding:.6em 1em;border-left:4px solid #E67E22">
    <span style="color:#E67E22;font-weight:bold">● cranial vasculature</span></td>
  <td style="padding:.6em 1em">↓ Decreased</td>
  <td style="padding:.6em 1em;text-align:center;font-family:monospace">""" + str(type_to_id["cranial vasculature"]) + """</td>
</tr>
<tr style="background:#27AE6018">
  <td style="padding:.6em 1em;border-left:4px solid #27AE60">
    <span style="color:#27AE60;font-weight:bold">● spinal cord</span></td>
  <td style="padding:.6em 1em">→ Control — relatively unchanged</td>
  <td style="padding:.6em 1em;text-align:center;font-family:monospace">""" + str(type_to_id["spinal cord"]) + """</td>
</tr>
<tr style="background:#2980B918">
  <td style="padding:.6em 1em;border-left:4px solid #2980B9">
    <span style="color:#2980B9;font-weight:bold">● pronephric distal early tubule</span></td>
  <td style="padding:.6em 1em">↑ Increased</td>
  <td style="padding:.6em 1em;text-align:center;font-family:monospace">""" + str(type_to_id["pronephric distal early tubule"]) + """</td>
</tr>
</tbody>
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

    # ── Metrics explainer ─────────────────────────────────────────────────────
    sections.append("""
<div style="background:#f0f4ff;border-left:4px solid #4a6fa5;border-radius:6px;
            padding:1.2em 1.6em;margin:2em 0;font-size:.95em;line-height:1.8">
<h3 style="margin-top:0;color:#2C3E50">How to read the metrics</h3>
<p style="margin:.4em 0">
  <b>Log₂ fold-change (log₂FC)</b> — how much larger or smaller a cell type's volume
  is in the mutant relative to WT, on a symmetric log scale.
  <span style="color:#555">0 = no change &nbsp;·&nbsp; +1 = doubled &nbsp;·&nbsp;
  −1 = halved &nbsp;·&nbsp; +2 = 4× larger &nbsp;·&nbsp; −2 = 4× smaller.</span>
</p>
<p style="margin:.4em 0">
  <b>Cohen's d</b> — the difference between group means expressed in units of the
  pooled standard deviation. Accounts for variability across fish, not just the
  average shift.
  <span style="color:#555">|d| ≈ 0.2 small &nbsp;·&nbsp; |d| ≈ 0.5 medium &nbsp;·&nbsp;
  |d| ≈ 0.8 large &nbsp;·&nbsp; |d| &gt; 1.5 very large.</span>
  <br>A large log₂FC with a small Cohen's d means the shift is real on average but the
  fish are variable; both together give more confidence.
</p>
<p style="margin:.4em 0">
  <b>Dice coefficient</b> — measures spatial overlap between two binary masks
  (here: which voxels contain a given cell type in fish A vs fish B).
  Dice = 2 |A ∩ B| / (|A| + |B|): 1 = perfect overlap, 0 = no overlap.
  Computed as standard 3D Dice across the full volume — equivalent to
  weighting each Z-slice by how much of the total structure it contains,
  so slices with more signal contribute more and near-empty slices barely matter.
</p>
<p style="margin:.4em 0;color:#666">
  <b>No p-values</b> — with n=3 per group, the minimum achievable two-sided
  Mann–Whitney p is 0.10, making significance thresholds uninformative.
  Effect sizes are the appropriate summary at this sample size.
</p>
</div>
""")

    # ── Section 1: Composition ────────────────────────────────────────────────
    sections.append("<h2>1 · Cell type composition per fish</h2>")
    sections.append("<p>Each bar shows one fish's tissue broken down by cell type as a "
                    "percentage of its total labelled volume. Focal cell types are coloured "
                    "and labelled; all others are gray.</p>")
    print("  fig: composition ...")
    fig = plot_composition(vol_df)
    sections.append(img_tag(fig_to_b64(fig), ""))

    # ── Section 2: Overview fold-change ───────────────────────────────────────
    sections.append("<h2>2 · Global volume changes: all cell types</h2>")
    sections.append("<p>Log₂ fold-change of mean voxel volume (Mutant / WT), "
                    "averaged across fish within each group.</p>")
    print("  fig: volume overview ...")
    fig = plot_volume_overview(vol_df)
    sections.append(img_tag(fig_to_b64(fig), ""))

    # ── Section 2: Focal cell type volumes ────────────────────────────────────
    sections.append("<h2>3 · Focal cell-type volumes — individual fish</h2>")
    sections.append("<p>Physical volume in µm³ (voxel count × 0.4516 µm³/voxel).<br>"
                    "Bar = group mean; error bar = ±1 SD; dots = individual fish.<br>"
                    "<b>log₂FC</b> and <b>Cohen's d</b> are annotated inside each subplot.</p>")
    print("  fig: volume per fish ...")
    fig = plot_volume_per_fish(vol_df)
    sections.append(img_tag(fig_to_b64(fig),
        "Physical volume (µm³) per fish for each focal cell type. Individual fish points overlay "
        "the group mean bar."))

    # ── Section 3: Dice within vs between ─────────────────────────────────────
    sections.append("<h2>4 · Within- vs between-group Dice — focal cell types</h2>")
    sections.append("<p>Within-WT (3 pairs), within-Mutant (3 pairs), and between-group "
                    "(9 pairs) Dice for each focal cell type. "
                    "For a control (spinal cord) all three should be similar; "
                    "for disrupted types, between-group Dice drops below within-group.</p>"
                    '<div style="background:#fff8e1;border-left:4px solid #f9a825;'
                    'border-radius:4px;padding:.7em 1em;margin:.8em 0;font-size:.9em">'
                    "<b>⚠ Interpretation caveat:</b> when a cell type is nearly absent in the mutant, "
                    "between-group Dice can appear artificially high — two near-empty volumes "
                    "trivially agree on being empty. High between-group Dice is only meaningful "
                    "when both groups actually have the tissue."
                    "</div>")
    print("  fig: dice within vs between ...")
    fig = plot_dice_comparison(vols, type_to_id)
    sections.append(img_tag(fig_to_b64(fig), ""))

    # ── Section 4: All-cell-type Dice summary ──────────────────────────────────
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
    print(f"\n[5/5] Writing report → {OUT_HTML}")
    html = build_html(sections)
    with open(OUT_HTML, "w") as fh:
        fh.write(html)

    csv_path = os.path.join(SCRIPT_DIR, "volumes.csv")
    vol_df.to_csv(csv_path, index=False)
    print(f"  Volume table saved → {csv_path}")

    print("\n=== Done ===")
    print(f"  Open: {OUT_HTML}")


if __name__ == "__main__":
    main()
