"""
prep_organella.py
=================
Prepare per-fish organella input from analysis/5_consensus outputs.

For each fish and each focal cell type, extracts a binary mask from
cell_type_label.tif and runs connected components to assign a unique
integer ID to each spatially distinct cluster. These instance label
volumes are what organella needs to measure and mesh individual objects.

Also writes a binary tissue mask (all non-zero labels merged) so the
full fish body can be shown as context in the 3D view.

Output layout  (analysis/organella/{fish}/)
-------------------------------------------
  source.tif                                  symlink to cell_type_label.tif
  source_blood_label.tif
  source_cranial_vasculature_label.tif
  source_hematopoietic_cell_label.tif
  source_pronephric_distal_early_tubule_label.tif
  source_spinal_cord_label.tif
  source_tissue_mask.tif                      binary mask — all tissue

Usage
-----
  python3 prep_organella.py
"""

import csv
import logging
import os
import re

import numpy as np
import tifffile
from scipy.ndimage import label as nd_label

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(os.path.dirname(SCRIPT_DIR))   # zebrafish_registration/
CONSENSUS  = os.path.join(os.path.dirname(REPO_ROOT), "analysis", "5_consensus")
OUT_BASE   = os.path.join(os.path.dirname(REPO_ROOT), "analysis", "organella")
ANNOTS_CSV = os.path.join(REPO_ROOT, "leiden10annots.csv")

# ── config ─────────────────────────────────────────────────────────────────────

ALL_FISH = [1, 2, 3, 4, 5, 6]

FOCAL_TYPES = [
    "blood",
    "hematopoietic cell",
    "cranial vasculature",
    "spinal cord",
    "pronephric distal early tubule",
]

# ── helpers ────────────────────────────────────────────────────────────────────

def build_label_map(annots_csv: str) -> dict:
    """Map cell type name → 1-based label ID (sorted alphabetically)."""
    cell_types = set()
    with open(annots_csv) as f:
        for row in csv.DictReader(f):
            cell_types.add(row["leiden10annots"])
    return {ct: i + 1 for i, ct in enumerate(sorted(cell_types))}


def safe_name(cell_type: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", cell_type.lower()).strip("_")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    type_to_id = build_label_map(ANNOTS_CSV)
    focal_ids  = {ct: type_to_id[ct] for ct in FOCAL_TYPES}
    log.info("Focal label IDs: %s", {ct: focal_ids[ct] for ct in FOCAL_TYPES})

    for fish in ALL_FISH:
        fish_in  = os.path.join(CONSENSUS, str(fish))
        fish_out = os.path.join(OUT_BASE,  str(fish))
        os.makedirs(fish_out, exist_ok=True)

        label_path = os.path.join(fish_in, "cell_type_label.tif")
        log.info("Fish %d: loading %s", fish, label_path)
        vol = tifffile.imread(label_path)   # (Z, H, W) uint8
        log.info("  shape %s  dtype %s", vol.shape, vol.dtype)

        # Symlink so organella has a source image with matching shape
        src_link = os.path.join(fish_out, "source.tif")
        if os.path.lexists(src_link):
            os.remove(src_link)
        os.symlink(label_path, src_link)

        # Per focal type: connected components → uint32 instance label volume
        for ct, ct_id in focal_ids.items():
            log.info("  [%s] label %d …", ct, ct_id)
            binary = (vol == ct_id)
            n_vox  = int(binary.sum())
            if n_vox == 0:
                log.warning("    no voxels — skipping")
                continue
            labeled, n_instances = nd_label(binary)
            log.info("    %d voxels → %d instances", n_vox, n_instances)
            out_path = os.path.join(fish_out, f"source_{safe_name(ct)}_label.tif")
            tifffile.imwrite(out_path, labeled.astype(np.uint32), compression="zlib")
            log.info("    → %s", out_path)

        # All tissue: single binary mask for context rendering
        log.info("  [tissue] all labels merged …")
        tissue = (vol > 0).astype(np.uint8)
        out_path = os.path.join(fish_out, "source_tissue_mask.tif")
        tifffile.imwrite(out_path, tissue, compression="zlib")
        log.info("  → %s", out_path)

        del vol

    log.info("Done. Output in %s", OUT_BASE)


if __name__ == "__main__":
    main()
