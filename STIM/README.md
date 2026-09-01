# STIM residual refinement

## What this script does

`2b_stim_refinement.py` attempts to correct residual slice-to-slice misalignment
on top of script 2's elastix registration, using STIM (a spatial transcriptomics
alignment tool based on SIFT + RANSAC).

Steps:
1. **build** — project cell centroids + gene counts through script 2's transforms into the registered canvas space, write to STIM's N5 container
2. **stim** — run STIM's `st-align-pairs` to find pairwise rigid transforms between adjacent slices using gene expression patterns
3. **chain** — sequentially compose pairwise transforms outward from the reference slice
4. **globalopt** — least-squares global optimizer over the full pairwise constraint graph (more robust to drift than chain)
5. **apply** — apply per-slice correction transforms to the registered TIFFs
6. **stack** — assemble corrected 3D stacks

Globalopt is automatically skipped with a warning if chain corrections are too large (a sign STIM found noise rather than real residuals).

## Why it didn't work for our data

STIM-SIFT needs meaningful misalignment between slices to find reliable feature
matches. After script 2's good elastix registration, the residual misalignment
is only a few pixels. With ~200 cells per slice and 10 µm z-spacing, SIFT cannot
distinguish real residuals from random feature matches at that scale — so the
pairwise transforms are noise, and applying them makes the registration worse.
