# Zebrafish tail registration pipeline

Registers Xenium spatial transcriptomics data from multiple zebrafish tails into a common space, so transcripts and segmentations from different fish can be compared directly.

## Requirements

Python 3.13, environment managed with [uv](https://github.com/astral-sh/uv).

```bash
uv venv
uv pip install -r requirements.txt
```

Always run scripts with `.venv/bin/python3`, not bare `python3`.

Data lives in `../data/`, outputs go to `../analysis/`.

## The pipeline

Run the scripts in order. Use `run_pipeline.sh` to chain them automatically.

```bash
bash run_pipeline.sh
```

Or run individually:

```bash
.venv/bin/python3 1_crop_and_tag_2d_slices.py --from-step tiles
.venv/bin/python3 2_rigid_registration.py
.venv/bin/python3 3_registration_experiments.py
.venv/bin/python3 4_cross_fish_registration.py
```

---

### Script 1: crop and tag 2D slices

`1_crop_and_tag_2d_slices.py`

Reads the raw Xenium morphology images (3 runs, 6 fish each). Detects which fish is which in each image by clustering cell centroids, crops each fish out, assigns a consistent fish ID (1-6) across runs, and saves per-fish 2D slice stacks.

Output: `analysis/individual_fish_2d/{fish}/c{channel}/{global_slice}.tif`

Config: `INPUT_FOLDERS`, `NUM_CHANNELS`. Run from scratch with `--from-step tiles`. Use `--from-step tag` etc. to resume from a later step.

---

### Script 2: per-slice 2D registration

`2_rigid_registration.py`

For each fish, registers every 2D slice independently. Two steps:

1. Rotates each slice to align the dorsal-ventral axis, then runs elastix rigid registration to align slices to a common canvas.
2. Applies the same transforms to segmentation outputs (seg_cells, seg_nuclei, cell_type_label, tissue_map). Label images use nearest-neighbour interpolation throughout so label boundaries stay sharp.

Also generates per-gene expression images for use in script 3.

Output: `analysis/2_registered/{fish}/`

Config: `NUM_CHANNELS`, `STEPS`. Run all fish by default; use `--fish 3` for a single fish.

---

### Script 3: per-slice correction experiments

`3_registration_experiments.py`

Tries different driving images for a second-pass 2D rigid correction on top of script 2. The idea: script 2 aligns slices within one fish; script 3 tries to improve cross-slice alignment by using richer image content (gene expression, cell-type maps).

Experiments:
- `dapi_blend`: DAPI blended with cell-type map
- `gene_composite`: composite of structurally-informative genes (tp63, myod1, tbxta, sox3)
- `tissue_mask`: cell-type label map (per-pixel cell type ID)
- `multi_metric`: combination of metrics

Outputs a correction transform and evaluation metrics (NCC, smoothness) for each experiment so you can pick the best one.

Output: `analysis/3_improved_registration/{experiment}/{fish}/`

Config: `EXPERIMENTS`, `STEPS`. Run all by default; filter with `--experiments dapi_blend`.

---

### Script 4: cross-fish 3D registration

`4_cross_fish_registration.py`

Registers all fish into a shared 3D space using elastix. Each fish is registered to a reference fish (default: fish 1). Tries combinations of driving image and registration type.

Driving images:
- `dapi` / `dapi_blend`: fluorescence channel, uses NCC metric
- `cell_type_map`: cell-type label image, uses Mutual Information (correct for discrete labels)

Registration types: `rigid` or `rigid_affine`.

Z-spacing is confirmed 10 µm. Registration runs at full XY resolution.

All label/seg outputs use nearest-neighbour interpolation. Fluorescence channels use linear.

Output: `analysis/4_registered/{experiment}/{fish}/`

Config: `WT_FISH`, `MUTANT_FISH`, `DRIVING_OPTIONS`, `STAGES_OPTIONS`. Filter with `--driving dapi_blend --stages rigid_affine`.

---

### Script 5: apply registration to new data

`5_apply_registration.py`

Applies the full registration stack to new data: transcripts, cell centroids, or any TIFF image.

All three registration layers are composed and applied in a single pass (no cascaded interpolation errors):
- Layer 1 (script 2): per-slice 2D rigid
- Layer 2 (script 3): per-slice 2D rigid correction
- Layer 3 (script 4): 3D cross-fish rigid/affine

For images: one `map_coordinates` call using backward-mapping through the composed transform.
For points: exact coordinate math, no interpolation at any step.

Config block at the top of the file. Set `FISH`, `RUN`, `SCRIPT3_EXP`, `SCRIPT4_EXP`, and the `APPLY_TO_*` flags, then run:

```bash
.venv/bin/python3 5_apply_registration.py
```

Output: `analysis/5_applied/`

Set `NN_INTERPOLATION = True` for label images.

---

## Z-spacing

Confirmed 10 µm between slices (verified with colleague). XY pixel size is 0.2125 µm.

## Fish numbering

Fish 1-6 are consistent across all scripts and runs. Fish 1 is the registration reference.
