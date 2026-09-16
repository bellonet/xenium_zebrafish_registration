# 3D visualisation with Organella

## Data prep

`prep_organella.py` — run once, reads `analysis/5_consensus/{fish}/cell_type_label.tif` per fish.

For each focal cell type it runs connected components on the binary mask to assign a unique integer ID to each spatially distinct cluster. This is the instance label format organella expects. Also writes a binary tissue mask (all labels merged) for context rendering.

Output: `analysis/organella/{1..6}/`

## Running organella

```bash
organella process analysis/organella/ \
  -o zebrafish_registration/visualization/organella/report.parquet \
  --voxel-size-um 10,0.2125,0.2125 \
  --with-mesh \
  --max-workers 6
```

Then view:
```bash
organella view zebrafish_registration/visualization/organella/report.parquet
```

## Key notes

- **No downsampling** — full resolution (XY = 0.2125 µm, Z = 10 µm).
- **Voxels are highly anisotropic** (47× in Z vs XY). Organella handles this correctly via `--voxel-size-um` — smoothing and meshing are done in physical µm. The Z resolution limit (10 µm slabs) is real and no smoothing can recover it.
- **Instances = connected components** of each cell type's binary mask, not individual cells from a cell segmentation. Touching cells of the same type merge into one instance.
- **tissue** entity is a whole-structure mask (no instances), used only for context in the 3D view.
- Organella is installed from source at `scripts/organella/` into `scripts/.venv`.
