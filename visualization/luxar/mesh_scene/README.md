# Mesh-based Luxar scene

This directory is a self-contained, reproducible build for an interactive Luxar
comparison of mutant fish 3 and WT fish 4. It uses the registered tissue/tubule
meshes plus Xenium transcripts and cell centroids belonging to cells annotated as
`pronephric distal early tubule`.

## Rebuild

From this directory:

```bash
make all
make verify
make serve
```

`make prepare` uses the registration project's existing environment.
`make scene` uses the shared Luxar environment in the parent directory. Both can
be overridden:

```bash
make all PREP_PYTHON=/path/to/python LUXAR_PYTHON=/path/to/python
```

The pipeline is:

1. Match the Leiden annotation IDs to the correct Xenium run automatically.
2. Assign target cells to serial-section tiles from the fish bounding-box table.
3. Transform cell centroids and their QV-filtered transcripts through the saved
   slice and cross-fish registrations.
4. Decode the already-generated Organella tissue and pronephric meshes.
5. Choose the most abundant genes in the target cells and compile a side-by-side
   `.luxar.zarr` scene.

The original inputs are never modified. Intermediate portable NumPy files go to
`build/processed/`; the final scene is `build/zebrafish_pronephric.luxar.zarr`.
`build/manifest.json` records input paths, selected genes, counts, package
versions, and SHA-256 hashes of the generated intermediates.

## Scene layers

- translucent grey tissue surface;
- cyan pronephric distal early tubule surface;
- white target-cell centroids;
- one independently toggleable colored point layer per selected gene.

Fish 3 (mutant) and fish 4 (WT) are placed side by side but retain the common
registered coordinate system within each specimen.

## Configuration

Edit `config.toml` to change fish, target annotation, QV threshold, number of
genes, point sizes, colors, or comparison spacing. Then rerun `make all`.

`make serve` serves an existing build without rebuilding it.
