# Image-derived Luxar pipeline

This is independent of the mesh-based scene in `../mesh_scene/`. It does not
read or overwrite that scene.

The pipeline demonstrates Luxar's native image workflow:

1. Read the registered DAPI TIFF for fish 3 and fish 4.
2. Downsample it reproducibly and save a compact NumPy volume.
3. Run `luxar gsplat fit` to reconstruct each image volume as oriented Gaussian
   splats.
4. Run `luxar gsplat lod --recipe stream` to build progressive, energy-ordered
   streaming levels.
5. Compile those image-derived splats with pronephric cell centroids and exactly
   two Xenium gene layers (`hmga1a`, `mef2ca`).

The final scene contains no triangle meshes. DAPI is the tissue context; cells
and transcripts remain accurately registered point layers.

## Reproduce

Install Luxar's optional CPU fitting dependencies once:

```bash
make deps
```

Build and verify everything:

```bash
make all
```

Serve on ports distinct from the original scene:

```bash
make serve
```

Open <http://127.0.0.1:5174/?src=http://127.0.0.1:8001>.

`make serve` serves an existing build without repeating the CPU fits.

All choices are in `config.toml`. CPU fitting is intentionally a modest preview
(6,000 splats per fish, 500 iterations). Increase `seeds` and `iterations` for a
publication-quality reconstruction. Luxar's full scientific model-selection
workflow begins with `luxar gsplat cal`; it is omitted from the default build
because its multi-fit Noise2Self sweep is substantially more expensive.
