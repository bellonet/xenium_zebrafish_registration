# Zebrafish Luxar visualizations

This directory contains two scene-building workflows for the same
mutant-versus-WT pronephric comparison. They share the top-level `.venv` but
keep their configuration, generated data, and scene outputs separate. The image
workflow reuses the mesh workflow's tested registration and point-preparation
helpers so both use identical coordinate logic.

## Pipelines

### `mesh_scene/` — preferred viewer

Uses the registered tissue and pronephric surface meshes with cell centroids and
Xenium transcript points. This is the clearer and more polished visualization.

```bash
make serve-mesh
```

Build it with `make mesh`. See [`mesh_scene/README.md`](mesh_scene/README.md).

### `image_splats/` — experimental reconstruction

Fits Gaussian splats directly to the registered DAPI volumes and combines them
with cell and transcript points. The result is useful for evaluating Luxar's
native image workflow, but the CPU build takes roughly 25–30 minutes.

```bash
make serve-image
```

Build it with `make image-splats`. See
[`image_splats/README.md`](image_splats/README.md).

The serve targets never rebuild. If a scene is missing, they report which build
target to run.

## Shared files

- `.venv/`: Luxar and image-fitting environment
- `requirements-luxar.txt`: Luxar runtime dependencies
- `requirements-preparation.txt`: data-preparation dependencies
- `.gitignore`: generated-build exclusions
