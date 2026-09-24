ROOT := $(abspath ..)
PREP_PYTHON := $(abspath $(ROOT)/../../.venv/bin/python)
LUXAR_PYTHON := $(ROOT)/.venv/bin/python
LUXAR := $(ROOT)/.venv/bin/luxar

.PHONY: all deps prepare fit scene verify serve clean

all: scene verify

deps:
	$(LUXAR_PYTHON) -m ensurepip
	$(LUXAR_PYTHON) -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3"
	$(LUXAR_PYTHON) -m pip install "scipy>=1.9,<2" "tifffile>=2023.1"

prepare:
	$(PREP_PYTHON) scripts/prepare_inputs.py --config config.toml

fit: prepare
	$(LUXAR_PYTHON) scripts/fit_volumes.py --config config.toml

scene: fit
	$(LUXAR_PYTHON) scripts/build_scene.py --config config.toml

verify:
	$(LUXAR_PYTHON) scripts/verify.py --config config.toml

serve:
	@test -f build/zebrafish_image_splats.luxar.zarr/zarr.json || { \
		echo "Scene not found; run 'make all' first." >&2; \
		exit 1; \
	}
	$(LUXAR) serve build/zebrafish_image_splats.luxar.zarr --viewer --port 8001 --viewer-port 5174

clean:
	$(LUXAR_PYTHON) -c 'import shutil; shutil.rmtree("build", ignore_errors=True)'
