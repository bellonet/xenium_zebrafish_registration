ROOT := $(abspath ..)
PREP_PYTHON ?= $(abspath $(ROOT)/../../.venv/bin/python)
LUXAR_PYTHON ?= $(ROOT)/.venv/bin/python
LUXAR ?= $(ROOT)/.venv/bin/luxar

.PHONY: all prepare scene verify serve clean

all: scene

prepare:
	$(PREP_PYTHON) scripts/prepare_data.py --config config.toml

scene: prepare
	$(LUXAR_PYTHON) scripts/build_scene.py --config config.toml

verify:
	$(LUXAR_PYTHON) scripts/verify_build.py --config config.toml

serve:
	@test -f build/zebrafish_pronephric.luxar.zarr/zarr.json || { \
		echo "Scene not found; run 'make all' first." >&2; \
		exit 1; \
	}
	$(LUXAR) serve build/zebrafish_pronephric.luxar.zarr --viewer

clean:
	$(LUXAR_PYTHON) scripts/clean_build.py
