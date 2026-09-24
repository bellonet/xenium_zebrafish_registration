.PHONY: help mesh image-splats verify-mesh verify-image serve-mesh serve-image

help:
	@echo "Luxar pipelines:"
	@echo "  make mesh          Build the preferred mesh-based scene"
	@echo "  make image-splats  Build the experimental image-derived scene (slow on CPU)"
	@echo "  make serve-mesh    Serve the existing mesh scene"
	@echo "  make serve-image   Serve the existing image-splat scene"
	@echo "  make verify-mesh   Verify the mesh scene"
	@echo "  make verify-image  Verify the image-splat scene"

mesh:
	$(MAKE) -C mesh_scene all

image-splats:
	$(MAKE) -C image_splats all

verify-mesh:
	$(MAKE) -C mesh_scene verify

verify-image:
	$(MAKE) -C image_splats verify

serve-mesh:
	$(MAKE) -C mesh_scene serve

serve-image:
	$(MAKE) -C image_splats serve
