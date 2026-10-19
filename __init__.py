"""Xiaomi-Lens-0 for ComfyUI (unofficial integration of https://github.com/SeerRay-Lab/Xiaomi-Lens-0)."""

import logging

try:
    from .xiaomi_lens_comfy.nodes import comfy_entrypoint  # noqa: F401
except ImportError as e:  # ComfyUI without the V3 node API, or a missing dependency
    logging.error("[Xiaomi-Lens] not loaded: %s", e)
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
