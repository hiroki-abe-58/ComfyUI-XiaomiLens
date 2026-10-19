"""Test setup. References come from environment variables, never from this repository:

XL_TINY_DIR      a tiny checkpoint built by scripts/make_tiny_model.py (from tools/download_model.py --no-weights; 84 MB)
XL_UPSTREAM_DIR  SeerRay-Lab/Xiaomi-Lens-0 checkout at the commit in scripts/vendor_sync.py (the official infer.py and package)
XL_COMFYUI_DIR   ComfyUI checkout (v0.38.0 or later)
XL_REQUIRE_REFERENCES=1  fail instead of skip when one of them is missing (CI sets this)
The tests run on the CPU and compare the pack with the official infer.py functions (imported from XL_UPSTREAM_DIR) on the same model
in the same process. The real model is compared on a GPU outside CI (docs/VERIFICATION.md).
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))
REQUIRE = os.environ.get("XL_REQUIRE_REFERENCES") == "1"


def _path(var):
    p = os.environ.get(var)
    return p if p and Path(p).exists() else None


COMFY = _path("XL_COMFYUI_DIR")
TINY = _path("XL_TINY_DIR")
UPSTREAM = _path("XL_UPSTREAM_DIR")
if COMFY:
    sys.path.insert(1, COMFY)
    import comfy.options

    comfy.options.enable_args_parsing(False)
    import comfy.cli_args

    comfy.cli_args.args.cpu = True


def _need(value, what):
    if value:
        return value
    if REQUIRE:
        pytest.fail(f"{what} is required (XL_REQUIRE_REFERENCES=1)")
    pytest.skip(f"{what} not available")


@pytest.fixture(scope="session")
def comfy_dir():
    return _need(COMFY, "ComfyUI checkout (XL_COMFYUI_DIR)")


@pytest.fixture(scope="session")
def tiny_dir():
    return Path(_need(TINY, "tiny checkpoint (XL_TINY_DIR)"))


@pytest.fixture(scope="session")
def upstream_dir():
    return Path(_need(UPSTREAM, "Xiaomi-Lens-0 checkout (XL_UPSTREAM_DIR)"))


@pytest.fixture(scope="session")
def official(upstream_dir):
    """The official infer.py of the upstream checkout (it puts the checkout's src/ on sys.path itself)."""
    sys.path.insert(0, str(upstream_dir))
    import infer

    return infer


@pytest.fixture(scope="session")
def tiny(tiny_dir):
    """(model, processor, mask_token_id) of the tiny checkpoint, loaded by the pack."""
    from xiaomi_lens_comfy import runtime

    return runtime.load(str(tiny_dir), "sdpa")


def image(height=256, width=320, seed=0):
    import numpy as np
    from PIL import Image

    return Image.fromarray((np.random.default_rng(seed).random((height, width, 3)) * 255).astype(np.uint8))
