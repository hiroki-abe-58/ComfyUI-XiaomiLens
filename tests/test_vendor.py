"""The vendored code is the pinned upstream commit, byte for byte, except the one documented patch (scripts/vendor_sync.py)."""

import hashlib
from pathlib import Path

import vendor_sync

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "xiaomi_lens_comfy" / "vendor"


def test_vendor_tree_equals_upstream(upstream_dir):
    assert vendor_sync.main(["--upstream", str(upstream_dir), "--check"]) == 0


def test_manifest_lists_every_vendored_file():
    manifest = dict(reversed(line.split("  ", 1)) for line in (VENDOR / "MANIFEST.sha256").read_text(encoding="ascii").splitlines())
    files = {p.relative_to(VENDOR).as_posix() for p in VENDOR.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    assert files - {"MANIFEST.sha256", "UPSTREAM.md", "__init__.py"} == set(manifest)
    for name, digest in manifest.items():
        assert hashlib.sha256((VENDOR / name).read_bytes()).hexdigest() == digest, name


def test_the_patch_only_fixes_the_decoder_kernel_choice():
    patched = (VENDOR / "xiaomi_lens" / "model" / "pixel_mask_decoder.py").read_text(encoding="utf-8")
    assert "Modified by ComfyUI-XiaomiLens" in patched and 'backend = "flash"' in patched and "os.environ" not in patched
    from xiaomi_lens_comfy.vendor.xiaomi_lens.model import pixel_mask_decoder

    assert pixel_mask_decoder._DECODER_ATTN == "flash"  # upstream's default: flash-attn when installed, else MultiheadAttention
