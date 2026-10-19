"""Rebuild xiaomi_lens_comfy/vendor/xiaomi_lens from an upstream checkout and rewrite vendor/MANIFEST.sha256 (maintainer tool).

    python scripts/vendor_sync.py --upstream <SeerRay-Lab/Xiaomi-Lens-0 checkout at UPSTREAM_COMMIT> [--check]

Only the model package that Hugging Face inference needs is vendored (`src/xiaomi_lens` without `integrations/`, which is the
vLLM serving code). Every file is copied byte for byte except `model/pixel_mask_decoder.py`, which gets one PATCH: upstream
picks the mask decoder's attention kernel from the XIAOMI_LENS_DECODER_ATTN environment variable at import time (default
"flash", i.e. flash-attn when it is installed, PyTorch's MultiheadAttention otherwise). The vendored copy always uses that
default, so the installed package reads no environment variables. The kernels give the same results up to rounding; the default
is what upstream's infer.py runs unless the variable is set.
tests/test_vendor.py rebuilds the tree from the pinned upstream commit in CI and compares every byte.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "xiaomi_lens_comfy" / "vendor"
UPSTREAM_COMMIT = "157b7bcb0b5010e81df858f5e9e4aa55a2fa9819"

COPIED = [
    "xiaomi_lens/__init__.py", "xiaomi_lens/constants.py", "xiaomi_lens/model/__init__.py", "xiaomi_lens/model/config.py",
    "xiaomi_lens/model/modeling_xiaomi_lens.py",
]
PATCHED = {
    "xiaomi_lens/model/pixel_mask_decoder.py": [
        ('"""MaskFormer-style dynamic classifier',
         '# Modified by ComfyUI-XiaomiLens: _resolve_attention_backend() returns the upstream default ("flash") instead of reading\n'
         '# the XIAOMI_LENS_DECODER_ATTN variable of the process. Nothing else is changed.\n\n"""MaskFormer-style dynamic classifier'),
        ('    backend = os.environ.get("XIAOMI_LENS_DECODER_ATTN", "").lower() or "flash"\n',
         '    backend = "flash"\n'),
    ],
}
LICENSE_FILES = {"LICENSE": "LICENSE"}  # vendored name: upstream path


def expected_tree(upstream: Path) -> dict[str, bytes]:
    src = upstream / "src"
    out = {rel: (src / rel).read_bytes() for rel in COPIED}
    for rel, edits in PATCHED.items():
        text = (src / rel).read_text(encoding="utf-8")
        for old, new in edits:
            if text.count(old) != 1:
                raise SystemExit(f"{rel}: patch anchor not found exactly once: {old[:60]!r}")
            text = text.replace(old, new)
        out[rel] = text.encode("utf-8")
    for name, upstream_name in LICENSE_FILES.items():
        out[name] = (upstream / upstream_name).read_bytes()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--check", action="store_true", help="only compare, write nothing")
    args = ap.parse_args(argv)
    tree = expected_tree(args.upstream)
    bad = [rel for rel, data in tree.items() if not (VENDOR / rel).is_file() or (VENDOR / rel).read_bytes() != data]
    if args.check:
        print("\n".join(f"DIFFERS {rel}" for rel in bad) or "vendor tree matches the upstream checkout")
        return 1 if bad else 0
    for rel in bad:
        (VENDOR / rel).parent.mkdir(parents=True, exist_ok=True)
        (VENDOR / rel).write_bytes(tree[rel])
        print("wrote", rel)
    lines = [f"{hashlib.sha256((VENDOR / n).read_bytes()).hexdigest()}  {n}" for n in sorted(tree)]
    (VENDOR / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")
    print(f"MANIFEST.sha256: {len(lines)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
