"""Download Xiaomi-Lens-0 into ComfyUI's models/xiaomi_lens, and verify it.

    python tools/download_model.py --models-dir <ComfyUI>/models/xiaomi_lens [--no-weights] [--verify-only]

Source (pinned revision; Apache-2.0, see the model card):
  https://huggingface.co/SeerRay-Lab/Xiaomi-Lens-0   9.2 GB (three bfloat16 safetensors shards + tokenizer and processor files)
Result:
  models/xiaomi_lens/Xiaomi-Lens-0/config.json, model-0000*-of-00003.safetensors, tokenizer.json, ... (the repository files)
Every file is checked against its SHA-256 in tools/model_files.sha256. --no-weights fetches only the configuration, tokenizer and
processor files (20 MB; what the CPU tests build their tiny model from). Needs `huggingface_hub` (installed with ComfyUI's
requirements). Nothing is downloaded unless you run this script.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO, REVISION = "SeerRay-Lab/Xiaomi-Lens-0", "9452efab96fa8b886e7a0f89e390219ba59deddb"
NAME = "Xiaomi-Lens-0"
MANIFEST = Path(__file__).resolve().with_name("model_files.sha256")


def expected_files() -> dict[str, str]:
    """{file name: sha256} from tools/model_files.sha256."""
    out = {}
    for line in MANIFEST.read_text(encoding="ascii").splitlines():
        if line and not line.startswith("#"):
            digest, _, name = line.partition("  ")
            out[name.split("/", 1)[1]] = digest
    return out


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", type=Path, required=True, help="ComfyUI's models/xiaomi_lens folder")
    parser.add_argument("--no-weights", action="store_true", help="skip the safetensors files (configuration, tokenizer, processor only)")
    parser.add_argument("--verify-only", action="store_true", help="check existing files, download nothing")
    args = parser.parse_args(argv)

    folder = args.models_dir / NAME
    ok = True
    for name, digest in expected_files().items():
        if args.no_weights and name.endswith(".safetensors"):
            continue
        target = folder / name
        if not args.verify_only and not (target.is_file() and sha256(target) == digest):
            from huggingface_hub import hf_hub_download

            hf_hub_download(repo_id=REPO, filename=name, revision=REVISION, local_dir=str(folder))
        good = target.is_file() and sha256(target) == digest
        ok &= good
        print(f"{'ok      ' if good else ('MISMATCH' if target.is_file() else 'MISSING ')} {NAME}/{name}", flush=True)
    print("all files verified" if ok else "verification FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
