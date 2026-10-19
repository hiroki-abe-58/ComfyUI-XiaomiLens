"""tools/download_model.py: the manifest covers the pinned repository files, and verification catches missing or changed files."""

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import download_model  # noqa: E402


def test_manifest_has_the_repository_files():
    files = download_model.expected_files()
    assert {"config.json", "tokenizer.json", "processor_config.json", "chat_template.jinja", "model.safetensors.index.json",
            "LICENSE", "README.md"} <= set(files)
    assert sorted(f for f in files if f.endswith(".safetensors")) == [f"model-0000{i}-of-00003.safetensors" for i in (1, 2, 3)]
    assert all(len(d) == 64 for d in files.values())


def test_verify_only_reports_missing_and_changed_files(tmp_path, capsys):
    folder = tmp_path / "Xiaomi-Lens-0"
    folder.mkdir()
    (folder / "config.json").write_text("{}", encoding="utf-8")
    assert download_model.main(["--models-dir", str(tmp_path), "--no-weights", "--verify-only"]) == 1
    out = capsys.readouterr().out
    assert "MISMATCH Xiaomi-Lens-0/config.json" in out and "MISSING  Xiaomi-Lens-0/tokenizer.json" in out
    assert "-of-00003.safetensors" not in out  # --no-weights skips the shards


def test_verify_only_accepts_matching_files(tmp_path, monkeypatch):
    folder = tmp_path / "Xiaomi-Lens-0"
    folder.mkdir()
    (folder / "a.json").write_bytes(b"x")
    monkeypatch.setattr(download_model, "expected_files", lambda: {"a.json": hashlib.sha256(b"x").hexdigest()})
    assert download_model.main(["--models-dir", str(tmp_path), "--verify-only"]) == 0
