from __future__ import annotations

from pathlib import Path

from adetailer.common import get_models


def test_get_models_supports_subfolders(tmp_path: Path) -> None:
    (tmp_path / "root.pt").write_bytes(b"")
    (tmp_path / "bbox").mkdir()
    (tmp_path / "bbox" / "sub.pt").write_bytes(b"")

    models = get_models(tmp_path, huggingface=False)

    assert models["root.pt"] == str(tmp_path / "root.pt")
    # Must use POSIX-style separators for stable keys across OSes.
    assert models["bbox/sub.pt"] == str(tmp_path / "bbox" / "sub.pt")
    assert list(models).index("root.pt") < list(models).index("bbox/sub.pt")


def test_get_models_allows_same_filename_in_subfolder(tmp_path: Path) -> None:
    (tmp_path / "dup.pt").write_bytes(b"")
    (tmp_path / "segm").mkdir()
    (tmp_path / "segm" / "dup.pt").write_bytes(b"")

    models = get_models(tmp_path, huggingface=False)

    assert "dup.pt" in models
    assert "segm/dup.pt" in models
    assert models["dup.pt"] == str(tmp_path / "dup.pt")
    assert models["segm/dup.pt"] == str(tmp_path / "segm" / "dup.pt")


def test_get_models_extra_dirs_key_mode_prefix(tmp_path: Path) -> None:
    base_dir = tmp_path / "base"
    extra_dir = tmp_path / "extra_dir"
    base_dir.mkdir()
    extra_dir.mkdir()

    (base_dir / "dup.pt").write_bytes(b"")
    (extra_dir / "dup.pt").write_bytes(b"")

    models = get_models(base_dir, extra_dir, huggingface=False, extra_key_mode="prefix")

    assert models["dup.pt"] == str(base_dir / "dup.pt")
    assert models["extra/dup.pt"] == str(extra_dir / "dup.pt")


def test_get_models_extra_dirs_key_mode_legacy_skips_on_conflict(tmp_path: Path) -> None:
    base_dir = tmp_path / "base"
    extra_dir = tmp_path / "extra_dir"
    base_dir.mkdir()
    extra_dir.mkdir()

    (base_dir / "dup.pt").write_bytes(b"")
    (extra_dir / "dup.pt").write_bytes(b"")

    models = get_models(base_dir, extra_dir, huggingface=False, extra_key_mode="legacy")

    assert models["dup.pt"] == str(base_dir / "dup.pt")
    assert "extra/dup.pt" not in models
