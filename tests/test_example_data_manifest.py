"""Integrity checks for the redistributable example-data asset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

DATA_ROOT = Path(__file__).parents[1] / "docs" / "examples" / "data"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_redistributable_example_data_matches_manifest() -> None:
    manifest = json.loads((DATA_ROOT / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["schemaVersion"] == 1
    assert manifest["files"]
    for entry in manifest["files"]:
        path = DATA_ROOT / entry["path"]
        assert path.is_file(), entry["path"]
        assert path.stat().st_size == entry["bytes"], entry["path"]
        assert _sha256(path) == entry["sha256"], entry["path"]
        assert entry["license"] == "Apache-2.0"
