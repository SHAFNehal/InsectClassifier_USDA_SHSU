from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any


def slugify(text: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "-" for char in text.strip())
    compact = "-".join(part for part in normalized.split("-") if part)
    return compact or "item"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed)


def link_or_copy_file(source: Path, destination: Path, copy_files: bool) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    ensure_dir(destination.parent)
    if copy_files:
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source.resolve())


def validate_prepared_root(prepared_root: Path) -> Path:
    prepared_root = prepared_root.resolve()
    required = [
        prepared_root / "dataset.yaml",
        prepared_root / "metadata.json",
        prepared_root / "splits" / "train.json",
        prepared_root / "splits" / "val.json",
        prepared_root / "splits" / "test.json",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        expected = prepared_root / "dataset.yaml"
        raise FileNotFoundError(
            "Prepared dataset directory is invalid. "
            f"Expected files under {prepared_root}, including {expected}. "
            "If you already ran prepare, you probably want '--prepared-root artifacts/prepared'."
        )
    return prepared_root
