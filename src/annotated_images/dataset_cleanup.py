from __future__ import annotations

import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from .utils import ensure_dir


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _normalize_folder_name(name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", name.strip())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "Unknown"


def _next_ood_target(ood_dir: Path, folder_name: str, suffix: str) -> Path:
    base = f"Image_{_normalize_folder_name(folder_name)}_"
    existing_numbers: list[int] = []
    for path in ood_dir.iterdir():
        if not path.is_file():
            continue
        if not path.name.startswith(base):
            continue
        stem_suffix = path.name[len(base) :]
        match = re.fullmatch(r"(\d{3})\.[^.]+", stem_suffix)
        if match:
            existing_numbers.append(int(match.group(1)))
    next_number = max(existing_numbers, default=0) + 1
    return ood_dir / f"{base}{next_number:03d}{suffix.lower()}"


def clean_dataset(dataset_root: Path, ood_dir: Path) -> dict[str, object]:
    dataset_root = dataset_root.resolve()
    ood_dir = ensure_dir(ood_dir.resolve())

    moved_to_ood: list[dict[str, str]] = []
    deleted_xml: list[str] = []
    fixed_name_xml_files: set[str] = set()
    fixed_object_names = 0
    removed_1x1_boxes = 0
    per_reason = Counter()
    per_ood_folder = Counter()

    for folder in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        if folder.resolve() == ood_dir:
            continue

        files = list(folder.iterdir())
        images = sorted(path for path in files if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        xmls_by_stem = {path.stem: path for path in files if path.is_file() and path.suffix.lower() == ".xml"}

        for image_path in images:
            if image_path.stem in xmls_by_stem:
                continue
            target = _next_ood_target(ood_dir, folder.name, image_path.suffix)
            shutil.move(str(image_path), str(target))
            moved_to_ood.append({"source": str(image_path), "target": str(target), "reason": "missing_xml"})
            per_reason["missing_xml"] += 1
            per_ood_folder[_normalize_folder_name(folder.name)] += 1

        current_images = {
            path.stem: path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        current_xmls = {
            path.stem: path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() == ".xml"
        }

        for stem, xml_path in sorted(current_xmls.items()):
            image_path = current_images.get(stem)
            if image_path is None:
                xml_path.unlink(missing_ok=True)
                deleted_xml.append(str(xml_path))
                per_reason["orphan_xml_deleted"] += 1
                continue

            tree = ET.parse(xml_path)
            root = tree.getroot()
            xml_changed = False

            folder_el = root.find("folder")
            if folder_el is not None and folder_el.text != folder.name:
                folder_el.text = folder.name
                xml_changed = True

            filename_el = root.find("filename")
            if filename_el is not None and filename_el.text != image_path.name:
                filename_el.text = image_path.name
                xml_changed = True

            for obj in list(root.findall("./object")):
                name_el = obj.find("name")
                if name_el is not None and name_el.text != folder.name:
                    name_el.text = folder.name
                    xml_changed = True
                    fixed_object_names += 1
                    fixed_name_xml_files.add(str(xml_path))

                box = obj.find("bndbox")
                if box is None:
                    continue

                xmin = float(box.findtext("xmin", "0"))
                ymin = float(box.findtext("ymin", "0"))
                xmax = float(box.findtext("xmax", "0"))
                ymax = float(box.findtext("ymax", "0"))
                if (xmax - xmin) == 1 and (ymax - ymin) == 1:
                    root.remove(obj)
                    xml_changed = True
                    removed_1x1_boxes += 1

            if not root.findall("./object"):
                target = _next_ood_target(ood_dir, folder.name, image_path.suffix)
                shutil.move(str(image_path), str(target))
                moved_to_ood.append(
                    {
                        "source": str(image_path),
                        "target": str(target),
                        "reason": "empty_after_1x1_cleanup",
                    }
                )
                per_reason["empty_after_1x1_cleanup"] += 1
                per_ood_folder[_normalize_folder_name(folder.name)] += 1
                xml_path.unlink(missing_ok=True)
                deleted_xml.append(str(xml_path))
                continue

            if xml_changed:
                tree.write(xml_path, encoding="utf-8", xml_declaration=True)

    active_counts = {}
    for folder in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        if folder.resolve() == ood_dir:
            continue
        image_count = sum(1 for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        xml_count = sum(1 for path in folder.iterdir() if path.is_file() and path.suffix.lower() == ".xml")
        active_counts[folder.name] = {"images": image_count, "xml": xml_count}

    return {
        "dataset_root": str(dataset_root),
        "ood_dir": str(ood_dir),
        "moved_to_ood": len(moved_to_ood),
        "deleted_xml": len(deleted_xml),
        "fixed_name_xml_files": len(fixed_name_xml_files),
        "fixed_object_names": fixed_object_names,
        "removed_1x1_boxes": removed_1x1_boxes,
        "moved_to_ood_by_reason": dict(per_reason),
        "ood_counts": dict(per_ood_folder),
        "active_counts": active_counts,
    }
