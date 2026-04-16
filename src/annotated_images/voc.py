from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from .types import AnnotationObject, Sample


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


class AnnotationError(RuntimeError):
    pass


def _resolve_image_path(xml_path: Path, root: ET.Element) -> Path:
    filename = (root.findtext("./filename", default="") or "").strip()
    if filename:
        candidate = xml_path.with_name(filename)
        if candidate.exists():
            return candidate

    for suffix in IMAGE_EXTENSIONS:
        candidate = xml_path.with_suffix(suffix)
        if candidate.exists():
            return candidate

    raise AnnotationError(f"Missing image for annotation: {xml_path}")


def parse_voc_xml(xml_path: Path, class_name: str, split: str) -> Sample:
    try:
        root = ET.fromstring(xml_path.read_text(encoding="utf-8"))
    except ET.ParseError as exc:
        raise AnnotationError(f"Failed to parse XML: {xml_path}") from exc

    image_path = _resolve_image_path(xml_path, root)

    width = int(root.findtext("./size/width", default="0"))
    height = int(root.findtext("./size/height", default="0"))
    if width <= 0 or height <= 0:
        raise AnnotationError(f"Invalid image size in {xml_path}")

    objects: list[AnnotationObject] = []
    annotation_names: list[str] = []
    for element in root.findall("./object"):
        name = (element.findtext("name", default="") or "").strip()
        annotation_names.append(name)
        box = element.find("bndbox")
        if box is None:
            raise AnnotationError(f"Missing bndbox in {xml_path}")

        xmin = float(box.findtext("xmin", default="0"))
        ymin = float(box.findtext("ymin", default="0"))
        xmax = float(box.findtext("xmax", default="0"))
        ymax = float(box.findtext("ymax", default="0"))
        if xmax <= xmin or ymax <= ymin:
            raise AnnotationError(f"Degenerate bounding box in {xml_path}")
        if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
            raise AnnotationError(f"Out-of-bounds box in {xml_path}")
        objects.append(
            AnnotationObject(
                class_name=class_name,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
            )
        )

    if not objects:
        raise AnnotationError(f"No objects found in {xml_path}")

    sample_id = f"{class_name}/{xml_path.stem}"
    return Sample(
        sample_id=sample_id,
        split=split,
        class_name=class_name,
        image_path=image_path,
        xml_path=xml_path,
        width=width,
        height=height,
        objects=objects,
        annotation_names=annotation_names,
    )
