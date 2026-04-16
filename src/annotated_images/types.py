from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class AnnotationObject:
    class_name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def to_xyxy(self) -> list[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]


@dataclass(slots=True)
class Sample:
    sample_id: str
    split: str
    class_name: str
    image_path: Path
    xml_path: Path
    width: int
    height: int
    objects: list[AnnotationObject]
    annotation_names: list[str]

    def to_dict(self, root: Path) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "class_name": self.class_name,
            "image_path": str(self.image_path.relative_to(root)),
            "xml_path": str(self.xml_path.relative_to(root)),
            "width": self.width,
            "height": self.height,
            "objects": [asdict(obj) for obj in self.objects],
            "annotation_names": self.annotation_names,
        }
