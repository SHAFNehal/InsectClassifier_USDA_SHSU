from __future__ import annotations

import random
import shutil
from pathlib import Path

import yaml

from .types import Sample
from .utils import ensure_dir, link_or_copy_file, set_seed, slugify, write_json
from .voc import parse_voc_xml


def _split_counts(total: int, val_fraction: float, test_fraction: float) -> tuple[int, int, int]:
    test_count = int(round(total * test_fraction))
    val_count = int(round(total * val_fraction))
    train_count = total - val_count - test_count
    if train_count <= 0:
        raise ValueError("Split fractions leave no training data.")
    return train_count, val_count, test_count


def _make_sample_name(sample: Sample) -> str:
    return f"{slugify(sample.class_name)}--{slugify(sample.image_path.stem)}{sample.image_path.suffix.lower()}"


def _yolo_labels(sample: Sample, class_to_id: dict[str, int]) -> str:
    class_id = class_to_id[sample.class_name]
    lines: list[str] = []
    for obj in sample.objects:
        x_center = ((obj.xmin + obj.xmax) / 2.0) / sample.width
        y_center = ((obj.ymin + obj.ymax) / 2.0) / sample.height
        width = (obj.xmax - obj.xmin) / sample.width
        height = (obj.ymax - obj.ymin) / sample.height
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return "\n".join(lines) + "\n"


def _coco_annotation(sample: Sample, image_id: int, start_annotation_id: int, class_to_id: dict[str, int]) -> tuple[list[dict[str, object]], int]:
    annotations: list[dict[str, object]] = []
    annotation_id = start_annotation_id
    category_id = class_to_id[sample.class_name] + 1
    for obj in sample.objects:
        width = obj.xmax - obj.xmin
        height = obj.ymax - obj.ymin
        annotations.append(
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [obj.xmin, obj.ymin, width, height],
                "area": width * height,
                "iscrowd": 0,
            }
        )
        annotation_id += 1
    return annotations, annotation_id


def prepare_dataset(
    dataset_root: Path,
    output_dir: Path,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    copy_files: bool,
) -> dict[str, object]:
    if val_fraction < 0 or test_fraction < 0 or (val_fraction + test_fraction) >= 1:
        raise ValueError("val_fraction + test_fraction must be in [0, 1).")

    set_seed(seed)
    classes = sorted(path.name for path in dataset_root.iterdir() if path.is_dir())
    if not classes:
        raise ValueError(f"No class directories found in {dataset_root}")
    class_to_id = {name: index for index, name in enumerate(classes)}

    split_samples: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
    warnings: list[str] = []

    for class_name in classes:
        xml_files = sorted((dataset_root / class_name).glob("*.xml"))
        total = len(xml_files)
        if total == 0:
            continue
        train_count, val_count, test_count = _split_counts(total, val_fraction, test_fraction)
        split_schedule = ["train"] * train_count + ["val"] * val_count + ["test"] * test_count
        if len(split_schedule) < total:
            split_schedule.extend(["train"] * (total - len(split_schedule)))
        shuffled = xml_files[:]
        random.shuffle(shuffled)
        for xml_path, split in zip(shuffled, split_schedule, strict=True):
            sample = parse_voc_xml(xml_path=xml_path, class_name=class_name, split=split)
            invalid_names = sorted({name for name in sample.annotation_names if name and name != class_name})
            if invalid_names:
                warnings.append(
                    f"{xml_path}: annotation names {invalid_names} differ from folder label {class_name!r}; folder label used."
                )
            split_samples[split].append(sample)

    for split in split_samples:
        split_samples[split].sort(key=lambda item: item.sample_id)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)
    for split in ("train", "val", "test"):
        ensure_dir(output_dir / "images" / split)
        ensure_dir(output_dir / "labels" / split)
    ensure_dir(output_dir / "splits")
    ensure_dir(output_dir / "coco")

    for split, samples in split_samples.items():
        manifest: list[dict[str, object]] = []
        coco_images: list[dict[str, object]] = []
        coco_annotations: list[dict[str, object]] = []
        next_annotation_id = 1
        for image_id, sample in enumerate(samples, start=1):
            exported_name = _make_sample_name(sample)
            exported_image = output_dir / "images" / split / exported_name
            exported_label = output_dir / "labels" / split / f"{Path(exported_name).stem}.txt"
            link_or_copy_file(sample.image_path, exported_image, copy_files=copy_files)
            exported_label.write_text(_yolo_labels(sample, class_to_id), encoding="utf-8")

            manifest.append(
                {
                    **sample.to_dict(dataset_root.parent),
                    "exported_image": str(exported_image.relative_to(output_dir)),
                    "exported_label": str(exported_label.relative_to(output_dir)),
                }
            )
            coco_images.append(
                {
                    "id": image_id,
                    "file_name": str(exported_image.relative_to(output_dir)),
                    "width": sample.width,
                    "height": sample.height,
                }
            )
            annotations, next_annotation_id = _coco_annotation(sample, image_id, next_annotation_id, class_to_id)
            coco_annotations.extend(annotations)

        write_json(output_dir / "splits" / f"{split}.json", manifest)
        write_json(
            output_dir / "coco" / f"{split}.json",
            {
                "images": coco_images,
                "annotations": coco_annotations,
                "categories": [{"id": index + 1, "name": name} for name, index in class_to_id.items()],
            },
        )

    dataset_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for name, index in class_to_id.items()},
    }
    (output_dir / "dataset.yaml").write_text(yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8")

    summary = {
        "dataset_root": str(dataset_root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "classes": classes,
        "class_to_id": class_to_id,
        "counts": {split: len(samples) for split, samples in split_samples.items()},
        "warnings": warnings,
    }
    write_json(output_dir / "metadata.json", summary)
    return summary
