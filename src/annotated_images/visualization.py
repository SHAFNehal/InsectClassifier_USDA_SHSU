from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from .utils import ensure_dir, read_json, write_json


def _label_name(label: int | str, class_names: list[str]) -> str:
    if isinstance(label, int):
        return class_names[label]
    return label


def draw_comparison_image(
    image_path: Path,
    class_names: list[str],
    ground_truth: list[dict[str, object]],
    predictions: list[dict[str, object]],
    output_path: Path,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for item in ground_truth:
        box = item["box"]
        label = _label_name(item["label"], class_names)
        draw.rectangle(box, outline="lime", width=3)
        draw.text((box[0], max(0, box[1] - 14)), f"GT: {label}", fill="lime")

    for item in predictions:
        box = item["box"]
        label = _label_name(item["label"], class_names)
        score = item.get("score")
        suffix = f" {score:.2f}" if isinstance(score, float) else ""
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], min(image.height - 14, box[3] + 2)), f"Pred: {label}{suffix}", fill="red")

    ensure_dir(output_path.parent)
    image.save(output_path)


def render_panel(
    image_path: Path,
    class_names: list[str],
    title: str,
    ground_truth: list[dict[str, object]] | None,
    predictions: list[dict[str, object]] | None,
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    canvas = Image.new("RGB", (image.width, image.height + 32), "white")
    canvas.paste(image, (0, 32))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill="black")

    for item in ground_truth or []:
        box = item["box"]
        shifted = [box[0], box[1] + 32, box[2], box[3] + 32]
        label = _label_name(item["label"], class_names)
        draw.rectangle(shifted, outline="lime", width=3)
        draw.text((shifted[0], max(32, shifted[1] - 14)), f"GT: {label}", fill="lime")

    for item in predictions or []:
        box = item["box"]
        shifted = [box[0], box[1] + 32, box[2], box[3] + 32]
        label = _label_name(item["label"], class_names)
        score = item.get("score")
        suffix = f" {score:.2f}" if isinstance(score, float) else ""
        draw.rectangle(shifted, outline="red", width=3)
        draw.text((shifted[0], min(canvas.height - 14, shifted[3] + 2)), f"Pred: {label}{suffix}", fill="red")

    return canvas


def save_side_by_side_comparisons(
    prepared_root: Path,
    split: str,
    class_names: list[str],
    prediction_files: dict[str, Path],
    output_dir: Path,
    limit: int | None,
) -> Path:
    samples = read_json(prepared_root / "splits" / f"{split}.json")
    if limit is not None:
        samples = samples[:limit]
    prediction_payloads = {name: read_json(path) for name, path in prediction_files.items()}
    manifest: list[dict[str, object]] = []

    for sample in samples:
        image_path = prepared_root / sample["exported_image"]
        ground_truth = [
            {"box": [obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"]], "label": sample["class_name"]}
            for obj in sample["objects"]
        ]
        panels = [render_panel(image_path, class_names, "Ground Truth", ground_truth, [])]
        for model_name, payload in prediction_payloads.items():
            panels.append(
                render_panel(
                    image_path=image_path,
                    class_names=class_names,
                    title=model_name,
                    ground_truth=[],
                    predictions=payload.get(sample["exported_image"], []),
                )
            )

        width = sum(panel.width for panel in panels)
        height = max(panel.height for panel in panels)
        canvas = Image.new("RGB", (width, height), "white")
        x_offset = 0
        for panel in panels:
            canvas.paste(panel, (x_offset, 0))
            x_offset += panel.width

        rendered_name = f"{Path(sample['exported_image']).stem}.png"
        rendered_path = output_dir / rendered_name
        ensure_dir(rendered_path.parent)
        canvas.save(rendered_path)
        manifest.append(
            {
                "image": sample["exported_image"],
                "rendered": str(rendered_path.relative_to(output_dir)),
                "models": ["Ground Truth", *prediction_payloads.keys()],
            }
        )

    write_json(output_dir / "manifest.json", manifest)
    return output_dir
