from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .fasterrcnn import build_model
from .ultralytics_pipeline import _load_ultralytics_model
from .utils import read_json


MODEL_LABELS = {
    "yolo": "YOLOv8n",
    "rtdetr": "RT-DETR",
    "fasterrcnn": "Faster R-CNN",
}


def discover_latest_checkpoints(runs_dir: Path) -> dict[str, Path]:
    patterns = {
        "yolo": ["**/yolo-train/weights/best.pt"],
        "rtdetr": ["**/rtdetr-train/weights/best.pt"],
        "fasterrcnn": ["**/fasterrcnn/best.pt"],
    }
    latest: dict[str, Path] = {}
    for model_type, glob_patterns in patterns.items():
        candidates: list[Path] = []
        for pattern in glob_patterns:
            candidates.extend(path for path in runs_dir.glob(pattern) if path.is_file())
        if candidates:
            latest[model_type] = max(candidates, key=lambda path: path.stat().st_mtime)
    return latest


def load_class_names(prepared_root: Path, checkpoint: Path, model_type: str) -> list[str]:
    metadata_path = prepared_root / "metadata.json"
    if metadata_path.exists():
        metadata = read_json(metadata_path)
        return list(metadata["classes"])

    if model_type == "fasterrcnn":
        import torch

        payload = torch.load(checkpoint, map_location="cpu")
        classes = payload.get("classes")
        if classes:
            return list(classes)

    model = _load_ultralytics_model(model_type, str(checkpoint))
    names = getattr(model, "names", None) or getattr(model.model, "names", None)
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    if isinstance(names, list):
        return [str(name) for name in names]
    raise FileNotFoundError(
        f"Could not resolve class names. Expected {metadata_path} or embedded class names in {checkpoint}."
    )


def load_inference_model(model_type: str, checkpoint: Path, class_names: list[str], device: str = "cpu") -> Any:
    if model_type in {"yolo", "rtdetr"}:
        return _load_ultralytics_model(model_type, str(checkpoint))

    import torch

    payload = torch.load(checkpoint, map_location=device)
    model = build_model(num_classes=len(class_names) + 1, pretrained=False)
    model.load_state_dict(payload["model_state_dict"])
    model = model.to(torch.device(device))
    model.eval()
    return model


def predict_uploaded_image(
    model_type: str,
    model: Any,
    image: Image.Image,
    class_names: list[str],
    score_threshold: float,
    device: str = "cpu",
) -> list[dict[str, object]]:
    if model_type in {"yolo", "rtdetr"}:
        result = model.predict(source=image, conf=score_threshold, device=device, verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().tolist() if result.boxes is not None else []
        scores = result.boxes.conf.cpu().tolist() if result.boxes is not None else []
        labels = result.boxes.cls.cpu().tolist() if result.boxes is not None else []
        return [
            {
                "box": [float(value) for value in box],
                "score": float(score),
                "label": class_names[int(label)],
            }
            for box, score, label in zip(boxes, scores, labels, strict=True)
        ]

    import torch
    from torchvision.transforms import functional as F

    with torch.no_grad():
        tensor = F.to_tensor(image).to(torch.device(device))
        output = model([tensor])[0]

    predictions: list[dict[str, object]] = []
    for box, score, label in zip(
        output["boxes"].detach().cpu().tolist(),
        output["scores"].detach().cpu().tolist(),
        output["labels"].detach().cpu().tolist(),
        strict=True,
    ):
        if score < score_threshold:
            continue
        predictions.append(
            {
                "box": [float(value) for value in box],
                "score": float(score),
                "label": class_names[label - 1],
            }
        )
    return predictions


def draw_uploaded_prediction(image: Image.Image, predictions: list[dict[str, object]]) -> Image.Image:
    rendered = image.copy().convert("RGB")
    draw = ImageDraw.Draw(rendered)
    for item in predictions:
        box = item["box"]
        label = str(item["label"])
        score = float(item["score"])
        draw.rectangle(box, outline="#ff4b4b", width=4)
        draw.text((box[0], max(0, box[1] - 16)), f"{label} {score:.2f}", fill="#ff4b4b")
    return rendered
