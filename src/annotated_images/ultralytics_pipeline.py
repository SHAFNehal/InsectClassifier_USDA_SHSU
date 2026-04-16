from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image, ImageDraw

from .metrics import evaluate_predictions
from .utils import ensure_dir, read_json, validate_prepared_root, write_json
from .visualization import draw_comparison_image


def _load_ultralytics_model(model_type: str, weights: str):
    if model_type == "yolo":
        from ultralytics import YOLO

        return YOLO(weights)
    if model_type == "rtdetr":
        from ultralytics import RTDETR

        return RTDETR(weights)
    raise ValueError(f"Unsupported Ultralytics model type: {model_type}")


def default_weights_for(model_type: str) -> str:
    if model_type == "yolo":
        return "yolov8n.pt"
    if model_type == "rtdetr":
        return "rtdetr-l.pt"
    raise ValueError(f"Unsupported model type: {model_type}")


def _ground_truth(prepared_root: Path, split: str, class_to_id: dict[str, int]) -> dict[str, list[dict[str, object]]]:
    samples = read_json(prepared_root / "splits" / f"{split}.json")
    return {
        sample["exported_image"]: [
            {
                "box": [obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"]],
                "label": class_to_id[sample["class_name"]],
            }
            for obj in sample["objects"]
        ]
        for sample in samples
    }


def _ultralytics_train_kwargs(
    model_type: str,
    prepared_root: Path,
    runs_dir: Path,
    run_name: str,
    epochs: int,
    batch_size: int,
    imgsz: int,
    device: str,
    workers: int,
    attempt: int,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "data": str(prepared_root / "dataset.yaml"),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch_size,
        "device": device,
        "workers": workers,
        "project": str(runs_dir),
        "name": run_name,
        "exist_ok": True,
    }
    if model_type == "rtdetr":
        kwargs.update(
            {
                "deterministic": False,
                "amp": False,
                "optimizer": "AdamW",
                "lr0": 5e-4 if attempt == 1 else 1e-4,
            }
        )
    return kwargs


def _is_nan_training_failure(exc: RuntimeError) -> bool:
    message = str(exc)
    return "NaN/Inf" in message or "Loss NaN/Inf detected" in message


def train_ultralytics(
    model_type: str,
    prepared_root: Path,
    runs_dir: Path,
    epochs: int,
    batch_size: int,
    imgsz: int,
    device: str,
    workers: int,
    pretrained_weights: str | None,
) -> Path:
    prepared_root = validate_prepared_root(prepared_root)
    runs_dir = runs_dir.resolve()
    run_name = f"{model_type}-train"
    run_dir = runs_dir / run_name
    attempts = [(1, batch_size)]
    if model_type == "rtdetr":
        attempts.append((2, max(1, batch_size // 2)))

    last_error: RuntimeError | None = None
    for attempt, attempt_batch_size in attempts:
        model = _load_ultralytics_model(model_type, pretrained_weights or default_weights_for(model_type))
        try:
            model.train(
                **_ultralytics_train_kwargs(
                    model_type=model_type,
                    prepared_root=prepared_root,
                    runs_dir=runs_dir,
                    run_name=run_name,
                    epochs=epochs,
                    batch_size=attempt_batch_size,
                    imgsz=imgsz,
                    device=device,
                    workers=workers,
                    attempt=attempt,
                )
            )
            save_dir = Path(model.trainer.save_dir)
            return save_dir / "weights" / "best.pt"
        except RuntimeError as exc:
            last_error = exc
            if model_type != "rtdetr" or not _is_nan_training_failure(exc) or attempt == len(attempts):
                raise
            if run_dir.exists():
                shutil.rmtree(run_dir)

    assert last_error is not None
    raise last_error


def evaluate_ultralytics(
    model_type: str,
    prepared_root: Path,
    checkpoint: Path,
    split: str,
    output_dir: Path,
    score_threshold: float,
) -> dict[str, object]:
    prepared_root = validate_prepared_root(prepared_root)
    metadata = read_json(prepared_root / "metadata.json")
    class_names = metadata["classes"]
    class_to_id = metadata["class_to_id"]
    samples = read_json(prepared_root / "splits" / f"{split}.json")
    model = _load_ultralytics_model(model_type, str(checkpoint))

    predictions: dict[str, list[dict[str, object]]] = {}
    for sample in samples:
        image_path = prepared_root / sample["exported_image"]
        result = model.predict(source=str(image_path), conf=score_threshold, verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().tolist() if result.boxes is not None else []
        scores = result.boxes.conf.cpu().tolist() if result.boxes is not None else []
        labels = result.boxes.cls.cpu().tolist() if result.boxes is not None else []
        predictions[sample["exported_image"]] = [
            {"box": box, "score": score, "label": int(label)}
            for box, score, label in zip(boxes, scores, labels, strict=True)
        ]

    metrics = evaluate_predictions(
        ground_truth_by_image=_ground_truth(prepared_root, split, class_to_id),
        predictions_by_image=predictions,
        class_names=class_names,
    )
    ensure_dir(output_dir)
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "predictions.json", predictions)
    return metrics


def predict_ultralytics(
    model_type: str,
    prepared_root: Path,
    checkpoint: Path,
    source: Path,
    output_dir: Path,
    score_threshold: float,
) -> Path:
    prepared_root = validate_prepared_root(prepared_root)
    metadata = read_json(prepared_root / "metadata.json")
    class_names = metadata["classes"]
    model = _load_ultralytics_model(model_type, str(checkpoint))
    ensure_dir(output_dir)
    ensure_dir(output_dir / "renders")

    predictions: dict[str, list[dict[str, object]]] = {}
    for image_path in sorted(path for path in source.iterdir() if path.suffix.lower() == ".png"):
        result = model.predict(source=str(image_path), conf=score_threshold, verbose=False)[0]
        render = Image.open(image_path).convert("RGB")
        drawer = ImageDraw.Draw(render)
        boxes = result.boxes.xyxy.cpu().tolist() if result.boxes is not None else []
        scores = result.boxes.conf.cpu().tolist() if result.boxes is not None else []
        labels = result.boxes.cls.cpu().tolist() if result.boxes is not None else []
        payload_items = []
        for box, score, label in zip(boxes, scores, labels, strict=True):
            label_name = class_names[int(label)]
            payload_items.append({"box": box, "score": score, "label": label_name})
            drawer.rectangle(box, outline="red", width=3)
            drawer.text((box[0], max(0, box[1] - 14)), f"{label_name} {score:.2f}", fill="red")
        predictions[image_path.name] = payload_items
        render.save(output_dir / "renders" / image_path.name)

    write_json(output_dir / "predictions.json", predictions)
    return output_dir / "predictions.json"


def visualize_ultralytics_split(
    model_type: str,
    prepared_root: Path,
    checkpoint: Path,
    split: str,
    output_dir: Path,
    score_threshold: float,
    limit: int | None,
) -> Path:
    prepared_root = validate_prepared_root(prepared_root)
    metadata = read_json(prepared_root / "metadata.json")
    class_names = metadata["classes"]
    class_to_id = metadata["class_to_id"]
    samples = read_json(prepared_root / "splits" / f"{split}.json")
    if limit is not None:
        samples = samples[:limit]
    model = _load_ultralytics_model(model_type, str(checkpoint))

    ensure_dir(output_dir)
    manifest: list[dict[str, object]] = []
    for sample in samples:
        image_path = prepared_root / sample["exported_image"]
        result = model.predict(source=str(image_path), conf=score_threshold, verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().tolist() if result.boxes is not None else []
        scores = result.boxes.conf.cpu().tolist() if result.boxes is not None else []
        labels = result.boxes.cls.cpu().tolist() if result.boxes is not None else []
        predictions = [
            {"box": box, "score": float(score), "label": int(label)}
            for box, score, label in zip(boxes, scores, labels, strict=True)
        ]
        ground_truth = [
            {
                "box": [obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"]],
                "label": class_to_id[sample["class_name"]],
            }
            for obj in sample["objects"]
        ]
        rendered_name = f"{Path(sample['exported_image']).stem}.png"
        rendered_path = output_dir / rendered_name
        draw_comparison_image(
            image_path=image_path,
            class_names=class_names,
            ground_truth=ground_truth,
            predictions=predictions,
            output_path=rendered_path,
        )
        manifest.append(
            {
                "image": sample["exported_image"],
                "rendered": str(rendered_path.relative_to(output_dir)),
                "ground_truth_count": len(ground_truth),
                "prediction_count": len(predictions),
            }
        )

    write_json(output_dir / "manifest.json", manifest)
    return output_dir
