from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from tqdm import tqdm

from .metrics import evaluate_predictions
from .utils import ensure_dir, read_json, validate_prepared_root, write_json
from .visualization import draw_comparison_image


def collate_fn(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]]):
    images, targets, metas = zip(*batch, strict=True)
    return list(images), list(targets), list(metas)


class DetectionDataset(Dataset):
    def __init__(self, prepared_root: Path, split: str, class_to_id: dict[str, int]) -> None:
        self.prepared_root = prepared_root
        self.class_to_id = class_to_id
        self.samples = read_json(prepared_root / "splits" / f"{split}.json")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image_path = self.prepared_root / sample["exported_image"]
        image = Image.open(image_path).convert("RGB")
        boxes = []
        labels = []
        for obj in sample["objects"]:
            boxes.append([obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"]])
            labels.append(self.class_to_id[sample["class_name"]] + 1)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
        }
        return F.to_tensor(image), target, sample


def build_model(num_classes: int, pretrained: bool) -> torch.nn.Module:
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def _ground_truth_format(samples: list[dict[str, Any]], class_to_id: dict[str, int]) -> dict[str, list[dict[str, object]]]:
    payload: dict[str, list[dict[str, object]]] = {}
    for sample in samples:
        payload[sample["exported_image"]] = [
            {
                "box": [obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"]],
                "label": class_to_id[sample["class_name"]],
            }
            for obj in sample["objects"]
        ]
    return payload


def _prediction_format(
    outputs: list[dict[str, torch.Tensor]],
    metas: list[dict[str, Any]],
    score_threshold: float = 0.0,
) -> dict[str, list[dict[str, object]]]:
    predictions: dict[str, list[dict[str, object]]] = {}
    for output, meta in zip(outputs, metas, strict=True):
        boxes = output["boxes"].detach().cpu().tolist()
        scores = output["scores"].detach().cpu().tolist()
        labels = output["labels"].detach().cpu().tolist()
        predictions[meta["exported_image"]] = [
            {"box": box, "score": score, "label": label - 1}
            for box, score, label in zip(boxes, scores, labels, strict=True)
            if score >= score_threshold
        ]
    return predictions


def train_fasterrcnn(
    prepared_root: Path,
    runs_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    num_workers: int,
    device: str,
    pretrained: bool,
) -> Path:
    prepared_root = validate_prepared_root(prepared_root)
    metadata = read_json(prepared_root / "metadata.json")
    class_names = metadata["classes"]
    class_to_id = metadata["class_to_id"]
    train_dataset = DetectionDataset(prepared_root, "train", class_to_id)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    target_device = torch.device(device)

    model = build_model(num_classes=len(class_names) + 1, pretrained=pretrained).to(target_device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)

    run_dir = ensure_dir(runs_dir / "fasterrcnn")
    best_checkpoint = run_dir / "best.pt"
    best_map50 = -1.0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, targets, _metas in tqdm(train_loader, desc=f"fasterrcnn train epoch {epoch}", leave=False):
            images = [image.to(target_device) for image in images]
            targets = [{key: value.to(target_device) for key, value in target.items()} for target in targets]
            losses = model(images, targets)
            loss = sum(losses.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu().item())

        metrics = evaluate_fasterrcnn(
            prepared_root=prepared_root,
            checkpoint=None,
            split="val",
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            output_dir=run_dir / "val_predictions",
            model=model,
        )
        average_loss = running_loss / max(1, len(train_loader))
        history.append({"epoch": float(epoch), "loss": average_loss, "val_map50": float(metrics["map50"])})
        if metrics["map50"] > best_map50:
            best_map50 = float(metrics["map50"])
            torch.save({"model_state_dict": model.state_dict(), "classes": class_names}, best_checkpoint)

    write_json(run_dir / "history.json", history)
    return best_checkpoint


def evaluate_fasterrcnn(
    prepared_root: Path,
    checkpoint: Path | None,
    split: str,
    device: str,
    batch_size: int,
    num_workers: int,
    output_dir: Path,
    score_threshold: float = 0.25,
    model: torch.nn.Module | None = None,
) -> dict[str, object]:
    prepared_root = validate_prepared_root(prepared_root)
    metadata = read_json(prepared_root / "metadata.json")
    class_names = metadata["classes"]
    class_to_id = metadata["class_to_id"]
    dataset = DetectionDataset(prepared_root, split, class_to_id)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    target_device = torch.device(device)

    if model is None:
        if checkpoint is None:
            raise ValueError("checkpoint is required when model is not supplied")
        payload = torch.load(checkpoint, map_location=target_device)
        model = build_model(num_classes=len(class_names) + 1, pretrained=False)
        model.load_state_dict(payload["model_state_dict"])
    model = model.to(target_device)
    model.eval()

    predictions: dict[str, list[dict[str, object]]] = {}
    with torch.no_grad():
        for images, _targets, metas in tqdm(loader, desc=f"fasterrcnn eval {split}", leave=False):
            images = [image.to(target_device) for image in images]
            outputs = model(images)
            predictions.update(_prediction_format(outputs, metas, score_threshold=score_threshold))

    metrics = evaluate_predictions(
        ground_truth_by_image=_ground_truth_format(dataset.samples, class_to_id),
        predictions_by_image=predictions,
        class_names=class_names,
    )
    ensure_dir(output_dir)
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "predictions.json", predictions)
    return metrics


def predict_fasterrcnn(
    prepared_root: Path,
    checkpoint: Path,
    source: Path,
    output_dir: Path,
    device: str,
    score_threshold: float,
) -> Path:
    prepared_root = validate_prepared_root(prepared_root)
    metadata = read_json(prepared_root / "metadata.json")
    class_names = metadata["classes"]
    target_device = torch.device(device)

    payload = torch.load(checkpoint, map_location=target_device)
    model = build_model(num_classes=len(class_names) + 1, pretrained=False)
    model.load_state_dict(payload["model_state_dict"])
    model = model.to(target_device)
    model.eval()

    ensure_dir(output_dir)
    ensure_dir(output_dir / "renders")
    predictions: dict[str, list[dict[str, object]]] = {}
    with torch.no_grad():
        for image_path in tqdm(sorted(path for path in source.iterdir() if path.suffix.lower() == ".png"), desc="fasterrcnn predict", leave=False):
            image = Image.open(image_path).convert("RGB")
            tensor = F.to_tensor(image).to(target_device)
            output = model([tensor])[0]
            render = image.copy()
            drawer = ImageDraw.Draw(render)
            payload_items = []
            for box, score, label in zip(
                output["boxes"].detach().cpu().tolist(),
                output["scores"].detach().cpu().tolist(),
                output["labels"].detach().cpu().tolist(),
                strict=True,
            ):
                if score < score_threshold:
                    continue
                label_name = class_names[label - 1]
                payload_items.append({"box": box, "score": score, "label": label_name})
                drawer.rectangle(box, outline="red", width=3)
                drawer.text((box[0], max(0, box[1] - 14)), f"{label_name} {score:.2f}", fill="red")
            predictions[image_path.name] = payload_items
            render.save(output_dir / "renders" / image_path.name)

    write_json(output_dir / "predictions.json", predictions)
    return output_dir / "predictions.json"


def visualize_fasterrcnn_split(
    prepared_root: Path,
    checkpoint: Path,
    split: str,
    output_dir: Path,
    device: str,
    score_threshold: float,
    limit: int | None,
) -> Path:
    prepared_root = validate_prepared_root(prepared_root)
    metadata = read_json(prepared_root / "metadata.json")
    class_names = metadata["classes"]
    class_to_id = metadata["class_to_id"]
    target_device = torch.device(device)

    payload = torch.load(checkpoint, map_location=target_device)
    model = build_model(num_classes=len(class_names) + 1, pretrained=False)
    model.load_state_dict(payload["model_state_dict"])
    model = model.to(target_device)
    model.eval()

    samples = read_json(prepared_root / "splits" / f"{split}.json")
    if limit is not None:
        samples = samples[:limit]
    ensure_dir(output_dir)
    manifest: list[dict[str, object]] = []

    with torch.no_grad():
        for sample in tqdm(samples, desc=f"fasterrcnn visualize {split}", leave=False):
            image_path = prepared_root / sample["exported_image"]
            image = Image.open(image_path).convert("RGB")
            tensor = F.to_tensor(image).to(target_device)
            output = model([tensor])[0]
            predictions = []
            for box, score, label in zip(
                output["boxes"].detach().cpu().tolist(),
                output["scores"].detach().cpu().tolist(),
                output["labels"].detach().cpu().tolist(),
                strict=True,
            ):
                if score < score_threshold:
                    continue
                predictions.append({"box": box, "score": float(score), "label": label - 1})
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
