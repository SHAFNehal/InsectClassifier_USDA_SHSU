from __future__ import annotations

import csv
from pathlib import Path

from .fasterrcnn import evaluate_fasterrcnn, train_fasterrcnn
from .prepare import prepare_dataset
from .ultralytics_pipeline import evaluate_ultralytics, train_ultralytics
from .utils import ensure_dir, read_json, validate_prepared_root, write_json
from .visualization import save_side_by_side_comparisons


def _write_evaluation_table(output_path: Path, rows: list[dict[str, object]]) -> None:
    ensure_dir(output_path.parent)
    fieldnames = ["model", "checkpoint", "map50", "split"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def run_all_models(
    dataset_root: Path,
    prepared_root: Path,
    output_dir: Path,
    runs_dir: Path,
    prepare_if_needed: bool,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    device: str,
    workers: int,
    score_threshold: float,
    split: str,
    limit: int | None,
    yolo_epochs: int,
    yolo_batch_size: int,
    yolo_imgsz: int,
    rtdetr_epochs: int,
    rtdetr_batch_size: int,
    rtdetr_imgsz: int,
    fasterrcnn_epochs: int,
    fasterrcnn_batch_size: int,
    fasterrcnn_learning_rate: float,
    fasterrcnn_weight_decay: float,
    fasterrcnn_pretrained: bool,
) -> dict[str, object]:
    if prepare_if_needed and not (prepared_root / "metadata.json").exists():
        prepare_dataset(
            dataset_root=dataset_root,
            output_dir=prepared_root,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
            copy_files=False,
        )
    prepared_root = validate_prepared_root(prepared_root)
    output_dir = ensure_dir(output_dir)

    checkpoints: dict[str, Path] = {}
    checkpoints["yolo"] = train_ultralytics(
        model_type="yolo",
        prepared_root=prepared_root,
        runs_dir=runs_dir,
        epochs=yolo_epochs,
        batch_size=yolo_batch_size,
        imgsz=yolo_imgsz,
        device=device,
        workers=workers,
        pretrained_weights=None,
    )
    checkpoints["rtdetr"] = train_ultralytics(
        model_type="rtdetr",
        prepared_root=prepared_root,
        runs_dir=runs_dir,
        epochs=rtdetr_epochs,
        batch_size=rtdetr_batch_size,
        imgsz=rtdetr_imgsz,
        device=device,
        workers=workers,
        pretrained_weights=None,
    )
    checkpoints["fasterrcnn"] = train_fasterrcnn(
        prepared_root=prepared_root,
        runs_dir=runs_dir,
        epochs=fasterrcnn_epochs,
        batch_size=fasterrcnn_batch_size,
        learning_rate=fasterrcnn_learning_rate,
        weight_decay=fasterrcnn_weight_decay,
        num_workers=workers,
        device=device,
        pretrained=fasterrcnn_pretrained,
    )

    evaluation_rows: list[dict[str, object]] = []
    prediction_files: dict[str, Path] = {}
    for model_name, checkpoint in checkpoints.items():
        evaluation_dir = output_dir / "evaluation" / model_name / split
        if model_name in {"yolo", "rtdetr"}:
            metrics = evaluate_ultralytics(
                model_type=model_name,
                prepared_root=prepared_root,
                checkpoint=checkpoint,
                split=split,
                output_dir=evaluation_dir,
                score_threshold=score_threshold,
            )
        else:
            metrics = evaluate_fasterrcnn(
                prepared_root=prepared_root,
                checkpoint=checkpoint,
                split=split,
                device=device,
                batch_size=fasterrcnn_batch_size,
                num_workers=workers,
                output_dir=evaluation_dir,
                score_threshold=score_threshold,
            )
        prediction_files[model_name] = evaluation_dir / "predictions.json"
        evaluation_rows.append(
            {
                "model": model_name,
                "checkpoint": str(checkpoint),
                "map50": float(metrics["map50"]),
                "split": split,
                "metrics": metrics,
            }
        )

    write_json(output_dir / "evaluation" / "summary.json", evaluation_rows)
    _write_evaluation_table(output_dir / "evaluation" / "summary.csv", evaluation_rows)

    metadata = read_json(prepared_root / "metadata.json")
    comparison_dir = save_side_by_side_comparisons(
        prepared_root=prepared_root,
        split=split,
        class_names=metadata["classes"],
        prediction_files=prediction_files,
        output_dir=output_dir / "comparisons" / split,
        limit=limit,
    )

    result = {
        "prepared_root": str(prepared_root),
        "checkpoints": {name: str(path) for name, path in checkpoints.items()},
        "evaluation_summary_json": str((output_dir / "evaluation" / "summary.json").resolve()),
        "evaluation_summary_csv": str((output_dir / "evaluation" / "summary.csv").resolve()),
        "comparison_dir": str(comparison_dir.resolve()),
    }
    write_json(output_dir / "run_all_summary.json", result)
    return result
