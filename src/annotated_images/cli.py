from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="annotated-images")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Validate VOC annotations and export train/val/test artifacts")
    prepare_parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    prepare_parser.add_argument("--output-dir", type=Path, default=Path("artifacts/prepared"))
    prepare_parser.add_argument("--val-fraction", type=float, default=0.2)
    prepare_parser.add_argument("--test-fraction", type=float, default=0.1)
    prepare_parser.add_argument("--seed", type=int, default=42)
    prepare_parser.add_argument("--copy-files", action="store_true")

    cleanup_parser = subparsers.add_parser(
        "clean-dataset",
        help="Normalize XML labels, remove 1x1 boxes, and move unannotated images to the OOD folder",
    )
    cleanup_parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    cleanup_parser.add_argument("--ood-dir", type=Path, default=Path("OOD_Test_Files"))

    train_parser = subparsers.add_parser("train", help="Train a detector")
    train_parser.add_argument("--model-type", choices=["yolo", "rtdetr", "fasterrcnn"], required=True)
    train_parser.add_argument("--prepared-root", type=Path, default=Path("artifacts/prepared"))
    train_parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--imgsz", type=int, default=960)
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument("--workers", type=int, default=4)
    train_parser.add_argument("--learning-rate", type=float, default=0.005)
    train_parser.add_argument("--weight-decay", type=float, default=0.0005)
    train_parser.add_argument("--pretrained", action="store_true")
    train_parser.add_argument("--pretrained-weights", default=None)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained detector")
    eval_parser.add_argument("--model-type", choices=["yolo", "rtdetr", "fasterrcnn"], required=True)
    eval_parser.add_argument("--prepared-root", type=Path, default=Path("artifacts/prepared"))
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    eval_parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation"))
    eval_parser.add_argument("--device", default="cpu")
    eval_parser.add_argument("--batch-size", type=int, default=4)
    eval_parser.add_argument("--workers", type=int, default=4)
    eval_parser.add_argument("--score-threshold", type=float, default=0.25)

    predict_parser = subparsers.add_parser("predict", help="Run inference on a folder of images")
    predict_parser.add_argument("--model-type", choices=["yolo", "rtdetr", "fasterrcnn"], required=True)
    predict_parser.add_argument("--prepared-root", type=Path, default=Path("artifacts/prepared"))
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument("--source", type=Path, required=True)
    predict_parser.add_argument("--output-dir", type=Path, required=True)
    predict_parser.add_argument("--device", default="cpu")
    predict_parser.add_argument("--score-threshold", type=float, default=0.25)

    visualize_parser = subparsers.add_parser("visualize-split", help="Render images with ground-truth and predicted boxes")
    visualize_parser.add_argument("--model-type", choices=["yolo", "rtdetr", "fasterrcnn"], required=True)
    visualize_parser.add_argument("--prepared-root", type=Path, default=Path("artifacts/prepared"))
    visualize_parser.add_argument("--checkpoint", type=Path, required=True)
    visualize_parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    visualize_parser.add_argument("--output-dir", type=Path, default=Path("artifacts/visualizations"))
    visualize_parser.add_argument("--device", default="cpu")
    visualize_parser.add_argument("--score-threshold", type=float, default=0.25)
    visualize_parser.add_argument("--limit", type=int, default=None)

    run_all_parser = subparsers.add_parser("run-all", help="Prepare, train, evaluate, and compare all detector families")
    run_all_parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    run_all_parser.add_argument("--prepared-root", type=Path, default=Path("artifacts/prepared"))
    run_all_parser.add_argument("--output-dir", type=Path, default=Path("artifacts/run-all"))
    run_all_parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    run_all_parser.add_argument("--prepare-if-needed", action="store_true")
    run_all_parser.add_argument("--val-fraction", type=float, default=0.2)
    run_all_parser.add_argument("--test-fraction", type=float, default=0.1)
    run_all_parser.add_argument("--seed", type=int, default=42)
    run_all_parser.add_argument("--device", default="cpu")
    run_all_parser.add_argument("--workers", type=int, default=4)
    run_all_parser.add_argument("--score-threshold", type=float, default=0.25)
    run_all_parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    run_all_parser.add_argument("--limit", type=int, default=None)
    run_all_parser.add_argument("--yolo-epochs", type=int, default=100)
    run_all_parser.add_argument("--yolo-batch-size", type=int, default=16)
    run_all_parser.add_argument("--yolo-imgsz", type=int, default=960)
    run_all_parser.add_argument("--rtdetr-epochs", type=int, default=100)
    run_all_parser.add_argument("--rtdetr-batch-size", type=int, default=8)
    run_all_parser.add_argument("--rtdetr-imgsz", type=int, default=960)
    run_all_parser.add_argument("--fasterrcnn-epochs", type=int, default=25)
    run_all_parser.add_argument("--fasterrcnn-batch-size", type=int, default=4)
    run_all_parser.add_argument("--fasterrcnn-learning-rate", type=float, default=0.005)
    run_all_parser.add_argument("--fasterrcnn-weight-decay", type=float, default=0.0005)
    run_all_parser.add_argument("--fasterrcnn-pretrained", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare":
        from .prepare import prepare_dataset

        summary = prepare_dataset(
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
            copy_files=args.copy_files,
        )
        print(f"Prepared dataset at {summary['output_dir']}")
        print(f"Classes: {', '.join(summary['classes'])}")
        print(f"Split counts: {summary['counts']}")
        if summary["warnings"]:
            print(f"Warnings: {len(summary['warnings'])} (see metadata.json)")
        return

    if args.command == "clean-dataset":
        from .dataset_cleanup import clean_dataset

        summary = clean_dataset(dataset_root=args.dataset_root, ood_dir=args.ood_dir)
        print(summary)
        return

    if args.command == "train":
        if args.model_type in {"yolo", "rtdetr"}:
            from .ultralytics_pipeline import train_ultralytics

            checkpoint = train_ultralytics(
                model_type=args.model_type,
                prepared_root=args.prepared_root,
                runs_dir=args.runs_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                imgsz=args.imgsz,
                device=args.device,
                workers=args.workers,
                pretrained_weights=args.pretrained_weights,
            )
        else:
            from .fasterrcnn import train_fasterrcnn

            checkpoint = train_fasterrcnn(
                prepared_root=args.prepared_root,
                runs_dir=args.runs_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                num_workers=args.workers,
                device=args.device,
                pretrained=args.pretrained,
            )
        print(checkpoint)
        return

    if args.command == "evaluate":
        output_dir = args.output_dir / args.model_type / args.split
        if args.model_type in {"yolo", "rtdetr"}:
            from .ultralytics_pipeline import evaluate_ultralytics

            metrics = evaluate_ultralytics(
                model_type=args.model_type,
                prepared_root=args.prepared_root,
                checkpoint=args.checkpoint,
                split=args.split,
                output_dir=output_dir,
                score_threshold=args.score_threshold,
            )
        else:
            from .fasterrcnn import evaluate_fasterrcnn

            metrics = evaluate_fasterrcnn(
                prepared_root=args.prepared_root,
                checkpoint=args.checkpoint,
                split=args.split,
                device=args.device,
                batch_size=args.batch_size,
                num_workers=args.workers,
                output_dir=output_dir,
                score_threshold=args.score_threshold,
            )
        print(metrics)
        return

    if args.command == "predict":
        if args.model_type in {"yolo", "rtdetr"}:
            from .ultralytics_pipeline import predict_ultralytics

            output = predict_ultralytics(
                model_type=args.model_type,
                prepared_root=args.prepared_root,
                checkpoint=args.checkpoint,
                source=args.source,
                output_dir=args.output_dir,
                score_threshold=args.score_threshold,
            )
        else:
            from .fasterrcnn import predict_fasterrcnn

            output = predict_fasterrcnn(
                prepared_root=args.prepared_root,
                checkpoint=args.checkpoint,
                source=args.source,
                output_dir=args.output_dir,
                device=args.device,
                score_threshold=args.score_threshold,
            )
        print(output)
        return

    if args.command == "visualize-split":
        output_dir = args.output_dir / args.model_type / args.split
        if args.model_type in {"yolo", "rtdetr"}:
            from .ultralytics_pipeline import visualize_ultralytics_split

            output = visualize_ultralytics_split(
                model_type=args.model_type,
                prepared_root=args.prepared_root,
                checkpoint=args.checkpoint,
                split=args.split,
                output_dir=output_dir,
                score_threshold=args.score_threshold,
                limit=args.limit,
            )
        else:
            from .fasterrcnn import visualize_fasterrcnn_split

            output = visualize_fasterrcnn_split(
                prepared_root=args.prepared_root,
                checkpoint=args.checkpoint,
                split=args.split,
                output_dir=output_dir,
                device=args.device,
                score_threshold=args.score_threshold,
                limit=args.limit,
            )
        print(output)
        return

    if args.command == "run-all":
        from .orchestrator import run_all_models

        summary = run_all_models(
            dataset_root=args.dataset_root,
            prepared_root=args.prepared_root,
            output_dir=args.output_dir,
            runs_dir=args.runs_dir,
            prepare_if_needed=args.prepare_if_needed,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
            device=args.device,
            workers=args.workers,
            score_threshold=args.score_threshold,
            split=args.split,
            limit=args.limit,
            yolo_epochs=args.yolo_epochs,
            yolo_batch_size=args.yolo_batch_size,
            yolo_imgsz=args.yolo_imgsz,
            rtdetr_epochs=args.rtdetr_epochs,
            rtdetr_batch_size=args.rtdetr_batch_size,
            rtdetr_imgsz=args.rtdetr_imgsz,
            fasterrcnn_epochs=args.fasterrcnn_epochs,
            fasterrcnn_batch_size=args.fasterrcnn_batch_size,
            fasterrcnn_learning_rate=args.fasterrcnn_learning_rate,
            fasterrcnn_weight_decay=args.fasterrcnn_weight_decay,
            fasterrcnn_pretrained=args.fasterrcnn_pretrained,
        )
        print(summary)
        return

    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
