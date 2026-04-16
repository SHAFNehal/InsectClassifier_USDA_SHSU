## Annotated Images Detection Pipeline

This repository contains an insect object-detection dataset in Pascal VOC format plus a `uv`-managed training and evaluation pipeline for:

- `YOLOv8n`
- `RT-DETR`
- `Faster R-CNN`

The project covers dataset cleanup, dataset preparation, model training, evaluation, batch prediction, split visualization, side-by-side model comparison, and a small Streamlit inference app.

## Repository Layout

- `dataset/`: active class folders used for training and evaluation
- `OOD_Test_Files/`: images excluded from the active dataset because they are missing usable annotations
- `artifacts/prepared/`: prepared train/val/test exports and metadata
- `artifacts/run-all/`: combined evaluation and comparison outputs
- `runs/`: training outputs and checkpoints
- `src/annotated_images/`: CLI and pipeline implementation
- `streamlit_app.py`: manual inference UI

## Python

Use Python `3.11` or `3.12`.

## Install

```bash
uv sync
```

## Dataset Layout

Each active class has its own folder under `dataset/`. Inside each class folder:

- each image is stored beside its Pascal VOC `.xml`
- image and XML filenames must share the same stem
- XML `<object><name>` values are expected to match the folder name

Images that should not remain in the active dataset are moved to the repo-root `OOD_Test_Files/` folder.

### Active Class Folders

- `CFB-A`
- `Coconut Rhinoceros Beetle`
- `European Grapevine Moth`
- `Giant African Snail`
- `Golden Twin Spotted Moth`
- `New Guinea Sugarcane Weevil`
- `Oak Splendour Beetle-A`
- `Small Brown Planthopper`
- `Spotted Lanternfly`

### Current Counts

- `CFB-A`: `121`
- `Coconut Rhinoceros Beetle`: `126`
- `European Grapevine Moth`: `106`
- `Giant African Snail`: `214`
- `Golden Twin Spotted Moth`: `230`
- `New Guinea Sugarcane Weevil`: `253`
- `Oak Splendour Beetle-A`: `340`
- `Small Brown Planthopper`: `241`
- `Spotted Lanternfly`: `198`
- `OOD_Test_Files`: `75`

At the last check:

- active dataset images: `1829`
- missing image/XML pairs in active class folders: `0`
- folder-label mismatches in active class folders: `0`
- remaining `1x1` boxes in active class folders: `0`

## Dataset Cleanup

Run cleanup after adding or updating files under `dataset/`.

```bash
uv run annotated-images clean-dataset \
  --dataset-root dataset \
  --ood-dir OOD_Test_Files
```

This command:

- fixes XML object names so they match the parent folder name
- updates XML `<folder>` and `<filename>` fields to match the on-disk sample
- removes only `1x1` bounding boxes
- moves images with missing XML files to `OOD_Test_Files/`
- moves images whose XML becomes empty after cleanup to `OOD_Test_Files/`
- deletes orphan XML files
- renames moved OOD images as `Image_{folder_name}_{NNN}` while preserving the original extension

Recommended workflow after new uploads:

1. Add or update class folders under `dataset/`.
2. Run `uv run annotated-images clean-dataset --dataset-root dataset --ood-dir OOD_Test_Files`.
3. Run `uv run annotated-images prepare --dataset-root dataset --output-dir artifacts/prepared --val-fraction 0.2 --test-fraction 0.1 --seed 42`.

## Prepare The Dataset

Preparation validates the cleaned dataset, creates train/val/test splits, exports Ultralytics labels, writes COCO-style split manifests, and builds metadata used by the rest of the pipeline.

```bash
uv run annotated-images prepare \
  --dataset-root dataset \
  --output-dir artifacts/prepared \
  --val-fraction 0.2 \
  --test-fraction 0.1 \
  --seed 42
```

Optional:

- add `--copy-files` to copy images instead of creating symlinks

Prepared output includes:

- `artifacts/prepared/dataset.yaml`
- `artifacts/prepared/metadata.json`
- `artifacts/prepared/splits/{train,val,test}.json`
- `artifacts/prepared/coco/{train,val,test}.json`
- `artifacts/prepared/images/{train,val,test}/`
- `artifacts/prepared/labels/{train,val,test}/`

## Train

### YOLOv8n

```bash
uv run annotated-images train \
  --model-type yolo \
  --prepared-root artifacts/prepared \
  --runs-dir runs \
  --epochs 100 \
  --imgsz 960 \
  --batch-size 16 \
  --device cpu
```

### RT-DETR

```bash
uv run annotated-images train \
  --model-type rtdetr \
  --prepared-root artifacts/prepared \
  --runs-dir runs \
  --epochs 100 \
  --imgsz 960 \
  --batch-size 8 \
  --device cpu
```

### Faster R-CNN

```bash
uv run annotated-images train \
  --model-type fasterrcnn \
  --prepared-root artifacts/prepared \
  --runs-dir runs \
  --epochs 25 \
  --batch-size 4 \
  --pretrained \
  --device cuda
  --workers 0
```

Default checkpoints are written to:

- `runs/yolo-train/weights/best.pt`
- `runs/rtdetr-train/weights/best.pt`
- `runs/fasterrcnn/best.pt`

Notes:

- `YOLOv8n` and `RT-DETR` use Ultralytics training
- `Faster R-CNN` uses `torchvision`
- RT-DETR training includes a retry path with a lower batch size if training fails due to NaN/Inf loss

## Evaluate

### YOLOv8n

```bash
uv run annotated-images evaluate \
  --model-type yolo \
  --prepared-root artifacts/prepared \
  --checkpoint runs/yolo-train/weights/best.pt \
  --split test \
  --output-dir artifacts/evaluation
```

### RT-DETR

```bash
uv run annotated-images evaluate \
  --model-type rtdetr \
  --prepared-root artifacts/prepared \
  --checkpoint runs/rtdetr-train/weights/best.pt \
  --split test \
  --output-dir artifacts/evaluation
```

### Faster R-CNN

```bash
uv run annotated-images evaluate \
  --model-type fasterrcnn \
  --prepared-root artifacts/prepared \
  --checkpoint runs/fasterrcnn/best.pt \
  --split test \
  --output-dir artifacts/evaluation
```

Evaluation writes:

- `artifacts/evaluation/<model>/<split>/metrics.json`
- `artifacts/evaluation/<model>/<split>/predictions.json`

Metrics are based on the shared internal evaluator and report `map50` plus per-class precision, recall, AP50, ground-truth count, and prediction count.

## Predict On A Folder

Run inference over a directory of images and save predictions plus rendered previews.

```bash
uv run annotated-images predict \
  --model-type yolo \
  --prepared-root artifacts/prepared \
  --checkpoint runs/yolo-train/weights/best.pt \
  --source dataset/Spotted\ Lanternfly \
  --output-dir artifacts/predictions/yolo \
  --score-threshold 0.25
```

Prediction writes:

- `artifacts/predictions/<model>/predictions.json`
- `artifacts/predictions/<model>/renders/*`

## Visualize A Prepared Split

This renders images from a prepared split with:

- green ground-truth boxes labeled `GT: <class>`
- red predicted boxes labeled `Pred: <class> <score>`

```bash
uv run annotated-images visualize-split \
  --model-type yolo \
  --prepared-root artifacts/prepared \
  --checkpoint runs/yolo-train/weights/best.pt \
  --split test \
  --output-dir artifacts/visualizations \
  --score-threshold 0.25
```

Limit the number of rendered samples during review:

```bash
uv run annotated-images visualize-split \
  --model-type yolo \
  --prepared-root artifacts/prepared \
  --checkpoint runs/yolo-train/weights/best.pt \
  --split test \
  --output-dir artifacts/visualizations \
  --score-threshold 0.25 \
  --limit 25
```

Outputs are written under:

- `artifacts/visualizations/<model>/<split>/`
- `artifacts/visualizations/<model>/<split>/manifest.json`

## Run Everything End To End

`run-all` prepares the dataset if needed, trains all three model families, evaluates them on one split, writes summary tables, and renders side-by-side comparisons.

```bash
uv run annotated-images run-all \
  --prepared-root artifacts/prepared \
  --output-dir artifacts/run-all \
  --runs-dir runs \
  --split test \
  --score-threshold 0.25 \
  --device cpu
```

If `artifacts/prepared/` does not exist yet, add `--prepare-if-needed` and provide `--dataset-root`:

```bash
uv run annotated-images run-all \
  --dataset-root dataset \
  --prepared-root artifacts/prepared \
  --output-dir artifacts/run-all \
  --runs-dir runs \
  --prepare-if-needed \
  --split test
```

`run-all` writes:

- `artifacts/run-all/run_all_summary.json`
- `artifacts/run-all/evaluation/summary.json`
- `artifacts/run-all/evaluation/summary.csv`
- `artifacts/run-all/evaluation/<model>/<split>/metrics.json`
- `artifacts/run-all/evaluation/<model>/<split>/predictions.json`
- `artifacts/run-all/comparisons/<split>/*.png`
- `artifacts/run-all/comparisons/<split>/manifest.json`

## Streamlit Demo

The Streamlit app is for quick manual inference against the latest available trained checkpoints.

It:

- discovers the newest checkpoint for each supported model under `runs/`
- only shows models that currently have a checkpoint
- loads class names from `artifacts/prepared/metadata.json` when available
- accepts `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tif`, `.tiff`

Run it with:

```bash
uv run streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

Then open:

```text
http://localhost:8501
```

The app expects:

- trained checkpoints under `runs/`
- prepared metadata at `artifacts/prepared/metadata.json`

To share it through `ngrok`:

```bash
ngrok http 8501
```

## CLI Summary

Available commands:

- `annotated-images clean-dataset`
- `annotated-images prepare`
- `annotated-images train`
- `annotated-images evaluate`
- `annotated-images predict`
- `annotated-images visualize-split`
- `annotated-images run-all`
