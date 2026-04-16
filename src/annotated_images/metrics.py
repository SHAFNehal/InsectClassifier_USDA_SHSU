from __future__ import annotations


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _average_precision(recalls: list[float], precisions: list[float]) -> float:
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for index in range(len(mpre) - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])
    area = 0.0
    for index in range(1, len(mrec)):
        if mrec[index] != mrec[index - 1]:
            area += (mrec[index] - mrec[index - 1]) * mpre[index]
    return area


def evaluate_predictions(
    ground_truth_by_image: dict[str, list[dict[str, object]]],
    predictions_by_image: dict[str, list[dict[str, object]]],
    class_names: list[str],
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    class_metrics: dict[str, dict[str, float]] = {}
    ap_values: list[float] = []
    for class_id, class_name in enumerate(class_names):
        gt_by_image: dict[str, list[dict[str, object]]] = {}
        total_gt = 0
        for image_path, items in ground_truth_by_image.items():
            class_items = [{"box": item["box"], "matched": False} for item in items if item["label"] == class_id]
            gt_by_image[image_path] = class_items
            total_gt += len(class_items)

        preds: list[dict[str, object]] = []
        for image_path, items in predictions_by_image.items():
            for item in items:
                if item["label"] == class_id:
                    preds.append(
                        {
                            "image_path": image_path,
                            "box": item["box"],
                            "score": float(item["score"]),
                        }
                    )
        preds.sort(key=lambda item: item["score"], reverse=True)

        true_positive: list[float] = []
        false_positive: list[float] = []
        for prediction in preds:
            candidates = gt_by_image.get(prediction["image_path"], [])
            best_iou = 0.0
            best_match_index = None
            for index, candidate in enumerate(candidates):
                if candidate["matched"]:
                    continue
                iou = compute_iou(prediction["box"], candidate["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_match_index = index
            if best_match_index is not None and best_iou >= iou_threshold:
                candidates[best_match_index]["matched"] = True
                true_positive.append(1.0)
                false_positive.append(0.0)
            else:
                true_positive.append(0.0)
                false_positive.append(1.0)

        recalls: list[float] = []
        precisions: list[float] = []
        running_tp = 0.0
        running_fp = 0.0
        for tp, fp in zip(true_positive, false_positive, strict=True):
            running_tp += tp
            running_fp += fp
            recall = running_tp / total_gt if total_gt else 0.0
            precision = running_tp / (running_tp + running_fp) if (running_tp + running_fp) else 0.0
            recalls.append(recall)
            precisions.append(precision)

        ap50 = _average_precision(recalls, precisions) if total_gt else 0.0
        final_tp = running_tp
        final_fp = running_fp
        class_metrics[class_name] = {
            "ap50": ap50,
            "precision": final_tp / (final_tp + final_fp) if (final_tp + final_fp) else 0.0,
            "recall": final_tp / total_gt if total_gt else 0.0,
            "ground_truth_objects": float(total_gt),
            "predictions": float(len(preds)),
        }
        ap_values.append(ap50)

    return {
        "map50": sum(ap_values) / len(ap_values) if ap_values else 0.0,
        "per_class": class_metrics,
    }
