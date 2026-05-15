import csv
import json
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEQUENCE = "boxing-fisheye"

# Part 2 predicted union mask directory.
PRED_MASK_DIR = PROJECT_ROOT / "outputs" / "part2_masks" / f"{SEQUENCE}_sam2" / "masks_union"

# DAVIS GT mask directory.
GT_MASK_DIR = PROJECT_ROOT / "data" / "DAVIS" / "Annotations" / "480p" / SEQUENCE

# Output directory.
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "part2_eval" / f"{SEQUENCE}_sam2_eval"

# IoU >= 这个阈值，记为 recall 命中
JR_THRESHOLD = 0.5

# 如果 GT 是多实例标注（像素值 0,1,2,...），是否把所有非 0 都当成前景
MERGE_ALL_GT_OBJECTS = True

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_mask_as_binary(path: Path, merge_all_objects: bool = True) -> np.ndarray:
    """
    读取单通道 mask，转成 bool 二值图。
    - 对 prediction: 非零即前景
    - 对 GT:
        * 若 merge_all_objects=True，则非零即前景
        * 否则同样先按非零处理（这里先统一成 union 评估）
    """
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Cannot read mask: {path}")

    # 如果是彩色图，转灰度
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    if merge_all_objects:
        return mask > 0
    else:
        return mask > 0


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    IoU for binary masks.
    特殊情况：
    - pred 空, gt 空 -> IoU = 1.0
    - 一个空一个非空 -> IoU = 0.0
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 1.0
    return float(inter / union)


def build_file_map(mask_dir: Path) -> dict[str, Path]:
    """
    用 stem 作为 key，便于匹配:
    00000.png -> key '00000'
    frame_000000.png -> key 'frame_000000'
    """
    file_map = {}
    for p in sorted(mask_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            file_map[p.stem] = p
    return file_map


def resize_if_needed(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    如果尺寸不一致，把 pred resize 到 gt 尺寸。
    mask resize 用 nearest，避免插值污染。
    """
    if pred.shape == gt.shape:
        return pred, gt

    pred_u8 = pred.astype(np.uint8) * 255
    pred_resized = cv2.resize(
        pred_u8,
        (gt.shape[1], gt.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )
    pred_resized = pred_resized > 0
    return pred_resized, gt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pred_dir = Path(PRED_MASK_DIR)
    gt_dir = Path(GT_MASK_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    assert pred_dir.exists(), f"Prediction dir not found: {pred_dir}"
    assert gt_dir.exists(), f"GT dir not found: {gt_dir}"

    pred_map = build_file_map(pred_dir)
    gt_map = build_file_map(gt_dir)

    if not pred_map:
        raise RuntimeError(f"No prediction masks found in: {pred_dir}")
    if not gt_map:
        raise RuntimeError(f"No GT masks found in: {gt_dir}")

    common_keys = sorted(set(pred_map.keys()) & set(gt_map.keys()))
    pred_only = sorted(set(pred_map.keys()) - set(gt_map.keys()))
    gt_only = sorted(set(gt_map.keys()) - set(pred_map.keys()))

    if not common_keys:
        raise RuntimeError(
            "No matched filenames between prediction and GT.\n"
            f"Pred dir: {pred_dir}\nGT dir: {gt_dir}"
        )

    print(f"Matched frames : {len(common_keys)}")
    print(f"Pred-only      : {len(pred_only)}")
    print(f"GT-only        : {len(gt_only)}")

    per_frame_rows = []
    ious = []

    for key in common_keys:
        pred_path = pred_map[key]
        gt_path = gt_map[key]

        pred_mask = load_mask_as_binary(pred_path, merge_all_objects=True)
        gt_mask = load_mask_as_binary(gt_path, merge_all_objects=MERGE_ALL_GT_OBJECTS)

        pred_mask, gt_mask = resize_if_needed(pred_mask, gt_mask)

        iou = compute_iou(pred_mask, gt_mask)
        ious.append(iou)

        per_frame_rows.append({
            "frame": key,
            "pred_file": pred_path.name,
            "gt_file": gt_path.name,
            "iou": iou,
            "pred_area": int(pred_mask.sum()),
            "gt_area": int(gt_mask.sum()),
        })

    ious_np = np.array(ious, dtype=np.float32)

    jm = float(ious_np.mean())
    jr = float((ious_np >= JR_THRESHOLD).mean())
    j_decay = float(ious_np[: len(ious_np)//4].mean() - ious_np[-len(ious_np)//4 :].mean()) if len(ious_np) >= 4 else 0.0
    j_min = float(ious_np.min())
    j_max = float(ious_np.max())
    j_std = float(ious_np.std())

    summary = {
        "num_matched_frames": len(common_keys),
        "num_pred_only_frames": len(pred_only),
        "num_gt_only_frames": len(gt_only),
        "JM": jm,
        "JR": jr,
        "JR_threshold": JR_THRESHOLD,
        "J_min": j_min,
        "J_max": j_max,
        "J_std": j_std,
        "J_decay": j_decay,
        "pred_dir": str(pred_dir),
        "gt_dir": str(gt_dir),
    }

    # 保存 CSV
    csv_path = out_dir / "per_frame_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame", "pred_file", "gt_file", "iou", "pred_area", "gt_area"]
        )
        writer.writeheader()
        writer.writerows(per_frame_rows)

    # 保存 JSON
    json_path = out_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # 保存 TXT
    txt_path = out_dir / "summary.txt"
    txt_lines = [
        "Part 2 Mask Evaluation",
        "======================",
        f"Prediction dir : {pred_dir}",
        f"GT dir         : {gt_dir}",
        f"Matched frames : {len(common_keys)}",
        f"Pred-only      : {len(pred_only)}",
        f"GT-only        : {len(gt_only)}",
        "",
        f"JM (IoU mean)  : {jm:.4f}",
        f"JR (IoU recall): {jr:.4f}   (threshold={JR_THRESHOLD})",
        f"J_min          : {j_min:.4f}",
        f"J_max          : {j_max:.4f}",
        f"J_std          : {j_std:.4f}",
        f"J_decay        : {j_decay:.4f}",
        "",
        f"Saved CSV      : {csv_path}",
        f"Saved JSON     : {json_path}",
    ]
    txt_path.write_text("\n".join(txt_lines))

    print("\n===== Evaluation Done =====")
    print(f"JM (IoU mean)   : {jm:.4f}")
    print(f"JR (IoU recall) : {jr:.4f}  (threshold={JR_THRESHOLD})")
    print(f"J_min           : {j_min:.4f}")
    print(f"J_max           : {j_max:.4f}")
    print(f"J_std           : {j_std:.4f}")
    print(f"J_decay         : {j_decay:.4f}")
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
