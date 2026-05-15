import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Config — 修改这里
# ---------------------------------------------------------------------------
PROJ_ROOT = PROJECT_ROOT
VIDEO_NAME = "boxing-fisheye"
PRED_MASK_DIR = PROJECT_ROOT / "outputs" / "part1_masks" / "boxing-fisheye_yolov8seg" / "masks_union"
GT_MASK_DIR = PROJECT_ROOT / "data" / "DAVIS" / "Annotations" / "480p" / "boxing-fisheye"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "part1_eval" / "boxing-fisheye_yolov8seg_eval"

JR_THRESHOLD = 0.5
FOREGROUND_IDS = None
MERGE_ALL_GT_OBJECTS = True


def default_config():
    return SimpleNamespace(
        proj_root=PROJ_ROOT,
        video_name=VIDEO_NAME,
        pred_dir=PRED_MASK_DIR,
        gt_dir=GT_MASK_DIR,
        output_dir=OUTPUT_DIR,
        jr_threshold=JR_THRESHOLD,
        foreground_ids=FOREGROUND_IDS,
        merge_all_gt_objects=MERGE_ALL_GT_OBJECTS,
    )


def load_mask_as_binary(path: Path, merge_all_objects: bool = True, foreground_ids=None) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Cannot read mask: {path}")

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    if foreground_ids:
        return np.isin(mask, np.array(foreground_ids))

    if merge_all_objects:
        return mask > 0
    return mask > 0


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 1.0
    return float(inter / union)


def build_file_map(mask_dir: Path) -> dict[str, Path]:
    file_map = {}
    for p in sorted(mask_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            file_map[p.stem] = p
    return file_map


def resize_if_needed(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if pred.shape == gt.shape:
        return pred, gt

    pred_u8 = pred.astype(np.uint8) * 255
    pred_resized = cv2.resize(
        pred_u8,
        (gt.shape[1], gt.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )
    return pred_resized > 0, gt


def main(cfg=None):
    if cfg is None:
        cfg = default_config()

    pred_dir = Path(cfg.pred_dir)
    gt_dir = Path(cfg.gt_dir)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction dir not found: {pred_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"GT dir not found: {gt_dir}")

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
        gt_mask = load_mask_as_binary(
            gt_path,
            merge_all_objects=cfg.merge_all_gt_objects,
            foreground_ids=cfg.foreground_ids,
        )

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
    jr = float((ious_np >= cfg.jr_threshold).mean())
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
        "JR_threshold": cfg.jr_threshold,
        "J_min": j_min,
        "J_max": j_max,
        "J_std": j_std,
        "J_decay": j_decay,
        "pred_dir": str(pred_dir),
        "gt_dir": str(gt_dir),
    }
    if cfg.foreground_ids:
        summary["foreground_ids"] = cfg.foreground_ids
    else:
        summary["foreground_ids"] = "all non-zero ids"

    csv_path = out_dir / "per_frame_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame", "pred_file", "gt_file", "iou", "pred_area", "gt_area"]
        )
        writer.writeheader()
        writer.writerows(per_frame_rows)

    json_path = out_dir / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    txt_path = out_dir / "summary.txt"
    txt_lines = [
        "Part 1 Mask Evaluation",
        "======================",
        f"Prediction dir : {pred_dir}",
        f"GT dir         : {gt_dir}",
        f"Matched frames : {len(common_keys)}",
        f"Pred-only      : {len(pred_only)}",
        f"GT-only        : {len(gt_only)}",
        "",
        f"JM (IoU mean)  : {jm:.4f}",
        f"JR (IoU recall): {jr:.4f}   (threshold={cfg.jr_threshold})",
        f"J_min          : {j_min:.4f}",
        f"J_max          : {j_max:.4f}",
        f"J_std          : {j_std:.4f}",
        f"J_decay        : {j_decay:.4f}",
        "",
        f"Saved CSV      : {csv_path}",
        f"Saved JSON     : {json_path}",
    ]
    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")

    print("\n===== Evaluation Done =====")
    print(f"JM (IoU mean)   : {jm:.4f}")
    print(f"JR (IoU recall) : {jr:.4f}  (threshold={cfg.jr_threshold})")
    print(f"J_min           : {j_min:.4f}")
    print(f"J_max           : {j_max:.4f}")
    print(f"J_std           : {j_std:.4f}")
    print(f"J_decay         : {j_decay:.4f}")
    print(f"\nSaved to: {out_dir}")

    return summary


if __name__ == "__main__":
    main()
