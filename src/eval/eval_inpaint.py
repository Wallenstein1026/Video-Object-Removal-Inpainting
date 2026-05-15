import csv
import json
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VIDEO_NAME = "bear"

# DiffuEraser 输出的最终帧目录
#PRED_FRAME_DIR = (PROJECT_ROOT / "outputs" / "part3_diffueraser_inpaint" / VIDEO_NAME / "final_frames")
PRED_FRAME_DIR = PROJECT_ROOT / "outputs" / "part2_inpaint" / VIDEO_NAME / VIDEO_NAME / "frames"
# 原始 DAVIS 帧目录。有 GT 的实验才会使用。
GT_FRAME_DIR = PROJECT_ROOT / "data" / "DAVIS" / "JPEGImages" / "480p" / VIDEO_NAME

# Mask directory. Set to None to evaluate the whole frame against a clean
# background GT image/sequence.
MASK_DIR = PROJECT_ROOT / "outputs" / "part2_masks" / f"{VIDEO_NAME}_sam2" / "masks_union"

# 输出目录
#OUTPUT_DIR = PROJECT_ROOT / "outputs" / "part3_eval" / f"{VIDEO_NAME}_diffueraser_eval"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "part2_eval" / f"{VIDEO_NAME}_inpaint_eval"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
MASK_DILATION_ITER = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sorted_image_paths(folder: Path) -> list[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    )


def normalize_key(stem: str) -> str:
    if stem.isdigit():
        return str(int(stem))
    return stem


def frame_sort_key(key: str) -> tuple[int, str]:
    if key.isdigit():
        return int(key), key
    return 10**9, key


def build_image_map(folder: Path) -> dict[str, Path]:
    return {normalize_key(p.stem): p for p in sorted_image_paths(folder)}


def load_bgr_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def load_binary_mask(path: Path, shape_hw: tuple[int, int] | None = None) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to read mask: {path}")
    if shape_hw is not None and mask.shape != shape_hw:
        h, w = shape_hw
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def resize_if_needed(image: np.ndarray, target_shape_hw: tuple[int, int]) -> np.ndarray:
    if image.shape[:2] == target_shape_hw:
        return image
    h, w = target_shape_hw
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)


def masked_psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    mask = mask.astype(bool)
    if not np.any(mask):
        return float("nan")
    diff = a.astype(np.float64) - b.astype(np.float64)
    if diff.ndim == 3:
        diff = diff[mask]
    else:
        diff = diff[mask]
    mse = float(np.mean(diff * diff))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def masked_l1(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    mask = mask.astype(bool)
    if not np.any(mask):
        return float("nan")
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    if diff.ndim == 3:
        diff = diff[mask]
    else:
        diff = diff[mask]
    return float(np.mean(diff))


def ssim_single_channel(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    ksize = (11, 11)
    sigma = 1.5

    mu_x = cv2.GaussianBlur(x, ksize, sigma)
    mu_y = cv2.GaussianBlur(y, ksize, sigma)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = cv2.GaussianBlur(x * x, ksize, sigma) - mu_x2
    sigma_y2 = cv2.GaussianBlur(y * y, ksize, sigma) - mu_y2
    sigma_xy = cv2.GaussianBlur(x * y, ksize, sigma) - mu_xy

    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = numerator / (denominator + 1e-12)
    return float(np.mean(ssim_map))


def masked_ssim(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    mask = mask.astype(bool)
    if not np.any(mask):
        return float("nan")

    a_eval = np.zeros_like(a)
    b_eval = np.zeros_like(b)
    a_eval[mask] = a[mask]
    b_eval[mask] = b[mask]

    if a_eval.ndim == 2:
        return ssim_single_channel(a_eval, b_eval)

    values = [
        ssim_single_channel(a_eval[:, :, c], b_eval[:, :, c])
        for c in range(a_eval.shape[2])
    ]
    return float(np.mean(values))


def dilate_mask(mask: np.ndarray, iterations: int = MASK_DILATION_ITER) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=iterations) > 0


def temporal_l1(curr: np.ndarray, prev: np.ndarray, mask: np.ndarray) -> float:
    mask = mask.astype(bool)
    if not np.any(mask):
        return float("nan")
    diff = np.abs(curr.astype(np.float64) - prev.astype(np.float64))
    if diff.ndim == 3:
        diff = diff[mask]
    else:
        diff = diff[mask]
    return float(np.mean(diff))


def mean_or_nan(values: list[float]) -> float:
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    pred_dir = Path(PRED_FRAME_DIR)
    gt_dir = Path(GT_FRAME_DIR) if GT_FRAME_DIR is not None else None
    mask_dir = Path(MASK_DIR) if MASK_DIR is not None else None
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction frame dir not found: {pred_dir}")
    if mask_dir is not None and not mask_dir.exists():
        raise FileNotFoundError(f"Mask dir not found: {mask_dir}")

    pred_map = build_image_map(pred_dir)
    mask_map = build_image_map(mask_dir) if mask_dir is not None else {}
    has_gt = gt_dir is not None and gt_dir.exists()
    gt_map = build_image_map(gt_dir) if has_gt else {}

    if not pred_map:
        raise RuntimeError(f"No prediction frames found in: {pred_dir}")
    if mask_dir is not None and not mask_map:
        raise RuntimeError(f"No masks found in: {mask_dir}")
    if has_gt and not gt_map:
        raise RuntimeError(f"No GT frames found in: {gt_dir}")

    if mask_dir is None:
        common_set = set(pred_map) & (set(gt_map) if has_gt else set(pred_map))
    else:
        common_set = set(pred_map) & set(mask_map) & (set(gt_map) if has_gt else set(pred_map))
    common_keys = sorted(common_set, key=frame_sort_key)
    pred_only = sorted(set(pred_map) - (set(gt_map) if has_gt else set()), key=frame_sort_key)
    gt_only = sorted(set(gt_map) - set(pred_map), key=frame_sort_key) if has_gt else []
    mask_only = sorted(set(mask_map) - set(pred_map), key=frame_sort_key) if mask_dir is not None else []

    if not common_keys:
        if has_gt:
            raise RuntimeError(
                "No matched filenames across prediction, GT, and mask directories.\n"
                f"Pred dir: {pred_dir}\nGT dir: {gt_dir}\nMask dir: {mask_dir}"
            )
        if mask_dir is None:
            raise RuntimeError(f"No matched filenames in prediction directory: {pred_dir}")
        raise RuntimeError(
            "No matched filenames across prediction and mask directories.\n"
            f"Pred dir: {pred_dir}\nMask dir: {mask_dir}"
        )

    print(f"Matched frames : {len(common_keys)}")
    print(f"Pred-only      : {len(pred_only)}")
    if has_gt:
        print(f"GT-only        : {len(gt_only)}")
    if mask_dir is None:
        print("Mask mode      : full-frame edit region")
    else:
        print(f"Mask-only      : {len(mask_only)}")
    if not has_gt:
        print("GT metrics     : skipped (no ground truth frames found)")

    per_frame_rows = []
    bg_l1_list = []
    bg_psnr_list = []
    bg_ssim_list = []
    ring_l1_list = []
    ring_psnr_list = []
    ring_ssim_list = []
    edit_l1_list = []
    edit_psnr_list = []
    edit_ssim_list = []
    temporal_edit_l1_list = []
    temporal_ring_l1_list = []

    prev_pred = None
    prev_edit_mask = None
    prev_ring_mask = None

    for key in common_keys:
        pred_path = pred_map[key]
        mask_path = mask_map[key] if mask_dir is not None else None

        pred = load_bgr_image(pred_path)
        gt = load_bgr_image(gt_map[key]) if has_gt else None
        if mask_path is None:
            mask = np.ones(gt.shape[:2] if has_gt else pred.shape[:2], dtype=bool)
        else:
            mask = load_binary_mask(mask_path, shape_hw=gt.shape[:2] if has_gt else pred.shape[:2])

        if has_gt:
            pred = resize_if_needed(pred, gt.shape[:2])

        bg_mask = ~mask
        ring_mask = dilate_mask(mask, iterations=MASK_DILATION_ITER) & (~mask)

        bg_l1 = masked_l1(pred, gt, bg_mask) if has_gt else float("nan")
        bg_psnr = masked_psnr(pred, gt, bg_mask) if has_gt else float("nan")
        bg_ssim = masked_ssim(pred, gt, bg_mask) if has_gt else float("nan")
        ring_l1 = masked_l1(pred, gt, ring_mask) if has_gt else float("nan")
        ring_psnr = masked_psnr(pred, gt, ring_mask) if has_gt else float("nan")
        ring_ssim = masked_ssim(pred, gt, ring_mask) if has_gt else float("nan")
        edit_l1 = masked_l1(pred, gt, mask) if has_gt else float("nan")
        edit_psnr = masked_psnr(pred, gt, mask) if has_gt else float("nan")
        edit_ssim = masked_ssim(pred, gt, mask) if has_gt else float("nan")

        bg_l1_list.append(bg_l1)
        bg_psnr_list.append(bg_psnr)
        bg_ssim_list.append(bg_ssim)
        ring_l1_list.append(ring_l1)
        ring_psnr_list.append(ring_psnr)
        ring_ssim_list.append(ring_ssim)
        edit_l1_list.append(edit_l1)
        edit_psnr_list.append(edit_psnr)
        edit_ssim_list.append(edit_ssim)

        temporal_edit_l1 = float("nan")
        temporal_ring_l1 = float("nan")
        if prev_pred is not None and prev_edit_mask is not None and prev_ring_mask is not None:
            temporal_edit_mask = mask | prev_edit_mask
            temporal_ring_mask = ring_mask | prev_ring_mask
            temporal_edit_l1 = temporal_l1(pred, prev_pred, temporal_edit_mask)
            temporal_ring_l1 = temporal_l1(pred, prev_pred, temporal_ring_mask)
            temporal_edit_l1_list.append(temporal_edit_l1)
            temporal_ring_l1_list.append(temporal_ring_l1)

        per_frame_rows.append(
            {
                "frame": key,
                "pred_file": pred_path.name,
                "gt_file": gt_map[key].name if has_gt else "",
                "mask_file": mask_path.name if mask_path is not None else "FULL_FRAME",
                "mask_area": int(mask.sum()),
                "ring_area": int(ring_mask.sum()),
                "bg_l1": bg_l1,
                "bg_psnr": bg_psnr,
                "bg_ssim": bg_ssim,
                "ring_l1": ring_l1,
                "ring_psnr": ring_psnr,
                "ring_ssim": ring_ssim,
                "edit_l1": edit_l1,
                "edit_psnr": edit_psnr,
                "edit_ssim": edit_ssim,
                "temporal_edit_l1": temporal_edit_l1,
                "temporal_ring_l1": temporal_ring_l1,
            }
        )

        prev_pred = pred
        prev_edit_mask = mask
        prev_ring_mask = ring_mask

    summary = {
        "video_name": VIDEO_NAME,
        "has_ground_truth": has_gt,
        "num_matched_frames": len(common_keys),
        "num_pred_only_frames": len(pred_only),
        "num_gt_only_frames": len(gt_only),
        "num_mask_only_frames": len(mask_only),
        "mask_mode": "full_frame" if mask_dir is None else "mask_dir",
        "bg_l1_mean": mean_or_nan(bg_l1_list),
        "bg_psnr_mean": mean_or_nan(bg_psnr_list),
        "bg_ssim_mean": mean_or_nan(bg_ssim_list),
        "ring_l1_mean": mean_or_nan(ring_l1_list),
        "ring_psnr_mean": mean_or_nan(ring_psnr_list),
        "ring_ssim_mean": mean_or_nan(ring_ssim_list),
        "edit_l1_mean": mean_or_nan(edit_l1_list),
        "edit_psnr_mean": mean_or_nan(edit_psnr_list),
        "edit_ssim_mean": mean_or_nan(edit_ssim_list),
        "temporal_edit_l1_mean": mean_or_nan(temporal_edit_l1_list),
        "temporal_ring_l1_mean": mean_or_nan(temporal_ring_l1_list),
        "pred_dir": str(pred_dir),
        "gt_dir": str(gt_dir) if gt_dir is not None else None,
        "mask_dir": str(mask_dir),
    }

    csv_path = out_dir / "per_frame_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "pred_file",
                "gt_file",
                "mask_file",
                "mask_area",
                "ring_area",
                "bg_l1",
                "bg_psnr",
                "bg_ssim",
                "ring_l1",
                "ring_psnr",
                "ring_ssim",
                "edit_l1",
                "edit_psnr",
                "edit_ssim",
                "temporal_edit_l1",
                "temporal_ring_l1",
            ],
        )
        writer.writeheader()
        writer.writerows(per_frame_rows)

    json_path = out_dir / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    txt_path = out_dir / "summary.txt"
    txt_lines = [
        "Inpaint Evaluation",
        "===================",
        f"Prediction dir         : {pred_dir}",
        f"GT dir                 : {gt_dir}",
        f"Mask dir               : {mask_dir if mask_dir is not None else 'None (full-frame edit region)'}",
        f"Has ground truth       : {has_gt}",
        f"Matched frames         : {len(common_keys)}",
        f"Pred-only              : {len(pred_only)}",
        f"GT-only                : {len(gt_only)}" if has_gt else "GT-only                : N/A",
        f"Mask-only              : {len(mask_only)}" if mask_dir is not None else "Mask-only              : N/A",
        "",
        f"BG L1 mean             : {summary['bg_l1_mean']:.4f}" if has_gt else "BG L1 mean             : N/A",
        f"BG PSNR mean           : {summary['bg_psnr_mean']:.4f}" if has_gt else "BG PSNR mean           : N/A",
        f"BG SSIM mean           : {summary['bg_ssim_mean']:.4f}" if has_gt else "BG SSIM mean           : N/A",
        f"Ring L1 mean           : {summary['ring_l1_mean']:.4f}" if has_gt else "Ring L1 mean           : N/A",
        f"Ring PSNR mean         : {summary['ring_psnr_mean']:.4f}" if has_gt else "Ring PSNR mean         : N/A",
        f"Ring SSIM mean         : {summary['ring_ssim_mean']:.4f}" if has_gt else "Ring SSIM mean         : N/A",
        f"Edit L1 mean           : {summary['edit_l1_mean']:.4f}" if has_gt else "Edit L1 mean           : N/A",
        f"Edit PSNR mean         : {summary['edit_psnr_mean']:.4f}" if has_gt else "Edit PSNR mean         : N/A",
        f"Edit SSIM mean         : {summary['edit_ssim_mean']:.4f}" if has_gt else "Edit SSIM mean         : N/A",
        f"Temporal edit L1 mean   : {summary['temporal_edit_l1_mean']:.4f}",
        f"Temporal ring L1 mean   : {summary['temporal_ring_l1_mean']:.4f}",
        "",
        f"Saved CSV              : {csv_path}",
        f"Saved JSON             : {json_path}",
    ]
    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")

    print("\n===== Evaluation Done =====")
    if has_gt:
        print(f"BG L1 mean            : {summary['bg_l1_mean']:.4f}")
        print(f"BG PSNR mean          : {summary['bg_psnr_mean']:.4f}")
        print(f"BG SSIM mean          : {summary['bg_ssim_mean']:.4f}")
        print(f"Ring L1 mean          : {summary['ring_l1_mean']:.4f}")
        print(f"Ring PSNR mean        : {summary['ring_psnr_mean']:.4f}")
        print(f"Ring SSIM mean        : {summary['ring_ssim_mean']:.4f}")
        print(f"Edit L1 mean          : {summary['edit_l1_mean']:.4f}")
        print(f"Edit PSNR mean        : {summary['edit_psnr_mean']:.4f}")
        print(f"Edit SSIM mean        : {summary['edit_ssim_mean']:.4f}")
    else:
        print("GT metrics            : skipped")
    print(f"Temporal edit L1 mean : {summary['temporal_edit_l1_mean']:.4f}")
    print(f"Temporal ring L1 mean : {summary['temporal_ring_l1_mean']:.4f}")
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
