"""
01_yolo_detect.py
-----------------
Step 1 of Part 2 pipeline: Run YOLOv8-Seg on every frame of a video
OR a folder of JPEG/PNG images (e.g. DAVIS JPEGImages), filter detections
to dynamic classes, and export per-frame bounding-box prompts ready for SAM 2.

Auto-detects input type:
  - If --input is a video file  → extract frames first, then detect
  - If --input is a directory   → use images directly (sorted by filename)

Output layout
-------------
<output_dir>/
    frames/          raw extracted frames  (frame_000000.jpg …)  [video only]
    yolo_vis/        debug frames with boxes drawn
    prompts.json     [{frame_idx, track_id, cls, box_xyxy, conf, point}]
    summary.txt      per-class detection counts
"""

import argparse
import json
import shutil
from itertools import groupby
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Dynamic object classes we care about (COCO class names used by YOLOv8)
# ---------------------------------------------------------------------------
DYNAMIC_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "truck",
    "bird", "cat", "dog", "horse", "sheep", "cow", "bear",
    "tennis racket", "suitcase", "refrigerator"

}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = str(PROJECT_ROOT / "data" / "wild" / "JPEGImages" / "480p" / "swimming")
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "outputs" / "part2_masks" / "swimming_yolo")
DEFAULT_MODEL_WEIGHTS = "yolo11x.pt"
DEFAULT_CONF_THRESH = 0.7
DEFAULT_IOU_THRESH = 0.45
DEFAULT_MERGE_IOU = 0.5
DEFAULT_TARGET_FPS = None
DEFAULT_ENABLE_DYNAMIC_FILTER = True
DEFAULT_COMPENSATE_GLOBAL_MOTION = True
DEFAULT_TAU_MOTION = 1.5
DEFAULT_MOTION_PERCENTILE = 75.0


# ---------------------------------------------------------------------------
# Frame loading: two strategies, unified interface
# ---------------------------------------------------------------------------

def load_frames_from_video(video_path: Path, frames_dir: Path, fps: float = None) -> list[Path]:
    """Extract frames from a video file and save as JPEGs."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / fps)) if fps else 1
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    saved = []
    idx = 0
    pbar = tqdm(total=total, desc="Extracting frames", unit="f")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            out_path = frames_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved.append(out_path)
        idx += 1
        pbar.update(1)
    cap.release()
    pbar.close()
    print(f"  → Extracted {len(saved)} frames from {total} total (step={step})")
    return sorted(saved)


def load_frames_from_dir(image_dir: Path) -> list[Path]:
    """Collect all images from a directory, sorted by filename.

    Works directly on DAVIS JPEGImages/<sequence>/ folders.
    No copying — returns paths in-place.
    """
    frames = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not frames:
        raise RuntimeError(f"No images found in {image_dir}")
    print(f"  → Found {len(frames)} images in {image_dir}")
    return frames


def resolve_frames(input_path: Path, out_dir: Path, fps: float = None, keep_frames: bool = False) -> list[Path]:
    """Auto-detect input type and return a sorted list of frame paths."""
    if input_path.is_file():
        print(f"Input detected as VIDEO: {input_path}")
        frames_dir = out_dir / "frames"
        if frames_dir.exists() and any(frames_dir.iterdir()) and not keep_frames:
            print(f"Frames already exist in {frames_dir}, skipping extraction.")
            return sorted(frames_dir.glob("frame_*.jpg"))
        return load_frames_from_video(input_path, frames_dir, fps=fps)

    elif input_path.is_dir():
        print(f"Input detected as IMAGE FOLDER: {input_path}")
        return load_frames_from_dir(input_path)

    else:
        raise FileNotFoundError(f"Input not found: {input_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def box_center(box_xyxy: list[float]) -> list[float]:
    x1, y1, x2, y2 = box_xyxy
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


def draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    vis = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box_xyxy"]]
        label = f"{det['cls']} {det['conf']:.2f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, label, (x1, max(y1 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return vis


# ---------------------------------------------------------------------------
# Dynamic filtering helpers
# ---------------------------------------------------------------------------

def estimate_global_transform(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    """Estimate global affine motion from prev -> curr using sparse LK + RANSAC."""
    pts_prev = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=1000,
        qualityLevel=0.01,
        minDistance=8,
        blockSize=7
    )

    identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    if pts_prev is None or len(pts_prev) < 6:
        return identity

    pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        pts_prev,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    if pts_curr is None or status is None:
        return identity

    good_prev = pts_prev[status.reshape(-1) == 1].reshape(-1, 2)
    good_curr = pts_curr[status.reshape(-1) == 1].reshape(-1, 2)

    if len(good_prev) < 6 or len(good_curr) < 6:
        return identity

    M, _ = cv2.estimateAffinePartial2D(
        good_prev,
        good_curr,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
        confidence=0.99
    )

    if M is None:
        return identity

    return M.astype(np.float32)


def compute_motion_map(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    compensate_global_motion: bool = False
) -> np.ndarray:
    """
    Return per-pixel motion magnitude map between prev and curr.
    If compensate_global_motion=True, first warp prev_gray to align with curr_gray.
    """
    if compensate_global_motion:
        M = estimate_global_transform(prev_gray, curr_gray)
        h, w = curr_gray.shape
        prev_gray = cv2.warpAffine(
            prev_gray,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )
    mag = cv2.magnitude(flow[..., 0], flow[..., 1])
    mag = cv2.GaussianBlur(mag, (5, 5), 0)
    return mag


def box_motion_score(
    motion_map: np.ndarray,
    box_xyxy: list[float],
    percentile: float = 75.0
) -> float:
    """Use a high percentile of motion magnitude inside the box as motion score."""
    h, w = motion_map.shape
    x1, y1, x2, y2 = box_xyxy

    x1 = max(0, min(w - 1, int(np.floor(x1))))
    y1 = max(0, min(h - 1, int(np.floor(y1))))
    x2 = max(0, min(w,     int(np.ceil(x2))))
    y2 = max(0, min(h,     int(np.ceil(y2))))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    roi = motion_map[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    return float(np.percentile(roi, percentile))


def is_dynamic_box(
    motion_map: np.ndarray,
    box_xyxy: list[float],
    tau_motion: float = 2.0,
    percentile: float = 75.0
) -> bool:
    """Keep detection only if motion score inside box exceeds threshold."""
    score = box_motion_score(motion_map, box_xyxy, percentile=percentile)
    return score >= tau_motion


# ---------------------------------------------------------------------------
# Core detection loop
# ---------------------------------------------------------------------------

def run_detection(
    frames,
    model,
    target_classes,
    conf_threshold,
    iou_threshold,
    vis_dir,
    enable_dynamic_filter=False,
    compensate_global_motion=False,
    tau_motion=2.0,
    motion_percentile=75.0,
):
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    cid_to_name = model.names
    all_prompts = []

    prev_gray = None

    for frame_idx, frame_path in enumerate(tqdm(frames, desc="Detecting", unit="frame")):
        frame_bgr = cv2.imread(str(frame_path))
        if frame_bgr is None:
            print(f"  [warn] Cannot read {frame_path}, skipping.")
            continue

        curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        motion_map = None
        if enable_dynamic_filter and prev_gray is not None:
            motion_map = compute_motion_map(
                prev_gray,
                curr_gray,
                compensate_global_motion=compensate_global_motion
            )

        results = model.predict(
            source=frame_bgr,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=1280,
            augment=True,
            verbose=False,
        )[0]
        frame_dets = []

        if results.boxes is not None:
            boxes_xyxy = results.boxes.xyxy.cpu().numpy()
            boxes_xywh = results.boxes.xywh.cpu().numpy()
            confs      = results.boxes.conf.cpu().numpy()
            cls_ids    = results.boxes.cls.cpu().numpy().astype(int)
            track_ids  = (
                results.boxes.id.cpu().numpy().astype(int).tolist()
                if results.boxes.id is not None
                else [None] * len(cls_ids)
            )

            for i, cls_id in enumerate(cls_ids):
                cls_name = cid_to_name.get(cls_id, str(cls_id))
                if cls_name not in target_classes:
                    continue

                # 动态性过滤：从第 2 帧开始才有 prev_gray 可以判断
                if enable_dynamic_filter and motion_map is not None:
                    if not is_dynamic_box(
                        motion_map,
                        boxes_xyxy[i].tolist(),
                        tau_motion=tau_motion,
                        percentile=motion_percentile
                    ):
                        continue

                record = {
                    "frame_idx":  frame_idx,
                    "frame_name": frame_path.name,
                    "frame_path": str(frame_path.resolve()),
                    "track_id":   track_ids[i],
                    "cls":        cls_name,
                    "cls_id":     int(cls_id),
                    "conf":       float(confs[i]),
                    "box_xyxy":   boxes_xyxy[i].tolist(),
                    "box_xywh":   boxes_xywh[i].tolist(),
                    "point":      box_center(boxes_xyxy[i].tolist()),
                }
                all_prompts.append(record)
                frame_dets.append(record)

        if vis_dir:
            vis = draw_detections(frame_bgr, frame_dets)
            cv2.imwrite(str(vis_dir / frame_path.name), vis)

        prev_gray = curr_gray

    return all_prompts


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def merge_overlapping_boxes(prompts, iou_threshold=0.5):
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter)

    key_fn = lambda r: (r["frame_idx"], r["cls"])
    sorted_p = sorted(prompts, key=key_fn)
    merged = []
    for _, group in groupby(sorted_p, key=key_fn):
        group = list(group)
        keep = [True] * len(group)
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if not keep[i] or not keep[j]:
                    continue
                if iou(group[i]["box_xyxy"], group[j]["box_xyxy"]) > iou_threshold:
                    if group[i]["conf"] >= group[j]["conf"]:
                        keep[j] = False
                    else:
                        keep[i] = False
        merged.extend([g for g, k in zip(group, keep) if k])
    return merged


def build_summary(prompts, total_frames):
    cls_counts   = Counter(p["cls"] for p in prompts)
    frame_counts = Counter(p["frame_idx"] for p in prompts)
    lines = [
        f"Total prompts  : {len(prompts)}",
        f"Frames with det: {len(frame_counts)} / {total_frames}",
        "Per-class counts:",
    ]
    for cls, cnt in sorted(cls_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {cls:<20s} {cnt:>6d}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ⚙️ Config — 修改这里
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Part 2 YOLO detection for DAVIS frames")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT,
                        help=f"Input video file or image folder (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory for YOLO results (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--weights", type=str, default=DEFAULT_MODEL_WEIGHTS,
                        help="YOLO weights file or model name")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESH,
                        help="Confidence threshold for YOLO")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU_THRESH,
                        help="NMS IOU threshold for YOLO")
    parser.add_argument("--merge-iou", type=float, default=DEFAULT_MERGE_IOU,
                        help="IOU threshold for merging overlapping boxes")
    parser.add_argument("--fps", type=float, default=DEFAULT_TARGET_FPS,
                        help="Target frame rate for video extraction")
    parser.add_argument("--no-vis", action="store_true", help="Disable YOLO visualization frames")
    parser.add_argument("--keep-frames", action="store_true", help="Keep existing extracted frames")
    parser.add_argument("--enable-dynamic-filter", action="store_true",
                        dest="enable_dynamic_filter", default=DEFAULT_ENABLE_DYNAMIC_FILTER,
                        help="Enable dynamic object filtering")
    parser.add_argument("--disable-dynamic-filter", action="store_false",
                        dest="enable_dynamic_filter",
                        help="Disable dynamic object filtering")
    parser.add_argument("--compensate-global-motion", action="store_true",
                        dest="compensate_global_motion", default=DEFAULT_COMPENSATE_GLOBAL_MOTION,
                        help="Enable global motion compensation for dynamic filtering")
    parser.add_argument("--no-compensate-global-motion", action="store_false",
                        dest="compensate_global_motion",
                        help="Disable global motion compensation for dynamic filtering")
    parser.add_argument("--tau-motion", type=float, default=DEFAULT_TAU_MOTION,
                        help="Motion threshold for dynamic filtering")
    parser.add_argument("--motion-percentile", type=float, default=DEFAULT_MOTION_PERCENTILE,
                        help="Motion percentile for dynamic filtering")
    return parser.parse_args()


def main(args: argparse.Namespace | None = None):
    if args is None:
        args = parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output)
    vis_dir = None if args.no_vis else out_dir / "yolo_vis"
    out_dir.mkdir(parents=True, exist_ok=True)

    target_classes = DYNAMIC_CLASSES

    print(f"Target classes : {sorted(target_classes)}")

    frames = resolve_frames(input_path, out_dir, fps=args.fps, keep_frames=args.keep_frames)

    print(f"\nLoading YOLO model: {args.weights}")
    model = YOLO(args.weights)

    print(
        f"\nRunning detection (conf={args.conf}, iou={args.iou}, "
        f"dynamic_filter={args.enable_dynamic_filter}, "
        f"compensate_global_motion={args.compensate_global_motion}, "
        f"tau_motion={args.tau_motion}) …"
    )

    prompts = run_detection(
        frames,
        model,
        target_classes,
        args.conf,
        args.iou,
        vis_dir,
        enable_dynamic_filter=args.enable_dynamic_filter,
        compensate_global_motion=args.compensate_global_motion,
        tau_motion=args.tau_motion,
        motion_percentile=args.motion_percentile,
    )

    before = len(prompts)
    prompts = merge_overlapping_boxes(prompts, iou_threshold=args.merge_iou)
    print(f"  → Merged duplicates: {before} → {len(prompts)} prompts")

    prompts_path = out_dir / "prompts.json"
    with open(prompts_path, "w") as f:
        json.dump(prompts, f, indent=2)
    print(f"\nSaved prompts  → {prompts_path}")

    summary = build_summary(prompts, len(frames))
    summary_path = out_dir / "summary.txt"
    summary_path.write_text(summary)
    print(summary)

    if vis_dir:
        print(f"Saved vis frames → {vis_dir}")


if __name__ == "__main__":
    main()
