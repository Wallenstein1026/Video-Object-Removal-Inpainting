import sys
import csv
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.io_utils import ensure_dir, load_input_frames, save_mask, save_image, overlay_mask


# ---------------------------------------------------------------------------
# Config — 修改这里
# ---------------------------------------------------------------------------
PROJ_ROOT = PROJECT_ROOT
INPUT = PROJECT_ROOT / "data" / "DAVIS" / "JPEGImages" / "480p" / "boxing-fisheye"
VIDEO_NAME = "boxing-fisheye"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "part1_masks"

YOLO_WEIGHTS = "yolov8s-seg.pt"
DEVICE = "0"
CLASSES = None
CONF_THRES = 0.25
IOU_THRES = 0.60

TAU_MOTION = 2.0
MIN_MASK_AREA = 150
MIN_TRACK_POINTS = 8
MAX_TRACK_POINTS = 80
WEAK_PATIENCE = 2
MATCH_IOU_THRES = 0.20

DILATE_KERNEL = 9
CLOSE_KERNEL = 7
MIN_COMPONENT_AREA = 120

COMPENSATE_GLOBAL_MOTION = False
USE_MOTION_FALLBACK = False
MOTION_FALLBACK_THR = 1.5


def default_config():
    return SimpleNamespace(
        proj_root=PROJ_ROOT,
        input=INPUT,
        video_name=VIDEO_NAME,
        output_root=OUTPUT_ROOT,
        yolo_weights=YOLO_WEIGHTS,
        device=DEVICE,
        classes=CLASSES,
        conf_thres=CONF_THRES,
        iou_thres=IOU_THRES,
        tau_motion=TAU_MOTION,
        min_mask_area=MIN_MASK_AREA,
        min_track_points=MIN_TRACK_POINTS,
        max_track_points=MAX_TRACK_POINTS,
        weak_patience=WEAK_PATIENCE,
        match_iou_thres=MATCH_IOU_THRES,
        dilate_kernel=DILATE_KERNEL,
        close_kernel=CLOSE_KERNEL,
        min_component_area=MIN_COMPONENT_AREA,
        compensate_global_motion=COMPENSATE_GLOBAL_MOTION,
        use_motion_fallback=USE_MOTION_FALLBACK,
        motion_fallback_thr=MOTION_FALLBACK_THR,
    )


def mask_iou(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 0.0 if union == 0 else inter / union


def remove_small_components(mask, min_area=100):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask, dtype=np.uint8)

    for lab in range(1, num_labels):
        area = stats[lab, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == lab] = 1
    return out


def postprocess_mask(mask, close_kernel=7, dilate_kernel=9, min_component_area=100):
    mask = (mask > 0).astype(np.uint8)

    if close_kernel > 1:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

    if dilate_kernel > 1:
        k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
        mask = cv2.dilate(mask, k_dilate, iterations=1)

    mask = remove_small_components(mask, min_area=min_component_area)
    return mask.astype(np.uint8)


def detect_instances(model, frame_bgr, allowed_classes, conf_thres, iou_thres, device):
    result = model.predict(
        source=frame_bgr,
        conf=conf_thres,
        iou=iou_thres,
        verbose=False,
        device=device
    )[0]

    if result.boxes is None or result.masks is None:
        return []

    h, w = frame_bgr.shape[:2]
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    masks = result.masks.data.cpu().numpy()
    names_map = result.names

    if allowed_classes is not None:
        allowed_classes = {c.lower() for c in allowed_classes}

    outputs = []
    for cls_id, conf, box, mask in zip(cls_ids, confs, boxes, masks):
        class_name = str(names_map[int(cls_id)]).lower()

        if allowed_classes is not None and class_name not in allowed_classes:
            continue

        mask = (mask > 0.5).astype(np.uint8)
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0).astype(np.uint8)

        outputs.append({
            "label": class_name,
            "conf": float(conf),
            "box": box,
            "mask": mask
        })

    return outputs


def estimate_global_motion(source_gray, target_gray, max_corners=400):
    pts = cv2.goodFeaturesToTrack(
        source_gray,
        maxCorners=max_corners,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7
    )
    if pts is None or len(pts) < 8:
        return np.zeros(2, dtype=np.float32)

    nxt, st, _ = cv2.calcOpticalFlowPyrLK(source_gray, target_gray, pts, None)
    if nxt is None:
        return np.zeros(2, dtype=np.float32)

    pts0 = pts[st == 1].reshape(-1, 2)
    pts1 = nxt[st == 1].reshape(-1, 2)
    if len(pts0) < 8:
        return np.zeros(2, dtype=np.float32)

    flow = pts1 - pts0
    return np.median(flow, axis=0).astype(np.float32)


def compute_instance_motion(source_gray, target_gray, mask, max_track_points=80, global_motion=None):
    mask_u8 = (mask > 0).astype(np.uint8) * 255

    pts = cv2.goodFeaturesToTrack(
        source_gray,
        maxCorners=max_track_points,
        qualityLevel=0.01,
        minDistance=4,
        mask=mask_u8,
        blockSize=5
    )

    if pts is None or len(pts) == 0:
        return 0.0, 0

    nxt, st, _ = cv2.calcOpticalFlowPyrLK(source_gray, target_gray, pts, None)
    if nxt is None:
        return 0.0, 0

    pts0 = pts[st == 1].reshape(-1, 2)
    pts1 = nxt[st == 1].reshape(-1, 2)
    if len(pts0) == 0:
        return 0.0, 0

    flow = pts1 - pts0
    if global_motion is not None:
        flow = flow - global_motion[None, :]

    mag = np.linalg.norm(flow, axis=1)
    return float(mag.mean()), int(len(mag))


def dense_motion_fallback(source_gray, target_gray, mag_thr=1.5):
    flow = cv2.calcOpticalFlowFarneback(
        source_gray, target_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    mask = (mag > mag_thr).astype(np.uint8)

    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2)
    return mask


def match_previous_state(curr_label, curr_mask, prev_states, used_prev, iou_thres=0.2):
    best_idx = -1
    best_iou = 0.0

    for i, state in enumerate(prev_states):
        if i in used_prev:
            continue
        if state["label"] != curr_label:
            continue
        iou = mask_iou(curr_mask, state["mask"])
        if iou > best_iou:
            best_iou = iou
            best_idx = i

    if best_idx >= 0 and best_iou >= iou_thres:
        used_prev.add(best_idx)
        return prev_states[best_idx], best_iou

    return None, 0.0


def main(args=None):
    if args is None:
        args = default_config()
    proj_root = Path(args.proj_root)

    frames, names, fps, inferred_video_name = load_input_frames(
        args.input,
        processed_root=proj_root / "data" / "processed"
    )
    video_name = args.video_name if args.video_name else inferred_video_name

    out_dir = Path(args.output_root) / f"{video_name}_yolov8seg"
    mask_dir = out_dir / "masks_union"
    overlay_dir = out_dir / "masks_vis"
    stats_csv = out_dir / "motion_stats.csv"

    ensure_dir(mask_dir)
    ensure_dir(overlay_dir)

    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    model = YOLO(args.yolo_weights)

    prev_states = []
    csv_rows = []

    if len(frames) == 1:
        empty = np.zeros(gray_frames[0].shape, dtype=np.uint8)
        save_mask(empty, mask_dir / f"{names[0]}.png")
        save_image(overlay_mask(frames[0], empty), overlay_dir / f"{names[0]}.png")
        print("Only one frame found. Saved empty mask.")
        return

    for t in tqdm(range(len(frames)), desc=f"[Part1-Mask] {video_name}"):
        frame = frames[t]
        curr_gray = gray_frames[t]

        if t == 0:
            neighbor_idx = 1
        else:
            neighbor_idx = t - 1

        neighbor_gray = gray_frames[neighbor_idx]

        global_motion = None
        if args.compensate_global_motion:
            global_motion = estimate_global_motion(curr_gray, neighbor_gray)

        detections = detect_instances(
            model=model,
            frame_bgr=frame,
            allowed_classes=args.classes,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            device=args.device
        )

        union_mask = np.zeros(curr_gray.shape, dtype=np.uint8)
        used_prev = set()
        curr_states = []

        for det_idx, det in enumerate(detections):
            inst_mask = det["mask"].astype(np.uint8)
            area = int(inst_mask.sum())
            if area < args.min_mask_area:
                csv_rows.append({
                    "frame": t,
                    "name": names[t],
                    "det_id": det_idx,
                    "label": det["label"],
                    "conf": f"{det['conf']:.4f}",
                    "area": area,
                    "track_points": 0,
                    "mean_motion": "0.0000",
                    "matched_prev_iou": "0.0000",
                    "is_dynamic": 0,
                    "reason": "mask_too_small"
                })
                continue

            mean_motion, n_points = compute_instance_motion(
                source_gray=curr_gray,
                target_gray=neighbor_gray,
                mask=inst_mask,
                max_track_points=args.max_track_points,
                global_motion=global_motion
            )

            prev_state, prev_iou = match_previous_state(
                curr_label=det["label"],
                curr_mask=inst_mask,
                prev_states=prev_states,
                used_prev=used_prev,
                iou_thres=args.match_iou_thres
            )

            dynamic_now = (n_points >= args.min_track_points) and (mean_motion > args.tau_motion)

            prev_dynamic = prev_state["is_dynamic"] if prev_state is not None else False
            prev_weak = prev_state["weak_count"] if prev_state is not None else 0

            if dynamic_now:
                weak_count = 0
                is_dynamic = True
                reason = "dynamic_motion"
            else:
                weak_count = prev_weak + 1 if prev_state is not None else 1
                if prev_dynamic and weak_count < args.weak_patience:
                    is_dynamic = True
                    reason = "temporal_keep"
                else:
                    is_dynamic = False
                    reason = "weak_motion"

            if n_points < args.min_track_points and not (prev_dynamic and weak_count < args.weak_patience):
                is_dynamic = False
                reason = "too_few_track_points"

            if is_dynamic:
                union_mask = np.maximum(union_mask, inst_mask)

            curr_states.append({
                "label": det["label"],
                "mask": inst_mask,
                "is_dynamic": is_dynamic,
                "weak_count": weak_count
            })

            csv_rows.append({
                "frame": t,
                "name": names[t],
                "det_id": det_idx,
                "label": det["label"],
                "conf": f"{det['conf']:.4f}",
                "area": area,
                "track_points": n_points,
                "mean_motion": f"{mean_motion:.4f}",
                "matched_prev_iou": f"{prev_iou:.4f}",
                "is_dynamic": int(is_dynamic),
                "reason": reason
            })

        if union_mask.sum() == 0 and args.use_motion_fallback:
            motion_mask = dense_motion_fallback(
                source_gray=curr_gray,
                target_gray=neighbor_gray,
                mag_thr=args.motion_fallback_thr
            )
            motion_mask = remove_small_components(
                motion_mask,
                min_area=args.min_component_area
            )
            union_mask = np.maximum(union_mask, motion_mask)

        final_mask = postprocess_mask(
            union_mask,
            close_kernel=args.close_kernel,
            dilate_kernel=args.dilate_kernel,
            min_component_area=args.min_component_area
        )

        save_mask(final_mask, mask_dir / f"{names[t]}.png")
        save_image(overlay_mask(frame, final_mask), overlay_dir / f"{names[t]}.png")

        prev_states = curr_states

    with open(stats_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame", "name", "det_id", "label", "conf", "area",
            "track_points", "mean_motion", "matched_prev_iou",
            "is_dynamic", "reason"
        ])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Saved masks to      : {mask_dir}")
    print(f"Saved overlays to   : {overlay_dir}")
    print(f"Saved motion stats  : {stats_csv}")


if __name__ == "__main__":
    main()
