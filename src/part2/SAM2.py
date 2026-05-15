"""
02_sam2_track.py
----------------
Step 2 of Part 2 pipeline: Load YOLO prompts.json -> assign object IDs via
a lightweight IoU tracker -> feed box+point prompts into SAM 2 video predictor ->
propagate masks through the entire video -> save results for ProPainter.

Improvements over the original:
1) Remove duplicated SAM 2 initialization
2) Prefer SAM 2.1 assets automatically if available, else fallback to SAM 2
3) Choose the BEST prompt frame per track (highest conf, tie-break by larger area)
4) Feed both box + positive center point to SAM 2
5) Forward/backward propagation are fused by mask quality score
6) Save union mask for EVERY frame (blank mask for frames without objects)
7) Optionally enable VOS-optimized predictor when supported
"""

import argparse
import json
import os
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor


# ---------------------------------------------------------------------------
# ⚙️ Config — modify here
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Step 1 output
PROMPTS_JSON = str(PROJECT_ROOT / "outputs" / "part2_masks" / "swimming_yolo" / "prompts.json")

# Video frames
FRAMES_DIR = str(PROJECT_ROOT / "data" / "wild" / "JPEGImages" / "480p" / "swimming")

# SAM 2 repo/checkpoints
SAM2_REPO_DIR = str(PROJECT_ROOT / "repos" / "sam2")

# Auto-prefer SAM 2.1 if files exist; otherwise fallback to SAM 2 legacy
SAM2_CFG_CANDIDATES = [
    "configs/sam2.1/sam2.1_hiera_l.yaml",
    "configs/sam2/sam2_hiera_l.yaml",
]
SAM2_CHECKPOINT_CANDIDATES = [
    str(PROJECT_ROOT / "repos" / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt"),
    str(PROJECT_ROOT / "repos" / "sam2" / "checkpoints" / "sam2_hiera_large.pt"),
]

OUTPUT_DIR = str(PROJECT_ROOT / "outputs" / "part2_masks" / "swimming_sam2")

# Tracker params
IOU_MATCH_THRESH = 0.3
MAX_MISS_FRAMES = 5

# SAM 2 params
MASK_THRESHOLD = 0.0
PROPAGATE_BACKWARD = True
USE_BOX_PROMPT = True
USE_POINT_PROMPT = True
VOS_OPTIMIZED = False   # if your installed sam2 version supports it

# Visualisation
SAVE_VIS = True
VIS_ALPHA = 0.5

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

@contextmanager
def temporary_cwd(path: str):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def maybe_autocast(device: str):
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def box_area(box_xyxy: list[float]) -> float:
    x1, y1, x2, y2 = box_xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_center(box_xyxy: list[float]) -> list[float]:
    x1, y1, x2, y2 = box_xyxy
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


def resolve_sam2_assets(repo_dir: str = SAM2_REPO_DIR):
    """
    Prefer SAM 2.1 (cfg + checkpoint) if both exist; otherwise fallback to SAM 2 legacy.

    Important:
    - build_sam2_video_predictor expects model_cfg like "configs/sam2.1/xxx.yaml"
    - but the physical file inside the repo lives under "sam2/configs/..."
    """
    candidates = list(zip(SAM2_CFG_CANDIDATES, SAM2_CHECKPOINT_CANDIDATES))

    tried = []
    for cfg_rel, ckpt_path in candidates:
        cfg_abs = Path(repo_dir) / "sam2" / cfg_rel
        ckpt_abs = Path(ckpt_path)
        tried.append(f"cfg={cfg_abs} | ckpt={ckpt_abs}")

        if cfg_abs.exists() and ckpt_abs.exists():
            return cfg_rel, str(ckpt_abs)

    raise FileNotFoundError(
        "Cannot find a valid SAM2 config/checkpoint pair.\nTried:\n  - " + "\n  - ".join(tried)
    )


def build_predictor(repo_dir: str, model_cfg_rel: str, checkpoint: str, device: str, vos_optimized: bool):
    """
    Build predictor inside the SAM2 repo dir to avoid Hydra config path issues.
    Supports old/new sam2 signatures.
    """
    with temporary_cwd(repo_dir):
        try:
            predictor = build_sam2_video_predictor(
                model_cfg_rel,
                checkpoint,
                device=device,
                vos_optimized=vos_optimized,
            )
        except TypeError:
            predictor = build_sam2_video_predictor(
                model_cfg_rel,
                checkpoint,
                device=device,
            )
    return predictor


# ---------------------------------------------------------------------------
# Tiny IoU Tracker with best-prompt selection
# ---------------------------------------------------------------------------

class Track:
    _next_id = 1

    def __init__(self, det: dict):
        self.id = Track._next_id
        Track._next_id += 1

        self.cls = det["cls"]
        self.first_frame = det["frame_idx"]
        self.last_frame = det["frame_idx"]
        self.last_box = det["box_xyxy"]

        self.detections = [det]
        self.best_det = det

    def update(self, det: dict):
        self.last_box = det["box_xyxy"]
        self.last_frame = det["frame_idx"]
        self.detections.append(det)

        if self._is_better_prompt(det, self.best_det):
            self.best_det = det

    @staticmethod
    def _is_better_prompt(det_a: dict, det_b: dict) -> bool:
        conf_a = float(det_a.get("conf", 0.0))
        conf_b = float(det_b.get("conf", 0.0))
        if conf_a != conf_b:
            return conf_a > conf_b

        area_a = box_area(det_a["box_xyxy"])
        area_b = box_area(det_b["box_xyxy"])
        return area_a > area_b

    @property
    def prompt_frame(self) -> int:
        return self.best_det["frame_idx"]

    @property
    def prompt_box(self) -> list[float]:
        return self.best_det["box_xyxy"]

    @property
    def prompt_point(self) -> list[float]:
        return self.best_det.get("point", box_center(self.best_det["box_xyxy"]))


def box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def run_iou_tracker(
    prompts: list[dict],
    total_frames: int,
    iou_thresh: float = IOU_MATCH_THRESH,
    max_miss: int = MAX_MISS_FRAMES,
) -> list[Track]:
    Track._next_id = 1

    by_frame: dict[int, list[dict]] = defaultdict(list)
    for p in prompts:
        by_frame[int(p["frame_idx"])].append(p)

    active_tracks: list[Track] = []
    all_tracks: list[Track] = []

    for fi in range(total_frames):
        dets = by_frame.get(fi, [])

        matched_track_ids = set()
        matched_det_idxs = set()

        scores = []
        for ti, trk in enumerate(active_tracks):
            for di, det in enumerate(dets):
                if det["cls"] != trk.cls:
                    continue
                iou = box_iou(trk.last_box, det["box_xyxy"])
                if iou >= iou_thresh:
                    scores.append((iou, ti, di))

        scores.sort(reverse=True)
        for _, ti, di in scores:
            if ti in matched_track_ids or di in matched_det_idxs:
                continue
            active_tracks[ti].update(dets[di])
            matched_track_ids.add(ti)
            matched_det_idxs.add(di)

        for di, det in enumerate(dets):
            if di not in matched_det_idxs:
                trk = Track(det)
                active_tracks.append(trk)
                all_tracks.append(trk)

        still_active = []
        for trk in active_tracks:
            if fi - trk.last_frame <= max_miss:
                still_active.append(trk)
        active_tracks = still_active

    for trk in active_tracks:
        if trk not in all_tracks:
            all_tracks.append(trk)

    print(f"  Tracker: {len(all_tracks)} unique tracks found")
    for trk in all_tracks:
        print(
            f"    obj_{trk.id:03d}  cls={trk.cls:<12s}  "
            f"first_frame={trk.first_frame:<3d}  last_frame={trk.last_frame:<3d}  "
            f"prompt_frame={trk.prompt_frame:<3d}  dets={len(trk.detections):<3d}  "
            f"best_conf={trk.best_det.get('conf', 0.0):.3f}"
        )
    return all_tracks


# ---------------------------------------------------------------------------
# Colour palette for visualisation
# ---------------------------------------------------------------------------

PALETTE = [
    (255,  82,  82), (255, 165,   0), ( 57, 255,  20),
    (  0, 191, 255), (148,   0, 211), (255, 20, 147),
    (  0, 255, 127), (255, 215,   0), (100, 149, 237),
    (255, 127,  80),
]


def obj_colour(obj_id: int) -> tuple[int, int, int]:
    return PALETTE[(obj_id - 1) % len(PALETTE)]


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------


def save_mask(mask: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (mask > 0).astype(np.uint8) * 255)


def save_union_mask(masks_by_obj: dict[int, np.ndarray], path: Path, hw: tuple[int, int]):
    h, w = hw
    union = np.zeros((h, w), dtype=bool)
    for m in masks_by_obj.values():
        union |= m.astype(bool)
    save_mask(union, path)


def save_vis(frame_bgr: np.ndarray, masks_by_obj: dict[int, np.ndarray], path: Path, alpha: float = VIS_ALPHA):
    vis = frame_bgr.copy().astype(np.float32)
    for obj_id, mask in masks_by_obj.items():
        colour = np.array(obj_colour(obj_id), dtype=np.float32)[::-1]
        vis[mask > 0] = vis[mask > 0] * (1 - alpha) + colour * alpha
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), vis.astype(np.uint8))


def logits_to_mask_and_score(mask_logits_2d: torch.Tensor, mask_threshold: float):
    logits_np = mask_logits_2d.detach().float().cpu().numpy()
    mask = logits_np > mask_threshold

    if mask.any():
        pos_mean = float(logits_np[mask].mean())
        area_bonus = 1e-4 * float(mask.sum()) ** 0.5
        score = pos_mean + area_bonus
    else:
        score = float(logits_np.max()) if logits_np.size > 0 else -1e9

    return mask.astype(bool), score


def update_mask_candidate(frame_masks, frame_scores, frame_idx, obj_id, mask, score):
    old_score = frame_scores[frame_idx].get(obj_id, -1e18)
    if score > old_score:
        frame_masks[frame_idx][obj_id] = mask
        frame_scores[frame_idx][obj_id] = score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Part 2 SAM2 mask propagation for DAVIS frames")
    parser.add_argument("--prompts", type=str, default=PROMPTS_JSON,
                        help=f"YOLO prompts.json path (default: {PROMPTS_JSON})")
    parser.add_argument("--frames-dir", type=str, default=FRAMES_DIR,
                        help=f"Frames folder path (default: {FRAMES_DIR})")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR,
                        help=f"Output folder for SAM2 masks (default: {OUTPUT_DIR})")
    parser.add_argument("--sam2-repo-dir", type=str, default=SAM2_REPO_DIR,
                        help=f"SAM2 repo root path (default: {SAM2_REPO_DIR})")
    parser.add_argument("--mask-threshold", type=float, default=MASK_THRESHOLD,
                        help="SAM2 mask threshold")
    parser.add_argument("--use-box-prompt", action="store_true", default=USE_BOX_PROMPT,
                        help="Enable box prompts")
    parser.add_argument("--no-box-prompt", action="store_false", dest="use_box_prompt",
                        help="Disable box prompts")
    parser.add_argument("--use-point-prompt", action="store_true", default=USE_POINT_PROMPT,
                        help="Enable point prompts")
    parser.add_argument("--no-point-prompt", action="store_false", dest="use_point_prompt",
                        help="Disable point prompts")
    parser.add_argument("--propagate-backward", action="store_true", dest="propagate_backward",
                        default=PROPAGATE_BACKWARD, help="Enable backward propagation")
    parser.add_argument("--no-propagate-backward", action="store_false", dest="propagate_backward",
                        help="Disable backward propagation")
    parser.add_argument("--vos-optimized", action="store_true", dest="vos_optimized",
                        default=VOS_OPTIMIZED, help="Enable VOS-optimized SAM2 predictor")
    parser.add_argument("--no-vis", action="store_true", help="Disable visualizations")
    return parser.parse_args()


def main(args: argparse.Namespace | None = None):
    if args is None:
        args = parse_args()

    prompts_path = Path(args.prompts)
    frames_dir = Path(args.frames_dir)
    out_dir = Path(args.output)

    dir_per_obj = out_dir / "masks_per_obj"
    dir_union = out_dir / "masks_union"
    dir_vis = out_dir / "masks_vis"

    for d in [dir_per_obj, dir_union, dir_vis]:
        d.mkdir(parents=True, exist_ok=True)

    assert prompts_path.exists(), f"prompts.json not found: {prompts_path}"
    assert frames_dir.exists(), f"Frames dir not found: {frames_dir}"

    with open(prompts_path) as f:
        prompts: list[dict] = json.load(f)

    frames = sorted(
        p for p in frames_dir.iterdir()
        if p.suffix.lower() in IMAGE_SUFFIXES
    )
    total_frames = len(frames)
    print(f"Frames: {total_frames}  |  Prompts: {len(prompts)}")

    if total_frames == 0:
        raise RuntimeError(f"No frames found in {frames_dir}")

    first_frame = cv2.imread(str(frames[0]))
    if first_frame is None:
        raise RuntimeError(f"Cannot read first frame: {frames[0]}")
    frame_hw = first_frame.shape[:2]

    save_vis_enabled = not args.no_vis

    if not prompts:
        print("[warn] No prompts found — saving blank masks only.")
        for frame_path in frames:
            stem = frame_path.stem
            save_union_mask({}, dir_union / f"{stem}.png", frame_hw)
            if save_vis_enabled:
                frame_bgr = cv2.imread(str(frame_path))
                if frame_bgr is not None:
                    save_vis(frame_bgr, {}, dir_vis / frame_path.name)
        return

    print("\n[1/5] Running IoU tracker …")
    tracks = run_iou_tracker(prompts, total_frames)

    print("\n[2/5] Resolving SAM 2 assets …")
    model_cfg_rel, sam2_checkpoint = resolve_sam2_assets(args.sam2_repo_dir)
    print(f"  Using config     : {model_cfg_rel}")
    print(f"  Using checkpoint : {sam2_checkpoint}")

    print("\n[3/5] Initialising SAM 2 …")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    predictor = build_predictor(
        repo_dir=args.sam2_repo_dir,
        model_cfg_rel=model_cfg_rel,
        checkpoint=sam2_checkpoint,
        device=device,
        vos_optimized=args.vos_optimized,
    )

    frame_masks: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    frame_scores: dict[int, dict[int, float]] = defaultdict(dict)

    print("\n[4/5] Adding prompts to SAM 2 …")
    with torch.inference_mode(), maybe_autocast(device):
        inference_state = predictor.init_state(video_path=str(frames_dir))
        predictor.reset_state(inference_state)

        for trk in tracks:
            kwargs = {
                "inference_state": inference_state,
                "frame_idx": trk.prompt_frame,
                "obj_id": trk.id,
            }
            if args.use_box_prompt:
                kwargs["box"] = np.array(trk.prompt_box, dtype=np.float32)

            if args.use_point_prompt:
                point = np.array([trk.prompt_point], dtype=np.float32)
                labels = np.array([1], dtype=np.int32)
                kwargs["points"] = point
                kwargs["labels"] = labels

            predictor.add_new_points_or_box(**kwargs)

            print(
                f"  Added obj_{trk.id:03d} ({trk.cls}) "
                f"@ frame {trk.prompt_frame}  "
                f"conf={trk.best_det.get('conf', 0.0):.3f}  "
                f"box={[round(v, 1) for v in trk.prompt_box]}  "
                f"point={[round(v, 1) for v in trk.prompt_point]}"
            )

        print("\n[5/5] Propagating forward …")
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
            for oi, obj_id in enumerate(obj_ids):
                mask, score = logits_to_mask_and_score(mask_logits[oi, 0], args.mask_threshold)
                update_mask_candidate(frame_masks, frame_scores, frame_idx, obj_id, mask, score)

        if args.propagate_backward:
            print("      Propagating backward and fusing …")
            for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(
                inference_state, reverse=True
            ):
                for oi, obj_id in enumerate(obj_ids):
                    mask, score = logits_to_mask_and_score(mask_logits[oi, 0], args.mask_threshold)
                    update_mask_candidate(frame_masks, frame_scores, frame_idx, obj_id, mask, score)

    print("\n[6/5] Saving masks …")
    for frame_idx, frame_path in enumerate(frames):
        stem = frame_path.stem
        masks_this_frame = frame_masks.get(frame_idx, {})
        save_union_mask(masks_this_frame, dir_union / f"{stem}.png", frame_hw)
        if save_vis_enabled:
            frame_bgr = cv2.imread(str(frame_path))
            if frame_bgr is not None:
                save_vis(frame_bgr, masks_this_frame, dir_vis / frame_path.name)

    print(f"Saved union masks -> {dir_union}")
    if save_vis_enabled:
        print(f"Saved visualizations -> {dir_vis}")


if __name__ == "__main__":
    main()
