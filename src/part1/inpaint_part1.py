import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.io_utils import (
    ensure_dir,
    load_input_frames,
    save_image,
    save_video,
    overlay_mask,
)


# ---------------------------------------------------------------------------
# Config — 修改这里
# ---------------------------------------------------------------------------
PROJ_ROOT = PROJECT_ROOT
INPUT = PROJECT_ROOT / "data" / "DAVIS" / "JPEGImages" / "480p" / "boxing-fisheye"
VIDEO_NAME = "boxing-fisheye"
MASKS_ROOT = PROJECT_ROOT / "outputs" / "part1_masks"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "part1_inpaint"
FIGURES_ROOT = PROJECT_ROOT / "outputs" / "figures"

FPS = 15.0
SEARCH_RADIUS = 1
INPAINT_METHOD = "telea"
INPAINT_RADIUS = 3
SAVE_SPATIAL_ONLY = False
NUM_KEYFRAMES = 4


def default_config():
    return SimpleNamespace(
        proj_root=PROJ_ROOT,
        input=INPUT,
        video_name=VIDEO_NAME,
        masks_root=MASKS_ROOT,
        output_root=OUTPUT_ROOT,
        figures_root=FIGURES_ROOT,
        fps=FPS,
        search_radius=SEARCH_RADIUS,
        inpaint_method=INPAINT_METHOD,
        inpaint_radius=INPAINT_RADIUS,
        save_spatial_only=SAVE_SPATIAL_ONLY,
        num_keyframes=NUM_KEYFRAMES,
    )


def load_masks(mask_dir, names, shape_hw):
    h, w = shape_hw
    masks = []

    for name in names:
        p = Path(mask_dir) / f"{name}.png"
        if not p.exists():
            masks.append(np.zeros((h, w), dtype=np.uint8))
            continue

        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            masks.append(np.zeros((h, w), dtype=np.uint8))
            continue

        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        masks.append((m > 0).astype(np.uint8))

    return masks


def inpaint_one(frame, mask, method="telea", radius=3):
    if mask.sum() == 0:
        return frame.copy()

    flag = cv2.INPAINT_TELEA if method.lower() == "telea" else cv2.INPAINT_NS
    return cv2.inpaint(frame, (mask > 0).astype(np.uint8) * 255, radius, flag)


def temporal_borrow(frames, masks, t, search_radius=1):
    current = frames[t].copy()
    hole = masks[t].astype(bool).copy()

    if not hole.any():
        return current, np.zeros_like(masks[t], dtype=np.uint8)

    h, w = masks[t].shape
    false_map = np.zeros((h, w), dtype=bool)

    for d in range(1, search_radius + 1):
        prev_valid = false_map
        next_valid = false_map
        prev_img = None
        next_img = None

        if t - d >= 0:
            prev_valid = masks[t - d] == 0
            prev_img = frames[t - d]

        if t + d < len(frames):
            next_valid = masks[t + d] == 0
            next_img = frames[t + d]

        both = hole & prev_valid & next_valid
        only_prev = hole & prev_valid & (~next_valid)
        only_next = hole & next_valid & (~prev_valid)

        if both.any():
            avg_pixels = (
                prev_img[both].astype(np.float32) +
                next_img[both].astype(np.float32)
            ) / 2.0
            current[both] = avg_pixels.astype(np.uint8)
            hole[both] = False

        if only_prev.any():
            current[only_prev] = prev_img[only_prev]
            hole[only_prev] = False

        if only_next.any():
            current[only_next] = next_img[only_next]
            hole[only_next] = False

        if not hole.any():
            break

    remaining = hole.astype(np.uint8)
    return current, remaining


def make_keyframe_strip(frames, masks, spatial_frames, final_frames, out_path, num_keyframes=4):
    ensure_dir(Path(out_path).parent)

    n = len(frames)
    if n == 0:
        return

    if n <= num_keyframes:
        idxs = list(range(n))
    else:
        idxs = np.linspace(0, n - 1, num_keyframes, dtype=int).tolist()

    rows = []
    for idx in idxs:
        orig = frames[idx].copy()
        mask_vis = overlay_mask(frames[idx], masks[idx], color=(0, 0, 255), alpha=0.45)
        spatial = spatial_frames[idx].copy()
        final = final_frames[idx].copy()

        panels = [orig, mask_vis, spatial, final]
        labels = [
            f"orig [{idx:05d}]",
            "mask",
            "spatial_only",
            "temporal+spatial"
        ]

        labeled = []
        for img, text in zip(panels, labels):
            canvas = img.copy()
            cv2.putText(canvas, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 255, 255), 2, cv2.LINE_AA)
            labeled.append(canvas)

        row = np.concatenate(labeled, axis=1)
        rows.append(row)

    grid = np.concatenate(rows, axis=0)
    cv2.imwrite(str(out_path), grid)


def main(args=None):
    if args is None:
        args = default_config()
    proj_root = Path(args.proj_root)

    frames, names, input_fps, inferred_video_name = load_input_frames(
        args.input,
        processed_root=proj_root / "data" / "processed"
    )
    video_name = args.video_name if args.video_name else inferred_video_name
    fps = args.fps if args.fps > 0 else (input_fps if input_fps is not None else 15.0)

    h, w = frames[0].shape[:2]
    mask_dir = Path(args.masks_root) / f"{video_name}_yolov8seg" / "masks_union"
    out_dir = Path(args.output_root) / video_name
    final_frame_dir = out_dir / "frames"
    final_video_path = out_dir / f"{video_name}.mp4"

    spatial_frame_dir = out_dir / "frames_spatial_only"
    spatial_video_path = out_dir / f"{video_name}__spatial_only.mp4"

    figure_path = Path(args.figures_root) / f"part1_{video_name}_comparison.jpg"

    ensure_dir(final_frame_dir)
    ensure_dir(final_video_path.parent)
    ensure_dir(figure_path.parent)

    masks = load_masks(mask_dir, names, shape_hw=(h, w))

    spatial_only_frames = []
    final_frames = []

    for t in tqdm(range(len(frames)), desc=f"[Part1-Inpaint] {video_name}"):
        frame = frames[t]
        mask = masks[t]

        spatial = inpaint_one(
            frame=frame,
            mask=mask,
            method=args.inpaint_method,
            radius=args.inpaint_radius
        )
        spatial_only_frames.append(spatial)

        borrowed, remaining = temporal_borrow(
            frames=frames,
            masks=masks,
            t=t,
            search_radius=args.search_radius
        )
        final = inpaint_one(
            frame=borrowed,
            mask=remaining,
            method=args.inpaint_method,
            radius=args.inpaint_radius
        )
        final_frames.append(final)

    for name, img in zip(names, final_frames):
        save_image(img, final_frame_dir / f"{name}.png")

    save_video(final_frames, final_video_path, fps=fps)

    if args.save_spatial_only:
        ensure_dir(spatial_frame_dir)
        for name, img in zip(names, spatial_only_frames):
            save_image(img, spatial_frame_dir / f"{name}.png")
        save_video(spatial_only_frames, spatial_video_path, fps=fps)

    make_keyframe_strip(
        frames=frames,
        masks=masks,
        spatial_frames=spatial_only_frames,
        final_frames=final_frames,
        out_path=figure_path,
        num_keyframes=args.num_keyframes
    )

    print(f"Saved final inpaint frames : {final_frame_dir}")
    print(f"Saved final video          : {final_video_path}")
    if args.save_spatial_only:
        print(f"Saved spatial-only frames  : {spatial_frame_dir}")
        print(f"Saved spatial-only video   : {spatial_video_path}")
    print(f"Saved comparison figure    : {figure_path}")


if __name__ == "__main__":
    main()
