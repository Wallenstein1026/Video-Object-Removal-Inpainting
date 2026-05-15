import os
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Config: modify here only
# ============================================================

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])

# Option 1: convert one video
INPUT_VIDEO = f"{PROJECT_ROOT}/data/wild/swimming.mp4"

# Output sequence name. If None, use video filename without extension.
SEQUENCE_NAME = None

# DAVIS-style output root
DAVIS_ROOT = f"{PROJECT_ROOT}/data/wild"

# Resize setting
# DAVIS 480p usually means height = 480 and width is resized proportionally.
RESIZE_TO_HEIGHT = 480

# Frame sampling
# 1 = keep every frame
# 2 = keep every 2 frames
# 5 = keep every 5 frames
FRAME_STRIDE = 1

# Output JPEG quality
JPEG_QUALITY = 95

# Whether to create empty annotation masks
# For wild videos without GT, keep True if your code expects Annotations folder.
CREATE_EMPTY_MASKS = True

# Whether to overwrite existing extracted frames
OVERWRITE = True

# Whether to copy original video into data/wild/
COPY_VIDEO_TO_WILD = True

# ImageSets file name
IMAGESET_NAME = "swimming"


# ============================================================
# Utility functions
# ============================================================

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def resize_keep_aspect(frame, target_height: int):
    h, w = frame.shape[:2]

    if target_height is None or target_height <= 0:
        return frame

    if h == target_height:
        return frame

    scale = target_height / float(h)
    new_w = int(round(w * scale))

    # Keep width even. This helps later video encoding with ffmpeg/libx264.
    if new_w % 2 == 1:
        new_w += 1

    resized = cv2.resize(frame, (new_w, target_height), interpolation=cv2.INTER_AREA)
    return resized


def clear_folder(folder: str):
    folder = Path(folder)
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)


def write_imageset(davis_root: str, sequence_name: str, imageset_name: str):
    imageset_dir = Path(davis_root) / "ImageSets" / "2017"
    ensure_dir(str(imageset_dir))

    imageset_path = imageset_dir / imageset_name

    existing = []
    if imageset_path.exists():
        with open(imageset_path, "r", encoding="utf-8") as f:
            existing = [line.strip() for line in f.readlines() if line.strip()]

    if sequence_name not in existing:
        existing.append(sequence_name)

    with open(imageset_path, "w", encoding="utf-8") as f:
        for name in existing:
            f.write(name + "\n")

    return imageset_path


def save_metadata(
    out_seq_dir: str,
    input_video: str,
    sequence_name: str,
    original_fps: float,
    saved_fps: float,
    total_input_frames: int,
    total_saved_frames: int,
    frame_stride: int,
    resize_to_height: int,
):
    meta = {
        "input_video": input_video,
        "sequence_name": sequence_name,
        "original_fps": original_fps,
        "saved_fps": saved_fps,
        "total_input_frames": total_input_frames,
        "total_saved_frames": total_saved_frames,
        "frame_stride": frame_stride,
        "resize_to_height": resize_to_height,
    }

    meta_path = Path(out_seq_dir) / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta_path


# ============================================================
# Main conversion
# ============================================================

def convert_video_to_davis():
    input_video = Path(INPUT_VIDEO)

    if not input_video.exists():
        raise FileNotFoundError(f"Input video does not exist: {input_video}")

    sequence_name = SEQUENCE_NAME
    if sequence_name is None:
        sequence_name = input_video.stem

    jpeg_seq_dir = Path(DAVIS_ROOT) / "JPEGImages" / "480p" / sequence_name
    ann_seq_dir = Path(DAVIS_ROOT) / "Annotations" / "480p" / sequence_name

    if OVERWRITE:
        clear_folder(str(jpeg_seq_dir))
        if CREATE_EMPTY_MASKS:
            clear_folder(str(ann_seq_dir))
    else:
        ensure_dir(str(jpeg_seq_dir))
        if CREATE_EMPTY_MASKS:
            ensure_dir(str(ann_seq_dir))

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_video}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0 or np.isnan(original_fps):
        original_fps = 24.0

    total_input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved_fps = original_fps / FRAME_STRIDE

    input_idx = 0
    save_idx = 0

    print(f"[INFO] Input video: {input_video}")
    print(f"[INFO] Sequence name: {sequence_name}")
    print(f"[INFO] Output frames: {jpeg_seq_dir}")
    print(f"[INFO] Original FPS: {original_fps:.3f}")
    print(f"[INFO] Frame stride: {FRAME_STRIDE}")
    print(f"[INFO] Saved FPS: {saved_fps:.3f}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if input_idx % FRAME_STRIDE != 0:
            input_idx += 1
            continue

        frame = resize_keep_aspect(frame, RESIZE_TO_HEIGHT)

        frame_name = f"{save_idx:05d}"
        jpg_path = jpeg_seq_dir / f"{frame_name}.jpg"

        cv2.imwrite(
            str(jpg_path),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
        )

        if CREATE_EMPTY_MASKS:
            h, w = frame.shape[:2]
            empty_mask = np.zeros((h, w), dtype=np.uint8)
            mask_path = ann_seq_dir / f"{frame_name}.png"
            cv2.imwrite(str(mask_path), empty_mask)

        save_idx += 1
        input_idx += 1

    cap.release()

    imageset_path = write_imageset(DAVIS_ROOT, sequence_name, IMAGESET_NAME)

    meta_path = save_metadata(
        out_seq_dir=str(jpeg_seq_dir),
        input_video=str(input_video),
        sequence_name=sequence_name,
        original_fps=original_fps,
        saved_fps=saved_fps,
        total_input_frames=total_input_frames,
        total_saved_frames=save_idx,
        frame_stride=FRAME_STRIDE,
        resize_to_height=RESIZE_TO_HEIGHT,
    )

    if COPY_VIDEO_TO_WILD:
        wild_dir = Path(PROJECT_ROOT) / "data" / "wild"
        ensure_dir(str(wild_dir))
        dst_video = wild_dir / input_video.name
        if input_video.resolve() != dst_video.resolve():
            shutil.copy2(input_video, dst_video)

    print("[DONE] Video converted to DAVIS-style format.")
    print(f"[DONE] Frames saved to: {jpeg_seq_dir}")
    if CREATE_EMPTY_MASKS:
        print(f"[DONE] Empty masks saved to: {ann_seq_dir}")
    print(f"[DONE] ImageSet saved to: {imageset_path}")
    print(f"[DONE] Metadata saved to: {meta_path}")
    print(f"[DONE] Total saved frames: {save_idx}")


if __name__ == "__main__":
    convert_video_to_davis()
