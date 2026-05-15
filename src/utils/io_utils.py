from pathlib import Path
import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def is_video_file(path):
    return Path(path).suffix.lower() in VIDEO_EXTS


def sorted_image_paths(folder):
    folder = Path(folder)
    paths = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    return sorted(paths)


def load_frames_from_dir(folder):
    folder = Path(folder)
    image_paths = sorted_image_paths(folder)
    if not image_paths:
        raise FileNotFoundError(f"No image frames found in: {folder}")

    frames, names = [], []
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            raise RuntimeError(f"Failed to read image: {p}")
        frames.append(img)
        names.append(p.stem)

    return frames, names, None, folder.name


def load_frames_from_video(video_path, save_frames_dir=None):
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 15.0

    frames, names = [], []
    idx = 0

    if save_frames_dir is not None:
        ensure_dir(save_frames_dir)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        name = f"{idx:05d}"
        frames.append(frame)
        names.append(name)
        if save_frames_dir is not None:
            cv2.imwrite(str(Path(save_frames_dir) / f"{name}.jpg"), frame)
        idx += 1

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames decoded from video: {video_path}")

    return frames, names, fps, video_path.stem


def load_input_frames(input_path, processed_root=None):
    input_path = Path(input_path)

    if input_path.is_dir():
        return load_frames_from_dir(input_path)

    if input_path.is_file() and is_video_file(input_path):
        save_dir = None
        if processed_root is not None:
            save_dir = Path(processed_root) / input_path.stem / "frames"
        return load_frames_from_video(input_path, save_frames_dir=save_dir)

    raise ValueError(f"Unsupported input path: {input_path}")


def save_video(frames, out_path, fps=15):
    if len(frames) == 0:
        raise ValueError("No frames to save.")

    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))

    for frame in frames:
        writer.write(frame)
    writer.release()


def save_mask(mask, out_path):
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    cv2.imwrite(str(out_path), mask_u8)


def save_image(img, out_path):
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    cv2.imwrite(str(out_path), img)


def overlay_mask(frame, mask, color=(0, 255, 0), alpha=0.4):
    vis = frame.copy()
    colored = np.zeros_like(frame)
    colored[:, :] = color
    blended = cv2.addWeighted(frame, 1 - alpha, colored, alpha, 0)
    vis[mask > 0] = blended[mask > 0]
    return vis