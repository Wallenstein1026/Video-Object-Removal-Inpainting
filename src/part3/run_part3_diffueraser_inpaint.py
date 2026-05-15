"""
Part 3: SAM2 masks -> DiffuEraser video inpainting.

Edit the Config section below before running. This script intentionally has no
command-line arguments so that all project paths and hyperparameters are kept in
one place for reproducible experiments.

DiffuEraser currently expects an input video and a mask video, so this wrapper
packs DAVIS frames and Part2 union masks into MP4 files, runs DiffuEraser, then
extracts the output video back to frame files with the original frame names.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VIDEO_NAME = "bear"

FRAMES_DIR = PROJECT_ROOT / "data" / "DAVIS" / "JPEGImages" / "480p" / VIDEO_NAME
MASK_DIR = PROJECT_ROOT / "outputs" / "part2_masks" / f"{VIDEO_NAME}_sam2" / "masks_union"
DIFFUERASER_ROOT = PROJECT_ROOT / "repos" / "DiffuEraser"
PYTHON_EXE = Path(os.environ.get("DIFFUERASER_PYTHON", sys.executable))
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "part3_diffueraser_inpaint" / VIDEO_NAME

FPS = 15.0
MAX_FRAMES: int | None = None

# DiffuEraser / ProPainter parameters passed through to run_diffueraser.py.
MASK_DILATION_ITER = 4
MAX_IMG_SIZE = 960
REF_STRIDE = 5
NEIGHBOR_LENGTH = 20
SUBVIDEO_LENGTH = 80

# DiffuEraser model paths. These are relative to DIFFUERASER_ROOT by default,
# matching the upstream repository layout.
BASE_MODEL_PATH = DIFFUERASER_ROOT / "weights" / "stable-diffusion-v1-5"
VAE_PATH = DIFFUERASER_ROOT / "weights" / "sd-vae-ft-mse"
DIFFUERASER_MODEL_PATH = DIFFUERASER_ROOT / "weights" / "diffuEraser"
PROPAINTER_MODEL_DIR = DIFFUERASER_ROOT / "weights" / "propainter"


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class RunPaths:
    work_dir: Path
    input_video: Path
    input_mask_video: Path
    diffueraser_save_dir: Path
    diffueraser_result_video: Path
    final_video_path: Path
    final_frames: Path
    summary_path: Path


def ensure_runtime_dependencies() -> None:
    required = ["cv2", "numpy"]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            "Missing runtime dependencies in the current Python environment: "
            + ", ".join(missing)
            + "\nRun this script with the DiffuEraser environment, or edit "
            + "PYTHON_EXE in the Config section."
        )


def import_runtime_modules() -> dict[str, Any]:
    import cv2
    import numpy as np

    return {"cv2": cv2, "np": np}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sorted_image_paths(folder: Path) -> list[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def build_image_map(folder: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted_image_paths(folder)}


def build_paths() -> RunPaths:
    return RunPaths(
        work_dir=OUTPUT_DIR / "_work",
        input_video=OUTPUT_DIR / "_work" / "input_video.mp4",
        input_mask_video=OUTPUT_DIR / "_work" / "input_mask.mp4",
        diffueraser_save_dir=OUTPUT_DIR / "diffueraser_raw",
        diffueraser_result_video=OUTPUT_DIR / "diffueraser_raw" / "diffueraser_result.mp4",
        final_video_path=OUTPUT_DIR / f"{VIDEO_NAME}_diffueraser.mp4",
        final_frames=OUTPUT_DIR / "final_frames",
        summary_path=OUTPUT_DIR / "summary.json",
    )


def validate_static_inputs() -> None:
    if not FRAMES_DIR.exists():
        raise FileNotFoundError(f"Frames directory not found: {FRAMES_DIR}")
    if not MASK_DIR.exists():
        raise FileNotFoundError(f"Part2 union mask directory not found: {MASK_DIR}")
    if not DIFFUERASER_ROOT.exists():
        raise FileNotFoundError(
            f"DiffuEraser repository not found: {DIFFUERASER_ROOT}\n"
            "Download it with:\n"
            f"  cd {PROJECT_ROOT}\n"
            "  mkdir -p repos\n"
            "  git clone https://github.com/lixiaowen-xw/DiffuEraser.git repos/DiffuEraser"
        )
    if not (DIFFUERASER_ROOT / "run_diffueraser.py").exists():
        raise FileNotFoundError(f"DiffuEraser entry script not found: {DIFFUERASER_ROOT / 'run_diffueraser.py'}")

    for model_path in [BASE_MODEL_PATH, VAE_PATH, DIFFUERASER_MODEL_PATH, PROPAINTER_MODEL_DIR]:
        if not model_path.exists():
            raise FileNotFoundError(
                f"DiffuEraser model path not found: {model_path}\n"
                "Place the required weights under repos/DiffuEraser/weights, "
                "or edit the model paths in the Config section."
            )


def collect_inputs() -> tuple[list[Path], list[Path], list[str]]:
    validate_static_inputs()

    frame_paths_all = sorted_image_paths(FRAMES_DIR)
    mask_map = build_image_map(MASK_DIR)
    if not frame_paths_all:
        raise FileNotFoundError(f"No input frames found in: {FRAMES_DIR}")
    if not mask_map:
        raise FileNotFoundError(f"No Part2 masks found in: {MASK_DIR}")

    selected_frame_paths = frame_paths_all[:MAX_FRAMES] if MAX_FRAMES is not None else frame_paths_all
    frame_names = [p.stem for p in selected_frame_paths]
    missing_masks = [name for name in frame_names if name not in mask_map]
    if missing_masks:
        preview = ", ".join(missing_masks[:10])
        more = "" if len(missing_masks) <= 10 else f" ... (+{len(missing_masks) - 10} more)"
        raise FileNotFoundError(f"Part2 masks are missing for frames: {preview}{more}")

    selected_mask_paths = [mask_map[name] for name in frame_names]
    return selected_frame_paths, selected_mask_paths, frame_names


def load_bgr_image(path: Path, cv2: Any) -> Any:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def load_binary_mask_bgr(path: Path, shape_hw: tuple[int, int], cv2: Any, np: Any) -> Any:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to read mask: {path}")
    h, w = shape_hw
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.uint8) * 255
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def make_video_writer(path: Path, fps: float, size_wh: tuple[int, int], cv2: Any) -> Any:
    ensure_dir(path.parent)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), size_wh)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    return writer


def pack_inputs_to_videos(paths: RunPaths, frame_paths: list[Path], mask_paths: list[Path], modules: dict[str, Any]) -> None:
    cv2 = modules["cv2"]
    np = modules["np"]

    first_frame = load_bgr_image(frame_paths[0], cv2)
    h, w = first_frame.shape[:2]

    frame_writer = make_video_writer(paths.input_video, FPS, (w, h), cv2)
    mask_writer = make_video_writer(paths.input_mask_video, FPS, (w, h), cv2)

    try:
        for frame_path, mask_path in zip(frame_paths, mask_paths):
            frame = load_bgr_image(frame_path, cv2)
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_CUBIC)
            mask_bgr = load_binary_mask_bgr(mask_path, (h, w), cv2, np)
            frame_writer.write(frame)
            mask_writer.write(mask_bgr)
    finally:
        frame_writer.release()
        mask_writer.release()


def run_diffueraser(paths: RunPaths, frame_count: int) -> None:
    ensure_dir(paths.diffueraser_save_dir)
    for stale_name in ["priori.mp4", "diffueraser_result.mp4"]:
        stale_path = paths.diffueraser_save_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    cmd = [
        str(PYTHON_EXE),
        str(DIFFUERASER_ROOT / "run_diffueraser.py"),
        "--input_video",
        str(paths.input_video),
        "--input_mask",
        str(paths.input_mask_video),
        "--video_length",
        str(frame_count),
        "--mask_dilation_iter",
        str(MASK_DILATION_ITER),
        "--max_img_size",
        str(MAX_IMG_SIZE),
        "--save_path",
        str(paths.diffueraser_save_dir),
        "--ref_stride",
        str(REF_STRIDE),
        "--neighbor_length",
        str(NEIGHBOR_LENGTH),
        "--subvideo_length",
        str(SUBVIDEO_LENGTH),
        "--base_model_path",
        str(BASE_MODEL_PATH),
        "--vae_path",
        str(VAE_PATH),
        "--diffueraser_path",
        str(DIFFUERASER_MODEL_PATH),
        "--propainter_model_dir",
        str(PROPAINTER_MODEL_DIR),
    ]

    print("[Part3-DiffuEraser] Running DiffuEraser:")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(DIFFUERASER_ROOT), check=True)

    if not paths.diffueraser_result_video.exists():
        raise RuntimeError(f"DiffuEraser did not generate result video: {paths.diffueraser_result_video}")


def extract_result_frames(paths: RunPaths, frame_names: list[str], modules: dict[str, Any]) -> None:
    cv2 = modules["cv2"]
    ensure_dir(paths.final_frames)
    for old in sorted_image_paths(paths.final_frames):
        old.unlink()

    capture = cv2.VideoCapture(str(paths.diffueraser_result_video))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open DiffuEraser result video: {paths.diffueraser_result_video}")

    written = 0
    try:
        for name in frame_names:
            ok, frame = capture.read()
            if not ok:
                break
            cv2.imwrite(str(paths.final_frames / f"{name}.png"), frame)
            written += 1
    finally:
        capture.release()

    if written != len(frame_names):
        raise RuntimeError(
            f"Extracted {written} frames from DiffuEraser result, but expected {len(frame_names)}."
        )


def copy_final_video(paths: RunPaths) -> None:
    ensure_dir(paths.final_video_path.parent)
    shutil.copy2(paths.diffueraser_result_video, paths.final_video_path)


def write_summary(paths: RunPaths, frame_names: list[str]) -> None:
    summary = {
        "video_name": VIDEO_NAME,
        "num_frames": len(frame_names),
        "frame_names": frame_names,
        "fps": FPS,
        "max_frames": MAX_FRAMES,
        "frames_dir": str(FRAMES_DIR),
        "mask_dir": str(MASK_DIR),
        "diffueraser_root": str(DIFFUERASER_ROOT),
        "python_exe": str(PYTHON_EXE),
        "output_dir": str(OUTPUT_DIR),
        "outputs": {
            "input_video": str(paths.input_video),
            "input_mask_video": str(paths.input_mask_video),
            "diffueraser_save_dir": str(paths.diffueraser_save_dir),
            "diffueraser_result_video": str(paths.diffueraser_result_video),
            "final_video": str(paths.final_video_path),
            "final_frames": str(paths.final_frames),
            "summary": str(paths.summary_path),
        },
        "diffueraser": {
            "video_length": len(frame_names),
            "mask_dilation_iter": MASK_DILATION_ITER,
            "max_img_size": MAX_IMG_SIZE,
            "ref_stride": REF_STRIDE,
            "neighbor_length": NEIGHBOR_LENGTH,
            "subvideo_length": SUBVIDEO_LENGTH,
            "base_model_path": str(BASE_MODEL_PATH),
            "vae_path": str(VAE_PATH),
            "diffueraser_path": str(DIFFUERASER_MODEL_PATH),
            "propainter_model_dir": str(PROPAINTER_MODEL_DIR),
        },
    }
    ensure_dir(paths.summary_path.parent)
    paths.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    configured_python = PYTHON_EXE.expanduser().resolve()
    current_python = Path(sys.executable).resolve()
    if configured_python != current_python:
        print(f"[Part3-DiffuEraser] Re-running with configured PYTHON_EXE: {configured_python}")
        subprocess.run([str(configured_python), str(Path(__file__).resolve())], check=True)
        return

    ensure_runtime_dependencies()
    modules = import_runtime_modules()
    paths = build_paths()
    ensure_dir(OUTPUT_DIR)
    ensure_dir(paths.work_dir)

    frame_paths, mask_paths, frame_names = collect_inputs()
    print(f"[Part3-DiffuEraser] Video: {VIDEO_NAME}")
    print(f"[Part3-DiffuEraser] Frames: {len(frame_names)}")
    print(f"[Part3-DiffuEraser] Frames dir: {FRAMES_DIR}")
    print(f"[Part3-DiffuEraser] Mask dir: {MASK_DIR}")

    pack_inputs_to_videos(paths, frame_paths, mask_paths, modules)
    run_diffueraser(paths, len(frame_names))
    extract_result_frames(paths, frame_names, modules)
    copy_final_video(paths)
    write_summary(paths, frame_names)

    print("\n[Part3-DiffuEraser] Done")
    print(f"  Input video       : {paths.input_video}")
    print(f"  Input mask video  : {paths.input_mask_video}")
    print(f"  Raw result video  : {paths.diffueraser_result_video}")
    print(f"  Final video       : {paths.final_video_path}")
    print(f"  Final frames      : {paths.final_frames}")
    print(f"  Summary           : {paths.summary_path}")


if __name__ == "__main__":
    main()
