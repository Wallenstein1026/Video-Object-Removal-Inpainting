"""
Part 3: SAM2 mask -> ProPainter -> SDXL keyframe refinement -> local propagation.

Edit the Config section below before running. This script intentionally has no
command-line arguments so that all project paths and hyperparameters are kept in
one place for reproducible experiments.
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
CACHE_ROOT = PROJECT_ROOT / ".cache"
VIDEO_NAME = "bear"

FRAMES_DIR = PROJECT_ROOT / "data" / "DAVIS" / "JPEGImages" / "480p" / VIDEO_NAME
SAM2_MASK_DIR = PROJECT_ROOT / "outputs" / "part2_masks" / f"{VIDEO_NAME}_sam2" / "masks_union"
PROPAINTER_ROOT = PROJECT_ROOT / "repos" / "ProPainter"
SDXL_MODEL_DIR = PROJECT_ROOT / "models" / "sdxl-inpainting"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "part3_inpaint" / VIDEO_NAME

# Use the Python executable from the environment that has ProPainter, torch,
# diffusers, cv2, PIL, and numpy installed. If this script is started from a
# different interpreter, it will re-execute itself with PYTHON_EXE.
PYTHON_EXE = Path(os.environ.get("PART123_PYTHON", sys.executable))

FPS = 15.0
MAX_FRAMES: int | None = None

# Keyframe selection: choose every N frames, plus the last frame.
KEYFRAME_STRIDE = 8

# ProPainter parameters.
PROPAINTER_MASK_DILATION = 4
PROPAINTER_REF_STRIDE = 10
PROPAINTER_NEIGHBOR_LENGTH = 10
PROPAINTER_SUBVIDEO_LENGTH = 80
PROPAINTER_RAFT_ITER = 20
PROPAINTER_FP16 = True
PROPAINTER_SAVE_FRAMES = True

# SDXL inpainting parameters.
SDXL_PROMPT = (
    "clean natural background, realistic video frame, temporally consistent, "
    "high quality inpainting, no object remnants"
)
SDXL_NEGATIVE_PROMPT = (
    "object, person, animal, vehicle, artifact, blur, distortion, duplicate, "
    "ghosting, watermark, text"
)
SDXL_NUM_INFERENCE_STEPS = 30
SDXL_GUIDANCE_SCALE = 7.5
SDXL_STRENGTH = 0.85
SDXL_SEED = 1234
SDXL_MASK_BLUR = 15
SDXL_MASK_DILATE = 8

# Propagate SDXL keyframe repairs to frames within this temporal radius.
PROPAGATION_RADIUS = 3


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def configure_local_caches() -> dict[str, str]:
    cache_paths = {
        "HF_HOME": CACHE_ROOT / "huggingface",
        "HUGGINGFACE_HUB_CACHE": CACHE_ROOT / "huggingface" / "hub",
        "TRANSFORMERS_CACHE": CACHE_ROOT / "huggingface" / "transformers",
        "DIFFUSERS_CACHE": CACHE_ROOT / "huggingface" / "diffusers",
        "MODELSCOPE_CACHE": CACHE_ROOT / "modelscope",
        "TORCH_HOME": CACHE_ROOT / "torch",
        "XDG_CACHE_HOME": CACHE_ROOT / "xdg",
        "PIP_CACHE_DIR": CACHE_ROOT / "pip",
        "MPLCONFIGDIR": CACHE_ROOT / "matplotlib",
        "TMPDIR": STORAGE_ROOT / "tmp",
        "TEMP": STORAGE_ROOT / "tmp",
        "TMP": STORAGE_ROOT / "tmp",
    }
    for path in cache_paths.values():
        path.mkdir(parents=True, exist_ok=True)
    for key, path in cache_paths.items():
        os.environ[key] = str(path)
    return {key: str(path) for key, path in cache_paths.items()}


LOCAL_CACHE_ENV = configure_local_caches()


@dataclass(frozen=True)
class RunPaths:
    work_dir: Path
    propainter_output_root: Path
    propainter_raw_frames: Path
    propainter_frames: Path
    sdxl_keyframes: Path
    final_frames: Path
    comparison_path: Path
    final_video_path: Path
    summary_path: Path


def ensure_runtime_dependencies() -> None:
    required = ["cv2", "numpy", "PIL", "torch", "diffusers"]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            "Missing runtime dependencies in the current Python environment: "
            + ", ".join(missing)
            + "\nRun this script with the environment that has ProPainter and "
            + "SDXL dependencies installed, or edit PYTHON_EXE / interpreter setup "
            + "in the Config section."
        )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sorted_image_paths(folder: Path) -> list[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def build_image_map(folder: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted_image_paths(folder)}


def import_runtime_modules() -> dict[str, Any]:
    import cv2
    import numpy as np
    import torch
    from diffusers import StableDiffusionXLInpaintPipeline
    from PIL import Image

    return {
        "cv2": cv2,
        "np": np,
        "torch": torch,
        "Image": Image,
        "StableDiffusionXLInpaintPipeline": StableDiffusionXLInpaintPipeline,
    }


def validate_inputs(frame_paths: list[Path], mask_paths: list[Path]) -> None:
    if not FRAMES_DIR.exists():
        raise FileNotFoundError(f"Frames directory not found: {FRAMES_DIR}")
    if not SAM2_MASK_DIR.exists():
        raise FileNotFoundError(f"SAM2 union mask directory not found: {SAM2_MASK_DIR}")
    if not PROPAINTER_ROOT.exists():
        raise FileNotFoundError(f"ProPainter root not found: {PROPAINTER_ROOT}")
    if not (PROPAINTER_ROOT / "inference_propainter.py").exists():
        raise FileNotFoundError(f"ProPainter inference script not found: {PROPAINTER_ROOT / 'inference_propainter.py'}")
    for weight_name in ["ProPainter.pth", "recurrent_flow_completion.pth", "raft-things.pth"]:
        weight_path = PROPAINTER_ROOT / "weights" / weight_name
        if not weight_path.exists():
            raise FileNotFoundError(f"ProPainter weight not found: {weight_path}")
    if not SDXL_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"SDXL local model directory not found: {SDXL_MODEL_DIR}\n"
            "Place the local SDXL inpainting diffusers model there, or edit SDXL_MODEL_DIR in the Config section."
        )
    if not frame_paths:
        raise FileNotFoundError(f"No input frames found in: {FRAMES_DIR}")
    if not mask_paths:
        raise FileNotFoundError(f"No SAM2 masks found in: {SAM2_MASK_DIR}")

    selected_names = [p.stem for p in frame_paths]
    mask_map = {p.stem: p for p in mask_paths}
    missing = [name for name in selected_names if name not in mask_map]
    if missing:
        preview = ", ".join(missing[:10])
        more = "" if len(missing) <= 10 else f" ... (+{len(missing) - 10} more)"
        raise FileNotFoundError(f"SAM2 masks are missing for frames: {preview}{more}")


def collect_inputs() -> tuple[list[Path], list[Path]]:
    if not FRAMES_DIR.exists():
        raise FileNotFoundError(f"Frames directory not found: {FRAMES_DIR}")
    if not SAM2_MASK_DIR.exists():
        raise FileNotFoundError(f"SAM2 union mask directory not found: {SAM2_MASK_DIR}")
    return sorted_image_paths(FRAMES_DIR), sorted_image_paths(SAM2_MASK_DIR)


def build_paths() -> RunPaths:
    return RunPaths(
        work_dir=OUTPUT_DIR / "_work",
        propainter_output_root=OUTPUT_DIR / "propainter_raw",
        propainter_raw_frames=OUTPUT_DIR / "propainter_raw" / VIDEO_NAME / "frames",
        propainter_frames=OUTPUT_DIR / "propainter_frames",
        sdxl_keyframes=OUTPUT_DIR / "sdxl_keyframes",
        final_frames=OUTPUT_DIR / "final_frames",
        comparison_path=OUTPUT_DIR / "keyframe_comparison.jpg",
        final_video_path=OUTPUT_DIR / f"{VIDEO_NAME}_part3.mp4",
        summary_path=OUTPUT_DIR / "summary.json",
    )


def prepare_subset_inputs(paths: RunPaths, frame_paths: list[Path], mask_paths: list[Path]) -> tuple[Path, Path, list[str]]:
    frame_paths = frame_paths[:MAX_FRAMES] if MAX_FRAMES is not None else frame_paths
    names = [p.stem for p in frame_paths]
    mask_map = {p.stem: p for p in mask_paths}

    if MAX_FRAMES is None:
        return FRAMES_DIR, SAM2_MASK_DIR, names

    # Keep the subset frame directory name equal to VIDEO_NAME so ProPainter
    # still writes to propainter_raw/{VIDEO_NAME}/frames.
    subset_frames = paths.work_dir / VIDEO_NAME
    subset_masks = paths.work_dir / f"{VIDEO_NAME}_masks"
    ensure_dir(subset_frames)
    ensure_dir(subset_masks)

    for old in sorted_image_paths(subset_frames) + sorted_image_paths(subset_masks):
        old.unlink()

    for src in frame_paths:
        shutil.copy2(src, subset_frames / src.name)
    for name in names:
        src = mask_map[name]
        shutil.copy2(src, subset_masks / f"{name}{src.suffix.lower()}")

    return subset_frames, subset_masks, names


def run_propainter(paths: RunPaths, frames_input: Path, masks_input: Path) -> None:
    ensure_dir(paths.propainter_output_root)
    prior_sequence_output = paths.propainter_output_root / frames_input.name
    if prior_sequence_output.exists():
        shutil.rmtree(prior_sequence_output)
    env = os.environ.copy()
    env.update(LOCAL_CACHE_ENV)

    cmd = [
        str(PYTHON_EXE),
        str(PROPAINTER_ROOT / "inference_propainter.py"),
        "-i",
        str(frames_input),
        "-m",
        str(masks_input),
        "-o",
        str(paths.propainter_output_root),
        "--mask_dilation",
        str(PROPAINTER_MASK_DILATION),
        "--ref_stride",
        str(PROPAINTER_REF_STRIDE),
        "--neighbor_length",
        str(PROPAINTER_NEIGHBOR_LENGTH),
        "--subvideo_length",
        str(PROPAINTER_SUBVIDEO_LENGTH),
        "--raft_iter",
        str(PROPAINTER_RAFT_ITER),
    ]
    if PROPAINTER_FP16:
        cmd.append("--fp16")
    if PROPAINTER_SAVE_FRAMES:
        cmd.append("--save_frames")

    print("[Part3] Running ProPainter:")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(PROPAINTER_ROOT), check=True, env=env)


def normalize_propainter_frames(paths: RunPaths, frame_names: list[str]) -> None:
    raw_paths = sorted_image_paths(paths.propainter_raw_frames)
    if not raw_paths:
        raise RuntimeError(
            f"ProPainter did not generate saved frames at: {paths.propainter_raw_frames}\n"
            "Check the ProPainter environment and ensure PROPAINTER_SAVE_FRAMES=True."
        )
    if len(raw_paths) < len(frame_names):
        raise RuntimeError(
            f"ProPainter generated {len(raw_paths)} frames, but {len(frame_names)} are required."
        )

    ensure_dir(paths.propainter_frames)
    for old in sorted_image_paths(paths.propainter_frames):
        old.unlink()

    for raw_path, name in zip(raw_paths, frame_names):
        shutil.copy2(raw_path, paths.propainter_frames / f"{name}.png")


def choose_keyframes(frame_names: list[str]) -> list[str]:
    if KEYFRAME_STRIDE <= 0:
        raise ValueError(f"KEYFRAME_STRIDE must be positive, got {KEYFRAME_STRIDE}")
    keyframes = frame_names[::KEYFRAME_STRIDE]
    if frame_names[-1] not in keyframes:
        keyframes.append(frame_names[-1])
    return keyframes


def load_bgr_image(path: Path, cv2: Any) -> Any:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def load_mask_alpha(path: Path, shape_hw: tuple[int, int], cv2: Any, np: Any, dilate: int, blur: int) -> Any:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to read mask: {path}")
    h, w = shape_hw
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.uint8) * 255
    if dilate > 0:
        kernel_size = dilate * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel, iterations=1)
    if blur > 0:
        blur = blur if blur % 2 == 1 else blur + 1
        mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    return mask.astype(np.float32) / 255.0


def bgr_to_pil(image_bgr: Any, cv2: Any, Image: Any) -> Any:
    return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


def alpha_to_mask_pil(alpha: Any, np: Any, Image: Any) -> Any:
    mask_u8 = (np.clip(alpha, 0.0, 1.0) * 255).astype(np.uint8)
    return Image.fromarray(mask_u8, mode="L")


def pil_to_bgr(image: Any, cv2: Any, np: Any) -> Any:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def pad_for_sdxl(image_bgr: Any, alpha: Any, cv2: Any, np: Any, multiple: int = 8) -> tuple[Any, Any]:
    h, w = image_bgr.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return image_bgr, alpha

    padded_image = cv2.copyMakeBorder(
        image_bgr,
        0,
        pad_h,
        0,
        pad_w,
        borderType=cv2.BORDER_REFLECT_101,
    )
    padded_alpha = np.pad(alpha, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    return padded_image, padded_alpha


def run_sdxl_keyframes(paths: RunPaths, keyframes: list[str], modules: dict[str, Any]) -> None:
    cv2 = modules["cv2"]
    np = modules["np"]
    torch = modules["torch"]
    Image = modules["Image"]
    Pipeline = modules["StableDiffusionXLInpaintPipeline"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = Pipeline.from_pretrained(str(SDXL_MODEL_DIR), torch_dtype=dtype, local_files_only=True)
    pipe = pipe.to(device)

    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    ensure_dir(paths.sdxl_keyframes)
    generator = torch.Generator(device=device).manual_seed(SDXL_SEED)

    for name in keyframes:
        propainter_path = paths.propainter_frames / f"{name}.png"
        mask_path = SAM2_MASK_DIR / f"{name}.png"

        base_bgr = load_bgr_image(propainter_path, cv2)
        h, w = base_bgr.shape[:2]
        alpha = load_mask_alpha(
            mask_path,
            (h, w),
            cv2=cv2,
            np=np,
            dilate=SDXL_MASK_DILATE,
            blur=SDXL_MASK_BLUR,
        )
        sdxl_bgr, sdxl_alpha = pad_for_sdxl(base_bgr, alpha, cv2, np)

        result = pipe(
            prompt=SDXL_PROMPT,
            negative_prompt=SDXL_NEGATIVE_PROMPT,
            image=bgr_to_pil(sdxl_bgr, cv2, Image),
            mask_image=alpha_to_mask_pil(sdxl_alpha, np, Image),
            height=sdxl_bgr.shape[0],
            width=sdxl_bgr.shape[1],
            num_inference_steps=SDXL_NUM_INFERENCE_STEPS,
            guidance_scale=SDXL_GUIDANCE_SCALE,
            strength=SDXL_STRENGTH,
            generator=generator,
        ).images[0]

        refined_bgr = pil_to_bgr(result, cv2, np)
        refined_bgr = refined_bgr[:h, :w]
        if refined_bgr.shape[:2] != (h, w):
            refined_bgr = cv2.resize(refined_bgr, (w, h), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(paths.sdxl_keyframes / f"{name}.png"), refined_bgr)
        print(f"[Part3] Saved SDXL keyframe: {name}")


def propagate_keyframes(paths: RunPaths, frame_names: list[str], keyframes: list[str], modules: dict[str, Any]) -> None:
    cv2 = modules["cv2"]
    np = modules["np"]

    ensure_dir(paths.final_frames)
    keyframe_index = {name: i for i, name in enumerate(frame_names) if name in set(keyframes)}
    keyframe_images = {
        name: load_bgr_image(paths.sdxl_keyframes / f"{name}.png", cv2)
        for name in keyframes
    }

    for idx, name in enumerate(frame_names):
        base = load_bgr_image(paths.propainter_frames / f"{name}.png", cv2)
        h, w = base.shape[:2]
        mask_alpha = load_mask_alpha(
            SAM2_MASK_DIR / f"{name}.png",
            (h, w),
            cv2=cv2,
            np=np,
            dilate=SDXL_MASK_DILATE,
            blur=SDXL_MASK_BLUR,
        )

        weighted_sum = np.zeros_like(base, dtype=np.float32)
        total_weight = 0.0

        for key_name, key_idx in keyframe_index.items():
            distance = abs(idx - key_idx)
            if distance > PROPAGATION_RADIUS:
                continue
            weight = 1.0 / float(distance + 1)
            key_img = keyframe_images[key_name]
            if key_img.shape[:2] != (h, w):
                key_img = cv2.resize(key_img, (w, h), interpolation=cv2.INTER_CUBIC)
            weighted_sum += key_img.astype(np.float32) * weight
            total_weight += weight

        if total_weight > 0:
            propagated = weighted_sum / total_weight
            alpha3 = mask_alpha[:, :, None]
            final = base.astype(np.float32) * (1.0 - alpha3) + propagated * alpha3
            final = np.clip(final, 0, 255).astype(np.uint8)
        else:
            final = base

        cv2.imwrite(str(paths.final_frames / f"{name}.png"), final)


def save_video_from_frames(frame_names: list[str], paths: RunPaths, modules: dict[str, Any]) -> None:
    cv2 = modules["cv2"]
    first = load_bgr_image(paths.final_frames / f"{frame_names[0]}.png", cv2)
    h, w = first.shape[:2]
    ensure_dir(paths.final_video_path.parent)
    writer = cv2.VideoWriter(
        str(paths.final_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(FPS),
        (w, h),
    )
    for name in frame_names:
        frame = load_bgr_image(paths.final_frames / f"{name}.png", cv2)
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_CUBIC)
        writer.write(frame)
    writer.release()


def make_keyframe_comparison(paths: RunPaths, keyframes: list[str], modules: dict[str, Any]) -> None:
    cv2 = modules["cv2"]
    np = modules["np"]
    original_map = build_image_map(FRAMES_DIR)
    rows = []
    for name in keyframes:
        if name not in original_map:
            raise FileNotFoundError(f"Original frame not found for comparison: {name}")
        original = load_bgr_image(original_map[name], cv2)
        propainter = load_bgr_image(paths.propainter_frames / f"{name}.png", cv2)
        sdxl = load_bgr_image(paths.sdxl_keyframes / f"{name}.png", cv2)
        final = load_bgr_image(paths.final_frames / f"{name}.png", cv2)

        h, w = propainter.shape[:2]
        panels = []
        for label, image in [
            (f"orig {name}", original),
            ("ProPainter", propainter),
            ("SDXL keyframe", sdxl),
            ("final", final),
        ]:
            if image.shape[:2] != (h, w):
                image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
            canvas = image.copy()
            cv2.putText(canvas, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1, cv2.LINE_AA)
            panels.append(canvas)
        rows.append(np.concatenate(panels, axis=1))

    if rows:
        comparison = np.concatenate(rows, axis=0)
        cv2.imwrite(str(paths.comparison_path), comparison)


def write_summary(paths: RunPaths, frame_names: list[str], keyframes: list[str]) -> None:
    summary = {
        "video_name": VIDEO_NAME,
        "num_frames": len(frame_names),
        "keyframe_stride": KEYFRAME_STRIDE,
        "keyframes": keyframes,
        "propagation_radius": PROPAGATION_RADIUS,
        "frames_dir": str(FRAMES_DIR),
        "sam2_mask_dir": str(SAM2_MASK_DIR),
        "propainter_root": str(PROPAINTER_ROOT),
        "sdxl_model_dir": str(SDXL_MODEL_DIR),
        "output_dir": str(OUTPUT_DIR),
        "outputs": {
            "propainter_frames": str(paths.propainter_frames),
            "sdxl_keyframes": str(paths.sdxl_keyframes),
            "final_frames": str(paths.final_frames),
            "final_video": str(paths.final_video_path),
            "comparison": str(paths.comparison_path),
        },
        "propainter": {
            "mask_dilation": PROPAINTER_MASK_DILATION,
            "ref_stride": PROPAINTER_REF_STRIDE,
            "neighbor_length": PROPAINTER_NEIGHBOR_LENGTH,
            "subvideo_length": PROPAINTER_SUBVIDEO_LENGTH,
            "raft_iter": PROPAINTER_RAFT_ITER,
            "fp16": PROPAINTER_FP16,
        },
        "sdxl": {
            "prompt": SDXL_PROMPT,
            "negative_prompt": SDXL_NEGATIVE_PROMPT,
            "num_inference_steps": SDXL_NUM_INFERENCE_STEPS,
            "guidance_scale": SDXL_GUIDANCE_SCALE,
            "strength": SDXL_STRENGTH,
            "seed": SDXL_SEED,
            "mask_blur": SDXL_MASK_BLUR,
            "mask_dilate": SDXL_MASK_DILATE,
        },
    }
    ensure_dir(paths.summary_path.parent)
    paths.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    configured_python = PYTHON_EXE.expanduser().resolve()
    current_python = Path(sys.executable).resolve()
    if configured_python != current_python:
        print(f"[Part3] Re-running with configured PYTHON_EXE: {configured_python}")
        env = os.environ.copy()
        env.update(LOCAL_CACHE_ENV)
        subprocess.run([str(configured_python), str(Path(__file__).resolve())], check=True, env=env)
        return

    ensure_runtime_dependencies()
    modules = import_runtime_modules()
    paths = build_paths()
    ensure_dir(OUTPUT_DIR)
    ensure_dir(paths.work_dir)

    frame_paths_all, mask_paths_all = collect_inputs()
    selected_frame_paths = frame_paths_all[:MAX_FRAMES] if MAX_FRAMES is not None else frame_paths_all
    validate_inputs(selected_frame_paths, mask_paths_all)

    frames_input, masks_input, frame_names = prepare_subset_inputs(paths, frame_paths_all, mask_paths_all)
    keyframes = choose_keyframes(frame_names)

    print(f"[Part3] Video: {VIDEO_NAME}")
    print(f"[Part3] Frames: {len(frame_names)}")
    print(f"[Part3] Keyframes: {', '.join(keyframes)}")

    run_propainter(paths, frames_input, masks_input)
    normalize_propainter_frames(paths, frame_names)
    run_sdxl_keyframes(paths, keyframes, modules)
    propagate_keyframes(paths, frame_names, keyframes, modules)
    save_video_from_frames(frame_names, paths, modules)
    make_keyframe_comparison(paths, keyframes, modules)
    write_summary(paths, frame_names, keyframes)

    print("\n[Part3] Done")
    print(f"  ProPainter frames : {paths.propainter_frames}")
    print(f"  SDXL keyframes    : {paths.sdxl_keyframes}")
    print(f"  Final frames      : {paths.final_frames}")
    print(f"  Final video       : {paths.final_video_path}")
    print(f"  Comparison        : {paths.comparison_path}")
    print(f"  Summary           : {paths.summary_path}")


if __name__ == "__main__":
    main()
