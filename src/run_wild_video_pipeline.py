"""
Run the full wild-video pipeline:
  1) video -> DAVIS-style image sequence
  2) Part1 masks + inpaint
  3) Part2 YOLO -> SAM2 -> ProPainter inpaint
  4) Part3 diffusion refinement and DiffuEraser
  5) inpaint quality eval using a selected GT frame

Only edit the Config section for normal use.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from copy import copy
from pathlib import Path


# =============================================================================
# Config: edit here
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Video name without extension. The script expects data/wild/{VIDEO_NAME}.mp4
# unless VIDEO_PATH is set explicitly.
VIDEO_NAME = "walk"
VIDEO_PATH: Path | None = None

# GT_SOURCE:
#   "first" -> use first extracted frame as GT for all eval frames
#   "last"  -> use last extracted frame as GT for all eval frames
#   "image" -> use GT_IMAGE_PATH as GT for all eval frames
GT_SOURCE = "first"
GT_IMAGE_PATH: Path | None = None

# If False, eval compares the whole generated frame against the clean
# background GT. Use this when you do not have masks and only care about
# edit_psnr/edit_ssim against the clean background.
EVAL_USE_MASK = True

# Environment paths.
PART123_PYTHON = Path(os.environ.get("PART123_PYTHON", sys.executable))
DIFFUERASER_PYTHON = Path(os.environ.get("DIFFUERASER_PYTHON", sys.executable))

# Stage switches. Keep these True for a full run. Turn off completed expensive
# stages when resuming after a failure.
RUN_VIDEO_TO_IMAGES = True
RUN_PART1 = True
RUN_PART2_YOLO = True
RUN_PART2_SAM2 = True
RUN_PART2_INPAINT = True
RUN_PART3_DIFFUSION = True
RUN_PART3_DIFFUERASER = True
RUN_EVAL = True

# Video conversion.
RESIZE_TO_HEIGHT = 480
FRAME_STRIDE = 1
OVERWRITE_EXTRACTED_FRAMES = True

# Output / runtime options.
YOLOV8_SEG_WEIGHTS = "yolov8s-seg.pt"
YOLO11_WEIGHTS = PROJECT_ROOT / "yolo11x.pt"
PART2_PROPAINTER_FP16 = True
PART2_PROPAINTER_SAVE_FRAMES = True
NO_PART2_VIS = False


# =============================================================================
# Helpers
# =============================================================================

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def image_paths(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def require_dir(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {path}")
    return path


def require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    return path


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    print("\n$ " + " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=str(cwd) if cwd else None, check=True)


def resolve_video_path(video_name: str) -> Path:
    if VIDEO_PATH is not None:
        return require_file(Path(VIDEO_PATH), "Configured video")
    wild_root = PROJECT_ROOT / "data" / "wild"
    for ext in VIDEO_EXTS:
        candidate = wild_root / f"{video_name}{ext}"
        if candidate.exists():
            return candidate
    tried = ", ".join(str(wild_root / f"{video_name}{ext}") for ext in VIDEO_EXTS)
    raise FileNotFoundError(f"Cannot find video for {video_name}. Tried: {tried}")


def frames_dir(video_name: str) -> Path:
    return PROJECT_ROOT / "data" / "wild" / "JPEGImages" / "480p" / video_name


def part1_mask_dir(video_name: str) -> Path:
    return PROJECT_ROOT / "outputs" / "part1_masks" / f"{video_name}_yolov8seg" / "masks_union"


def part2_mask_dir(video_name: str) -> Path:
    return PROJECT_ROOT / "outputs" / "part2_masks" / f"{video_name}_sam2" / "masks_union"


def part2_inpaint_root(video_name: str) -> Path:
    return PROJECT_ROOT / "outputs" / "part2_inpaint" / video_name


def find_part2_frames(video_name: str) -> Path:
    root = part2_inpaint_root(video_name)
    candidates = [
        root / video_name / "frames",
        root / "frames",
        root / "frames" / "frames",
    ]
    if root.exists():
        candidates.extend(sorted(p for p in root.rglob("frames") if p.is_dir()))
    for candidate in candidates:
        if image_paths(candidate):
            return candidate
    raise FileNotFoundError(f"Cannot find Part2 inpaint frame directory under: {root}")


def make_clean_image_dir(src_dir: Path, dst_dir: Path, label: str) -> Path:
    clean_dir(dst_dir)
    imgs = image_paths(src_dir)
    if not imgs:
        raise RuntimeError(f"No images found for {label}: {src_dir}")
    for src in imgs:
        link_or_copy(src, dst_dir / src.name)
    return dst_dir


# =============================================================================
# Pipeline stages
# =============================================================================

def convert_video_to_images(video_name: str, video_path: Path) -> None:
    import src.preprocess.video_to_image as v

    v.PROJECT_ROOT = str(PROJECT_ROOT)
    v.INPUT_VIDEO = str(video_path)
    v.SEQUENCE_NAME = video_name
    v.DAVIS_ROOT = str(PROJECT_ROOT / "data" / "wild")
    v.RESIZE_TO_HEIGHT = RESIZE_TO_HEIGHT
    v.FRAME_STRIDE = FRAME_STRIDE
    v.CREATE_EMPTY_MASKS = True
    v.OVERWRITE = OVERWRITE_EXTRACTED_FRAMES
    v.COPY_VIDEO_TO_WILD = False
    v.IMAGESET_NAME = f"{video_name}.txt"
    v.convert_video_to_davis()


def run_part1(video_name: str) -> None:
    from src.part1.extract_masks_yolov8seg import default_config as mask_config
    from src.part1.extract_masks_yolov8seg import main as run_masks
    from src.part1.inpaint_part1 import default_config as inpaint_config
    from src.part1.inpaint_part1 import main as run_inpaint

    input_frames = require_dir(frames_dir(video_name), "Extracted frames")

    mask_cfg = copy(mask_config())
    mask_cfg.proj_root = PROJECT_ROOT
    mask_cfg.input = input_frames
    mask_cfg.video_name = video_name
    mask_cfg.output_root = PROJECT_ROOT / "outputs" / "part1_masks"
    mask_cfg.yolo_weights = YOLOV8_SEG_WEIGHTS
    run_masks(mask_cfg)

    inpaint_cfg = copy(inpaint_config())
    inpaint_cfg.proj_root = PROJECT_ROOT
    inpaint_cfg.input = input_frames
    inpaint_cfg.video_name = video_name
    inpaint_cfg.masks_root = PROJECT_ROOT / "outputs" / "part1_masks"
    inpaint_cfg.output_root = PROJECT_ROOT / "outputs" / "part1_inpaint"
    inpaint_cfg.figures_root = PROJECT_ROOT / "outputs" / "figures"
    run_inpaint(inpaint_cfg)


def run_part2(video_name: str) -> None:
    input_frames = require_dir(frames_dir(video_name), "Extracted frames")
    yolo_out = PROJECT_ROOT / "outputs" / "part2_masks" / f"{video_name}_yolo"
    sam2_out = PROJECT_ROOT / "outputs" / "part2_masks" / f"{video_name}_sam2"
    propainter_out = part2_inpaint_root(video_name)

    if RUN_PART2_YOLO:
        cmd = [
            PART123_PYTHON,
            PROJECT_ROOT / "src" / "part2" / "yolo.py",
            "--input", input_frames,
            "--output", yolo_out,
            "--weights", YOLO11_WEIGHTS,
        ]
        if NO_PART2_VIS:
            cmd.append("--no-vis")
        run_cmd(cmd, cwd=PROJECT_ROOT)

    if RUN_PART2_SAM2:
        cmd = [
            PART123_PYTHON,
            PROJECT_ROOT / "src" / "part2" / "SAM2.py",
            "--prompts", yolo_out / "prompts.json",
            "--frames-dir", input_frames,
            "--output", sam2_out,
        ]
        if NO_PART2_VIS:
            cmd.append("--no-vis")
        run_cmd(cmd, cwd=PROJECT_ROOT)

    if RUN_PART2_INPAINT:
        work = PROJECT_ROOT / "outputs" / "_wild_pipeline_work" / video_name / "part2_propainter"
        clean_frames = make_clean_image_dir(input_frames, work / "frames" / video_name, "Part2 frames")
        clean_masks = make_clean_image_dir(part2_mask_dir(video_name), work / "masks" / "masks_union", "Part2 masks")

        cmd = [
            PART123_PYTHON,
            PROJECT_ROOT / "src" / "part2" / "Propoint.py",
            "--frames", clean_frames,
            "--masks", clean_masks,
            "--output", propainter_out,
            "--python", PART123_PYTHON,
        ]
        if PART2_PROPAINTER_FP16:
            cmd.append("--fp16")
        if PART2_PROPAINTER_SAVE_FRAMES:
            cmd.append("--save_frames")
        run_cmd(cmd, cwd=PROJECT_ROOT)


def run_part3_diffusion(video_name: str) -> None:
    import src.part3.run_part3_diffusion_inpaint as p

    input_frames = require_dir(frames_dir(video_name), "Extracted frames")
    num_frames = len(image_paths(input_frames))
    if num_frames == 0:
        raise RuntimeError(f"No extracted images found: {input_frames}")

    p.PROJECT_ROOT = PROJECT_ROOT
    p.VIDEO_NAME = video_name
    p.FRAMES_DIR = input_frames
    p.SAM2_MASK_DIR = part2_mask_dir(video_name)
    p.OUTPUT_DIR = PROJECT_ROOT / "outputs" / "part3_inpaint" / video_name
    p.PYTHON_EXE = PART123_PYTHON
    # Force use of the script's clean subset work dir so metadata.json is never
    # passed to ProPainter.
    p.MAX_FRAMES = num_frames
    p.main()


def run_part3_diffueraser(video_name: str) -> None:
    require_dir(frames_dir(video_name), "Extracted frames")
    require_dir(part2_mask_dir(video_name), "Part2 SAM2 masks")

    # run_part3_diffueraser_inpaint.py re-execs itself when PYTHON_EXE differs
    # from the current interpreter. If we call d.main() directly from the
    # Part1/2/3 environment, that re-exec loses these runtime overrides and the
    # module falls back to its hardcoded VIDEO_NAME. Run a tiny configured
    # wrapper inside the DiffuEraser environment instead.
    runner_dir = PROJECT_ROOT / "outputs" / "_wild_pipeline_work" / video_name
    ensure_dir(runner_dir)
    runner = runner_dir / "run_configured_diffueraser.py"
    runner.write_text(
        f"""
from pathlib import Path
import sys

PROJECT_ROOT = Path({str(PROJECT_ROOT)!r})
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import part3.run_part3_diffueraser_inpaint as d

d.PROJECT_ROOT = PROJECT_ROOT
d.VIDEO_NAME = {video_name!r}
d.FRAMES_DIR = Path({str(frames_dir(video_name))!r})
d.MASK_DIR = Path({str(part2_mask_dir(video_name))!r})
d.OUTPUT_DIR = Path({str(PROJECT_ROOT / "outputs" / "part3_diffueraser_inpaint" / video_name)!r})
d.PYTHON_EXE = Path({str(DIFFUERASER_PYTHON)!r})

d.main()
""".lstrip(),
        encoding="utf-8",
    )
    run_cmd([DIFFUERASER_PYTHON, runner], cwd=PROJECT_ROOT)


def select_gt_image(video_name: str) -> Path:
    if GT_SOURCE not in {"first", "last", "image"}:
        raise ValueError(f"GT_SOURCE must be 'first', 'last', or 'image', got: {GT_SOURCE}")

    if GT_SOURCE == "image":
        if GT_IMAGE_PATH is None:
            raise ValueError("GT_IMAGE_PATH must be set when GT_SOURCE='image'")
        return require_file(Path(GT_IMAGE_PATH), "GT image")

    imgs = image_paths(require_dir(frames_dir(video_name), "Extracted frames"))
    if not imgs:
        raise RuntimeError(f"No extracted frames found for GT selection: {frames_dir(video_name)}")
    return imgs[0] if GT_SOURCE == "first" else imgs[-1]


def build_gt_dir(video_name: str, gt_image: Path, pred_dirs: list[Path]) -> Path:
    gt_dir = PROJECT_ROOT / "outputs" / "eval_gt" / f"{video_name}_{GT_SOURCE}_gt"
    clean_dir(gt_dir)

    frame_names: set[str] = set()
    for pred_dir in pred_dirs:
        imgs = image_paths(pred_dir)
        if not imgs:
            raise RuntimeError(f"No prediction images found: {pred_dir}")
        frame_names.update(p.stem for p in imgs)

    for name in sorted(frame_names):
        shutil.copy2(gt_image, gt_dir / f"{name}{gt_image.suffix.lower()}")
    return gt_dir


def run_inpaint_eval(video_name: str) -> None:
    import src.eval.eval_inpaint as e

    eval_mask_dir = part2_mask_dir(video_name) if EVAL_USE_MASK else None
    jobs = [
        (
            "part1",
            PROJECT_ROOT / "outputs" / "part1_inpaint" / video_name / "frames",
            part1_mask_dir(video_name) if EVAL_USE_MASK else None,
            PROJECT_ROOT / "outputs" / "part1_eval" / f"{video_name}_{GT_SOURCE}_gt_inpaint_eval",
        ),
        (
            "part2_propainter",
            find_part2_frames(video_name),
            eval_mask_dir,
            PROJECT_ROOT / "outputs" / "part2_eval" / f"{video_name}_{GT_SOURCE}_gt_inpaint_eval",
        ),
        (
            "part3_diffusion",
            PROJECT_ROOT / "outputs" / "part3_inpaint" / video_name / "final_frames",
            eval_mask_dir,
            PROJECT_ROOT / "outputs" / "part3_eval" / f"{video_name}_diffusion_{GT_SOURCE}_gt_inpaint_eval",
        ),
        (
            "part3_diffueraser",
            PROJECT_ROOT / "outputs" / "part3_diffueraser_inpaint" / video_name / "final_frames",
            eval_mask_dir,
            PROJECT_ROOT / "outputs" / "part3_eval" / f"{video_name}_diffueraser_{GT_SOURCE}_gt_inpaint_eval",
        ),
    ]

    gt_image = select_gt_image(video_name)
    pred_dirs = []
    for label, pred_dir, mask_dir, _ in jobs:
        pred_dirs.append(require_dir(pred_dir, f"{label} prediction frames"))
        if mask_dir is not None:
            require_dir(mask_dir, f"{label} masks")
    gt_dir = build_gt_dir(video_name, gt_image, pred_dirs)

    print(f"\nUsing GT image: {gt_image}")
    print(f"Expanded GT dir: {gt_dir}")

    for label, pred_dir, mask_dir, out_dir in jobs:
        print(f"\n===== eval {label} =====")
        print(f"Pred: {pred_dir}")
        print(f"Mask: {mask_dir if mask_dir is not None else 'None (full-frame)'}")
        print(f"Out : {out_dir}")
        e.VIDEO_NAME = video_name
        e.PRED_FRAME_DIR = pred_dir
        e.GT_FRAME_DIR = gt_dir
        e.MASK_DIR = mask_dir
        e.OUTPUT_DIR = out_dir
        e.main()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the wild-video Part1/2/3 + eval pipeline.")
    parser.add_argument("--video-name", default=VIDEO_NAME, help="Override VIDEO_NAME from the Config section.")
    parser.add_argument("--skip-existing-extract", action="store_true", help="Do not overwrite extracted frames.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_name = args.video_name
    if args.skip_existing_extract:
        global OVERWRITE_EXTRACTED_FRAMES
        OVERWRITE_EXTRACTED_FRAMES = False

    if str(PROJECT_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    video_path = resolve_video_path(video_name)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Video name  : {video_name}")
    print(f"Video path  : {video_path}")
    print(f"GT source   : {GT_SOURCE}")

    if RUN_VIDEO_TO_IMAGES:
        convert_video_to_images(video_name, video_path)
    if RUN_PART1:
        run_part1(video_name)
    if RUN_PART2_YOLO or RUN_PART2_SAM2 or RUN_PART2_INPAINT:
        run_part2(video_name)
    if RUN_PART3_DIFFUSION:
        run_part3_diffusion(video_name)
    if RUN_PART3_DIFFUERASER:
        run_part3_diffueraser(video_name)
    if RUN_EVAL:
        run_inpaint_eval(video_name)

    print("\nPipeline completed.")


if __name__ == "__main__":
    main()
