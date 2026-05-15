import argparse
import os
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# ProPainter wrapper for Part 2 inpainting
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROPAINTER_ROOT = PROJECT_ROOT / "repos" / "ProPainter"
if str(PROPAINTER_ROOT) not in sys.path:
    sys.path.insert(0, str(PROPAINTER_ROOT))

try:
    from model.propainter import InpaintGenerator
    from model.recurrent_flow_completion import RecurrentFlowCompleteNet
    print("✅ ProPainter module import verified.")
except Exception as e:
    print(f"⚠️ ProPainter import failed ({e}); will run via inference_propainter.py instead.")

# Built-in Part 2 defaults
DEFAULT_DAVIS_ROOT = PROJECT_ROOT / "data" / "DAVIS" / "JPEGImages" / "480p"
DEFAULT_MASKS_ROOT = PROJECT_ROOT / "outputs" / "part2_masks"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "part2_inpaint"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".MP4", ".AVI", ".MOV"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP"}


def is_video(path: Path) -> bool:
    return path.suffix in VIDEO_EXTS


def is_image(path: Path) -> bool:
    return path.suffix in IMAGE_EXTS


def resolve_input_path(path: Path, name: str) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    return path


def extract_video_to_frames(video_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_path = output_dir / f"{idx:06d}.png"
        cv2.imwrite(str(out_path), frame)
        idx += 1

    cap.release()
    if idx == 0:
        raise RuntimeError(f"No frames extracted from {video_path}")
    return output_dir


def prepare_mask_path(mask_path: Path, temp_dir: Path) -> Path:
    if mask_path.is_dir():
        return mask_path
    if is_image(mask_path):
        return mask_path
    if is_video(mask_path):
        mask_frames_dir = temp_dir / "mask_frames"
        print(f"📽️ Extracting mask video to images: {mask_path} -> {mask_frames_dir}")
        return extract_video_to_frames(mask_path, mask_frames_dir)
    raise ValueError(f"Unsupported mask path: {mask_path}")


def run_propaint(
    python_path: Path,
    propainter_root: Path,
    video_input: Path,
    mask_input: Path,
    output_dir: Path,
    height: int,
    width: int,
    fp16: bool,
    mask_dilation: int,
    ref_stride: int,
    neighbor_length: int,
    subvideo_length: int,
    raft_iter: int,
    save_frames: bool,
    background_prefill: bool,
) -> None:
    script = propainter_root / "inference_propainter.py"
    if not script.exists():
        raise FileNotFoundError(f"ProPainter inference script not found: {script}")

    cmd = [
        str(python_path),
        str(script),
        "-i",
        str(video_input),
        "-m",
        str(mask_input),
        "-o",
        str(output_dir),
        "--height",
        str(height),
        "--width",
        str(width),
        "--mask_dilation",
        str(mask_dilation),
        "--ref_stride",
        str(ref_stride),
        "--neighbor_length",
        str(neighbor_length),
        "--subvideo_length",
        str(subvideo_length),
        "--raft_iter",
        str(raft_iter),
    ]

    if fp16:
        cmd.append("--fp16")
    if save_frames:
        cmd.append("--save_frames")

    if background_prefill:
        print("⚠️ background_prefill is currently not supported by the bundled ProPainter inference script.")
        print("   The option will be ignored until inference_propainter.py adds support for it.")

    print("🚀 Running ProPainter inference:")
    print("  ", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(propainter_root))
    if result.returncode != 0:
        raise RuntimeError("ProPainter inference failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Part 2 ProPainter wrapper")
    parser.add_argument("--sequence", type=str, default="tennis",
                        help="DAVIS sequence name to process (used when frames/masks/output are not explicitly set)")
    parser.add_argument("--frames", type=str, default=None,
                        help="Input video file or frame folder")
    parser.add_argument("--masks", type=str, default=None,
                        help="Mask folder or single mask image")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for ProPainter results")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python interpreter to run ProPainter")
    parser.add_argument("--height", type=int, default=-1, help="Processing height for ProPainter")
    parser.add_argument("--width", type=int, default=-1, help="Processing width for ProPainter")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision")
    parser.add_argument("--mask_dilation", type=int, default=4, help="Mask dilation for ProPainter")
    parser.add_argument("--ref_stride", type=int, default=10, help="Reference stride for ProPainter")
    parser.add_argument("--neighbor_length", type=int, default=10, help="Neighbor length for ProPainter")
    parser.add_argument("--subvideo_length", type=int, default=80, help="Subvideo length for ProPainter")
    parser.add_argument("--raft_iter", type=int, default=20, help="RAFT iterations for ProPainter")
    parser.add_argument("--save_frames", action="store_true", help="Save output frames")
    parser.add_argument("--background_prefill", action="store_true", help="Enable background prefill for ProPainter")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequence = args.sequence

    default_frames = DEFAULT_DAVIS_ROOT / sequence
    default_masks = DEFAULT_MASKS_ROOT / f"{sequence}_sam2" / "masks_union"
    default_output = DEFAULT_OUTPUT_ROOT / sequence

    frames_path = resolve_input_path(Path(args.frames) if args.frames else default_frames, "Frames input")
    masks_path = resolve_input_path(Path(args.masks) if args.masks else default_masks, "Masks input")
    output_dir = Path(args.output if args.output else default_output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="propainter_") as temp_dir:
        temp_root = Path(temp_dir)
        mask_input = prepare_mask_path(masks_path, temp_root)
        run_propaint(
            python_path=Path(args.python),
            propainter_root=PROPAINTER_ROOT,
            video_input=frames_path,
            mask_input=mask_input,
            output_dir=output_dir,
            height=args.height,
            width=args.width,
            fp16=args.fp16,
            mask_dilation=args.mask_dilation,
            ref_stride=args.ref_stride,
            neighbor_length=args.neighbor_length,
            subvideo_length=args.subvideo_length,
            raft_iter=args.raft_iter,
            save_frames=args.save_frames,
            background_prefill=args.background_prefill,
        )

    print("\n✅ Part 2 ProPainter completed successfully")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
