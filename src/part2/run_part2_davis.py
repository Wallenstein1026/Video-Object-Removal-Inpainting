import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_ROOT.parents[1]
DEFAULT_DAVIS_ROOT = PROJECT_ROOT / "data" / "DAVIS"
DEFAULT_IMAGE_SUBDIR = Path("JPEGImages/480p")
DEFAULT_OUTPUT_MASKS = PROJECT_ROOT / "outputs" / "part2_masks"
DEFAULT_OUTPUT_INPAINT = PROJECT_ROOT / "outputs" / "part2_inpaint"
DEFAULT_PYTHON = sys.executable
DEFAULT_YOLO_WEIGHTS = "yolo11x.pt"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part2 pipeline over all DAVIS sequences")
    parser.add_argument("--davis-root", type=str, default=str(DEFAULT_DAVIS_ROOT),
                        help="DAVIS dataset root folder")
    parser.add_argument("--image-subdir", type=str, default=str(DEFAULT_IMAGE_SUBDIR),
                        help="Relative DAVIS frames subfolder under davis-root")
    parser.add_argument("--output-masks-root", type=str, default=str(DEFAULT_OUTPUT_MASKS),
                        help="Root output folder for part2 masks")
    parser.add_argument("--output-inpaint-root", type=str, default=str(DEFAULT_OUTPUT_INPAINT),
                        help="Root output folder for part2 inpaint results")
    parser.add_argument("--python", type=str, default=DEFAULT_PYTHON,
                        help="Python interpreter to run the part2 scripts")
    parser.add_argument("--weights", type=str, default=DEFAULT_YOLO_WEIGHTS,
                        help="YOLO model weights")
    parser.add_argument("--sequences", type=str, default="all",
                        help="Comma-separated DAVIS sequences to run, or 'all' (default)")
    parser.add_argument("--skip-yolo", action="store_true", help="Skip YOLO detection step")
    parser.add_argument("--skip-sam2", action="store_true", help="Skip SAM2 propagation step")
    parser.add_argument("--skip-propoint", action="store_true", help="Skip ProPainter inpainting step")
    parser.add_argument("--no-vis", action="store_true", help="Disable visualization outputs in YOLO/SAM2")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--background-prefill", action="store_true", help="Enable background prefill for ProPainter")
    return parser.parse_args()


def list_sequences(davis_root: Path, image_subdir: Path) -> list[str]:
    image_root = davis_root / image_subdir
    if not image_root.exists():
        raise FileNotFoundError(f"DAVIS image folder not found: {image_root}")
    return sorted([p.name for p in image_root.iterdir() if p.is_dir()])


def run_command(cmd: list[str], dry_run: bool = False) -> None:
    print("\n$ " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    davis_root = Path(args.davis_root).expanduser().resolve()
    image_subdir = Path(args.image_subdir)
    masks_root = Path(args.output_masks_root).expanduser().resolve()
    inpaint_root = Path(args.output_inpaint_root).expanduser().resolve()
    python_exe = Path(args.python)

    seqs = []
    if args.sequences.strip().lower() == "all":
        seqs = list_sequences(davis_root, image_subdir)
    else:
        seqs = [s.strip() for s in args.sequences.split(",") if s.strip()]
    if not seqs:
        raise ValueError("No DAVIS sequences selected.")

    masks_root.mkdir(parents=True, exist_ok=True)
    inpaint_root.mkdir(parents=True, exist_ok=True)

    yolo_script = SCRIPTS_ROOT / "yolo.py"
    sam2_script = SCRIPTS_ROOT / "SAM2.py"
    propoint_script = SCRIPTS_ROOT / "Propoint.py"

    print(f"Running Part2 on DAVIS sequences: {', '.join(seqs)}")
    print(f"DAVIS root         : {davis_root}")
    print(f"Frames subdir      : {image_subdir}")
    print(f"YOLO weights       : {args.weights}")
    print(f"Masks root         : {masks_root}")
    print(f"Inpaint root       : {inpaint_root}")
    print(f"Python executable  : {python_exe}")

    for seq in seqs:
        print(f"\n===== SEQUENCE: {seq} =====")

        frame_dir = davis_root / image_subdir / seq
        if not frame_dir.exists():
            raise FileNotFoundError(f"Frames folder not found for sequence: {frame_dir}")

        yolo_out = masks_root / f"{seq}_yolo"
        sam2_out = masks_root / f"{seq}_sam2"
        propoint_out = inpaint_root / seq
        mask_union_dir = sam2_out / "masks_union"

        if not args.skip_yolo:
            yolo_cmd = [
                str(python_exe),
                str(yolo_script),
                "--input", str(frame_dir),
                "--output", str(yolo_out),
                "--weights", args.weights,
            ]
            if args.no_vis:
                yolo_cmd.append("--no-vis")
            run_command(yolo_cmd, dry_run=args.dry_run)

        if not args.skip_sam2:
            sam2_cmd = [
                str(python_exe),
                str(sam2_script),
                "--prompts", str(yolo_out / "prompts.json"),
                "--frames-dir", str(frame_dir),
                "--output", str(sam2_out),
            ]
            if args.no_vis:
                sam2_cmd.append("--no-vis")
            run_command(sam2_cmd, dry_run=args.dry_run)

        if not args.skip_propoint:
            propoint_cmd = [
                str(python_exe),
                str(propoint_script),
                "--frames", str(frame_dir),
                "--masks", str(mask_union_dir),
                "--output", str(propoint_out),
                "--python", str(python_exe),
            ]
            if args.background_prefill:
                propoint_cmd.append("--background_prefill")
            run_command(propoint_cmd, dry_run=args.dry_run)

    print("\n===== Part2 DAVIS batch run completed =====")


if __name__ == "__main__":
    main()
