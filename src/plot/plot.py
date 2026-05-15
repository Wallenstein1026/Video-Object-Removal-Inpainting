from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DAVIS_SEQUENCES = [
    "bear",
    "bike-packing",
    "bmx-trees",
    "boxing-fisheye",
    "breakdance-flare",
    "crossing",
    "dog-agility",
    "drift-chicane",
    "tennis",
]
WILD_VIS_SEQUENCES = ["ski", "swimming"]
PART12_VIS_SEQUENCES = DAVIS_SEQUENCES + WILD_VIS_SEQUENCES
PART23_VIS_SEQUENCE = "bear"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

PART1_COLOR = "#2E7D32"
PART2_COLOR = "#1565C0"
DIFFUSION_COLOR = "#6A1B9A"
DIFFUERASER_COLOR = "#C62828"
GRID_BG = (245, 245, 245)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final report figures for Overleaf.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-keyframes", type=int, default=2)
    parser.add_argument("--panel-width", type=int, default=300)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf"
    return f"{v:.{digits}f}"


def sorted_image_paths(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def normalize_key(stem: str) -> str:
    return str(int(stem)) if stem.isdigit() else stem


def frame_sort_key(key: str) -> tuple[int, str]:
    if key.isdigit():
        return int(key), key
    match = re.search(r"(\d+)$", key)
    return (int(match.group(1)) if match else 10**9), key


def image_map(folder: Path) -> dict[str, Path]:
    return {normalize_key(p.stem): p for p in sorted_image_paths(folder)}


def frame_dir(root: Path, seq: str) -> Path:
    davis = root / "data" / "DAVIS" / "JPEGImages" / "480p" / seq
    if davis.exists():
        return davis
    return root / "data" / "wild" / "JPEGImages" / "480p" / seq


def gt_mask_dir(root: Path, seq: str) -> Path:
    davis = root / "data" / "DAVIS" / "Annotations" / "480p" / seq
    if davis.exists():
        return davis
    return root / "data" / "wild" / "Annotations" / "480p" / seq


def read_bgr(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def read_gray(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def read_by_key(folder: Path, key: str) -> np.ndarray | None:
    return read_bgr(image_map(folder).get(normalize_key(key)))


def read_video_frame(video_path: Path, key: str) -> np.ndarray | None:
    if not key.isdigit() or not video_path.exists():
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(key))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def part2_frame(root: Path, seq: str, key: str) -> np.ndarray | None:
    base = root / "outputs" / "part2_inpaint" / seq / seq
    image = read_by_key(base / "frames", key)
    if image is not None:
        return image
    return read_video_frame(base / "inpaint_out.mp4", key)


def overlay_mask(frame: np.ndarray, mask_path: Path | None, color: tuple[int, int, int]) -> np.ndarray | None:
    mask = read_gray(mask_path)
    if mask is None:
        return None
    if mask.shape != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    out = frame.copy()
    color_img = np.zeros_like(out)
    color_img[:, :] = color
    blended = cv2.addWeighted(out, 0.55, color_img, 0.45, 0)
    out[mask > 0] = blended[mask > 0]
    return out


def mask_overlay(root: Path, seq: str, key: str, kind: str, frame: np.ndarray) -> np.ndarray | None:
    if kind == "gt":
        folder = gt_mask_dir(root, seq)
        color = (255, 128, 0)
    elif kind == "part1":
        folder = root / "outputs" / "part1_masks" / f"{seq}_yolov8seg" / "masks_union"
        color = (0, 0, 255)
    elif kind == "part2":
        folder = root / "outputs" / "part2_masks" / f"{seq}_sam2" / "masks_union"
        color = (0, 180, 0)
    else:
        return None
    return overlay_mask(frame, image_map(folder).get(normalize_key(key)), color)


def choose_keys(folder: Path, count: int) -> list[str]:
    keys = sorted(image_map(folder).keys(), key=frame_sort_key)
    if not keys:
        return []
    if len(keys) <= count:
        return keys
    idxs = np.linspace(0, len(keys) - 1, count, dtype=int).tolist()
    return [keys[i] for i in idxs]


def resize_panel(image: np.ndarray, panel_width: int) -> np.ndarray:
    h, w = image.shape[:2]
    new_h = max(1, int(round(h * panel_width / float(w))))
    return cv2.resize(image, (panel_width, new_h), interpolation=cv2.INTER_AREA)


def label_panel(image: np.ndarray, text: str) -> np.ndarray:
    out = cv2.copyMakeBorder(image, 32, 0, 0, 0, cv2.BORDER_CONSTANT, value=(32, 32, 32))
    scale = 0.48
    while scale > 0.28:
        width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        if width <= out.shape[1] - 12:
            break
        scale -= 0.04
    cv2.putText(out, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def blank(shape_hw: tuple[int, int], text: str = "missing") -> np.ndarray:
    h, w = shape_hw
    out = np.full((h, w, 3), GRID_BG, dtype=np.uint8)
    cv2.putText(out, text, (12, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2, cv2.LINE_AA)
    return out


def render_grid(rows: list[list[tuple[str, np.ndarray | None]]], panel_width: int, out_path: Path) -> Path:
    rendered_rows = []
    for row in rows:
        base = next((img for _, img in row if img is not None), None)
        if base is None:
            continue
        target_hw = base.shape[:2]
        panels = []
        for label, image in row:
            if image is None:
                image = blank(target_hw)
            elif image.shape[:2] != target_hw:
                image = cv2.resize(image, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_AREA)
            panels.append(label_panel(resize_panel(image, panel_width), label))
        rendered_rows.append(np.concatenate(panels, axis=1))
    if not rendered_rows:
        return out_path
    grid = np.concatenate(rendered_rows, axis=0)
    cv2.imwrite(str(out_path), grid, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    return out_path


def mask_summary(root: Path, part: str, seq: str) -> dict[str, Any]:
    if part == "part1":
        path = root / "outputs" / "part1_eval" / f"{seq}_yolov8seg_eval" / "summary.json"
    else:
        path = root / "outputs" / "part2_eval" / f"{seq}_sam2_eval" / "summary.json"
    return load_json(path)


def save_plot(fig: plt.Figure, output_dir: Path, name: str, dpi: int) -> list[Path]:
    paths = [output_dir / f"{name}.jpg"]
    for path in paths:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return paths


def generate_mask_summary(root: Path, output_dir: Path, dpi: int) -> list[Path]:
    x = np.arange(len(DAVIS_SEQUENCES))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True)
    for ax, metric, title in zip(axes, ["JM", "JR"], ["Mean IoU $J_M$", "Recall $J_R@0.5$"]):
        y1 = [float(mask_summary(root, "part1", s)[metric]) for s in DAVIS_SEQUENCES]
        y2 = [float(mask_summary(root, "part2", s)[metric]) for s in DAVIS_SEQUENCES]
        ax.bar(x - width / 2, y1, width, color=PART1_COLOR, label="Part 1")
        ax.bar(x + width / 2, y2, width, color=PART2_COLOR, label="Part 2")
        ax.set_xticks(x)
        ax.set_xticklabels(DAVIS_SEQUENCES, rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)
        ax.set_title(title)
    axes[0].set_ylabel("Score")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return save_plot(fig, output_dir, "part12_davis_mask_metrics", dpi)


def generate_part12_visuals(root: Path, output_dir: Path, count: int, panel_width: int) -> list[Path]:
    paths = []
    for seq in PART12_VIS_SEQUENCES:
        fdir = frame_dir(root, seq)
        rows = []
        for key in choose_keys(fdir, count):
            original = read_by_key(fdir, key)
            if original is None:
                continue
            row = [
                (f"Original {int(key):05d}" if key.isdigit() else f"Original {key}", original),
            ]
            if seq in DAVIS_SEQUENCES:
                row.append(("GT mask", mask_overlay(root, seq, key, "gt", original)))
            row.extend(
                [
                    ("Part1 mask", mask_overlay(root, seq, key, "part1", original)),
                    ("Part2 mask", mask_overlay(root, seq, key, "part2", original)),
                    ("Part1 result", read_by_key(root / "outputs" / "part1_inpaint" / seq / "frames", key)),
                    ("Part2 result", part2_frame(root, seq, key)),
                ]
            )
            rows.append(row)
        out = output_dir / f"part12_visual_{seq}.jpg"
        paths.append(render_grid(rows, panel_width, out))
    return paths


def generate_part12_pipeline_visual(root: Path, output_dir: Path, count: int, panel_width: int) -> list[Path]:
    seq = "bmx-trees"
    fdir = frame_dir(root, seq)
    rows = []
    for key in choose_keys(fdir, count):
        original = read_by_key(fdir, key)
        if original is None:
            continue
        rows.append(
            [
                (f"Original {int(key):05d}" if key.isdigit() else f"Original {key}", original),
                ("Part1 dynamic mask", mask_overlay(root, seq, key, "part1", original)),
                ("Part1 Telea result", read_by_key(root / "outputs" / "part1_inpaint" / seq / "frames", key)),
                ("YOLO prompt vis", read_by_key(root / "outputs" / "part2_masks" / f"{seq}_yolo" / "yolo_vis", key)),
                ("SAM2 propagated mask", read_by_key(root / "outputs" / "part2_masks" / f"{seq}_sam2" / "masks_vis", key)),
                ("Part2 ProPainter result", part2_frame(root, seq, key)),
            ]
        )
    out = output_dir / "part12_pipeline_bmx-trees.jpg"
    return [render_grid(rows, panel_width, out)]


def generate_part23_visual(root: Path, output_dir: Path, count: int, panel_width: int) -> list[Path]:
    seq = PART23_VIS_SEQUENCE
    fdir = frame_dir(root, seq)
    rows = []
    for key in choose_keys(fdir, count):
        original = read_by_key(fdir, key)
        if original is None:
            continue
        rows.append(
            [
                (f"Original {int(key):05d}" if key.isdigit() else f"Original {key}", original),
                ("GT mask", mask_overlay(root, seq, key, "gt", original)),
                ("ProPainter", read_by_key(root / "outputs" / "part3_inpaint" / seq / "propainter_frames", key)),
                ("Diffusion", read_by_key(root / "outputs" / "part3_inpaint" / seq / "final_frames", key)),
                ("DiffuEraser", read_by_key(root / "outputs" / "part3_diffueraser_inpaint" / seq / "final_frames", key)),
            ]
        )
    out = output_dir / "part23_bear_visual_comparison.jpg"
    return [render_grid(rows, panel_width, out)]


def write_latex_snippets(output_dir: Path, paths: list[Path]) -> Path:
    image_paths = [p for p in paths if p.suffix.lower() in {".jpg", ".png", ".pdf"}]
    lines = [
        "% Upload this folder to Overleaf and add:",
        f"\\graphicspath{{{{{output_dir.name}/}}}}",
        "",
    ]
    for path in sorted(image_paths):
        lines.extend(
            [
                "\\begin{figure}[t]",
                "    \\centering",
                f"    \\includegraphics[width=\\linewidth]{{{path.name}}}",
                f"    \\caption{{TODO: {path.stem}}}",
                f"    \\label{{fig:{path.stem.replace('_', '-')}}}",
                "\\end{figure}",
                "",
            ]
        )
    out = output_dir / "latex_snippets.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    output_dir = (args.output_dir or (root / "overleaf_figures")).expanduser().resolve()
    ensure_dir(output_dir)

    generated: list[Path] = []
    generated.extend(generate_mask_summary(root, output_dir, args.dpi))
    generated.extend(generate_part12_visuals(root, output_dir, args.num_keyframes, args.panel_width))
    generated.extend(generate_part23_visual(root, output_dir, args.num_keyframes, args.panel_width))
    generated.extend(generate_part12_pipeline_visual(root, output_dir, args.num_keyframes, args.panel_width))

    print("Generated files:")
    for path in generated:
        print(f"  {path}")


if __name__ == "__main__":
    main()
