from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Config
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "vis_new"

DAVIS_MASK_EVAL_SEQUENCES = [
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

WILD_INPAINT_EVAL_SEQUENCES = [
    "Valorant",
    "walk",
    "walk_tree",
    "fish",
    "rotation",
]

MASK_GRID_SEQUENCES = [
    "bmx-trees",
    "tennis",
    "walk",
    "walk_tree",
    "dog-agility",
    "breakdance-flare",
]

INPAINT_GRID_SEQUENCES = [
    "bmx-trees",
    "tennis",
    "bear",
    "rotation",
    "crossing",
    "fish",
]

PIPELINE_SEQUENCE = "bmx-trees"
PIPELINE_FRAME_KEY = "5"
DEFAULT_FRAME_COUNT = 6
DEFAULT_PANEL_WIDTH = 220
DEFAULT_DPI = 220

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

COLORS = {
    "part1": "#2E7D32",
    "part2": "#1565C0",
    "part3_diffusion": "#6A1B9A",
    "part3_diffueraser": "#C62828",
    "gt": "#F9A825",
}

MASK_OVERLAY_BGR = {
    "gt": (37, 168, 249),
    "part1": (50, 125, 46),
    "part2": (192, 101, 21),
}

METHOD_LABELS = {
    "part1": "Part1",
    "part2": "Part2 ProPainter",
    "part3_diffusion": "Part3 Diffusion",
    "part3_diffueraser": "Part3 DiffuEraser",
}


@dataclass
class MissingAssets:
    items: list[str]

    def add(self, message: str) -> None:
        self.items.append(message)

    def has_any(self) -> bool:
        return bool(self.items)

    def report(self) -> str:
        if not self.items:
            return "No missing assets."
        lines = ["Missing assets:"]
        lines.extend(f"  - {item}" for item in self.items)
        return "\n".join(lines)


# =============================================================================
# Basic IO and path helpers
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report visualization figures.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--frame-count", type=int, default=DEFAULT_FRAME_COUNT)
    parser.add_argument("--panel-width", type=int, default=DEFAULT_PANEL_WIDTH)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--figures",
        default="all",
        help="Comma-separated subset: fig1,fig2,fig3,fig4,fig5,all",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return load_json(path)


def sorted_image_paths(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def normalize_key(stem: str) -> str:
    return str(int(stem)) if stem.isdigit() else stem


def frame_sort_key(key: str) -> tuple[int, str]:
    if key.isdigit():
        return int(key), key
    match = re.search(r"(\d+)$", key)
    return int(match.group(1)) if match else 10**9, key


def image_map(folder: Path) -> dict[str, Path]:
    return {normalize_key(path.stem): path for path in sorted_image_paths(folder)}


def choose_keys(folder: Path, count: int) -> list[str]:
    keys = sorted(image_map(folder).keys(), key=frame_sort_key)
    if not keys:
        return []
    if len(keys) <= count:
        return keys
    idxs = np.linspace(0, len(keys) - 1, count, dtype=int).tolist()
    return [keys[i] for i in idxs]


def read_bgr(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return image


def read_gray(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def read_by_key(folder: Path, key: str) -> np.ndarray | None:
    return read_bgr(image_map(folder).get(normalize_key(key)))


def read_gray_by_key(folder: Path, key: str) -> np.ndarray | None:
    return read_gray(image_map(folder).get(normalize_key(key)))


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


def part1_mask_dir(root: Path, seq: str) -> Path:
    return root / "outputs" / "part1_masks" / f"{seq}_yolov8seg" / "masks_union"


def part1_inpaint_dir(root: Path, seq: str) -> Path:
    return root / "outputs" / "part1_inpaint" / seq / "frames"


def part2_yolo_vis_dir(root: Path, seq: str) -> Path:
    return root / "outputs" / "part2_masks" / f"{seq}_yolo" / "yolo_vis"


def part2_mask_dir(root: Path, seq: str) -> Path:
    return root / "outputs" / "part2_masks" / f"{seq}_sam2" / "masks_union"


def part2_inpaint_dir(root: Path, seq: str) -> Path:
    base = root / "outputs" / "part2_inpaint" / seq
    candidates = [
        base / seq / "frames",
        base / "frames",
        base / "frames" / "frames",
    ]
    if base.exists():
        candidates.extend(sorted(path for path in base.rglob("frames") if path.is_dir()))
    for candidate in candidates:
        if sorted_image_paths(candidate):
            return candidate
    return candidates[0]


def part3_diffusion_dir(root: Path, seq: str) -> Path:
    return root / "outputs" / "part3_inpaint" / seq / "final_frames"


def part3_diffueraser_dir(root: Path, seq: str) -> Path:
    return root / "outputs" / "part3_diffueraser_inpaint" / seq / "final_frames"


def mask_eval_summary_path(root: Path, method: str, seq: str) -> Path:
    if method == "part1":
        return root / "outputs" / "part1_eval" / f"{seq}_yolov8seg_eval" / "summary.json"
    if method == "part2":
        return root / "outputs" / "part2_eval" / f"{seq}_sam2_eval" / "summary.json"
    raise ValueError(f"Unknown mask method: {method}")


def inpaint_eval_summary_candidates(root: Path, method: str, seq: str) -> list[Path]:
    if method == "part1":
        base = root / "outputs" / "part1_eval"
        patterns = [f"{seq}_*_gt_inpaint_eval/summary.json", f"{seq}_*frame_gt_inpaint_eval/summary.json"]
    elif method == "part2":
        base = root / "outputs" / "part2_eval"
        patterns = [f"{seq}_*_gt_inpaint_eval/summary.json", f"{seq}_*frame_gt_inpaint_eval/summary.json"]
    elif method == "part3_diffusion":
        base = root / "outputs" / "part3_eval"
        patterns = [f"{seq}_diffusion_*_gt_inpaint_eval/summary.json", f"{seq}_diffusion_*frame_gt_inpaint_eval/summary.json"]
    elif method == "part3_diffueraser":
        base = root / "outputs" / "part3_eval"
        patterns = [f"{seq}_diffueraser_*_gt_inpaint_eval/summary.json", f"{seq}_diffueraser_*frame_gt_inpaint_eval/summary.json"]
    else:
        raise ValueError(f"Unknown inpaint method: {method}")

    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(base.glob(pattern)))
    # Prefer summaries generated by the newer clean-background full-frame eval.
    return sorted(set(matches), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)


def inpaint_eval_summary_path(root: Path, method: str, seq: str) -> Path:
    candidates = inpaint_eval_summary_candidates(root, method, seq)
    return candidates[0] if candidates else Path("__missing__")


# =============================================================================
# Rendering helpers
# =============================================================================

def save_bar_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> list[Path]:
    ensure_dir(output_dir)
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    for path in paths:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return paths


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def resize_panel(image: np.ndarray, panel_width: int) -> np.ndarray:
    h, w = image.shape[:2]
    new_h = max(1, int(round(h * panel_width / float(w))))
    return cv2.resize(image, (panel_width, new_h), interpolation=cv2.INTER_AREA)


def label_panel(image: np.ndarray, text: str) -> np.ndarray:
    out = cv2.copyMakeBorder(image, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(32, 32, 32))
    scale = 0.46
    while scale > 0.25:
        text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        if text_width <= out.shape[1] - 12:
            break
        scale -= 0.03
    cv2.putText(out, text, (7, 20), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def blank_panel(shape_hw: tuple[int, int], text: str = "missing") -> np.ndarray:
    h, w = shape_hw
    image = np.full((h, w, 3), 245, dtype=np.uint8)
    cv2.putText(image, text, (10, max(24, h // 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 90), 2, cv2.LINE_AA)
    return image


def mask_to_bgr(mask: np.ndarray | None, shape_hw: tuple[int, int] | None = None) -> np.ndarray | None:
    if mask is None:
        return None
    if shape_hw is not None and mask.shape != shape_hw:
        mask = cv2.resize(mask, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    binary = (mask > 0).astype(np.uint8) * 255
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def overlay_mask(
    frame: np.ndarray | None,
    mask: np.ndarray | None,
    method: str,
    alpha: float = 0.68,
) -> np.ndarray | None:
    if frame is None or mask is None:
        return None
    if mask.shape != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    out = frame.copy()
    color = np.zeros_like(out)
    color[:, :] = MASK_OVERLAY_BGR[method]
    blended = cv2.addWeighted(out, 1.0 - alpha, color, alpha, 0)
    out[mask > 0] = blended[mask > 0]
    return out


def render_grid(rows: list[list[tuple[str, np.ndarray | None]]], panel_width: int, output_path: Path) -> Path:
    rendered_rows: list[np.ndarray] = []
    for row in rows:
        base = next((image for _, image in row if image is not None), None)
        if base is None:
            base = np.full((240, 426, 3), 245, dtype=np.uint8)
        target_hw = base.shape[:2]
        panels = []
        for label, image in row:
            if image is None:
                image = blank_panel(target_hw)
            elif image.shape[:2] != target_hw:
                image = cv2.resize(image, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_AREA)
            panels.append(label_panel(resize_panel(image, panel_width), label))
        rendered_rows.append(np.concatenate(panels, axis=1))

    if not rendered_rows:
        return output_path
    max_width = max(row.shape[1] for row in rendered_rows)
    padded_rows = []
    for row in rendered_rows:
        if row.shape[1] < max_width:
            pad = np.full((row.shape[0], max_width - row.shape[1], 3), 255, dtype=np.uint8)
            row = np.concatenate([row, pad], axis=1)
        padded_rows.append(row)
    grid = np.concatenate(padded_rows, axis=0)
    ensure_dir(output_path.parent)
    cv2.imwrite(str(output_path), grid, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    return output_path


def make_contact_sheet(image_paths_list: list[Path], output_path: Path, max_width: int = 1800) -> Path | None:
    images = [read_bgr(path) for path in image_paths_list if path.exists()]
    images = [image for image in images if image is not None]
    if not images:
        return None
    scaled = []
    for image in images:
        h, w = image.shape[:2]
        scale = min(1.0, max_width / float(w))
        scaled.append(cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA))
    width = max(image.shape[1] for image in scaled)
    padded = []
    for image in scaled:
        if image.shape[1] < width:
            pad = np.full((image.shape[0], width - image.shape[1], 3), 255, dtype=np.uint8)
            image = np.concatenate([image, pad], axis=1)
        padded.append(image)
    sheet = np.concatenate(padded, axis=0)
    ensure_dir(output_path.parent)
    cv2.imwrite(str(output_path), sheet, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    return output_path


# =============================================================================
# Validation
# =============================================================================

def require_file(path: Path, missing: MissingAssets, label: str) -> None:
    if not path.exists():
        missing.add(f"{label}: {path}")


def require_image_dir(path: Path, missing: MissingAssets, label: str) -> None:
    if not sorted_image_paths(path):
        missing.add(f"{label}: {path}")


def validate_assets(root: Path, figures: set[str]) -> MissingAssets:
    missing = MissingAssets([])

    if "fig1" in figures:
        for seq in DAVIS_MASK_EVAL_SEQUENCES:
            require_file(mask_eval_summary_path(root, "part1", seq), missing, f"fig1 Part1 mask summary {seq}")
            require_file(mask_eval_summary_path(root, "part2", seq), missing, f"fig1 Part2 mask summary {seq}")

    if "fig2" in figures:
        for seq in WILD_INPAINT_EVAL_SEQUENCES:
            for method in METHOD_LABELS:
                require_file(inpaint_eval_summary_path(root, method, seq), missing, f"fig2 {method} summary {seq}")

    if "fig3" in figures:
        seq = PIPELINE_SEQUENCE
        for label, folder in [
            ("fig3 original frames", frame_dir(root, seq)),
            ("fig3 GT masks", gt_mask_dir(root, seq)),
            ("fig3 Part1 masks", part1_mask_dir(root, seq)),
            ("fig3 Part1 result", part1_inpaint_dir(root, seq)),
            ("fig3 Part2 YOLO vis", part2_yolo_vis_dir(root, seq)),
            ("fig3 Part2 masks", part2_mask_dir(root, seq)),
            ("fig3 Part2 result", part2_inpaint_dir(root, seq)),
            ("fig3 Part3 diffusion result", part3_diffusion_dir(root, seq)),
            ("fig3 Part3 DiffuEraser result", part3_diffueraser_dir(root, seq)),
        ]:
            require_image_dir(folder, missing, f"{label} {seq}")

    if "fig4" in figures:
        for seq in MASK_GRID_SEQUENCES:
            for label, folder in [
                ("fig4 frames", frame_dir(root, seq)),
                ("fig4 GT masks", gt_mask_dir(root, seq)),
                ("fig4 Part1 masks", part1_mask_dir(root, seq)),
                ("fig4 Part2 masks", part2_mask_dir(root, seq)),
            ]:
                require_image_dir(folder, missing, f"{label} {seq}")

    if "fig5" in figures:
        for seq in INPAINT_GRID_SEQUENCES:
            for label, folder in [
                ("fig5 frames", frame_dir(root, seq)),
                ("fig5 Part1 result", part1_inpaint_dir(root, seq)),
                ("fig5 Part2 result", part2_inpaint_dir(root, seq)),
                ("fig5 Part3 diffusion result", part3_diffusion_dir(root, seq)),
                ("fig5 Part3 DiffuEraser result", part3_diffueraser_dir(root, seq)),
            ]:
                require_image_dir(folder, missing, f"{label} {seq}")

    return missing


# =============================================================================
# Figure generators
# =============================================================================

def generate_fig1(root: Path, output_dir: Path, dpi: int) -> list[Path]:
    metrics = [("JM", "Mean IoU $J_M$"), ("JR", "Recall $J_R$")]
    x = np.arange(len(DAVIS_MASK_EVAL_SEQUENCES))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)

    for ax, (metric, title) in zip(axes, metrics):
        part1 = [safe_float(load_json_or_empty(mask_eval_summary_path(root, "part1", seq)).get(metric)) for seq in DAVIS_MASK_EVAL_SEQUENCES]
        part2 = [safe_float(load_json_or_empty(mask_eval_summary_path(root, "part2", seq)).get(metric)) for seq in DAVIS_MASK_EVAL_SEQUENCES]
        ax.bar(x - width / 2, part1, width, color=COLORS["part1"], label="Part1")
        ax.bar(x + width / 2, part2, width, color=COLORS["part2"], label="Part2")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(DAVIS_MASK_EVAL_SEQUENCES, rotation=35, ha="right")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Score")
    axes[1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return save_bar_figure(fig, output_dir, "fig1_davis_mask_eval_part1_part2", dpi)


def generate_fig2(root: Path, output_dir: Path, dpi: int) -> list[Path]:
    methods = ["part1", "part2", "part3_diffusion", "part3_diffueraser"]
    metrics = [("edit_psnr_mean", "Edit PSNR ↑"), ("edit_ssim_mean", "Edit SSIM ↑")]
    x = np.arange(len(WILD_INPAINT_EVAL_SEQUENCES))
    width = 0.18
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))

    for ax, (metric, title) in zip(axes, metrics):
        for i, method in enumerate(methods):
            values = []
            for seq in WILD_INPAINT_EVAL_SEQUENCES:
                summary_path = inpaint_eval_summary_path(root, method, seq)
                values.append(safe_float(load_json_or_empty(summary_path).get(metric)))
            offset = (i - (len(methods) - 1) / 2) * width
            ax.bar(x + offset, values, width, color=COLORS[method], label=METHOD_LABELS[method])
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(WILD_INPAINT_EVAL_SEQUENCES, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
        if metric == "edit_ssim_mean":
            ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("Value")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return save_bar_figure(fig, output_dir, "fig2_wild_inpaint_eval_methods", dpi)


def generate_fig3(root: Path, output_dir: Path, panel_width: int) -> Path:
    seq = PIPELINE_SEQUENCE
    key = normalize_key(PIPELINE_FRAME_KEY)
    original = read_by_key(frame_dir(root, seq), key)
    gt_mask = overlay_mask(original, read_gray_by_key(gt_mask_dir(root, seq), key), "gt")
    part1_mask = overlay_mask(original, read_gray_by_key(part1_mask_dir(root, seq), key), "part1")
    part2_mask = overlay_mask(original, read_gray_by_key(part2_mask_dir(root, seq), key), "part2")

    rows = [[
        ("Original", original),
        ("GT overlay", gt_mask),
        ("Part1 overlay", part1_mask),
        ("Part1 result", read_by_key(part1_inpaint_dir(root, seq), key)),
        ("Part2 YOLO", read_by_key(part2_yolo_vis_dir(root, seq), key)),
        ("Part2 overlay", part2_mask),
        ("Part2 result", read_by_key(part2_inpaint_dir(root, seq), key)),
        ("Part3 diffusion", read_by_key(part3_diffusion_dir(root, seq), key)),
        ("Part3 DiffuEraser", read_by_key(part3_diffueraser_dir(root, seq), key)),
    ]]
    return render_grid(rows, panel_width, output_dir / "fig3_pipeline_bmx_trees_frame005.png")


def generate_fig4(root: Path, output_dir: Path, frame_count: int, panel_width: int) -> list[Path]:
    paths = []
    for seq in MASK_GRID_SEQUENCES:
        keys = choose_keys(frame_dir(root, seq), frame_count)
        rows = []
        frames = {key: read_by_key(frame_dir(root, seq), key) for key in keys}
        for row_label, folder, method in [
            ("GT", gt_mask_dir(root, seq), "gt"),
            ("Part1", part1_mask_dir(root, seq), "part1"),
            ("Part2", part2_mask_dir(root, seq), "part2"),
        ]:
            row = []
            for key in keys:
                panel = overlay_mask(frames[key], read_gray_by_key(folder, key), method)
                row.append((f"{row_label} {key}", panel))
            rows.append(row)
        out_path = output_dir / f"fig4_mask_grid_{seq}.png"
        paths.append(render_grid(rows, panel_width, out_path))
    sheet = make_contact_sheet(paths, output_dir / "fig4_mask_grid_contact_sheet.png")
    if sheet is not None:
        paths.append(sheet)
    return paths


def generate_fig5(root: Path, output_dir: Path, frame_count: int, panel_width: int) -> list[Path]:
    paths = []
    for seq in INPAINT_GRID_SEQUENCES:
        keys = choose_keys(frame_dir(root, seq), frame_count)
        rows = []
        for row_label, folder in [
            ("Original", frame_dir(root, seq)),
            ("Part1", part1_inpaint_dir(root, seq)),
            ("Part2", part2_inpaint_dir(root, seq)),
            ("Part3 diffusion", part3_diffusion_dir(root, seq)),
            ("Part3 DiffuEraser", part3_diffueraser_dir(root, seq)),
        ]:
            row = []
            for key in keys:
                row.append((f"{row_label} {key}", read_by_key(folder, key)))
            rows.append(row)
        out_path = output_dir / f"fig5_inpaint_grid_{seq}.png"
        paths.append(render_grid(rows, panel_width, out_path))
    sheet = make_contact_sheet(paths, output_dir / "fig5_inpaint_grid_contact_sheet.png")
    if sheet is not None:
        paths.append(sheet)
    return paths


def selected_figures(raw: str) -> set[str]:
    items = {item.strip().lower() for item in raw.split(",") if item.strip()}
    if not items or "all" in items:
        return {"fig1", "fig2", "fig3", "fig4", "fig5"}
    valid = {"fig1", "fig2", "fig3", "fig4", "fig5"}
    unknown = items - valid
    if unknown:
        raise ValueError(f"Unknown figures: {', '.join(sorted(unknown))}")
    return items


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    output_dir = (args.output_dir or (root / "outputs" / "vis_new")).resolve()
    figures = selected_figures(args.figures)

    missing = validate_assets(root, figures)
    if missing.has_any():
        print(missing.report(), file=sys.stderr)
        if not args.allow_missing:
            print("\nUse --allow-missing to render placeholder panels for qualitative figures.", file=sys.stderr)
            return 2
    else:
        print("Validation passed: all requested assets are present.")

    if args.validate_only:
        return 0 if not missing.has_any() else 2

    ensure_dir(output_dir)
    generated: list[Path] = []
    if "fig1" in figures:
        generated.extend(generate_fig1(root, output_dir, args.dpi))
    if "fig2" in figures:
        generated.extend(generate_fig2(root, output_dir, args.dpi))
    if "fig3" in figures:
        generated.append(generate_fig3(root, output_dir, args.panel_width))
    if "fig4" in figures:
        generated.extend(generate_fig4(root, output_dir, args.frame_count, args.panel_width))
    if "fig5" in figures:
        generated.extend(generate_fig5(root, output_dir, args.frame_count, args.panel_width))

    print("\nGenerated figures:")
    for path in generated:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
