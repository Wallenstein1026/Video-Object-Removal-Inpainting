import csv
import sys
from copy import copy
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from part1.extract_masks_yolov8seg import default_config as default_mask_config
from part1.extract_masks_yolov8seg import main as run_masks
from part1.inpaint_part1 import default_config as default_inpaint_config
from part1.inpaint_part1 import main as run_inpaint
from eval.eval_part1_masks import default_config as default_eval_config
from eval.eval_part1_masks import main as run_eval


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


# ---------------------------------------------------------------------------
# Config — 修改这里
# ---------------------------------------------------------------------------
PROJ_ROOT = PROJECT_ROOT
SOURCE = "davis"
VIDEOS = None
EXCLUDE = []

DO_MASKS = True
DO_INPAINT = True
DO_MASK_EVAL = True
SKIP_EXISTING = False
CONTINUE_ON_ERROR = True
DRY_RUN = False


def clone_config(cfg, **updates):
    out = copy(cfg)
    for key, value in updates.items():
        setattr(out, key, value)
    return out


def default_config():
    return SimpleNamespace(
        proj_root=PROJ_ROOT,
        source=SOURCE,
        videos=VIDEOS,
        exclude=EXCLUDE,
        do_masks=DO_MASKS,
        do_inpaint=DO_INPAINT,
        do_mask_eval=DO_MASK_EVAL,
        skip_existing=SKIP_EXISTING,
        continue_on_error=CONTINUE_ON_ERROR,
        dry_run=DRY_RUN,
        mask=default_mask_config(),
        inpaint=default_inpaint_config(),
        eval=default_eval_config(),
    )


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def list_davis_sequences(seq_root: Path):
    return sorted([p.name for p in seq_root.iterdir() if p.is_dir()])


def list_wild_inputs(wild_root: Path):
    items = []
    for p in sorted(wild_root.iterdir()):
        if p.is_dir():
            items.append(p)
        elif p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            items.append(p)
    return items


def main(cfg=None):
    if cfg is None:
        cfg = default_config()

    proj_root = Path(cfg.proj_root)
    seq_root = proj_root / "data" / "DAVIS" / "JPEGImages" / "480p"
    ann_root = proj_root / "data" / "DAVIS" / "Annotations" / "480p"
    wild_root = proj_root / "data" / "wild"

    if not (cfg.do_masks or cfg.do_inpaint or cfg.do_mask_eval):
        raise ValueError("At least one of do_masks / do_inpaint / do_mask_eval is required.")

    if cfg.source == "davis":
        if not seq_root.exists():
            raise FileNotFoundError(f"Missing DAVIS sequence root: {seq_root}")
        items = list_davis_sequences(seq_root)
        if cfg.videos:
            items = [v for v in items if v in set(cfg.videos)]
        if cfg.exclude:
            items = [v for v in items if v not in set(cfg.exclude)]
        print("Source          : DAVIS")
        print(f"Sequence root   : {seq_root}")
        print(f"Annotation root : {ann_root}")
        print(f"Num sequences   : {len(items)}")
    else:
        if not wild_root.exists():
            raise FileNotFoundError(f"Missing wild root: {wild_root}")
        items = list_wild_inputs(wild_root)
        if cfg.videos:
            items = [p for p in items if p.stem in set(cfg.videos) or p.name in set(cfg.videos)]
        if cfg.exclude:
            items = [p for p in items if (p.stem not in set(cfg.exclude) and p.name not in set(cfg.exclude))]
        print("Source        : wild")
        print(f"Wild root      : {wild_root}")
        print(f"Num items      : {len(items)}")

    summary_rows = []

    for item in items:
        if cfg.source == "davis":
            video_name = item
            input_path = seq_root / video_name
        else:
            input_path = item
            video_name = item.stem if item.is_file() else item.name

        print(f"\n========== {video_name} ==========")

        mask_out = Path(cfg.mask.output_root) / f"{video_name}_yolov8seg"
        mask_union_dir = mask_out / "masks_union"
        inpaint_out = Path(cfg.inpaint.output_root) / video_name
        eval_out = Path(cfg.eval.output_dir).parent / f"{video_name}_yolov8seg_eval"

        try:
            if cfg.do_masks:
                mask_probe = mask_union_dir / "00000.png"
                if cfg.skip_existing and mask_probe.exists():
                    print(f"[SKIP] masks exist for {video_name}")
                elif cfg.dry_run:
                    print(f"[DRY] masks: {input_path} -> {mask_out}")
                else:
                    run_masks(clone_config(
                        cfg.mask,
                        proj_root=proj_root,
                        input=input_path,
                        video_name=video_name,
                    ))

            if cfg.do_inpaint:
                video_probe = inpaint_out / f"{video_name}.mp4"
                if cfg.skip_existing and video_probe.exists():
                    print(f"[SKIP] inpaint video exists for {video_name}")
                elif cfg.dry_run:
                    print(f"[DRY] inpaint: {input_path} + {mask_union_dir} -> {inpaint_out}")
                else:
                    run_inpaint(clone_config(
                        cfg.inpaint,
                        proj_root=proj_root,
                        input=input_path,
                        video_name=video_name,
                        masks_root=Path(cfg.mask.output_root),
                    ))

            mask_metrics = {}
            if cfg.do_mask_eval:
                if cfg.source != "davis":
                    print(f"[INFO] skip mask eval for wild: {video_name}")
                elif not (ann_root / video_name).exists():
                    print(f"[WARN] missing GT masks for {video_name}, skip mask eval")
                elif cfg.dry_run:
                    print(f"[DRY] eval: {mask_union_dir} vs {ann_root / video_name} -> {eval_out}")
                else:
                    summary = run_eval(clone_config(
                        cfg.eval,
                        proj_root=proj_root,
                        video_name=video_name,
                        pred_dir=mask_union_dir,
                        gt_dir=ann_root / video_name,
                        output_dir=eval_out,
                    ))
                    mask_metrics = {
                        "JM": f"{summary['JM']:.6f}",
                        "JR": f"{summary['JR']:.6f}",
                    }

            summary_rows.append({
                "video_name": video_name,
                "JM": mask_metrics.get("JM", ""),
                "JR": mask_metrics.get("JR", ""),
                "status": "ok",
            })

        except Exception as e:
            print(f"[ERROR] {video_name} failed: {e}")
            summary_rows.append({
                "video_name": video_name,
                "JM": "",
                "JR": "",
                "status": f"failed({type(e).__name__})",
            })
            if not cfg.continue_on_error:
                raise

    if not cfg.dry_run:
        out_csv = proj_root / "outputs" / "figures" / f"part1_batch_summary_{cfg.source}.csv"
        ensure_dir(out_csv.parent)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["video_name", "JM", "JR", "status"])
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nSaved batch summary to: {out_csv}")


if __name__ == "__main__":
    main()
