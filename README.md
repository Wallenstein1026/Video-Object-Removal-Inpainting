# Video Object Removal and Inpainting

This repository contains the cleaned project code and final PDF report for a
three-part video object removal pipeline.

- Part 1: YOLOv8-Seg masks, sparse optical-flow dynamic filtering, temporal
  borrowing, and OpenCV Telea inpainting.
- Part 2: YOLO prompts, SAM2 mask propagation, and ProPainter video
  inpainting.
- Part 3: diffusion-based restoration variants using the Part 2 masks,
  including SDXL-based diffusion refinement and DiffuEraser.

The latest report evaluates mask quality on DAVIS and inpainting quality on
wild videos with manually selected clean-background frames. Inpainting scores
are computed in the edit region with `edit_psnr_mean` and `edit_ssim_mean`, not
over the full frame.

## Repository Layout

```text
src/
  preprocess/     video-to-DAVIS-style frame conversion
  part1/          classical mask extraction and inpainting baseline
  part2/          YOLO, SAM2, and ProPainter pipeline
  part3/          SDXL and DiffuEraser case-study wrappers
  eval/           mask and inpainting evaluation scripts
  plot/           figure generation utilities
  run_wild_video_pipeline.py
CV_report.pdf    final report
data/             local data placeholder, not tracked
models/           local model placeholder, not tracked
```

## Setup

Create an environment with the common dependencies:

```bash
pip install -r requirements.txt
```

Part 2 and Part 3 also require external repositories and their checkpoints:
SAM2, ProPainter, and DiffuEraser. See `models/README.md` for the expected
local layout. Large data, model weights, cache folders, and generated videos are
not included in Git.

## Running

For a complete wild-video run, edit the configuration block at the top of
`src/run_wild_video_pipeline.py` to choose:

- `VIDEO_NAME`
- `GT_SOURCE`: `first`, `last`, or `image`
- `GT_IMAGE_PATH`, when using a specific clean background image
- `EVAL_USE_MASK`: set to `True` for edit-region evaluation using the Part 2
  mask as the ROI

Then run:

```bash
python src/run_wild_video_pipeline.py --video-name walk
```

Part 1 on DAVIS:

```bash
python src/part1/run_part1_batch.py
```

Part 2 on selected DAVIS sequences:

```bash
python src/part2/run_part2_davis.py --sequences bear,tennis --weights yolo11x.pt
```

Part 3 case studies use the configuration blocks at the top of:

```bash
python src/part3/run_part3_diffusion_inpaint.py
python src/part3/run_part3_diffueraser_inpaint.py
```

Optionally generate report figures locally:

```bash
python src/plot/plot_figure.py --validate-only
python src/plot/plot_figure.py
```

The generated figures are written to `outputs/vis_new/`, which is ignored by
Git because it can become large.

## Results

The final report is provided as `CV_report.pdf`. The main mask-quality result
on nine DAVIS sequences is:

| Method | Mean JM | Mean JR |
| --- | ---: | ---: |
| Part 1 YOLOv8-Seg + flow | 0.5951 | 0.7393 |
| Part 2 YOLO + SAM2 | 0.8299 | 0.9062 |

For wild-video inpainting with clean-background references, the report compares
five videos: `Valorant`, `walk`, `walk_tree`, `fish`, and `rotation`.

| Method | Valid videos | Edit PSNR | Edit SSIM |
| --- | ---: | ---: | ---: |
| Part 1 Telea | 4 | 14.61 | 0.9710 |
| Part 2 ProPainter | 5 | 17.30 | 0.9824 |
| Part 3 Diffusion | 5 | 16.52 | 0.9788 |
| Part 3 DiffuEraser | 5 | 17.79 | 0.9848 |

Part 1 has no valid `Valorant` edit-region score because it did not produce a
valid edit mask/result under the selected evaluation setting.

## Report

See `CV_report.pdf` for the full write-up, figures, quantitative tables, and
analysis.
