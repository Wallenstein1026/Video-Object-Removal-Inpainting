# Model Weights

Large model files are intentionally excluded from Git.

Expected local paths:

```text
models/
  sdxl-inpainting/              # local SDXL inpainting Diffusers model
repos/
  sam2/                         # SAM2 repository with checkpoints/
  ProPainter/                   # ProPainter repository with weights/
  DiffuEraser/                  # DiffuEraser repository with weights/
```

YOLO weights such as `yolov8s-seg.pt` and `yolo11x.pt` can be placed at the
repository root or passed explicitly with `--weights`.

Recommended environment split:

- Use the main project environment for Part 1, Part 2, ProPainter, evaluation,
  and plotting.
- Use the DiffuEraser-compatible diffusion environment for DiffuEraser if your
  main environment has incompatible `diffusers`, `transformers`, or `peft`
  versions.

The wild-video runner exposes these paths in its top-level configuration:

- `PART23_PYTHON`
- `DIFFUERASER_PYTHON`
- `SAM2_ROOT`
- `PROPAINTER_ROOT`
- `DIFFUERASER_ROOT`
