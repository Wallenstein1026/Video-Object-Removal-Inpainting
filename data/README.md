# Data Layout

The repository does not include DAVIS or wild videos because they are large.
Place data in this layout before running the scripts:

```text
data/
  DAVIS/
    JPEGImages/480p/<sequence>/*.jpg
    Annotations/480p/<sequence>/*.png
  wild/
    <video>.mp4
    JPEGImages/480p/<sequence>/*.jpg
    ImageSets/2017/<sequence>
```

The report experiments used DAVIS sequences `bear`, `bike-packing`,
`bmx-trees`, `boxing-fisheye`, `breakdance-flare`, `crossing`,
`dog-agility`, `drift-chicane`, and `tennis`, plus wild videos `Valorant`,
`walk`, `walk_tree`, `fish`, and `rotation`.

For wild videos, `src/run_wild_video_pipeline.py` can convert an input video into
the DAVIS-style frame folder automatically. The clean-background evaluation
uses a selected first frame, last frame, or external clean image as the
reference background.
