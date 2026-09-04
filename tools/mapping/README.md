# tools/mapping — SurroundOcc-style nuScenes mapping

Static / dynamic LiDAR mapping aligned to the **last keyframe LIDAR_TOP** frame.
Uses official nuScenes poses + HEDNet/SimpleTrack `tracking_results.json`.
Does **not** re-run SLAM. Avoids mmcv (`points_in_boxes` is numpy/Open3D).

By default fuses **all LIDAR_TOP sweeps** (~10× keyframes) with boxes interpolated
between adjacent keyframes (lerp translation/size, Quaternion.slerp rotation).
Use `--keyframes-only` to restore the old 2 Hz keyframe-only path.

## Run

```bash
/data/software/conda/anaconda3/envs/mv2d/bin/python tools/mapping/run_mapping.py \
  --dataroot /data/data/automomous/nuscenes/v1.0-mini \
  --version v1.0-mini \
  --tracking-json output/track-gt-20260903-155401-CST/delivery/tracking_results.json \
  --scenes scene-0103,scene-0916 \
  --use-sweeps \
  --poisson
```

Flags:
- `--use-sweeps` (default on): walk `LIDAR_TOP` sample_data prev/next; interpolate boxes on non-keyframes
- `--keyframes-only`: disable sweeps (legacy keyframe loop)
- `--poisson` / `--poisson-depth N`: optional Open3D Poisson hole-filling on static (default depth 9)
- `--voxel-size 0.1`, `--box-expand 1.1`

See `docs/建图/在nuscenes数据集上进行建图.md`.
