# tools/mapping — SurroundOcc-style nuScenes mapping

Static / dynamic LiDAR mapping aligned to the **last keyframe LIDAR_TOP** frame.
Uses official nuScenes poses + HEDNet/SimpleTrack `tracking_results.json`.
Does **not** re-run SLAM. Avoids mmcv (`points_in_boxes` is numpy/Open3D).

## Run

```bash
/data/software/conda/anaconda3/envs/mv2d/bin/python tools/mapping/run_mapping.py \
  --dataroot /data/data/automomous/nuscenes/v1.0-mini \
  --version v1.0-mini \
  --tracking-json output/track-gt-20260903-155401-CST/delivery/tracking_results.json \
  --scenes scene-0103,scene-0916
```

Optional: `--poisson` (static only), `--voxel-size 0.1`, `--box-expand 1.1`.

See `docs/建图/在nuscenes数据集上进行建图.md`.
