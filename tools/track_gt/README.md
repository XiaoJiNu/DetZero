# track_gt — nuScenes HEDNet + SimpleTrack tracking GT

Product goal B: stable `track_id` + official nuScenes tracking JSON (+ AMOTA if env allows).

## Quick run

```bash
./tools/track_gt/run_pipeline.sh
# or
/data/software/conda/anaconda3/envs/mv2d/bin/python tools/track_gt/run_track_gt_pipeline.py
```

Outputs land in `output/track-gt-<stamp>-CST/` with `delivery/tracking_results.json`.

## Notes

- Detector: existing HEDNet `result.pkl` (no retrain).
- Tracker: `third_party/SimpleTrack` (MIT). Preprocess scripts patched for `--version` / `--split`.
- Official eval may need isolated env `nusc-track-gt` with `motmetrics==0.9.9` if mv2d's motmetrics 1.4 breaks.
