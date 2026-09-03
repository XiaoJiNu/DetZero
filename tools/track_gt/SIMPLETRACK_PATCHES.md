# SimpleTrack local patches

Upstream: https://github.com/tusen-ai/SimpleTrack @ `05c96bb7ed98fc179856f327544612a66c839b5e` (MIT)

Patched only under `third_party/SimpleTrack/preprocessing/nuscenes_data/`:

1. `token_info.py`, `time_stamp.py`, `sensor_calibration.py`, `ego_pose.py`, `gt_info.py`, `raw_pc.py`
   - Added `--version` (default `v1.0-trainval`) and `--split` (default `val`, or `mini_val` when version contains `mini`)
2. `detection.py`
   - Skip sample tokens not present in preprocessed `token_info` (avoids silent mis-assignment)
3. `nuscenes_preprocess.sh`
   - Optional 4th/5th args: version, split

Vendor tree is gitignored; clone URL + commit recorded in `output/.../assets_manifest.json` / `run_manifest.json`.
