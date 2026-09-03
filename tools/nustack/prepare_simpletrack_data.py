#!/usr/bin/env python3
"""N2a: build SimpleTrack's 2Hz preprocessed data layout for the nuScenes
mini_val scenes (scene-0103, scene-0916) using the devkit tables.

Mirrors third_party/SimpleTrack/preprocessing/nuscenes_data/*.py field-for-field
(2hz mode: iterate `sample` records; ego_info = LIDAR ego_pose translation+q;
ts_info = sample timestamp us; calib_info = LIDAR translation+q; gt_info per
frame via nusc.get_boxes; pc = raw lidar bin 5-col kept as (N,4)).

Then runs SimpleTrack's OWN detection.py to convert a nuScenes-format
detection json into the per-sequence dets npz. Third-party code unmodified.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from nuscenes.nuscenes import NuScenes

SCENES = ["scene-0103", "scene-0916"]  # default: mini_val


def frame_stream(nusc: NuScenes, scene_name: str):
    scene = next(s for s in nusc.scene if s["name"] == scene_name)
    cur = scene["first_sample_token"]
    while cur:
        sample = nusc.get("sample", cur)
        yield sample
        cur = sample["next"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--scenes", nargs="*", default=None,
                        help="scene names; default = mini_val pair; 'val' = all val scenes")
    parser.add_argument("--no-pc", action="store_true",
                        help="skip raw point clouds (G2 needs no pc; configs pc:false)")
    args = parser.parse_args()

    nusc = NuScenes(args.version, dataroot=str(args.dataroot), verbose=False)
    from nuscenes.utils import splits
    scenes = args.scenes or SCENES
    if scenes == ["val"]:
        scenes = splits.create_splits_scenes()["val"]
    out = args.out
    for sub in ("token_info", "ts_info", "ego_info", "calib_info", "gt_info",
                os.path.join("pc", "raw_pc")):
        (out / sub).mkdir(parents=True, exist_ok=True)

    for scene_name in scenes:
        frames = list(frame_stream(nusc, scene_name))
        tokens, tss = [], []
        ego_data, calib_data, pc_data = {}, {}, {}
        gt_ids, gt_types, gt_bboxes = [], [], []
        for i, sample in enumerate(frames):
            tokens.append(sample["token"])
            tss.append(sample["timestamp"])
            lidar = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            ego = nusc.get("ego_pose", lidar["ego_pose_token"])
            calib = nusc.get("calibrated_sensor", lidar["calibrated_sensor_token"])
            ego_data[str(i)] = ego["translation"] + ego["rotation"]
            calib_data[str(i)] = calib["translation"] + calib["rotation"]
            if not args.no_pc:
                pts = np.fromfile(str(args.dataroot / lidar["filename"]), dtype=np.float32)
                pc_data[str(i)] = pts.reshape(-1, 5)[:, :4]
            ids_f, types_f, boxes_f = [], [], []
            for box in nusc.get_boxes(sample["data"]["LIDAR_TOP"]):
                ids_f.append(box.token)
                types_f.append(box.name)
                boxes_f.append(box.center.tolist() + box.wlh.tolist()
                               + box.orientation.q.tolist())
            gt_ids.append(ids_f)
            gt_types.append(types_f)
            gt_bboxes.append(boxes_f)

        (out / "token_info" / f"{scene_name}.json").write_text(json.dumps(tokens))
        (out / "ts_info" / f"{scene_name}.json").write_text(json.dumps(tss))
        np.savez_compressed(out / "ego_info" / f"{scene_name}.npz", **ego_data)
        np.savez_compressed(out / "calib_info" / f"{scene_name}.npz", **calib_data)
        if not args.no_pc:
            np.savez_compressed(out / "pc" / "raw_pc" / f"{scene_name}.npz", **pc_data)
        np.savez_compressed(out / "gt_info" / f"{scene_name}.npz",
                            ids=np.array(gt_ids, dtype=object),
                            types=np.array(gt_types, dtype=object),
                            bboxes=np.array(gt_bboxes, dtype=object),
                            allow_pickle=True)
        print(f"{scene_name}: {len(frames)} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
