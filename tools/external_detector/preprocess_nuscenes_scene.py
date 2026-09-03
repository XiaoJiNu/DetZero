#!/usr/bin/env python3
"""Stream one nuScenes-mini scene into DetZero's Waymo-style data layout (S1).

Scene frames are resolved through the devkit scene.json/sample.json forward
sample chain, intersected with the info pickle. The generated root carries NO
ground truth; GT stays in the original nuScenes info pickle and is read only
by the S7 evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import os
import pickle
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.external_detector.pipeline import publish_preprocessed_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nuscenes-info", type=Path, required=True,
                        help="nuscenes_infos_10sweeps_val.pkl")
    parser.add_argument("--nuscenes-root", type=Path, required=True,
                        help="dir holding samples/ with the .pcd.bin files")
    parser.add_argument("--devkit-dir", type=Path,
                        help="dir with scene.json/sample.json (default: <nuscenes-info parent>/v1.0-mini)")
    parser.add_argument("--scene-name", type=str, required=True,
                        help="devkit scene name, e.g. scene-0103")
    parser.add_argument("--root-path", type=Path, required=True,
                        help="fresh DetZero-style data root for this scene")
    parser.add_argument("--expected-frames", type=int, required=True)
    return parser.parse_args()


def scene_tokens(info_dir: Path, scene_name: str) -> list[str]:
    scenes = json.load((info_dir / "scene.json").open())
    samples = {s["token"]: s for s in json.load((info_dir / "sample.json").open())}
    scene = next(s for s in scenes if s["name"] == scene_name)
    tokens = []
    token = scene["first_sample_token"]
    while token:
        if token not in samples:
            raise ValueError(f"broken sample chain at {token}")
        tokens.append(token)
        token = samples[token]["next"]
    return tokens


def main() -> int:
    args = parse_args()
    if not (args.nuscenes_root / "samples").is_dir():
        raise ValueError(f"missing samples/ under {args.nuscenes_root}")
    infos = pickle.load(args.nuscenes_info.open("rb"))
    by_token = {m["token"]: m for m in infos}
    devkit = args.devkit_dir or (args.nuscenes_info.parent / "v1.0-mini")
    tokens = scene_tokens(devkit, args.scene_name)
    missing = [t for t in tokens if t not in by_token]
    if missing:
        raise ValueError(f"scene tokens missing from info pickle: {missing[:5]}")
    frames = [by_token[t] for t in tokens]
    if len(frames) != args.expected_frames:
        raise ValueError(
            f"{args.scene_name}: {len(frames)} frames, expected {args.expected_frames}")
    stamps = [float(m["timestamp"]) for m in frames]
    if any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise ValueError("scene timestamps not strictly increasing along sample chain")

    def records():
        for frame_id, m in enumerate(frames):
            bin_path = args.nuscenes_root / m["lidar_path"]
            flat = np.fromfile(bin_path, dtype=np.float32)
            if flat.size == 0 or flat.size % 5 != 0:
                raise ValueError(f"{bin_path} is not a non-empty 5-column float32 cloud")
            pts5 = flat.reshape(-1, 5)
            # Waymo row contract x,y,z,intensity,elastic,NLZ. nuScenes column 4
            # is intensity in [0,1] -> x255 to Waymo magnitude; every point is
            # the first return, so NLZ = -1 keeps prepare_object_data's
            # first-return filter from dropping the whole frame.
            points = np.zeros((len(pts5), 6), dtype=np.float32)
            points[:, :3] = pts5[:, :3]
            points[:, 3] = pts5[:, 4] * 255.0
            points[:, 5] = -1.0
            # car_from_global = global->ego, ref_from_car = ego->lidar (verified
            # exactly against the devkit tables, risk R3): pose = lidar->global.
            # NOTE 2026-09-03: composition order FIXED. The global->lidar chain
            # is ref_from_car @ car_from_global (ego->lidar AFTER global->ego);
            # the previous swap produced a frame-wise inconsistent "pseudo
            # global" (~2 km off). Measured on all 81 v1.0-mini val frames:
            # corrected err vs devkit 3.4e-12, old order err ~2.0e3 m.
            # See tools/nustack/__init__.py + tests/test_nustack_pose.py.
            g2l = np.asarray(m["ref_from_car"], dtype=np.float64) @ \
                np.asarray(m["car_from_global"], dtype=np.float64)
            pose = np.ascontiguousarray(np.linalg.inv(g2l))
            yield {
                "frame_id": frame_id,
                "timestamp": np.int64(round(stamps[frame_id] * 1e6)),
                "pose": pose,
                "points": points,
                "num_points_of_each_lidar": [len(points), 0, 0, 0, 0],
            }

    manifest = publish_preprocessed_records(
        records(),
        args.root_path,
        sequence_name=args.scene_name,
        expected_frame_count=args.expected_frames,
        manifest_fields={
            "source": "nuscenes-mini",
            "nuscenes_info_sha256": _sha(args.nuscenes_info),
            "note": "generated root carries no ground truth",
        },
    )
    # tracking reads root/waymo_infos_<split>.pkl; publish wrote the per-segment
    # info file inside the segment dir — expose it under the Waymo name too.
    segment_info = (args.root_path / "waymo_processed_data" /
                    f"segment-{args.scene_name}" / f"{args.scene_name}.pkl")
    os.link(segment_info, args.root_path / "waymo_infos_test.pkl")
    with (args.root_path / "gt_tokens.json").open("x") as stream:
        json.dump({"scene_name": args.scene_name,
                   "tokens": [m["token"] for m in frames]}, stream, indent=1)
    print(json.dumps(manifest, allow_nan=False, sort_keys=True))
    return 0


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as s:
        for block in iter(lambda: s.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
