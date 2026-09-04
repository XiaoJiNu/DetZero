#!/usr/bin/env python3
"""CLI: SurroundOcc-style static/dynamic mapping on nuScenes scenes."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
from nuscenes.nuscenes import NuScenes

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from scene_map import build_scene_map, load_lidar_points, pose_mats  # noqa: E402
from geometry import global_to_lidar, quat_to_rot  # noqa: E402
from visualize_map_bev import visualize_scene_dir  # noqa: E402


CST = timezone(timedelta(hours=8))


def stamp_cst() -> str:
    return datetime.now(CST).strftime("%Y%m%d-%H%M%S") + "-CST"


def boxes_to_last_lidar(boxes_last: dict, cs_last: dict, pose_last: dict) -> list:
    """Transform global box centers to last lidar frame for BEV overlay."""
    cs_r, cs_t, ego_r, ego_t = pose_mats(cs_last, pose_last)
    out = []
    for tid, b in boxes_last.items():
        c_g = np.asarray(b["translation"], dtype=np.float64).reshape(1, 3)
        c_l = global_to_lidar(c_g, cs_r, cs_t, ego_r, ego_t)[0]
        # yaw in lidar ≈ yaw_global - ego_yaw - cs_yaw; for BEV overlay use relative
        # Approximate: rotate heading by inverse ego*cs
        from pyquaternion import Quaternion

        q_box = Quaternion(b["rotation"])
        q_ego = Quaternion(pose_last["rotation"])
        q_cs = Quaternion(cs_last["rotation"])
        q_l = q_cs.inverse * q_ego.inverse * q_box
        out.append(
            {
                "tracking_id": tid,
                "translation": c_l.tolist(),
                "size": b["size"],
                "rotation": [q_l.w, q_l.x, q_l.y, q_l.z],
                "tracking_name": b.get("tracking_name", ""),
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--tracking-json", required=True)
    ap.add_argument("--scenes", default="scene-0103,scene-0916")
    ap.add_argument("--out-dir", default="", help="default: output/mapping-<stamp>-CST")
    ap.add_argument("--voxel-size", type=float, default=0.1)
    ap.add_argument("--box-expand", type=float, default=1.1)
    ap.add_argument("--self-range", type=float, nargs=3, default=[3.0, 3.0, 3.0])
    ap.add_argument("--poisson", action="store_true", help="optional Poisson on static")
    ap.add_argument("--poisson-depth", type=int, default=9)
    ap.add_argument("--bev-range", type=float, default=50.0)
    ap.add_argument("--no-pcd", action="store_true")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[2]
    stamp = stamp_cst()
    out_root = Path(args.out_dir) if args.out_dir else (repo / "output" / f"mapping-{stamp}")
    if not out_root.is_absolute():
        out_root = repo / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    with open(args.tracking_json) as f:
        track_data = json.load(f)
    results = track_data["results"]

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]

    t0 = time.time()
    scene_summaries = []
    for name in scenes:
        scene_dir = out_root / name
        print(f"[mapping] {name} -> {scene_dir}")
        man = build_scene_map(
            nusc=nusc,
            scene_name=name,
            tracking_results=results,
            out_dir=scene_dir,
            box_expand=args.box_expand,
            voxel_size=args.voxel_size,
            self_range=args.self_range,
            use_poisson=args.poisson,
            poisson_depth=args.poisson_depth,
            write_pcd_also=not args.no_pcd,
        )

        with open(scene_dir / "boxes_last.json") as f:
            bl = json.load(f)
        # load last pose for box overlay
        _, cs_last, pose_last, _ = load_lidar_points(nusc, man["last_sample_token"])
        boxes_lidar = boxes_to_last_lidar(bl["boxes"], cs_last, pose_last)
        # Also dump lidar-frame boxes for debugging overlays
        with open(scene_dir / "boxes_last_lidar.json", "w") as f:
            json.dump({"scene": name, "boxes": boxes_lidar}, f, indent=2)

        vis = visualize_scene_dir(
            scene_dir,
            name,
            range_m=args.bev_range,
            boxes_lidar=boxes_lidar,
        )
        man["visuals"] = vis
        with open(scene_dir / "manifest.json", "w") as f:
            json.dump(man, f, indent=2)
        scene_summaries.append(man)
        print(
            f"  frames={man['n_frames']} static={man['n_static_after_voxel']} "
            f"dynamic={man['n_dynamic_after_place_voxel']} tracks={man['n_tracks_placed']} "
            f"time={man['elapsed_sec']}s"
        )

    root_manifest = {
        "stamp": stamp,
        "out_dir": str(out_root),
        "dataroot": args.dataroot,
        "version": args.version,
        "tracking_json": str(Path(args.tracking_json).resolve()),
        "scenes": scenes,
        "voxel_size": args.voxel_size,
        "box_expand": args.box_expand,
        "self_range": list(args.self_range),
        "poisson": args.poisson,
        "elapsed_sec": round(time.time() - t0, 3),
        "scenes_detail": [
            {
                "scene": m["scene"],
                "n_frames": m["n_frames"],
                "n_static": m["n_static_after_voxel"],
                "n_dynamic": m["n_dynamic_after_place_voxel"],
                "n_combined": m["n_combined"],
                "n_tracks_placed": m["n_tracks_placed"],
                "elapsed_sec": m["elapsed_sec"],
            }
            for m in scene_summaries
        ],
    }
    with open(out_root / "manifest.json", "w") as f:
        json.dump(root_manifest, f, indent=2)
    print(json.dumps(root_manifest, indent=2))


if __name__ == "__main__":
    main()
