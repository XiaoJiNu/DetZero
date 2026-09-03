#!/usr/bin/env python3
"""N1: HEDNet result.pkl -> nuScenes detection-json (global frame, 7 tracking classes).

Output follows the devkit detection result schema (the input expected by
SimpleTrack preprocessing/nuscenes_data/detection.py):
    {"meta": {...}, "results": {sample_token: [ {sample_token, translation,
      size(w,l,h), rotation(wxyz), velocity(vx,vy), detection_name,
      detection_score, attribute_name} ]}}

Coordinate convention is the SINGLE source of truth in tools.nustack
(corrected `ref_from_car @ car_from_global`; see that module's docstring).
traffic_cone / barrier are outside the official tracking task: dropped and
counted into the manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.nustack import (  # noqa: E402
    TRACKING_CLASSES,
    lidar_to_global,
    transform_boxes_lidar_to_global,
    yaw_to_quat,
)

DROPPED = ("traffic_cone", "barrier", "construction_vehicle")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hednet-result", type=Path, required=True)
    parser.add_argument("--nuscenes-info", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.1)
    args = parser.parse_args()

    results = pickle.load(args.hednet_result.open("rb"))
    by_token = {r["metadata"]["token"]: r for r in results}
    infos = pickle.load(args.nuscenes_info.open("rb"))

    out_results = {}
    class_counts = {c: 0 for c in TRACKING_CLASSES}
    dropped_counts = {c: 0 for c in DROPPED}
    for info in infos:
        token = info["token"]
        if token not in by_token:
            raise ValueError(f"token missing from HEDNet result: {token}")
        det = by_token[token]
        boxes = np.asarray(det["boxes_lidar"], dtype=np.float64)
        names = np.asarray(det["name"]).tolist()
        scores = np.asarray(det["score"], dtype=np.float64)
        centers, yaws, vels = transform_boxes_lidar_to_global(
            boxes, lidar_to_global(info))
        samples = []
        for row, name, score in zip(range(len(boxes)), names, scores):
            name = str(name)
            if score < args.score_threshold:
                continue
            if name in DROPPED:
                dropped_counts[name] = dropped_counts.get(name, 0) + 1
                continue
            if name not in TRACKING_CLASSES:
                raise ValueError(f"unmapped HEDNet class: {name!r}")
            l, w, h = boxes[row, 3:6]  # lidar boxes carry (l,w,h)
            samples.append({
                "sample_token": token,
                "translation": [float(v) for v in centers[row]],
                "size": [float(w), float(l), float(h)],
                "rotation": yaw_to_quat(float(yaws[row])),
                "velocity": [float(vels[row, 0]), float(vels[row, 1])],
                "detection_name": name,
                "detection_score": float(score),
                "attribute_name": "vehicle.parked" if name not in ("pedestrian", "bicycle", "motorcycle") else "human.pedestrian.moving",
            })
            class_counts[name] += 1
        out_results[token] = samples

    payload = {"meta": {"use_camera": False, "use_lidar": True,
                        "use_radar": False, "use_map": False,
                        "use_external": False},
               "results": out_results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload))
    manifest = {
        "frames": len(out_results),
        "score_threshold": args.score_threshold,
        "kept_by_class": class_counts,
        "dropped_by_class": dropped_counts,
        "coordinate_note": "global frame via tools.nustack.lidar_to_global (corrected composition)",
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
