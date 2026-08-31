#!/usr/bin/env python3
"""Adapt HEDNet nuScenes predictions to DetZero frame pickles (S2).

HEDNet already emits DetZero-shaped boxes [x,y,z,l,w,h,yaw,vx,vy] in the lidar
frame, so boxes pass through untouched; only the class map and the frame
envelope are new. One output pickle per scene, matching adapt_raw_predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import uuid

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.external_detector.pipeline import _validated_pose, rename_noreplace

CLASS_MAP = {
    "car": "Vehicle", "bus": "Vehicle", "truck": "Vehicle",
    "construction_vehicle": "Vehicle", "trailer": "Vehicle",
    "pedestrian": "Pedestrian",
    "bicycle": "Cyclist", "motorcycle": "Cyclist",
}
DROPPED = ("traffic_cone", "barrier")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hednet-result", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True,
                        help="S1 data root (gt_tokens.json + waymo_infos_test.pkl)")
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tokens = json.load((args.scene_root / "gt_tokens.json").open())
    infos = pickle.load((args.scene_root / "waymo_infos_test.pkl").open("rb"))
    if tokens["tokens"] and [int(i["sample_idx"]) for i in infos] != list(range(len(infos))):
        raise ValueError("scene info sample_idx must be contiguous from zero")

    results = pickle.load(args.hednet_result.open("rb"))
    by_token = {r["metadata"]["token"]: r for r in results}
    missing = [t for t in tokens["tokens"] if t not in by_token]
    if missing:
        raise ValueError(f"tokens missing from HEDNet result: {missing[:5]}")

    frames = []
    dropped_counts = {}
    class_counts = {name: 0 for name in ("Vehicle", "Pedestrian", "Cyclist")}
    for frame_id, token in enumerate(tokens["tokens"]):
        result = by_token[token]
        info = infos[frame_id]
        timestamp = int(info["time_stamp"])
        pose = _validated_pose(info["pose"])
        boxes = np.asarray(result["boxes_lidar"], dtype=np.float32)
        names = np.asarray(result["name"]).tolist()
        scores = np.asarray(result["score"], dtype=np.float32)
        keep = scores >= args.score_threshold
        kept_rows, kept_names = [], []
        for row in np.nonzero(keep)[0]:
            name = str(names[row])
            if name in DROPPED:
                dropped_counts[name] = dropped_counts.get(name, 0) + 1
                continue
            det_class = CLASS_MAP.get(name)
            if det_class is None:
                raise ValueError(f"unmapped HEDNet class: {name!r}")
            kept_rows.append(row)
            kept_names.append(det_class)
        kept_boxes = boxes[kept_rows] if kept_rows else np.zeros((0, 9), np.float32)
        if len(kept_boxes) and (kept_boxes.shape[1] != 9 or not np.isfinite(kept_boxes).all()):
            raise ValueError(f"invalid HEDNet boxes at frame {frame_id}")
        for det_class in kept_names:
            class_counts[det_class] += 1
        frames.append({
            "sequence_name": tokens["scene_name"],
            "sample_idx": frame_id,
            "frame_id": frame_id,
            "timestamp": timestamp,
            "pose": pose.copy(),
            "name": np.asarray(kept_names, dtype="<U10"),
            "score": scores[kept_rows] if kept_rows else np.zeros(0, np.float32),
            "boxes_lidar": kept_boxes,
        })

    if args.output_file.exists() or args.output_file.is_symlink():
        raise FileExistsError(args.output_file)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    stage = args.output_file.with_name(f".{args.output_file.name}.tmp-{uuid.uuid4().hex}")
    with stage.open("xb") as stream:
        pickle.dump(frames, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    rename_noreplace(stage, args.output_file)
    print(json.dumps({
        "scene_name": tokens["scene_name"],
        "frame_count": len(frames),
        "score_threshold": args.score_threshold,
        "class_counts": class_counts,
        "dropped_counts": dropped_counts,
        "box_schema": ["x", "y", "z", "length", "width", "height", "heading", "vx", "vy"],
        "yaw_conversion": "passthrough (HEDNet already lidar/PCDet convention)",
        "velocity_status": "from HEDNet output",
        "output_sha256": _sha(args.output_file),
    }, sort_keys=True))
    return 0


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as s:
        for block in iter(lambda: s.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
