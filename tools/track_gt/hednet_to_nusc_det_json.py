#!/usr/bin/env python3
"""Convert HEDNet result.pkl (lidar boxes) to nuScenes detection results JSON.

Boxes are transformed lidar -> ego -> global using NuScenes calib + ego_pose,
matching official / PCDet export conventions. Only tracking challenge classes
are kept; dropped classes are counted in a sidecar manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion


TRACKING_NAMES = (
    "bicycle",
    "bus",
    "car",
    "motorcycle",
    "pedestrian",
    "trailer",
    "truck",
)


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def boxes_lidar_to_nusc_boxes(det_info: dict) -> List[Box]:
    boxes3d = np.asarray(det_info["boxes_lidar"])
    scores = np.asarray(det_info["score"])
    labels = np.asarray(det_info.get("pred_labels", np.zeros(len(scores), dtype=np.int32)))
    box_list = []
    for k in range(boxes3d.shape[0]):
        quat = Quaternion(axis=[0, 0, 1], radians=float(boxes3d[k, 6]))
        velocity = (
            (*boxes3d[k, 7:9], 0.0) if boxes3d.shape[1] >= 9 else (0.0, 0.0, 0.0)
        )
        # PCDet/HEDNet dims are l,w,h; NuScenes Box expects wlh.
        box = Box(
            boxes3d[k, :3],
            boxes3d[k, [4, 3, 5]],
            quat,
            label=int(labels[k]),
            score=float(scores[k]),
            velocity=velocity,
        )
        box_list.append(box)
    return box_list


def lidar_boxes_to_global(nusc: NuScenes, boxes: List[Box], sample_token: str) -> List[Box]:
    s_record = nusc.get("sample", sample_token)
    sample_data_token = s_record["data"]["LIDAR_TOP"]
    sd_record = nusc.get("sample_data", sample_data_token)
    cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])

    out = []
    for box in boxes:
        box.rotate(Quaternion(cs_record["rotation"]))
        box.translate(np.array(cs_record["translation"]))
        box.rotate(Quaternion(pose_record["rotation"]))
        box.translate(np.array(pose_record["translation"]))
        out.append(box)
    return out


def convert(
    pkl_path: str,
    dataroot: str,
    version: str,
    score_thres: float,
    out_json: str,
    drop_manifest: str | None,
) -> dict:
    with open(pkl_path, "rb") as f:
        det_annos = pickle.load(f)

    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    results: Dict[str, list] = {}
    kept = Counter()
    dropped = Counter()
    frames = 0
    total_raw = 0

    for det in det_annos:
        token = det["metadata"]["token"]
        # Validate token exists in this dataroot.
        nusc.get("sample", token)
        names = np.asarray(det["name"]).tolist()
        boxes = boxes_lidar_to_nusc_boxes(det)
        boxes = lidar_boxes_to_global(nusc, boxes, token)
        scores = np.asarray(det["score"])

        annos = []
        for k, box in enumerate(boxes):
            total_raw += 1
            name = names[k]
            score = float(scores[k])
            if score < score_thres:
                dropped[f"low_score:{name}"] += 1
                continue
            if name not in TRACKING_NAMES:
                dropped[name] += 1
                continue
            kept[name] += 1
            annos.append(
                {
                    "sample_token": token,
                    "translation": box.center.tolist(),
                    "size": box.wlh.tolist(),
                    "rotation": box.orientation.elements.tolist(),
                    "velocity": box.velocity[:2].tolist(),
                    "detection_name": name,
                    "detection_score": score,
                    "attribute_name": "",
                }
            )
        results[token] = annos
        frames += 1

    submission = {
        "meta": {
            "use_camera": False,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        },
        "results": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_json)) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(submission, f)

    summary = {
        "pkl_path": pkl_path,
        "pkl_sha256": sha256_file(pkl_path),
        "dataroot": dataroot,
        "version": version,
        "score_thres": score_thres,
        "n_frames": frames,
        "n_tokens_unique": len(results),
        "total_raw_boxes": total_raw,
        "kept_class_counts": dict(kept),
        "dropped_class_counts": dict(dropped),
        "tracking_names": list(TRACKING_NAMES),
        "box_frame": "global",
        "geometry_source": "detector+motion_filter_later",
        "out_json": out_json,
    }
    if drop_manifest:
        os.makedirs(os.path.dirname(os.path.abspath(drop_manifest)) or ".", exist_ok=True)
        with open(drop_manifest, "w") as f:
            json.dump(summary, f, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--score-thres", type=float, default=0.1)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--drop-manifest", default=None)
    args = ap.parse_args()
    summary = convert(
        args.pkl,
        args.dataroot,
        args.version,
        args.score_thres,
        args.out_json,
        args.drop_manifest,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
