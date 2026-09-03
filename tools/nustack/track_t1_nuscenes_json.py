#!/usr/bin/env python3
"""Control-1 driver: T1-v2 self-developed Kalman tracker on HEDNet boxes,
emitting a nuScenes tracking submission json (7 classes, global frame).

Official LidarPointCloudTracker is NOT reproducible (removed from every
available nuscenes-devkit release — see completion report), so the lower
bound baseline against SimpleTrack is the in-house T1-v2 (same detector,
same corrected pose, same output schema). Association reuses
tools.external_detector.track_hednet_boxes.track_frames verbatim; the only
addition is a 7-class -> 3-noise-profile map for the per-class CV-Kalman.
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

from tools.external_detector.track_hednet_boxes import track_frames  # noqa: E402
from tools.nustack import (  # noqa: E402
    TRACKING_CLASSES,
    lidar_to_global,
    transform_boxes_lidar_to_global,
    yaw_to_quat,
)

NOISE_CLASS = {
    "car": "Vehicle", "bus": "Vehicle", "truck": "Vehicle", "trailer": "Vehicle",
    "pedestrian": "Pedestrian",
    "bicycle": "Cyclist", "motorcycle": "Cyclist",
}
# track_frames reads CLASS_NOISE via detector names; monkey-patch the module
# table so the 7 native classes route to the 3 measured noise profiles.
import tools.external_detector.track_hednet_boxes as T1  # noqa: E402
T1.CLASS_NOISE = {c: T1.CLASS_NOISE[NOISE_CLASS[c]] for c in TRACKING_CLASSES}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hednet-result", type=Path, required=True)
    ap.add_argument("--nuscenes-info", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--score-threshold", type=float, default=0.1)
    ap.add_argument("--max-age", type=int, default=2)
    ap.add_argument("--gate-chi2", type=float, default=T1.DEFAULT_GATE_CHI2)
    args = ap.parse_args()

    results = pickle.load(args.hednet_result.open("rb"))
    by_token = {r["metadata"]["token"]: r for r in results}
    infos = pickle.load(args.nuscenes_info.open("rb"))

    frames = []
    kept = []
    for fid, info in enumerate(infos):
        det = by_token[info["token"]]
        boxes = np.asarray(det["boxes_lidar"], dtype=np.float64)
        names = [str(n) for n in np.asarray(det["name"]).tolist()]
        scores = np.asarray(det["score"], dtype=np.float64)
        rows, nms, scs = [], [], []
        for i, (n, s) in enumerate(zip(names, scores)):
            if n in TRACKING_CLASSES and s >= args.score_threshold:
                rows.append(i)
                nms.append(n)
                scs.append(s)
        boxes = boxes[rows]
        frames.append({"frame_id": fid, "timestamp": int(round(float(info["timestamp"]) * 1e6)),
                       "pose": lidar_to_global(info), "boxes_lidar": boxes,
                       "name": np.asarray(nms, dtype=object),
                       "score": np.asarray(scs)})
        kept.append((info["token"], boxes, nms, scs))

    tracks = track_frames(frames, args.max_age, args.gate_chi2, 1.0)
    tid_of = {(fid, i): tid for tid, tr in tracks.items() for (fid, i) in tr["obs"]}
    dims_of = {tid: np.array([T1.weighted_median(
                   np.array([np.asarray(frames[o[0]]["boxes_lidar"])[o[1], 3:6] for o in tr["obs"]])[:, k],
                   np.array([np.asarray(frames[o[0]]["score"])[o[1]] for o in tr["obs"]]))
               for k in range(3)]) for tid, tr in tracks.items()}

    out_results = {}
    ntr = 0
    for fid, (token, boxes, nms, scs) in enumerate(kept):
        info = infos[fid]
        centers, yaws, vels = transform_boxes_lidar_to_global(boxes, lidar_to_global(info))
        out = []
        for i in range(len(boxes)):
            tid = tid_of[(fid, i)]
            l, w, h = dims_of[tid]
            out.append({
                "sample_token": token,
                "translation": [float(v) for v in centers[i]],
                "size": [float(w), float(l), float(h)],
                "rotation": yaw_to_quat(float(yaws[i])),
                "velocity": [float(vels[i, 0]), float(vels[i, 1])],
                "tracking_id": f"t1_{tid}",
                "tracking_name": nms[i],
                "tracking_score": float(scs[i]),
            })
        out_results[token] = out
        ntr += len(out)

    payload = {"meta": {"use_camera": False, "use_lidar": True, "use_radar": False,
                        "use_map": False, "use_external": False},
               "results": out_results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload))
    print(json.dumps({"frames": len(out_results), "boxes": ntr,
                      "tracks": len(tracks)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
