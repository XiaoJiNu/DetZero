#!/usr/bin/env python3
"""N3: nuScenes tracking json -> per-frame pkl with a track_ids (N,) column.

Same frame-list schema as S2/T1 truth frames (frame_id, timestamp, pose,
name, score, boxes_lidar) plus per-frame `track_ids` aligned row-for-row with
the surviving boxes. Boxes are brought back from global to the lidar frame
with the same pose used by N1 (lossless round-trip asserted on read).
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from pyquaternion import Quaternion

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.nustack import TRACKING_CLASSES, lidar_to_global  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track-json", type=Path, required=True,
                        help="nuScenes tracking submission json (results dict)")
    parser.add_argument("--nuscenes-info", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--filled-json", type=Path, default=None,
                        help="optional: write the tracking json back with derived velocities")
    args = parser.parse_args()

    results = json.loads(args.track_json.read_text())["results"]
    infos = pickle.load(args.nuscenes_info.open("rb"))
    info_ts = {m["token"]: float(m["timestamp"]) for m in infos}

    # velocity label: the official 2Hz result_creation hardcodes velocity=[0,0]
    # (SimpleTrack behaviour), so derive per-track finite differences of the
    # Kalman-smoothed global centers instead. NOT the HEDNet velocity head.
    traj: dict[str, list[tuple[float, np.ndarray]]] = {}
    for token, samples in results.items():
        for s in samples:
            traj.setdefault(str(s["tracking_id"]), []).append(
                (info_ts[token], np.asarray(s["translation"][:2])))
    vel_of = {}
    for tid, obs in traj.items():
        obs.sort(key=lambda o: o[0])
        for i, (ts, c) in enumerate(obs):
            if i == 0 and len(obs) > 1:
                j, k = 0, 1
            elif i == len(obs) - 1 and len(obs) > 1:
                j, k = i - 1, i
            elif len(obs) > 1:
                j, k = i - 1, i + 1
            else:
                vel_of[(tid, ts)] = [0.0, 0.0]
                continue
            dt = obs[k][0] - obs[j][0]
            v = ((obs[k][1] - obs[j][1]) / dt).tolist() if dt > 0 else [0.0, 0.0]
            vel_of[(tid, ts)] = v

    frames = []
    n_tracks = set()
    for frame_id, info in enumerate(infos):
        token = info["token"]
        pose_g2l = np.asarray(info["ref_from_car"], dtype=np.float64) @ \
            np.asarray(info["car_from_global"], dtype=np.float64)  # global->lidar
        rot, t = pose_g2l[:3, :3], pose_g2l[:3, 3]
        rows, names, scores, ids = [], [], [], []
        for s in results.get(token, []):
            if s["tracking_name"] not in TRACKING_CLASSES:
                continue
            center = np.asarray(s["translation"], dtype=np.float64) @ rot.T + t
            q = Quaternion(s["rotation"])
            # SimpleTrack's result_creation drops velocity; take detector's own
            # lidar-frame heading: yaw_g minus frame yaw, wrapped.
            frame_yaw = np.arctan2(pose_g2l[1, 0], pose_g2l[0, 0])
            yaw_l = (q.yaw_pitch_roll[0] - frame_yaw + np.pi) % (2 * np.pi) - np.pi
            w, l, h = s["size"]
            v_g = np.asarray(vel_of.get((str(s["tracking_id"]), info_ts[token]), [0.0, 0.0]))
            v_l = (np.append(v_g, 0.0) @ rot.T)[:2]  # global -> lidar velocity
            rows.append([center[0], center[1], center[2], l, w, h, yaw_l,
                         float(v_l[0]), float(v_l[1])])
            names.append(s["tracking_name"])
            scores.append(float(s["tracking_score"]))
            tid = str(s["tracking_id"])
            ids.append(tid)
            n_tracks.add(tid)
        frames.append({
            "frame_id": frame_id,
            "timestamp": int(round(float(info["timestamp"]) * Fraction(10**6))),
            "pose": lidar_to_global(info),
            "name": np.asarray(names, dtype=object),
            "score": np.asarray(scores, dtype=np.float64),
            "boxes_lidar": np.asarray(rows, dtype=np.float64).reshape(-1, 9),
            "track_ids": ids,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as fh:
        pickle.dump(frames, fh)
    # fill the official velocity field (global frame) in a copy of the json —
    # SimpleTrack's converter leaves [0,0]; the derived values above are ours.
    if args.filled_json is not None:
        for token, samples in results.items():
            for s in samples:
                s["velocity"] = [float(v) for v in
                                 vel_of.get((str(s["tracking_id"]), info_ts[token]), [0.0, 0.0])]
        args.filled_json.parent.mkdir(parents=True, exist_ok=True)
        args.filled_json.write_text(json.dumps(
            {"meta": {"use_camera": False, "use_lidar": True, "use_radar": False,
                      "use_map": False, "use_external": False},
             "results": results}))
    print(json.dumps({"frames": len(frames),
                      "boxes": int(sum(len(f["track_ids"]) for f in frames)),
                      "unique_tracks": len(n_tracks)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
