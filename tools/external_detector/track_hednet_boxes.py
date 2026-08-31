#!/usr/bin/env python3
"""T1: constant-velocity + Hungarian BEV gating on HEDNet lidar-frame boxes;
smoothed truth output in the S2 frame schema.

No learned weights, no DetZero cfg: motion comes from HEDNet vx/vy, dt from
real timestamps, gating from physical class sizes. Dimension smoothing =
score-weighted median per track; centers/heading/velocity pass through per
frame (they are the detector's own, already at 0.74/0.83 mAP quality).
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
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.external_detector.pipeline import rename_noreplace

# gate radius in metres on extrapolated BEV center distance. Sized from the
# MEASURED detector inter-frame jitter (adjacent-frame same-object center
# displacement 2-5m at 2Hz), not from physical object size.
GATES = {"Vehicle": 6.0, "Cyclist": 3.0, "Pedestrian": 2.0}
BIG = 1e6


def lidar_to_global(boxes, pose):
    """(N,9) lidar + lidar->global pose -> centers_g(N,3), yaws_g(N,), vels_g(N,2)."""
    if len(boxes) == 0:
        empty = np.zeros((0, 3))
        return empty, empty.ravel(), empty
    rot = pose[:3, :3]
    centers = np.concatenate([boxes[:, :3], np.ones((len(boxes), 1))], 1) @ pose.T
    yaws = (boxes[:, 6] + np.arctan2(rot[1, 0], rot[0, 0]) + np.pi) % (2 * np.pi) - np.pi
    v3 = np.concatenate([boxes[:, 7:9], np.zeros((len(boxes), 1))], 1) @ rot.T
    return centers[:, :3], yaws, v3[:, :2]


def weighted_median(values, weights):
    order = np.argsort(values)
    cum = np.cumsum(weights[order]) / max(weights[order].sum(), 1e-16)
    idx = np.searchsorted(cum, 0.5, side="left").clip(max=len(values) - 1)
    return float(values[order][idx])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detector-frames", type=Path, required=True,
                    help="S2 output hednet_frames_<scene>.pkl")
    ap.add_argument("--output-frames", type=Path, required=True)
    ap.add_argument("--output-tracks", type=Path, required=True)
    ap.add_argument("--max-age", type=int, default=3,
                    help="drop a track after this many consecutive missed frames")
    return ap.parse_args()


def track_frames(frames, max_age):
    """Motion model = EMA finite-difference velocity in the global frame.

    HEDNet vx/vy is NOT used for association: measured against GT-derived
    motion it carries a ~31 deg median angle error and produced more ID
    switches than plain position extrapolation. It is still passed through
    into the output boxes as a label attribute.
    """
    tracks, next_id = {}, 0
    for f in frames:
        fid, ts = int(f["frame_id"]), int(f["timestamp"])
        boxes = np.asarray(f["boxes_lidar"], np.float64)
        names = np.asarray(f["name"]).tolist()
        centers_g, _, vels_g = lidar_to_global(boxes, np.asarray(f["pose"], np.float64))
        for tid, tr in tracks.items():
            tr["stale"] = fid - tr["last_frame"] - 1 > max_age
        used = set()
        for cls in GATES:
            det_idx = [i for i, n in enumerate(names) if str(n) == cls]
            tr_ids = [t for t, tr in tracks.items()
                      if tr["cls"] == cls and not tr["stale"]]
            if det_idx and tr_ids:
                dt = np.array([(ts - tracks[t]["t_us"]) / 1e6 for t in tr_ids])[:, None]
                pred = np.array([tracks[t]["c_g"][:2] for t in tr_ids]) \
                    + np.array([tracks[t]["vel"] for t in tr_ids]) * dt
                cost = np.linalg.norm(centers_g[det_idx][:, None, :2] - pred[None], axis=2)
                cost = np.where(cost <= GATES[cls], cost, BIG)
                rows, cols = linear_sum_assignment(cost)  # rows=dets, cols=tracks
                for r, c in zip(rows, cols):
                    if cost[r, c] >= BIG:
                        continue
                    di = det_idx[r]
                    tr = tracks[tr_ids[c]]
                    dt_s = max((ts - tr["t_us"]) / 1e6, 1e-3)
                    tr["vel"] = 0.5 * tr["vel"] \
                        + 0.5 * (centers_g[di][:2] - tr["c_g"][:2]) / dt_s
                    tr.update(c_g=centers_g[di], t_us=ts,
                              last_frame=fid, hits=tr["hits"] + 1)
                    tr["obs"].append((fid, di))
                    used.add(di)
            for di in det_idx:
                if di in used:
                    continue
                tracks[next_id] = {"cls": cls, "c_g": centers_g[di],
                                   "vel": np.zeros(2), "t_us": ts,
                                   "last_frame": fid, "hits": 1,
                                   "obs": [(fid, di)], "stale": False}
                next_id += 1
    return tracks


def main() -> int:
    args = parse_args()
    frames = pickle.load(args.detector_frames.open("rb"))
    frames.sort(key=lambda f: int(f["frame_id"]))
    det_boxes = [np.asarray(f["boxes_lidar"], np.float64) for f in frames]
    det_scores = [np.asarray(f["score"], np.float64) for f in frames]

    tracks = track_frames(frames, args.max_age)

    # per-track score-weighted median dimensions (l,w,h): the only "optimization"
    dims = {}
    for tid, tr in tracks.items():
        vals = np.array([det_boxes[o[0]][o[1]][3:6] for o in tr["obs"]])
        wts = np.array([det_scores[o[0]][o[1]] for o in tr["obs"]])
        dims[tid] = np.array([weighted_median(vals[:, k], wts) for k in range(3)])

    obs_by_frame = {}
    for tid, tr in tracks.items():
        for fid, di in tr["obs"]:
            obs_by_frame.setdefault(fid, []).append((tid, di))

    out = []
    for f in frames:
        fid = int(f["frame_id"])
        items = sorted(obs_by_frame.get(fid, []))
        boxes = np.zeros((len(items), 9), np.float32)
        names = np.zeros(len(items), dtype="<U10")
        scores = np.zeros(len(items), np.float32)
        for row, (tid, di) in enumerate(items):
            src = det_boxes[fid][di]
            boxes[row] = [src[0], src[1], src[2], *dims[tid], src[6], src[7], src[8]]
            names[row] = tracks[tid]["cls"]
            scores[row] = det_scores[fid][di]
        out.append({k: f[k] for k in ("sequence_name", "sample_idx", "frame_id",
                                      "timestamp", "pose")} |
                   {"name": names, "score": scores, "boxes_lidar": boxes})

    for pth in (args.output_frames, args.output_tracks):
        if pth.exists() or pth.is_symlink():
            raise FileExistsError(pth)
        pth.parent.mkdir(parents=True, exist_ok=True)
    seq = str(frames[0]["sequence_name"])

    def dump(pth, obj):
        stage = pth.with_name(f".{pth.name}.tmp-{uuid.uuid4().hex}")
        with stage.open("xb") as s:
            pickle.dump(obj, s, protocol=pickle.HIGHEST_PROTOCOL)
            s.flush()
            os.fsync(s.fileno())
        rename_noreplace(stage, pth)

    dump(args.output_frames, out)
    track_dict = {seq: {tid: {
        "sequence_name": seq, "obj_id": tid,
        "name": [tracks[tid]["cls"]] * len(tracks[tid]["obs"]),
        "sample_idx": np.array([o[0] for o in tracks[tid]["obs"]], np.int64),
        "boxes_lidar": np.array([np.concatenate(
            [det_boxes[o[0]][o[1]][:3], dims[tid], det_boxes[o[0]][o[1]][6:9]])
            for o in tracks[tid]["obs"]], np.float32),
        "score": np.array([det_scores[o[0]][o[1]] for o in tracks[tid]["obs"]], np.float32),
    } for tid in tracks}}
    dump(args.output_tracks, track_dict)

    stats = {}
    for tr in tracks.values():
        s = stats.setdefault(tr["cls"], {"tracks": 0, "frames": 0, "longest": 0})
        s["tracks"] += 1
        s["frames"] += len(tr["obs"])
        s["longest"] = max(s["longest"], len(tr["obs"]))
    print(json.dumps({"scene_name": seq, "frame_count": len(out),
                      "track_count": len(tracks), "max_age": args.max_age,
                      "gates_m": GATES, "per_class": stats,
                      "output_frames_sha256": _sha(args.output_frames),
                      "output_tracks_sha256": _sha(args.output_tracks)}, sort_keys=True))
    return 0


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as s:
        for b in iter(lambda: s.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
