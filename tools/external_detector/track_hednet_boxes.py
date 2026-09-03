#!/usr/bin/env python3
"""T1: BEV tracker on HEDNet lidar-frame boxes; smoothed truth output.

Revision v2 (measured evidence): association uses a per-class constant-velocity
Kalman filter (filterpy, MIT) in the GLOBAL frame with a chi-square Mahalanobis
gate. v1 used EMA finite-difference velocity + fixed metric gates sized 2-5m;
measured 2Hz inter-frame detector jitter (median 2.4-3.3m, p90 3.3-4.8m by
class) exceeds those gates, which fragmented 78-98% of Ped/Cyc tracks into
single frames. The covariance-scaled gate also grows correctly over missed
frames, which a fixed radius cannot.

No learned weights, no DetZero/Waymo cfg. HEDNet vx/vy is NOT used for
association (measured ~31deg median angle error vs GT motion); it is passed
through into the output boxes as a label attribute. Dimension smoothing =
per-track score-weighted median (l,w,h); centers/heading pass through.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import os
import pickle
import sys
import uuid

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.external_detector.pipeline import rename_noreplace

# Per-class measurement noise sigma [m]: sqrt(halved) of measured adjacent-frame
# jitter of the SAME GT object (Veh 3.0 / Ped 2.4 / Cyc 3.3 median -> ~2.1/1.7/2.3
# per-box sigma); rounded up one step, 2.0..2.5m.
# Process noise = accel sigma [m/s^2] over the real dt (Veh accelerate slowly
# relative to 0.5s frames, VRUs change direction faster).
CLASS_NOISE = {
    "Vehicle": {"r_m": 2.5, "a_mps2": 1.0},
    "Cyclist": {"r_m": 2.5, "a_mps2": 1.5},
    "Pedestrian": {"r_m": 2.0, "a_mps2": 1.5},
}
# chi2 df=2 quantiles: 9.21 = 99%, 16 = 99.97%. Gating on extrapolation error.
DEFAULT_GATE_CHI2 = 9.21
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


def cv_filter(pos, r_m, a_mps2):
    """4-state [x, y, vx, vy] CV Kalman at measurement time; v unknown -> wide."""
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([[pos[0]], [pos[1]], [0.0], [0.0]])
    kf.H = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]])
    kf.R[:] = 0.0
    kf.R[0, 0] = kf.R[1, 1] = r_m ** 2
    kf.P[:] = 0.0
    kf.P[:2, :2] = np.eye(2) * r_m ** 2
    kf.P[2:, 2:] = np.eye(2) * (5.0 ** 2)  # allow up to ~5 m/s at birth
    kf.F = np.eye(4)
    kf.Q = np.eye(4)
    kf._r_m, kf._a = r_m, a_mps2
    return kf


def predict(kf, dt):
    kf.F[0, 2] = kf.F[1, 3] = dt
    a2 = kf._a ** 2
    q = np.array([[a2 * dt ** 3 / 3, 0, a2 * dt ** 2 / 2, 0],
                  [0, a2 * dt ** 3 / 3, 0, a2 * dt ** 2 / 2],
                  [a2 * dt ** 2 / 2, 0, a2 * dt, 0],
                  [0, a2 * dt ** 2 / 2, 0, a2 * dt]])
    kf.Q = q
    kf.predict()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detector-frames", type=Path, required=True,
                    help="S2 output hednet_frames_<scene>.pkl")
    ap.add_argument("--output-frames", type=Path, required=True)
    ap.add_argument("--output-tracks", type=Path, required=True)
    ap.add_argument("--max-age", type=int, default=3,
                    help="drop a track after this many consecutive missed frames")
    ap.add_argument("--gate-chi2", type=float, default=DEFAULT_GATE_CHI2)
    ap.add_argument("--r-scale", type=float, default=1.0,
                    help="multiplier on per-class measurement sigma (grid search)")
    return ap.parse_args()


def track_frames(frames, max_age, gate_chi2, r_scale):
    tracks = {}  # tid -> dict(cls, kf, t_us, last_frame, obs)
    next_id = 0
    for f in frames:
        fid, ts = int(f["frame_id"]), int(f["timestamp"])
        boxes = np.asarray(f["boxes_lidar"], np.float64)
        names = np.asarray(f["name"]).tolist()
        centers_g, _, _ = lidar_to_global(boxes, np.asarray(f["pose"], np.float64))
        for tr in tracks.values():
            tr["stale"] = fid - tr["last_frame"] - 1 > max_age
        used = set()
        for cls, noise in CLASS_NOISE.items():
            det_idx = [i for i, n in enumerate(names) if str(n) == cls]
            tr_ids = [t for t, tr in tracks.items()
                      if tr["cls"] == cls and not tr["stale"]]
            if det_idx and tr_ids:
                cost = np.full((len(det_idx), len(tr_ids)), BIG)
                for c, tid in enumerate(tr_ids):
                    tr = tracks[tid]
                    predict(tr["kf"], max((ts - tr["t_us"]) / 1e6, 1e-3))
                    pred = tr["kf"].x[:2].ravel()
                    S = (tr["kf"].P[:2, :2] + tr["kf"].R)  # innovation cov (H=[I|0])
                    inv = np.linalg.inv(S)
                    d = centers_g[det_idx][:, :2] - pred
                    cost[:, c] = np.einsum("ij,jk,ik->i", d, inv, d)
                rows, cols = linear_sum_assignment(cost)  # rows=dets, cols=tracks
                for r, c in zip(rows, cols):
                    if cost[r, c] > gate_chi2:
                        continue
                    di = det_idx[r]
                    tr = tracks[tr_ids[c]]
                    tr["kf"].update(centers_g[di][:2])
                    tr.update(t_us=ts, last_frame=fid)
                    tr["obs"].append((fid, di))
                    used.add(di)
            for di in det_idx:
                if di in used:
                    continue
                kf = cv_filter(centers_g[di][:2], noise["r_m"] * r_scale, noise["a_mps2"])
                tracks[next_id] = {"cls": cls, "kf": kf, "t_us": ts,
                                   "last_frame": fid, "obs": [(fid, di)],
                                   "stale": False}
                next_id += 1
    return {tid: tr for tid, tr in tracks.items()}


def weighted_median(values, weights):
    order = np.argsort(values)
    cum = np.cumsum(weights[order]) / max(weights[order].sum(), 1e-16)
    idx = np.searchsorted(cum, 0.5, side="left").clip(max=len(values) - 1)
    return float(values[order][idx])


def main() -> int:
    args = parse_args()
    frames = pickle.load(args.detector_frames.open("rb"))
    frames.sort(key=lambda f: int(f["frame_id"]))
    det_boxes = [np.asarray(f["boxes_lidar"], np.float64) for f in frames]
    det_scores = [np.asarray(f["score"], np.float64) for f in frames]

    tracks = track_frames(frames, args.max_age, args.gate_chi2, args.r_scale)

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
                      "motion_model": "cv-kalman-v2", "gate_chi2": args.gate_chi2,
                      "r_scale": args.r_scale,
                      "per_class": stats,
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
