#!/usr/bin/env python3
"""Self-check for T1 v2 (CV-Kalman + chi-square gate): synthetic 2Hz clip.
Two Vehicles move 0.75 m/frame with 1.5 m detector jitter (measured nuScenes
2Hz regime). Expect exactly 2 tracks covering all 40 detections, each track
locked to its own object (no ID switch, no fragmentation).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.external_detector.track_hednet_boxes import track_frames


def main():
    rng = np.random.default_rng(0)
    frames = []
    for fid in range(20):
        pose = np.eye(4)
        positions = [(10 + 0.75 * fid, 5.0), (30 + 0.75 * fid, -5.0)]
        obs = [p + rng.normal(0, 1.5, 2) for p in positions]
        boxes = np.array([[x, y, 0, 4, 2, 1.5, 0, 1.5, 0] for x, y in obs],
                         np.float32)
        frames.append({"frame_id": fid, "timestamp": fid * 500_000,
                       "pose": pose, "sample_idx": fid, "sequence_name": "synth",
                       "name": np.array(["Vehicle", "Vehicle"]),
                       "score": np.full(2, 0.9, np.float32),
                       "boxes_lidar": boxes})
    tracks = track_frames(frames, max_age=3, gate_chi2=9.21, r_scale=1.0)
    assert sum(len(tr["obs"]) for tr in tracks.values()) == 40, "dropped obs"
    lens = sorted((len(tr["obs"]) for tr in tracks.values()), reverse=True)
    assert lens == [20, 20], f"fragmented/merged: lens={lens}"
    for tr in tracks.values():
        xs = np.array([frames[o[0]]["boxes_lidar"][o[1]][0] for o in tr["obs"]])
        ys = np.array([frames[o[0]]["boxes_lidar"][o[1]][1] for o in tr["obs"]])
        assert (ys > 0).all() or (ys < 0).all(), "track switched objects"
        assert xs.max() - xs.min() > 8, "track must follow the moving object"
    print("track_hednet v2 self-check PASS: 2 tracks x 20 frames, 0 dropped")


if __name__ == "__main__":
    main()
