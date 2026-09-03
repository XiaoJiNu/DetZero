"""Shared conventions for the nuScenes-native tracking stack (nu-stack).

Single source of truth for the lidar<->global transform used by N1..N4.

PROVENANCE / BUGFIX: the HEDNet info pkl stores per-field-verified
`car_from_global` = global->ego and `ref_from_car` = ego->lidar. The
composition implemented in preprocess_nuscenes_scene.py was
`car_from_global @ ref_from_car` (wrong order); the correct global->lidar
composition is `ref_from_car @ car_from_global`. Measured against the devkit
tables on all 81 v1.0-mini val frames: correct order max-abs error 3.4e-12,
wrong order error ~2.0e3 m. tests/test_nustack_pose.py locks this down.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

# nuScenes native tracking classes (7; traffic_cone/barrier/construction
# excluded by the official tracking task).
TRACKING_CLASSES = ("bicycle", "bus", "car", "motorcycle", "pedestrian", "trailer", "truck")

# DetZero 3-class fold used by the S7 mAP bypass (same map as S2 adapter).
CLASS_MAP_3 = {
    "car": "Vehicle", "bus": "Vehicle", "truck": "Vehicle",
    "construction_vehicle": "Vehicle", "trailer": "Vehicle",
    "pedestrian": "Pedestrian",
    "bicycle": "Cyclist", "motorcycle": "Cyclist",
}


def load_infos(info_path: Path) -> dict[str, dict]:
    infos = pickle.load(Path(info_path).open("rb"))
    return {m["token"]: m for m in infos}


def lidar_to_global(info: dict) -> np.ndarray:
    """(4,4) lidar->global from an info-pkl frame (corrected composition)."""
    g2l = np.asarray(info["ref_from_car"], dtype=np.float64) @ \
        np.asarray(info["car_from_global"], dtype=np.float64)
    return np.linalg.inv(g2l)


def yaw_to_quat(yaw: float) -> list[float]:
    """Global-frame yaw -> (w,x,y,z) quaternion about +z (nuScenes convention)."""
    return [float(np.cos(yaw / 2.0)), 0.0, 0.0, float(np.sin(yaw / 2.0))]


def transform_boxes_lidar_to_global(boxes: np.ndarray, pose: np.ndarray):
    """(N,9) lidar [x,y,z,l,w,h,yaw,vx,vy] -> centers_g(N,3), yaws_g(N,), vels_g(N,2)."""
    boxes = np.asarray(boxes, dtype=np.float64)
    if len(boxes) == 0:
        z3 = np.zeros((0, 3))
        return z3, np.zeros(0), np.zeros((0, 2))
    rot = pose[:3, :3]
    centers = np.concatenate([boxes[:, :3], np.ones((len(boxes), 1))], 1) @ pose.T
    yaws = (boxes[:, 6] + np.arctan2(rot[1, 0], rot[0, 0]) + np.pi) % (2 * np.pi) - np.pi
    v3 = np.concatenate([boxes[:, 7:9], np.zeros((len(boxes), 1))], 1) @ rot.T
    return centers[:, :3], yaws, v3[:, :2]
