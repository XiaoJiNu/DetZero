"""Geometry helpers for SurroundOcc-style mapping (no mmcv)."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d
from pyquaternion import Quaternion


def quat_to_rot(q: Sequence[float]) -> np.ndarray:
    """nuScenes quaternion [w, x, y, z] -> 3x3."""
    return Quaternion(q).rotation_matrix


def transform_points(points: np.ndarray, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """Apply R @ p + t to (N,3)."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return pts.reshape(0, 3)
    return (rot @ pts.T).T + np.asarray(trans, dtype=np.float64).reshape(1, 3)


def invert_rt(rot: np.ndarray, trans: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r_inv = rot.T
    t_inv = -r_inv @ np.asarray(trans, dtype=np.float64).reshape(3)
    return r_inv, t_inv


def lidar_to_global(
    points_lidar: np.ndarray,
    cs_rot: np.ndarray,
    cs_trans: np.ndarray,
    ego_rot: np.ndarray,
    ego_trans: np.ndarray,
) -> np.ndarray:
    """LIDAR -> ego -> global."""
    pts = transform_points(points_lidar, cs_rot, cs_trans)
    return transform_points(pts, ego_rot, ego_trans)


def global_to_lidar(
    points_global: np.ndarray,
    cs_rot: np.ndarray,
    cs_trans: np.ndarray,
    ego_rot: np.ndarray,
    ego_trans: np.ndarray,
) -> np.ndarray:
    """global -> ego -> LIDAR (inverse of lidar_to_global)."""
    ego_r_inv, ego_t_inv = invert_rt(ego_rot, ego_trans)
    pts = transform_points(points_global, ego_r_inv, ego_t_inv)
    cs_r_inv, cs_t_inv = invert_rt(cs_rot, cs_trans)
    return transform_points(pts, cs_r_inv, cs_t_inv)


def yaw_from_quat(q: Sequence[float]) -> float:
    """Yaw (z-rotation) from nuScenes quaternion [w,x,y,z]."""
    return Quaternion(q).yaw_pitch_roll[0]


def points_in_oriented_boxes(
    points_xyz: np.ndarray,
    centers: np.ndarray,
    sizes_wlh: np.ndarray,
    rotations: Sequence[Sequence[float]],
    expand: float = 1.1,
) -> np.ndarray:
    """Return (N, M) bool mask: point i inside expanded box j.

    Box local frame (nuScenes): x=length forward, y=width left, z=height up.
    size = [w, l, h]. Uses full quaternion (not yaw-only) for containment.
    """
    pts = np.asarray(points_xyz, dtype=np.float64)
    n = pts.shape[0]
    m = 0 if centers is None else len(centers)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=bool)

    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    sizes = np.asarray(sizes_wlh, dtype=np.float64).reshape(-1, 3) * float(expand)
    # half extents: x=l/2, y=w/2, z=h/2
    half = np.stack([sizes[:, 1] * 0.5, sizes[:, 0] * 0.5, sizes[:, 2] * 0.5], axis=1)

    mask = np.zeros((n, m), dtype=bool)
    for j in range(m):
        r = quat_to_rot(rotations[j])
        local = (r.T @ (pts - centers[j]).T).T
        inside = (
            (np.abs(local[:, 0]) <= half[j, 0])
            & (np.abs(local[:, 1]) <= half[j, 1])
            & (np.abs(local[:, 2]) <= half[j, 2])
        )
        mask[:, j] = inside
    return mask


def world_to_object_local(
    points_global: np.ndarray,
    center: Sequence[float],
    rotation: Sequence[float],
) -> np.ndarray:
    """Undo box pose: R^T (p - c)."""
    r = quat_to_rot(rotation)
    c = np.asarray(center, dtype=np.float64).reshape(3)
    pts = np.asarray(points_global, dtype=np.float64)
    if pts.size == 0:
        return pts.reshape(0, 3)
    return (r.T @ (pts - c).T).T


def object_local_to_world(
    points_local: np.ndarray,
    center: Sequence[float],
    rotation: Sequence[float],
) -> np.ndarray:
    r = quat_to_rot(rotation)
    c = np.asarray(center, dtype=np.float64).reshape(3)
    pts = np.asarray(points_local, dtype=np.float64)
    if pts.size == 0:
        return pts.reshape(0, 3)
    return (r @ pts.T).T + c


def self_range_mask(
    points_lidar: np.ndarray,
    ranges: Sequence[float] = (3.0, 3.0, 3.0),
) -> np.ndarray:
    """True = keep (outside near-ego cube). SurroundOcc uses abs > range."""
    pts = np.asarray(points_lidar, dtype=np.float64)
    if pts.size == 0:
        return np.zeros((0,), dtype=bool)
    rx, ry, rz = ranges
    return (np.abs(pts[:, 0]) > rx) | (np.abs(pts[:, 1]) > ry) | (np.abs(pts[:, 2]) > rz)


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0 or voxel_size <= 0:
        return pts.reshape(0, 3) if pts.size == 0 else pts
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
    pcd = pcd.voxel_down_sample(voxel_size)
    return np.asarray(pcd.points, dtype=np.float64)


def optional_poisson(
    points: np.ndarray,
    depth: int = 9,
    min_density_quantile: float = 0.1,
) -> np.ndarray:
    """Poisson mesh vertices from static cloud; returns vertex array."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 100:
        return pts
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30)
    )
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 0.0]))
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, n_threads=8
    )
    if min_density_quantile is not None and len(densities) > 0:
        dens = np.asarray(densities)
        mesh.remove_vertices_by_mask(dens < np.quantile(dens, min_density_quantile))
    return np.asarray(mesh.vertices, dtype=np.float64)


def write_ply(path: str, points: np.ndarray, colors: Optional[np.ndarray] = None) -> int:
    pts = np.asarray(points, dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    if pts.size == 0:
        pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
    else:
        pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
        if colors is not None and len(colors) == len(pts):
            pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=False)
    return int(pts.shape[0]) if pts.size else 0


def write_pcd(path: str, points: np.ndarray) -> int:
    return write_ply(path, points)  # open3d infers by extension
