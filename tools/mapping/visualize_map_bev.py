"""Full-size BEV top-down PNGs for static / dynamic / combined maps."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def load_ply(path: Path) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points)
    return pts if pts.size else np.zeros((0, 3))


def draw_boxes_bev(ax, boxes: Dict[str, Any], color="C1", lw=1.0):
    for tid, b in boxes.items():
        c = np.asarray(b["translation"][:2], dtype=np.float64)
        # boxes_last are in global — BEV of map is in last lidar; skip if wrong frame.
        # Caller should pass boxes already transformed, or we only draw in lidar frame maps
        # without global boxes. Prefer drawing nothing here unless lidar-frame boxes provided.
        pass


def corners_bev_from_box_lidar(center, size_wlh, yaw, expand=1.0):
    w, l, h = np.asarray(size_wlh, dtype=np.float64) * expand
    # rectangle in xy: length along heading
    dx, dy = l / 2.0, w / 2.0
    corners = np.array(
        [[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy], [dx, dy]], dtype=np.float64
    )
    ca, sa = np.cos(yaw), np.sin(yaw)
    R = np.array([[ca, -sa], [sa, ca]])
    return (R @ corners.T).T + np.asarray(center[:2], dtype=np.float64)


def render_bev(
    points: np.ndarray,
    out_path: Path,
    title: str,
    boxes_lidar: Optional[Sequence[dict]] = None,
    range_m: float = 50.0,
    max_points: int = 400000,
    figsize=(12, 12),
    dpi: int = 150,
    point_color="0.35",
    point_size: float = 0.15,
):
    pts = np.asarray(points)
    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if pts.shape[0] > 0:
        ax.scatter(pts[:, 0], pts[:, 1], s=point_size, c=point_color, alpha=0.5, linewidths=0)
    ax.scatter([0], [0], c="red", s=40, marker="x", label="last lidar origin")

    if boxes_lidar:
        for b in boxes_lidar:
            from pyquaternion import Quaternion

            yaw = Quaternion(b["rotation"]).yaw_pitch_roll[0]
            # box center in last lidar frame expected
            poly = corners_bev_from_box_lidar(b["translation"], b["size"], yaw, expand=1.0)
            ax.plot(poly[:, 0], poly[:, 1], color="C1", linewidth=1.0, alpha=0.8)

    ax.set_aspect("equal")
    ax.set_xlim(-range_m, range_m)
    ax.set_ylim(-range_m, range_m)
    ax.set_xlabel("X lidar (m)")
    ax.set_ylabel("Y lidar (m)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def write_html_index(scene_dir: Path, scene_name: str):
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{scene_name} mapping</title>
<style>
body {{ font-family: sans-serif; margin: 16px; background: #111; color: #eee; }}
img {{ max-width: 100%; border: 1px solid #333; margin-bottom: 24px; }}
h2 {{ margin-top: 32px; }}
</style></head><body>
<h1>{scene_name}</h1>
<p>BEV in last-keyframe LIDAR_TOP frame. Full-size PNGs.</p>
<h2>Static</h2><img src="bev_static.png" alt="static"/>
<h2>Dynamic at last</h2><img src="bev_dynamic.png" alt="dynamic"/>
<h2>Combined</h2><img src="bev_combined.png" alt="combined"/>
</body></html>
"""
    (scene_dir / "index.html").write_text(html, encoding="utf-8")


def visualize_scene_dir(
    scene_dir: Path,
    scene_name: str,
    range_m: float = 50.0,
    boxes_lidar: Optional[list] = None,
):
    scene_dir = Path(scene_dir)
    static = load_ply(scene_dir / "static_map.ply")
    dynamic = load_ply(scene_dir / "dynamic_at_last.ply")
    combined = load_ply(scene_dir / "combined.ply")

    render_bev(
        static,
        scene_dir / "bev_static.png",
        title=f"{scene_name} static (n={len(static)})",
        boxes_lidar=None,
        range_m=range_m,
        point_color="0.4",
    )
    render_bev(
        dynamic,
        scene_dir / "bev_dynamic.png",
        title=f"{scene_name} dynamic@last (n={len(dynamic)})",
        boxes_lidar=boxes_lidar,
        range_m=range_m,
        point_color="C0",
        point_size=0.4,
    )
    render_bev(
        combined,
        scene_dir / "bev_combined.png",
        title=f"{scene_name} combined (n={len(combined)})",
        boxes_lidar=boxes_lidar,
        range_m=range_m,
        point_color="0.3",
    )
    write_html_index(scene_dir, scene_name)
    return {
        "bev_static.png": str(scene_dir / "bev_static.png"),
        "bev_dynamic.png": str(scene_dir / "bev_dynamic.png"),
        "bev_combined.png": str(scene_dir / "bev_combined.png"),
        "index.html": str(scene_dir / "index.html"),
    }
