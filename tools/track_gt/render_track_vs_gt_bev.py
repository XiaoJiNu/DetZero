#!/usr/bin/env python3
"""Full-size side-by-side BEV: GT (left) vs tracking (right). No contact-sheet downscale."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box, LidarPointCloud
from pyquaternion import Quaternion


TRACKING_NAMES = {
    "bicycle", "bus", "car", "motorcycle", "pedestrian", "trailer", "truck",
}


def color_for_id(tid: str):
    h = abs(hash(tid)) % (256 ** 3)
    return (((h >> 16) & 255) / 255.0, ((h >> 8) & 255) / 255.0, (h & 255) / 255.0)


def load_pc_global(nusc: NuScenes, sample_token: str):
    sample = nusc.get("sample", sample_token)
    sd_token = sample["data"]["LIDAR_TOP"]
    sd = nusc.get("sample_data", sd_token)
    pc = LidarPointCloud.from_file(nusc.get_sample_data_path(sd_token))
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    pose = nusc.get("ego_pose", sd["ego_pose_token"])
    pc.rotate(Quaternion(cs["rotation"]).rotation_matrix)
    pc.translate(np.array(cs["translation"]))
    pc.rotate(Quaternion(pose["rotation"]).rotation_matrix)
    pc.translate(np.array(pose["translation"]))
    ego_xy = np.array(pose["translation"][:2])
    pts = pc.points
    if pts.shape[1] > 120000:
        idx = np.random.RandomState(0).choice(pts.shape[1], 120000, replace=False)
        pts = pts[:, idx]
    return pts, ego_xy


def draw_box(ax, box: Box, color, label=None, lw=2.0, ls="-"):
    corners = box.bottom_corners()[:2, :].T
    xs = list(corners[:, 0]) + [corners[0, 0]]
    ys = list(corners[:, 1]) + [corners[0, 1]]
    ax.plot(xs, ys, color=color, linewidth=lw, linestyle=ls)
    if label:
        ax.text(box.center[0], box.center[1], label, color=color, fontsize=9,
                fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.55, edgecolor="none", pad=0.6))


def gt_boxes_for_sample(nusc: NuScenes, sample_token: str):
    sample = nusc.get("sample", sample_token)
    out = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        name = ann["category_name"].split(".")[1] if "." in ann["category_name"] else ann["category_name"]
        # map nuScenes category to tracking name
        # e.g. vehicle.car -> car, human.pedestrian.adult -> pedestrian
        cat = ann["category_name"]
        tname = None
        if cat.startswith("vehicle.car"):
            tname = "car"
        elif cat.startswith("vehicle.truck"):
            tname = "truck"
        elif cat.startswith("vehicle.bus"):
            tname = "bus"
        elif cat.startswith("vehicle.trailer"):
            tname = "trailer"
        elif cat.startswith("vehicle.motorcycle"):
            tname = "motorcycle"
        elif cat.startswith("vehicle.bicycle"):
            tname = "bicycle"
        elif "pedestrian" in cat:
            tname = "pedestrian"
        if tname is None or tname not in TRACKING_NAMES:
            continue
        box = Box(ann["translation"], ann["size"], Quaternion(ann["rotation"]), name=tname)
        out.append((box, tname, ann["instance_token"][:6]))
    return out


def setup_ax(ax, pts, ego_xy, radius, title):
    ax.scatter(pts[0], pts[1], s=0.08, c="0.45", alpha=0.35, rasterized=True)
    ax.scatter([ego_xy[0]], [ego_xy[1]], c="k", s=80, marker="x", zorder=5)
    ax.set_aspect("equal")
    ax.set_xlim(ego_xy[0] - radius, ego_xy[0] + radius)
    ax.set_ylim(ego_xy[1] - radius, ego_xy[1] + radius)
    ax.set_xlabel("X global (m)")
    ax.set_ylabel("Y global (m)")
    ax.set_title(title, fontsize=14)


def render_pair(nusc, sample_token, track_objs, out_path: Path, title_prefix: str, radius: float):
    pts, ego_xy = load_pc_global(nusc, sample_token)
    gt = gt_boxes_for_sample(nusc, sample_token)

    # Large figure, two full panels — no contact-sheet shrink
    fig, axes = plt.subplots(1, 2, figsize=(24, 12), dpi=150)
    setup_ax(axes[0], pts, ego_xy, radius,
             f"{title_prefix} | GT (green)  n={len(gt)}")
    setup_ax(axes[1], pts, ego_xy, radius,
             f"{title_prefix} | Track (by ID)  n={len(track_objs)}")

    for box, tname, iid in gt:
        draw_box(axes[0], box, color=(0.05, 0.65, 0.15), label=f"{tname[:3]}:{iid}", lw=2.2)

    for o in track_objs:
        box = Box(
            o["translation"], o["size"], Quaternion(o["rotation"]),
            name=o.get("tracking_name", ""), score=o.get("tracking_score", 0),
        )
        tid = str(o.get("tracking_id", "?"))
        short = tid.split("_")[-1]
        draw_box(axes[1], box, color_for_id(tid),
                 label=f"{o.get('tracking_name','')[:3]}:{short}", lw=2.0)

    fig.suptitle(
        "Left = official GT boxes | Right = SimpleTrack tracking GT  |  same LiDAR, global BEV",
        fontsize=16, y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", type=Path, required=True)
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--tracking-json", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--radius", type=float, default=40.0)
    ap.add_argument("--scenes", default="scene-0103,scene-0916")
    ap.add_argument("--frame-indices", default="0,7,15,23,31,39",
                    help="0-based keyframe indices within each scene (0916 may clamp)")
    args = ap.parse_args()

    nusc = NuScenes(version=args.version, dataroot=str(args.dataroot), verbose=False)
    results = json.loads(args.tracking_json.read_text())["results"]

    # scene name -> ordered sample tokens
    scene_tokens = {}
    for scene in nusc.scene:
        name = scene["name"]
        tokens = []
        token = scene["first_sample_token"]
        while token:
            tokens.append(token)
            token = nusc.get("sample", token)["next"]
        scene_tokens[name] = tokens

    indices = [int(x) for x in args.frame_indices.split(",") if x.strip() != ""]
    # scene-0916 in previous render used 0,8,16,... — allow per-scene via same list; clamp
    for scene_name in [s.strip() for s in args.scenes.split(",") if s.strip()]:
        tokens = scene_tokens[scene_name]
        # match prior sampling: for 0916 use step-friendly indices if last is 39 but len=41
        use_idx = []
        for i in indices:
            if i < len(tokens):
                use_idx.append(i)
            else:
                use_idx.append(len(tokens) - 1)
        # unique preserve order
        seen = set()
        use_idx2 = []
        for i in use_idx:
            if i not in seen:
                seen.add(i)
                use_idx2.append(i)

        for i in use_idx2:
            tok = tokens[i]
            objs = results.get(tok, [])
            out = args.output_dir / scene_name / f"frame_{i:03d}_{tok[:8]}_gt_vs_track.png"
            render_pair(nusc, tok, objs, out, f"{scene_name} frame={i}", args.radius)
            print("wrote", out, "n_gt_track", len(gt_boxes_for_sample(nusc, tok)), len(objs))


if __name__ == "__main__":
    main()
