#!/usr/bin/env python3
"""BEV visualization of SimpleTrack tracking results on nuScenes mini scenes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from pyquaternion import Quaternion


def color_for_id(tid: str):
    h = abs(hash(tid)) % (256 ** 3)
    r = ((h >> 16) & 255) / 255.0
    g = ((h >> 8) & 255) / 255.0
    b = (h & 255) / 255.0
    return (r, g, b)


def draw_box_bev(ax, box: Box, color, label=None, lw=1.5):
    corners = box.bottom_corners()[:2, :].T
    xs = list(corners[:, 0]) + [corners[0, 0]]
    ys = list(corners[:, 1]) + [corners[0, 1]]
    ax.plot(xs, ys, color=color, linewidth=lw)
    if label:
        ax.text(box.center[0], box.center[1], label, color=color, fontsize=6)


def render_frame(nusc, sample_token, objs, out_path, title):
    sample = nusc.get("sample", sample_token)
    sd_token = sample["data"]["LIDAR_TOP"]
    sd = nusc.get("sample_data", sd_token)
    pc_path = nusc.get_sample_data_path(sd_token)
    pc = LidarPointCloud.from_file(pc_path)

    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    pose = nusc.get("ego_pose", sd["ego_pose_token"])

    pc.rotate(Quaternion(cs["rotation"]).rotation_matrix)
    pc.translate(np.array(cs["translation"]))
    pc.rotate(Quaternion(pose["rotation"]).rotation_matrix)
    pc.translate(np.array(pose["translation"]))

    pts = pc.points
    if pts.shape[1] > 80000:
        idx = np.random.choice(pts.shape[1], 80000, replace=False)
        pts = pts[:, idx]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.scatter(pts[0], pts[1], s=0.05, c="0.5", alpha=0.4)
    ego_xy = np.array(pose["translation"][:2])
    ax.scatter([ego_xy[0]], [ego_xy[1]], c="k", s=30, marker="x")

    for o in objs:
        box = Box(
            o["translation"],
            o["size"],
            Quaternion(o["rotation"]),
            name=o.get("tracking_name", ""),
            score=o.get("tracking_score", 0),
        )
        tid = o.get("tracking_id", "?")
        short = tid.split("_")[-1] if isinstance(tid, str) else str(tid)
        draw_box_bev(ax, box, color_for_id(str(tid)), label=f"{o.get('tracking_name','')[:3]}:{short}")

    ax.set_aspect("equal")
    ax.set_title(title)
    r = 40
    ax.set_xlim(ego_xy[0] - r, ego_xy[0] + r)
    ax.set_ylim(ego_xy[1] - r, ego_xy[1] + r)
    ax.set_xlabel("X global (m)")
    ax.set_ylabel("Y global (m)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def contact_sheet(paths, out_path, cols=3):
    paths = [p for p in paths if Path(p).exists()]
    if not paths:
        return
    rows = int(np.ceil(len(paths) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < len(paths):
            img = plt.imread(paths[i])
            ax.imshow(img)
            ax.set_title(Path(paths[i]).name, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--tracking-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scenes", default="scene-0103,scene-0916")
    ap.add_argument("--max-frames-per-scene", type=int, default=6)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.tracking_json) as f:
        data = json.load(f)
    results = data["results"]

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    want = set(args.scenes.split(","))
    written = []

    for scene in nusc.scene:
        if scene["name"] not in want:
            continue
        tokens = []
        tok = scene["first_sample_token"]
        while tok:
            tokens.append(tok)
            tok = nusc.get("sample", tok)["next"]
        n = min(args.max_frames_per_scene, len(tokens))
        if n <= 0:
            continue
        idxs = np.linspace(0, len(tokens) - 1, n, dtype=int)
        scene_dir = out_dir / scene["name"]
        scene_dir.mkdir(parents=True, exist_ok=True)
        scene_paths = []
        for i in idxs:
            st = tokens[i]
            objs = results.get(st, [])
            outp = scene_dir / f"frame_{i:03d}_{st[:8]}.png"
            render_frame(
                nusc,
                st,
                objs,
                outp,
                title=f"{scene['name']} frame={i} n_trk={len(objs)}",
            )
            scene_paths.append(str(outp))
            written.append(str(outp))
        contact_sheet(scene_paths, out_dir / f"{scene['name']}_contact.png", cols=3)
        written.append(str(out_dir / f"{scene['name']}_contact.png"))

    print(json.dumps({"n_images": len(written), "files": written}, indent=2))


if __name__ == "__main__":
    main()
