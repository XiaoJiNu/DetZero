#!/usr/bin/env python3
"""Render nu-stack tracking truth frames: BEV cuboid wireframes + CAM_FRONT
projection (white bg, gray points, blue track boxes with id/score labels).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pyquaternion import Quaternion


def corners_global(translation, size_wlh, quat):
    w, l, h = size_wlh
    R = np.asarray(Quaternion(quat).rotation_matrix)
    x = np.array([l / 2, -l / 2])
    y = np.array([w / 2, -w / 2])
    z = np.array([-h / 2, h / 2])
    pts = np.array([[px, py, pz] for px in x for py in y for pz in z])
    return pts @ R.T + np.asarray(translation)


EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--track-json", type=Path, required=True)
    ap.add_argument("--dataroot", type=Path, default=Path("/data/data/automomous/nuscenes/v1.0-mini"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--every", type=int, default=1)
    args = ap.parse_args()

    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes("v1.0-mini", dataroot=str(args.dataroot), verbose=False)
    results = json.loads(args.track_json.read_text())["results"]
    args.out.mkdir(parents=True, exist_ok=True)

    lim = 60.0
    W = H = 720
    manifest = []
    for fi, sample in enumerate(nusc.sample):
        if fi % args.every:
            continue
        token = sample["token"]
        if token not in results:
            continue
        lidar = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        lcs = nusc.get("calibrated_sensor", lidar["calibrated_sensor_token"])
        ego = nusc.get("ego_pose", lidar["ego_pose_token"])
        E = np.eye(4)
        E[:3, :3] = Quaternion(ego["rotation"]).rotation_matrix
        E[:3, 3] = ego["translation"]
        L = np.eye(4)
        L[:3, :3] = Quaternion(lcs["rotation"]).rotation_matrix
        L[:3, 3] = lcs["translation"]
        inv_L_r = Quaternion(lcs["rotation"]).inverse.rotation_matrix
        inv_E_r = Quaternion(ego["rotation"]).inverse.rotation_matrix

        pts = np.fromfile(str(args.dataroot / lidar["filename"]), dtype=np.float32).reshape(-1, 5)
        p_lidar = pts[:, :3]  # raw bins are already in the lidar sample frame

        cam = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        cs = nusc.get("calibrated_sensor", cam["calibrated_sensor_token"])
        cego = nusc.get("ego_pose", cam["ego_pose_token"])
        Ce = np.eye(4)
        Ce[:3, :3] = Quaternion(cego["rotation"]).rotation_matrix
        Ce[:3, 3] = cego["translation"]
        Cc = np.eye(4)
        Cc[:3, :3] = Quaternion(cs["rotation"]).rotation_matrix
        Cc[:3, 3] = cs["translation"]
        g2cam = np.linalg.inv(Ce @ Cc)
        K = np.asarray(cs["camera_intrinsic"])
        CAM_H = 405
        img = Image.open(str(args.dataroot / cam["filename"])).convert("RGB").resize((W, CAM_H))
        sx = W / cam["width"]
        sy = CAM_H / cam["height"]

        # ---------- BEV panel ----------
        bev = Image.new("RGB", (W, H), (255, 255, 255))
        d = ImageDraw.Draw(bev)
        sel = (np.abs(p_lidar[:, 0]) < lim) & (np.abs(p_lidar[:, 1]) < lim)
        xy = p_lidar[sel][:, [0, 1]]
        coords = [(W / 2 + q[0] / lim * (W / 2 - 10), H / 2 - q[1] / lim * (H / 2 - 10)) for q in xy]
        for u, v in coords:
            d.point((u, v), fill=(120, 120, 120))

        dc = ImageDraw.Draw(img)
        for s in results[token]:
            c_ego = inv_E_r @ (np.asarray(s["translation"]) - E[:3, 3])
            c_lid = inv_L_r @ (c_ego - np.asarray(lcs["translation"]))
            if abs(c_lid[0]) < lim and abs(c_lid[1]) < lim:
                cg = corners_global(s["translation"], s["size"], s["rotation"])
                c_ego = (inv_E_r @ (cg - E[:3, 3]).T).T
                c_lid2 = (inv_L_r @ (c_ego - np.asarray(lcs["translation"])).T).T
                pts2 = [(W / 2 + q[0] / lim * (W / 2 - 10), H / 2 - q[1] / lim * (H / 2 - 10)) for q in c_lid2]
                for a, b in EDGES:
                    d.line([pts2[a], pts2[b]], fill=(30, 80, 220), width=2)
                cu = (W / 2 + c_lid[0] / lim * (W / 2 - 10), H / 2 - c_lid[1] / lim * (H / 2 - 10))
                d.text((cu[0] + 3, cu[1] - 9), str(s["tracking_id"]).split("_")[-1], fill=(210, 40, 40))
            pc = (g2cam[:3, :3] @ corners_global(s["translation"], s["size"], s["rotation"]).T + g2cam[:3, 3:4]).T
            if (pc[:, 2] > 0.5).any():
                uv = (K @ pc.T).T
                uv = uv[:, :2] / np.clip(uv[:, 2:3], 1e-9, None)
                uv[:, 0] *= sx
                uv[:, 1] *= sy
                vis = pc[:, 2] > 0.5
                for a, b in EDGES:
                    if vis[a] and vis[b]:
                        dc.line([uv[a].tolist(), uv[b].tolist()], fill=(30, 80, 220), width=2)

        d.text((10, 6), f"BEV lidar +/-{int(lim)}m", fill=(0, 0, 0))
        d.text((10, 20), f"frame {fi:03d}  tracks {len(results[token])}", fill=(0, 0, 0))
        canvas = Image.new("RGB", (W, H + 405))
        canvas.paste(bev, (0, 0))
        canvas.paste(img, (0, H))
        out = args.out / f"{fi:04d}_{token[:8]}.png"
        canvas.save(out)
        manifest.append({"frame_id": fi, "token": token, "png": str(out), "boxes": len(results[token])})

    (args.out / "render_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({"frames": len(manifest), "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
