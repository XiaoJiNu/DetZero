#!/usr/bin/env python3
"""Per-frame ID-correctness check for nu-stack tracking truth.

Matches every tracking box against nuScenes GT annotations (BEV centre
distance, Hungarian), records the tracking_id -> GT instance mapping per
frame, flags ID switches (same tracking_id matched to a different GT
instance in consecutive frames), and renders BEV + CAM_FRONT overlays
where each tracked box is COLOURED by its matched GT instance: a box
that changes colour between frames = an identity switch. Labels: red
number = tracking_id suffix, green number = GT instance index.

Writes <out>/idsw_report.json + <out>/NNNN_token.png (same frame
numbering as visuals/).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pyquaternion import Quaternion
from scipy.optimize import linear_sum_assignment

from render_tracks import corners_global, EDGES

TRACKING_CLASSES = {"car", "truck", "construction_vehicle", "bus",
                    "trailer", "barrier", "motorcycle", "bicycle",
                    "pedestrian", "traffic_cone"}
PRED_CLASSES = {"car", "truck", "bus", "trailer", "motorcycle", "bicycle",
                "pedestrian"}  # 7 eval classes (cone/barrier/cv excluded)

# tab20 hex -> rgb
PALETTE = [tuple(bytes.fromhex(h[1:])) for h in (
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
    "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
    "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
    "#17becf", "#9edae5")]


def gt_class(instance_name: str):
    n = instance_name
    if n.startswith("human.pedestrian"):
        return "pedestrian"
    if n.startswith("vehicle."):
        c = n[len("vehicle."):]
        return c if c in PRED_CLASSES else None
    return None


def bev_match(cost, thresh=2.0):
    """Hungarian on BEV distance matrix; returns {pred_i: (gt_i, dist)}."""
    if cost.size == 0:
        return {}
    big = 1e6
    c = np.where(cost > thresh, big, cost)
    rows, cols = linear_sum_assignment(c)
    return {int(r): (int(cc), float(cost[r, cc]))
            for r, cc in zip(rows, cols) if c[r, cc] < big}


def analyze_scene(frames, gt_by_token, pred_by_token, thresh=2.0):
    """frames: ordered list of sample tokens (one scene).
    Returns per-frame match info + id switch events."""
    last = {}            # tracking_id -> gt instance token (last match)
    out = []
    for token in frames:
        gts = gt_by_token[token]
        preds = pred_by_token[token]
        if gts and preds:
            cost = np.linalg.norm(
                np.array([p["translation"][:2] for p in preds])[:, None]
                - np.array([g["translation"][:2] for g in gts])[None],
                axis=2)
        else:
            cost = np.zeros((len(preds), len(gts)))
        m = bev_match(cost, thresh)
        switches = []
        assign = {}      # pred idx -> (gt idx, dist)
        for pi, (gi, dist) in m.items():
            tid = preds[pi]["tracking_id"]
            g = gts[gi]["token"]
            if tid in last and last[tid] != g:
                switches.append({"tracking_id": tid, "from_gt": last[tid],
                                 "to_gt": g, "name": preds[pi].get("tracking_name")})
            last[tid] = g
            assign[pi] = (gi, dist)
        out.append({"token": token, "assign": assign, "switches": switches,
                    "n_pred": len(preds), "n_gt": len(gts)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--track-json", type=Path, required=True)
    ap.add_argument("--dataroot", type=Path, default=Path("/data/data/automomous/nuscenes/v1.0-mini"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report-only", action="store_true", help="skip png rendering")
    args = ap.parse_args()

    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes("v1.0-mini", dataroot=str(args.dataroot), verbose=False)
    results = json.loads(args.track_json.read_text())["results"]

    samples = [s for s in nusc.sample if s["token"] in results]
    by_scene = defaultdict(list)
    for s in samples:
        by_scene[s["scene_token"]].append(s)

    gt_by_token, pred_by_token = {}, {}
    gt_idx, gt_color = {}, {}     # instance token -> short number / colour
    n_gt_used = 0
    for s in samples:
        token = s["token"]
        gts = []
        for at in s["anns"]:
            a = nusc.get("sample_annotation", at)
            if a["num_lidar_pts"] < 1:
                continue
            cls = gt_class(a["category_name"])
            if cls is None:
                continue
            inst = a["instance_token"]
            if inst not in gt_idx:
                gt_idx[inst] = n_gt_used
                gt_color[inst] = PALETTE[n_gt_used % len(PALETTE)]
                n_gt_used += 1
            gts.append({"token": inst, "translation": a["translation"],
                        "size": a["size"], "rotation": a["rotation"], "cls": cls})
        gt_by_token[token] = gts
        pred_by_token[token] = results[token]

    report_frames = {}
    events_total = 0
    for scene_tok, ss in by_scene.items():
        tokens = [s["token"] for s in ss]
        ana = analyze_scene(tokens, gt_by_token, pred_by_token)
        # conflict: two preds on one gt (devkit counts these as id switches too)
        for s_tok, fr in zip(tokens, ana):
            ev = list(fr["switches"])
            seen = defaultdict(list)
            for pi, (gi, _) in fr["assign"].items():
                seen[gt_by_token[s_tok][gi]["token"]].append(
                    pred_by_token[s_tok][pi]["tracking_id"])
            fr["conflicts"] = {g: ids for g, ids in seen.items() if len(ids) > 1}
            events_total += len(ev)
            report_frames[s_tok] = fr

    # ---------- render overlays ----------
    manifest = []
    if not args.report_only:
        args.out.mkdir(parents=True, exist_ok=True)
        lim, W, H, CAM_H = 60.0, 720, 720, 405
        for fi, sample in enumerate(nusc.sample):
            token = sample["token"]
            if token not in results:
                continue
            fr = report_frames[token]
            lidar = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            lcs = nusc.get("calibrated_sensor", lidar["calibrated_sensor_token"])
            ego = nusc.get("ego_pose", lidar["ego_pose_token"])
            inv_L_r = Quaternion(lcs["rotation"]).inverse.rotation_matrix
            inv_E_r = Quaternion(ego["rotation"]).inverse.rotation_matrix
            t_e, t_l = np.asarray(ego["translation"]), np.asarray(lcs["translation"])

            def g2lid(pts):
                return (inv_L_r @ ((inv_E_r @ (np.asarray(pts) - t_e).T).T - t_l).T).T

            pts = np.fromfile(str(args.dataroot / lidar["filename"]),
                              dtype=np.float32).reshape(-1, 5)[:, :3]
            bev = Image.new("RGB", (W, H), (255, 255, 255))
            d = ImageDraw.Draw(bev)
            def px(q):
                return (W / 2 + q[0] / lim * (W / 2 - 10), H / 2 - q[1] / lim * (W / 2 - 10))
            for q in pts[np.abs(pts[:, 0]) < lim]:
                if abs(q[1]) < lim:
                    d.point(px(q), fill=(120, 120, 120))

            cam = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
            cs = nusc.get("calibrated_sensor", cam["calibrated_sensor_token"])
            cego = nusc.get("ego_pose", cam["ego_pose_token"])
            Ce = np.eye(4); Ce[:3, :3] = Quaternion(cego["rotation"]).rotation_matrix; Ce[:3, 3] = cego["translation"]
            Cc = np.eye(4); Cc[:3, :3] = Quaternion(cs["rotation"]).rotation_matrix; Cc[:3, 3] = cs["translation"]
            g2cam = np.linalg.inv(Ce @ Cc); K = np.asarray(cs["camera_intrinsic"])
            img = Image.open(str(args.dataroot / cam["filename"])).convert("RGB").resize((W, CAM_H))
            dc = ImageDraw.Draw(img)
            sx, sy = W / cam["width"], CAM_H / cam["height"]
            switch_ids = {e["tracking_id"] for e in fr["switches"]}

            # GT boxes: dashed-ish green outlines + green number
            for gi, g in enumerate(gt_by_token[token]):
                cl = g2lid(np.asarray(g["translation"])[None])[0]
                if abs(cl[0]) < lim and abs(cl[1]) < lim:
                    cg = g2lid(corners_global(g["translation"], g["size"], g["rotation"]))
                    p2 = [px(q) for q in cg]
                    for a, b in EDGES:
                        d.line([p2[a], p2[b]], fill=(0, 150, 0), width=1)
                    d.text((cl[0] / lim * (W / 2 - 10) + W / 2 - 8,
                            H / 2 - cl[1] / lim * (W / 2 - 10) - 20),
                           str(gt_idx[g["token"]]), fill=(0, 130, 0))

            # predicted boxes: colour = matched GT instance, label = tracking id
            for pi, p in enumerate(pred_by_token[token]):
                gi = fr["assign"].get(pi)
                col = gt_color[gt_by_token[token][gi[0]]["token"]] if gi else (170, 170, 170)
                idsw = p["tracking_id"] in switch_ids
                cl = g2lid(np.asarray(p["translation"])[None])[0]
                if abs(cl[0]) < lim and abs(cl[1]) < lim:
                    cg = g2lid(corners_global(p["translation"], p["size"], p["rotation"]))
                    p2 = [px(q) for q in cg]
                    for a, b in EDGES:
                        d.line([p2[a], p2[b]], fill=col, width=3 if idsw else 2)
                    u, v = px(cl)
                    lab = str(p["tracking_id"].split("_")[-1]) + ("!!" if idsw else "")
                    d.text((u + 3, v - 9), lab, fill=(210, 0, 0) if idsw else col)
                    if gi:
                        d.text((u + 3, v + 2), f"d={gi[1]:.2f}", fill=(80, 80, 80))
                pc = (g2cam[:3, :3] @ corners_global(p["translation"], p["size"], p["rotation"]).T + g2cam[:3, 3:4]).T
                if (pc[:, 2] > 0.5).any():
                    uv = (K @ pc.T).T
                    uv = uv[:, :2] / np.clip(uv[:, 2:3], 1e-9, None)
                    uv[:, 0] *= sx; uv[:, 1] *= sy
                    vis = pc[:, 2] > 0.5
                    for a, b in EDGES:
                        if vis[a] and vis[b]:
                            dc.line([uv[a].tolist(), uv[b].tolist()], fill=col,
                                    width=3 if idsw else 2)

            n_sw = len(fr["switches"])
            d.text((10, 6), f"BEV lidar +/-{int(lim)}m  colour=GT object  red!!=IDSW", fill=(0, 0, 0))
            d.text((10, 20), f"frame {fi:03d}  pred {fr['n_pred']}  gt {fr['n_gt']}  idsw {n_sw}",
                   fill=(210, 0, 0) if n_sw else (0, 0, 0))
            canvas = Image.new("RGB", (W, H + CAM_H))
            canvas.paste(bev, (0, 0)); canvas.paste(img, (0, H))
            out = args.out / f"{fi:04d}_{token[:8]}.png"
            canvas.save(out)
            manifest.append({"frame_id": fi, "token": token, "png": str(out),
                             "id_switches": n_sw})

    report = {
        "method": "Hungarian BEV-centre match, <=2m; id switch = tracking_id matched to "
                  "different GT instance vs its previous appearance; GT filtered num_lidar_pts>=1, 7 tracking classes",
        "dataroot": str(args.dataroot),
        "n_gt_instances": n_gt_used,
        "auto_id_switch_total": events_total,
    }
    # frame-indexed list (manifest order == nusc.sample index)
    fi_of = {s["token"]: i for i, s in enumerate(nusc.sample) if s["token"] in results}
    report["per_frame"] = [
        {"frame_id": fi_of[tk], "token": tk, "n_pred": fr["n_pred"], "n_gt": fr["n_gt"],
         "matched": len(fr["assign"]), "unmatched_pred": fr["n_pred"] - len(fr["assign"]),
         "switches": fr["switches"], "conflicts": fr["conflicts"]}
        for tk, fr in sorted(report_frames.items(), key=lambda kv: fi_of[kv[0]])
    ]
    summary_p = args.out / "idsw_report.json" if not args.report_only else (args.out.parent / "idsw_report.json")
    summary_p.parent.mkdir(parents=True, exist_ok=True)
    summary_p.write_text(json.dumps(report, indent=1) + "\n")
    n_match = sum(f["matched"] for f in report["per_frame"])
    n_pred = sum(f["n_pred"] for f in report["per_frame"])
    print(json.dumps({"auto_id_switch_total": events_total,
                      "pred_matched_to_gt": f"{n_match}/{n_pred}",
                      "report": str(summary_p),
                      "frames_rendered": len(manifest) or None}, indent=1))
    return 0


# ---------------- self-check (synthetic) ----------------
def _self_check():
    # scene: 2 preds A,B, 2 GT g1,g2 over 3 frames; A jumps g1->g2 at f2
    def box(tid, x, y):
        return {"tracking_id": tid, "translation": [x, y, 0.0]}
    frames = ["t0", "t1", "t2"]
    gt = {"t0": [{"token": "g1", "translation": [0, 0, 0]}, {"token": "g2", "translation": [5, 0, 0]}],
          "t1": [{"token": "g1", "translation": [1, 0, 0]}, {"token": "g2", "translation": [5, 0, 0]}],
          "t2": [{"token": "g1", "translation": [2, 0, 0]}, {"token": "g2", "translation": [6, 0, 0]}]}
    pr = {"t0": [box("A", 0.2, 0), box("B", 4.8, 0)],
          "t1": [box("A", 4.9, 0), box("B", 1.1, 0)],      # both swap GTs
          "t2": [box("A", 6.1, 0), box("B", 2.0, 0)]}
    ana = analyze_scene(frames, gt, pr, thresh=2.0)
    assert ana[0]["assign"][0][0] == 0 and ana[0]["assign"][1][0] == 1  # A->g1 B->g2
    assert {e["tracking_id"] for e in ana[1]["switches"]} == {"A", "B"}, ana[1]["switches"]
    assert not ana[2]["switches"], ana[2]["switches"]          # stable again
    assert bev_match(np.array([[3.0]]), 2.0) == {}             # beyond thresh unmatched
    print("self-check PASS")


if __name__ == "__main__":
    import sys
    if "--self-check" in sys.argv:
        _self_check()
        raise SystemExit(0)
    raise SystemExit(main())
