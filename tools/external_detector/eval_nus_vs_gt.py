#!/usr/bin/env python3
"""Score detector / tracking / final boxes against nuScenes GT (S7).

The ONLY stage that reads ground truth. It never feeds any value back into
S1-S6 outputs; it exists to adjudicate risks R1 (tracking on 2Hz) and
R2 (Waymo-domain GRM/PRM on nuScenes). AP is nuScenes-style BEV with
center-distance thresholds, class-mapped the same way as the adapter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.external_detector.adapt_hednet_to_detzero import CLASS_MAP, DROPPED

DIST_THRESHOLDS = (0.5, 1.0, 2.0)
CLASSES = ("Vehicle", "Pedestrian", "Cyclist")


def load_frame_boxes(path: Path, tracking_style: bool):
    """-> {frame_id: {class: (boxes[N,3+], scores[N])}} in lidar frame."""
    data = pickle.load(path.open("rb"))
    if isinstance(data, list):
        tracking_style = False  # frame-list schema (detector / tracked frames)
    per_frame = {}
    if tracking_style:  # dict seq -> obj -> track arrays (global boxes)
        for tracks in data.values():
            for track in tracks.values():
                names = np.asarray(track["name"]).tolist()
                boxes_g = np.asarray(track["boxes_global"], dtype=np.float64)
                poses = np.asarray(track["pose"], dtype=np.float64)
                scores = np.asarray(track["score"], dtype=np.float64)
                for i, sample in enumerate(track["sample_idx"]):
                    frame_id = int(str(sample))
                    cls = str(names[i])
                    if cls not in CLASSES:
                        continue
                    inv = np.linalg.inv(poses[i])
                    center_g = np.append(boxes_g[i, :3], 1.0)
                    center_l = (inv @ center_g)[:3]
                    entry = per_frame.setdefault(frame_id, {}).setdefault(cls, ([], []))
                    entry[0].append(center_l)
                    entry[1].append(scores[i])
    else:  # detector/final frame list with boxes_lidar (N,9)
        for frame in data:
            frame_id = int(frame["frame_id"])
            boxes = np.asarray(frame["boxes_lidar"], dtype=np.float64)
            names = np.asarray(frame["name"]).tolist()
            scores = np.asarray(frame["score"], dtype=np.float64)
            for cls in CLASSES:
                mask = np.asarray([n == cls for n in names])
                if mask.any():
                    per_frame.setdefault(frame_id, {})[cls] = (
                        boxes[mask, :3], scores[mask])
    return {k: {c: (np.asarray(v[0]), np.asarray(v[1]))
                for c, v in d.items()} for k, d in per_frame.items()}


def load_gt(info_path: Path, tokens: list[str]):
    infos = pickle.load(info_path.open("rb"))
    by_token = {m["token"]: m for m in infos}
    per_frame = {}
    for frame_id, token in enumerate(tokens):
        m = by_token[token]
        boxes = np.asarray(m["gt_boxes"], dtype=np.float64)
        names = np.asarray(m["gt_names"]).tolist()
        # drop z and the invalid far/empty GT rows the way nuScenes does
        num_pts = np.asarray(m["num_lidar_pts"])
        keep_all = num_pts > -1
        for cls in CLASSES:
            mask = np.asarray([CLASS_MAP.get(str(n), None) == cls
                               and str(n) not in DROPPED for n in names]) & keep_all
            if mask.any():
                per_frame.setdefault(frame_id, {})[cls] = boxes[mask, :3]
    return per_frame


def average_precision(detections, ground_truths, threshold):
    """BEV center-distance AP, nuScenes recall interpolation, 101 points."""
    if len(ground_truths) == 0:
        return None  # not measurable on this class
    if len(detections) == 0:
        return 0.0
    scores = np.array([float(d[2]) for d in detections])
    order = np.argsort(-scores)
    matched = np.zeros((len(ground_truths),), dtype=bool)
    tp = np.zeros(len(order))
    fp = np.zeros(len(order))
    gt_by_frame = {}
    for i, gt in ground_truths:
        gt_by_frame.setdefault(i, []).append((gt, False))
    for rank, det_idx in enumerate(order):
        frame_id, det_boxes, _ = detections[det_idx]
        best, best_d = None, threshold
        for j, (gt, used) in enumerate(gt_by_frame.get(frame_id, [])):
            if used:
                continue
            d = np.linalg.norm(gt - det_boxes)
            if d < best_d:
                best, best_d = j, d
        if best is not None:
            tp[rank] = 1
            gt_by_frame[frame_id][best] = (gt_by_frame[frame_id][best][0], True)
        else:
            fp[rank] = 1
    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
    recall = tp_cum / len(ground_truths)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-16)
    # 101-point interpolation
    interp = np.zeros(101)
    recalls = np.linspace(0, 1, 101)
    for i, r in enumerate(recalls):
        p = precision[recall >= r]
        interp[i] = p.max() if len(p) else 0.0
    return float(interp.mean())


def score_stage(stage_name, path, tracking_style, gt, max_dist=None):
    preds = load_frame_boxes(path, tracking_style)
    pred_items = {}
    for frame_id, classes in preds.items():
        for cls, (boxes, scores) in classes.items():
            items = pred_items.setdefault(cls, [])
            items.extend((frame_id, b, s) for b, s in zip(boxes, scores))
    gt_items = {}
    for frame_id, classes in gt.items():
        for cls, boxes in classes.items():
            items = gt_items.setdefault(cls, [])
            items.extend((frame_id, b) for b in boxes)
    report = {}
    for cls in CLASSES:
        detections = [(f, b, s) for f, b, s in pred_items.get(cls, [])
                      if max_dist is None or np.linalg.norm(b[:2]) <= max_dist]
        ground = list(gt_items.get(cls, []))
        per_thr = {}
        for thr in DIST_THRESHOLDS:
            ap = average_precision(detections,
                                   [(f, b) for f, b in ground], thr)
            per_thr[str(thr)] = ap
        aps = [v for v in per_thr.values() if v is not None]
        report[cls] = {
            "num_det": len(detections), "num_gt": len(ground),
            "ap_by_distance": per_thr,
            "ap_mean": float(np.mean(aps)) if aps else None,
        }
    means = [report[c]["ap_mean"] for c in CLASSES if report[c]["ap_mean"] is not None]
    report["mAP"] = float(np.mean(means)) if means else None
    return {stage_name: report}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nuscenes-info", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--tracking", type=Path, required=True)
    parser.add_argument("--final-frames", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-dist", type=float, default=50.0,
                        help="clip GT/detections beyond this BEV radius (nuScenes default 50m)")
    args = parser.parse_args()

    tokens = json.load((args.scene_root / "gt_tokens.json").open())["tokens"]
    gt = load_gt(args.nuscenes_info, tokens)
    report = {
        "scene_name": tokens and json.load((args.scene_root / "gt_tokens.json").open())["scene_name"],
        "max_dist_m": args.max_dist,
        "distance_thresholds_m": list(DIST_THRESHOLDS),
        "stages": {},
    }
    report["stages"].update(score_stage("detector", args.detector, False, gt, args.max_dist))
    report["stages"].update(score_stage("tracking", args.tracking, True, gt, args.max_dist))
    report["stages"].update(score_stage("final_grm_prm", args.final_frames, False, gt, args.max_dist))
    d, t, f = (report["stages"][s]["mAP"] for s in
               ("detector", "tracking", "final_grm_prm"))
    report["risk_verdicts"] = {
        "R1_tracking_vs_detector": ("DEGRADED" if t is not None and d is not None and t < d - 0.02
                                    else "NOT_DEGRADED" if t is not None else "NOT_MEASURABLE"),
        "R2_refined_vs_tracking": ("DEGRADED" if f is not None and t is not None and f < t - 0.02
                                   else "NOT_DEGRADED" if f is not None else "NOT_MEASURABLE"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"mAP": {"detector": d, "tracking": t, "final": f},
                      "verdicts": report["risk_verdicts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
