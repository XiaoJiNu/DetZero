#!/usr/bin/env python3
"""G3' bypass: S7-style BEV center-distance mAP (3-class fold) for the nu-stack
frame pkl, vs the raw HEDNet detector under the same protocol.
Gate: tracked mAP >= detector mAP - 0.02.
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

from tools.external_detector.eval_nus_vs_gt import (  # noqa: E402
    CLASSES, DIST_THRESHOLDS, average_precision, load_gt)
from tools.nustack import CLASS_MAP_3, TRACKING_CLASSES  # noqa: E402

FOLD = {c: CLASS_MAP_3[c] for c in TRACKING_CLASSES}


def per_frame(path: Path, tracking: bool):
    data = pickle.load(path.open("rb"))
    out = {}
    for frame in data:
        fid = int(frame["frame_id"])
        boxes = np.asarray(frame["boxes_lidar"], dtype=np.float64)
        names = np.asarray(frame["name"]).tolist()
        scores = np.asarray(frame["score"], dtype=np.float64)
        for i, n in enumerate(names):
            cls = str(n) if str(n) in CLASSES else FOLD.get(str(n))
            if cls is None:
                continue
            out.setdefault(fid, {}).setdefault(cls, ([], []))
            out[fid][cls][0].append(boxes[i, :3])
            out[fid][cls][1].append(scores[i])
    return {k: {c: (np.asarray(v[0]), np.asarray(v[1]))
                for c, v in d.items()} for k, d in out.items()}


def score(path, gt, max_dist):
    preds = per_frame(path, True)
    pred_items, gt_items = {}, {}
    for fid, classes in preds.items():
        for cls, (boxes, scores) in classes.items():
            items = pred_items.setdefault(cls, [])
            items.extend((fid, b, s) for b, s in zip(boxes, scores))
    for fid, classes in gt.items():
        for cls, boxes in classes.items():
            gt_items.setdefault(cls, []).extend((fid, b) for b in boxes)
    report = {}
    means = []
    for cls in CLASSES:
        dets = [(f, b, s) for f, b, s in pred_items.get(cls, [])
                if np.linalg.norm(b[:2]) <= max_dist]
        g = gt_items.get(cls, [])
        aps = [average_precision(dets, g, t) for t in DIST_THRESHOLDS]
        aps = [a for a in aps if a is not None]
        m = float(np.mean(aps)) if aps else None
        report[cls] = m
        if m is not None:
            means.append(m)
    report["mAP"] = float(np.mean(means)) if means else None
    return report


def detector_frames(hednet_result: Path, info_path: Path, score_threshold: float):
    raw = pickle.load(hednet_result.open("rb"))
    by_token = {r["metadata"]["token"]: r for r in raw}
    infos = pickle.load(info_path.open("rb"))
    out = []
    for fid, info in enumerate(infos):
        det = by_token[info["token"]]
        boxes = np.asarray(det["boxes_lidar"], dtype=np.float64)
        names = np.asarray(det["name"]).tolist()
        scores = np.asarray(det["score"], dtype=np.float64)
        keep = (scores >= score_threshold) & np.isin(names, TRACKING_CLASSES)
        out.append({"frame_id": fid, "boxes_lidar": boxes[keep],
                    "name": np.asarray(names)[keep], "score": scores[keep]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracked-pkl", type=Path, required=True)
    ap.add_argument("--detector-pkl", type=Path, default=None)
    ap.add_argument("--hednet-result", type=Path, default=None,
                    help="alternative to --detector-pkl: raw HEDNet result.pkl")
    ap.add_argument("--nuscenes-info", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    infos = pickle.load(args.nuscenes_info.open("rb"))
    tokens = [m["token"] for m in infos]
    gt = load_gt(args.nuscenes_info, tokens)
    if args.detector_pkl is not None:
        det = score(args.detector_pkl, gt, 50.0)
    else:
        import tempfile
        frames = detector_frames(args.hednet_result, args.nuscenes_info, 0.1)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tf:
            pickle.dump(frames, tf)
            tmp = Path(tf.name)
        det = score(tmp, gt, 50.0)
        tmp.unlink()
    trk = score(args.tracked_pkl, gt, 50.0)
    verdict = ("PASS" if trk["mAP"] >= det["mAP"] - 0.02
               else "FAIL" if trk["mAP"] is not None else "NOT_MEASURABLE")
    out = {"detector_mAP": det, "tracked_mAP": trk,
           "G3_prime": verdict}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=1))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
