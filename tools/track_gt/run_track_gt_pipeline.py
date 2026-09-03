#!/usr/bin/env python3
"""Orchestrate HEDNet -> SimpleTrack 2Hz tracking-GT pipeline on nuScenes mini."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

CST = timezone(timedelta(hours=8))


def now_cst_stamp() -> str:
    return datetime.now(CST).strftime("%Y%m%d-%H%M%S") + "-CST"


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def run(cmd, cwd=None, env=None, log_path=None):
    print("+", " ".join(cmd), flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    dt = time.time() - t0
    out = (p.stdout or "") + (p.stderr or "")
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(out)
    if p.returncode != 0:
        print(out[-4000:])
        raise RuntimeError(f"cmd failed ({p.returncode}): {' '.join(cmd)}")
    print(f"  ok in {dt:.1f}s", flush=True)
    return {"cmd": cmd, "cwd": cwd, "returncode": p.returncode, "seconds": dt, "log": log_path}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worktree", default="/data/code/cv/AutoLabel/DetZero-nuscenes-simpletrack-trackgt")
    ap.add_argument("--python", default="/data/software/conda/anaconda3/envs/mv2d/bin/python")
    ap.add_argument(
        "--pkl",
        default="/data/code/cv/AutoLabel/BEV-OD/HEDNet-qwen/output/mini/eval/epoch_2/val/default/result.pkl",
    )
    ap.add_argument("--dataroot", default="/data/data/automomous/nuscenes/v1.0-mini")
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--split", default="mini_val")
    ap.add_argument("--score-thres", type=float, default=0.1)
    ap.add_argument("--process", type=int, default=1)
    ap.add_argument("--skip-preprocess", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--skip-viz", action="store_true")
    ap.add_argument("--out-root", default=None, help="override output/track-gt-<stamp>-CST")
    args = ap.parse_args()

    wt = Path(args.worktree)
    st = wt / "third_party" / "SimpleTrack"
    py = args.python
    stamp = now_cst_stamp()
    out_root = Path(args.out_root) if args.out_root else wt / "output" / f"track-gt-{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    data_2hz = out_root / "simpletrack_data_2hz"
    s1 = out_root / "s1_detections"
    s2 = out_root / "s2_simpletrack"
    delivery = out_root / "delivery"
    eval_dir = out_root / "eval"
    visuals = out_root / "visuals"
    logs = out_root / "logs"
    for d in [data_2hz, s1, s2, delivery, eval_dir, visuals, logs]:
        d.mkdir(parents=True, exist_ok=True)

    simpletrack_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(st), text=True
    ).strip()
    worktree_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(wt), text=True
    ).strip()

    assets = {
        "created_at_cst": datetime.now(CST).isoformat(),
        "hednet_pkl": {
            "path": args.pkl,
            "sha256": sha256_file(args.pkl),
        },
        "nuscenes": {
            "dataroot": args.dataroot,
            "version": args.version,
            "split": args.split,
            "scene_json": str(Path(args.dataroot) / args.version / "scene.json"),
        },
        "simpletrack": {
            "path": str(st),
            "commit": simpletrack_commit,
            "url": "https://github.com/tusen-ai/SimpleTrack.git",
            "license": "MIT",
        },
        "worktree_commit_at_start": worktree_commit,
    }
    (out_root / "assets_manifest.json").write_text(json.dumps(assets, indent=2))

    commands = []
    env = os.environ.copy()
    env["PYTHONPATH"] = str(st) + os.pathsep + env.get("PYTHONPATH", "")

    # ---- a) SimpleTrack 2Hz preprocess (skip raw_pc) ----
    if not args.skip_preprocess:
        pre = st / "preprocessing" / "nuscenes_data"
        for script in [
            "token_info.py",
            "time_stamp.py",
            "sensor_calibration.py",
            "ego_pose.py",
            "gt_info.py",
        ]:
            commands.append(
                run(
                    [
                        py,
                        str(pre / script),
                        "--raw_data_folder",
                        args.dataroot,
                        "--data_folder",
                        str(data_2hz),
                        "--mode",
                        "2hz",
                        "--version",
                        args.version,
                        "--split",
                        args.split,
                    ],
                    cwd=str(pre),
                    env=env,
                    log_path=str(logs / f"preprocess_{script}.log"),
                )
            )

    # ---- b) detection JSON + SimpleTrack detection preprocess ----
    det_json = s1 / "hednet_nusc_det.json"
    drop_manifest = delivery / "class_drop_manifest.json"
    commands.append(
        run(
            [
                py,
                str(wt / "tools" / "track_gt" / "hednet_to_nusc_det_json.py"),
                "--pkl",
                args.pkl,
                "--dataroot",
                args.dataroot,
                "--version",
                args.version,
                "--score-thres",
                str(args.score_thres),
                "--out-json",
                str(det_json),
                "--drop-manifest",
                str(drop_manifest),
            ],
            cwd=str(wt),
            env=env,
            log_path=str(logs / "hednet_to_nusc_det_json.log"),
        )
    )
    commands.append(
        run(
            [
                py,
                str(st / "preprocessing" / "nuscenes_data" / "detection.py"),
                "--raw_data_folder",
                args.dataroot,
                "--data_folder",
                str(data_2hz),
                "--det_name",
                "hednet",
                "--file_path",
                str(det_json),
                "--mode",
                "2hz",
                "--velo",
            ],
            cwd=str(st / "preprocessing" / "nuscenes_data"),
            env=env,
            log_path=str(logs / "detection_preprocess.log"),
        )
    )

    # ---- c) SimpleTrack main ----
    cfg = wt / "tools" / "track_gt" / "giou_nopc.yaml"
    commands.append(
        run(
            [
                py,
                str(st / "tools" / "main_nuscenes.py"),
                "--name",
                "SimpleTrack2Hz",
                "--det_name",
                "hednet",
                "--config_path",
                str(cfg),
                "--result_folder",
                str(s2),
                "--data_folder",
                str(data_2hz),
                "--process",
                str(args.process),
            ],
            cwd=str(st),
            env=env,
            log_path=str(logs / "main_nuscenes.log"),
        )
    )

    # ---- d) result creation + type merge ----
    commands.append(
        run(
            [
                py,
                str(st / "tools" / "nuscenes_result_creation.py"),
                "--name",
                "SimpleTrack2Hz",
                "--result_folder",
                str(s2),
                "--data_folder",
                str(data_2hz),
            ],
            cwd=str(st),
            env=env,
            log_path=str(logs / "result_creation.log"),
        )
    )
    commands.append(
        run(
            [
                py,
                str(st / "tools" / "nuscenes_type_merge.py"),
                "--name",
                "SimpleTrack2Hz",
                "--result_folder",
                str(s2),
            ],
            cwd=str(st),
            env=env,
            log_path=str(logs / "type_merge.log"),
        )
    )

    merged = s2 / "SimpleTrack2Hz" / "results" / "results.json"
    if not merged.exists():
        raise FileNotFoundError(merged)
    shutil.copy2(merged, delivery / "tracking_results.json")

    # ---- e) frames_with_track_id.pkl ----
    with open(delivery / "tracking_results.json") as f:
        track_json = json.load(f)
    frames = []
    track_lens = Counter()
    id_frames = defaultdict(list)
    for token, objs in track_json["results"].items():
        frame = {
            "sample_token": token,
            "boxes": [],
            "track_ids": [],
            "names": [],
            "scores": [],
        }
        for o in objs:
            frame["boxes"].append(
                o["translation"] + o["size"] + [0.0]  # placeholder yaw; keep quat separate
            )
            frame["track_ids"].append(o["tracking_id"])
            frame["names"].append(o["tracking_name"])
            frame["scores"].append(o["tracking_score"])
            id_frames[o["tracking_id"]].append(token)
        frame["boxes"] = np.asarray(frame["boxes"], dtype=np.float32) if frame["boxes"] else np.zeros((0, 7), np.float32)
        frames.append(
            {
                "sample_token": token,
                "tracking_id": frame["track_ids"],
                "tracking_name": frame["names"],
                "tracking_score": frame["scores"],
                "translation": [o["translation"] for o in objs],
                "size": [o["size"] for o in objs],
                "rotation": [o["rotation"] for o in objs],
                "velocity": [o["velocity"] for o in objs],
            }
        )
    for tid, toks in id_frames.items():
        track_lens[len(toks)] += 1
    with open(delivery / "frames_with_track_id.pkl", "wb") as f:
        pickle.dump(frames, f)

    track_stats = {
        "n_frames": len(frames),
        "n_tracks": len(id_frames),
        "track_length_hist": dict(sorted(track_lens.items())),
        "single_frame_track_ratio": (
            track_lens.get(1, 0) / max(len(id_frames), 1)
        ),
        "mean_track_length": (
            float(np.mean([len(v) for v in id_frames.values()])) if id_frames else 0.0
        ),
        "boxes_per_frame_mean": float(np.mean([len(fr["tracking_id"]) for fr in frames])) if frames else 0.0,
    }
    (delivery / "track_quality_stats.json").write_text(json.dumps(track_stats, indent=2))

    # ---- f) official tracking eval ----
    eval_info = {"attempted": False, "success": False, "error": None, "metrics": None}
    if not args.skip_eval:
        eval_info["attempted"] = True
        try:
            # Prefer isolated env if present
            eval_py = py
            iso = "/data/software/conda/anaconda3/envs/nusc-track-gt/bin/python"
            if Path(iso).exists():
                eval_py = iso
            commands.append(
                run(
                    [
                        eval_py,
                        str(wt / "tools" / "track_gt" / "eval_nusc_tracking.py"),
                        str(delivery / "tracking_results.json"),
                        "--output_dir",
                        str(eval_dir),
                        "--eval_set",
                        args.split,
                        "--dataroot",
                        args.dataroot,
                        "--version",
                        args.version,
                        "--render_curves",
                        "0",
                    ],
                    cwd=str(wt),
                    env=env,
                    log_path=str(logs / "official_tracking_eval.log"),
                )
            )
            metrics_path = eval_dir / "metrics_summary.json"
            if metrics_path.exists():
                eval_info["success"] = True
                eval_info["metrics"] = json.loads(metrics_path.read_text())
        except Exception as e:
            eval_info["error"] = repr(e)
            (eval_dir / "EVAL_ERROR.txt").write_text(str(e))
            # Try create isolated env once if motmetrics suspected
            err_s = str(e).lower()
            log_txt = ""
            logf = logs / "official_tracking_eval.log"
            if logf.exists():
                log_txt = logf.read_text()[-3000:].lower()
            if ("motmetrics" in err_s or "motmetrics" in log_txt or "pandas" in log_txt) and not Path(iso).exists():
                try:
                    print("Attempting isolated nusc-track-gt env for motmetrics pin...", flush=True)
                    commands.append(
                        run(
                            [
                                "/data/software/conda/anaconda3/bin/conda",
                                "create",
                                "-y",
                                "-n",
                                "nusc-track-gt",
                                "python=3.10",
                            ],
                            log_path=str(logs / "conda_create_nusc_track_gt.log"),
                        )
                    )
                    commands.append(
                        run(
                            [
                                iso if Path(iso).exists() else "/data/software/conda/anaconda3/envs/nusc-track-gt/bin/python",
                                "-m",
                                "pip",
                                "install",
                                "nuscenes-devkit",
                                "motmetrics==1.2.5",
                                "pandas==1.5.3",
                                "numpy==1.23.5",
                                "pyquaternion",
                                "matplotlib",
                                "tqdm",
                            ],
                            log_path=str(logs / "pip_nusc_track_gt.log"),
                        )
                    )
                    iso = "/data/software/conda/anaconda3/envs/nusc-track-gt/bin/python"
                    commands.append(
                        run(
                            [
                                iso,
                                str(wt / "tools" / "track_gt" / "eval_nusc_tracking.py"),
                                str(delivery / "tracking_results.json"),
                                "--output_dir",
                                str(eval_dir),
                                "--eval_set",
                                args.split,
                                "--dataroot",
                                args.dataroot,
                                "--version",
                                args.version,
                                "--render_curves",
                                "0",
                            ],
                            cwd=str(wt),
                            env=env,
                            log_path=str(logs / "official_tracking_eval_iso.log"),
                        )
                    )
                    metrics_path = eval_dir / "metrics_summary.json"
                    if metrics_path.exists():
                        eval_info["success"] = True
                        eval_info["error"] = None
                        eval_info["metrics"] = json.loads(metrics_path.read_text())
                        eval_info["eval_python"] = iso
                except Exception as e2:
                    eval_info["error"] = f"primary={e!r}; iso_retry={e2!r}"
                    (eval_dir / "EVAL_ERROR.txt").write_text(eval_info["error"])

    # ---- g) BEV visuals ----
    viz_info = {"attempted": False, "success": False, "error": None, "files": []}
    if not args.skip_viz:
        viz_info["attempted"] = True
        try:
            commands.append(
                run(
                    [
                        py,
                        str(wt / "tools" / "track_gt" / "render_track_bev.py"),
                        "--dataroot",
                        args.dataroot,
                        "--version",
                        args.version,
                        "--tracking-json",
                        str(delivery / "tracking_results.json"),
                        "--out-dir",
                        str(visuals),
                        "--scenes",
                        "scene-0103,scene-0916",
                        "--max-frames-per-scene",
                        "6",
                    ],
                    cwd=str(wt),
                    env=env,
                    log_path=str(logs / "render_track_bev.log"),
                )
            )
            viz_info["success"] = True
            viz_info["files"] = sorted([str(p.relative_to(out_root)) for p in visuals.rglob("*.png")])
        except Exception as e:
            viz_info["error"] = repr(e)

    run_manifest = {
        "created_at_cst": datetime.now(CST).isoformat(),
        "out_root": str(out_root),
        "python": py,
        "score_thres": args.score_thres,
        "process": args.process,
        "simpletrack_commit": simpletrack_commit,
        "worktree_commit_at_start": worktree_commit,
        "commands": commands,
        "delivery": {
            "tracking_results_json": str(delivery / "tracking_results.json"),
            "frames_with_track_id_pkl": str(delivery / "frames_with_track_id.pkl"),
            "class_drop_manifest": str(drop_manifest),
            "track_quality_stats": track_stats,
        },
        "eval": eval_info,
        "viz": viz_info,
        "geometry_source": "detector+motion_filter",
        "box_frame": "global",
        "tracker": "SimpleTrack2Hz",
        "config": str(cfg),
    }
    (out_root / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, default=str))
    print("DONE", out_root)
    print(json.dumps({"track_stats": track_stats, "eval": {k: eval_info[k] for k in eval_info if k != "metrics"}, "amota": (eval_info.get("metrics") or {}).get("amota")}, indent=2))


if __name__ == "__main__":
    main()
