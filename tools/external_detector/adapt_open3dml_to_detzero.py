#!/usr/bin/env python3
"""Adapt safe Open3D-ML predictions to DetZero's frame-pickle contract."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import uuid

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.external_detector.pipeline import (
    _sha256_file,
    adapt_raw_predictions,
    generation_relative_provenance,
    rename_noreplace,
)
from tools.external_detector.safe_io import safe_load_pickle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-predictions", type=Path, required=True)
    parser.add_argument("--waymo-info", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError(args.output_dir)
    infos = safe_load_pickle(args.waymo_info, 256 * 1024 * 1024)
    if not isinstance(infos, list):
        raise ValueError("Waymo info pickle must contain a frame list")
    frames = adapt_raw_predictions(
        args.raw_predictions, infos, expected_frame_count=args.expected_frames
    )

    stage = args.output_dir.with_name(
        f".{args.output_dir.name}.tmp-{uuid.uuid4().hex}"
    )
    stage.mkdir(parents=True)
    try:
        result_path = stage / "detzero_result.pkl"
        with result_path.open("xb") as stream:
            pickle.dump(frames, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        class_counts = {
            name: sum(np.count_nonzero(frame["name"] == name) for frame in frames)
            for name in ("Vehicle", "Pedestrian", "Cyclist")
        }
        manifest = {
            "schema_version": "open3dml-to-detzero-adapter-manifest-v1",
            "sequence_name": frames[0]["sequence_name"],
            "frame_count": len(frames),
            "box_count": sum(class_counts.values()),
            "class_counts": class_counts,
            "input_raw_predictions": generation_relative_provenance(
                args.raw_predictions, args.output_dir
            ),
            "input_raw_predictions_sha256": _sha256_file(args.raw_predictions),
            "input_waymo_info": generation_relative_provenance(
                args.waymo_info, args.output_dir
            ),
            "input_waymo_info_sha256": _sha256_file(args.waymo_info),
            "output_pickle": "detzero_result.pkl",
            "output_pickle_sha256": _sha256_file(result_path),
            "box_schema": ["x", "y", "z", "length", "width", "height", "heading", "vx", "vy"],
            "center_definition": "geometric box center",
            "yaw_conversion": "detzero_heading = wrap(-open3d_yaw - pi/2)",
            "size_conversion": "open3d [width,height,length] -> detzero [length,width,height]",
            "velocity_status": "unavailable; vx=vy=0",
        }
        manifest_path = stage / "adapter_manifest.json"
        with manifest_path.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, allow_nan=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        rename_noreplace(stage, args.output_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    print(json.dumps(manifest, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
