#!/usr/bin/env python3
"""Run Open3D-ML's native Waymo PointPillars model on DetZero point files."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.external_detector.pipeline import (
    generation_relative_provenance,
    rename_noreplace,
    save_raw_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessed-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-identity", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-identity", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--open3dml-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--open3dml-commit",
        default="fcf97c07bf7a113a47d0fcf63760b245c2a2784e",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_pinned_file(path: Path, expected_sha256: str) -> tuple[Path, str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"input must be a regular non-symlink file: {path}")
    path = path.resolve(strict=True)
    actual = sha256_file(path)
    if actual != expected_sha256.lower():
        raise ValueError(f"SHA-256 mismatch for {path}: {actual}")
    return path, actual


def require_runtime_module_source(module, declared_root: Path) -> Path:
    if declared_root.is_symlink() or not declared_root.is_dir():
        raise ValueError(f"invalid declared Open3D-ML root: {declared_root}")
    root = declared_root.resolve(strict=True)
    source = Path(getattr(module, "__file__", ""))
    if source.is_symlink() or not source.is_file():
        raise ValueError("Open3D-ML runtime module lacks a regular source file")
    source = source.resolve(strict=True)
    try:
        source.relative_to(root)
    except ValueError as error:
        raise ValueError("runtime module is outside declared Open3D-ML root") from error
    return source


def main() -> int:
    args = parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError(args.output_dir)
    declared_open3dml_root = args.open3dml_root.resolve(strict=True)
    controlled_root = os.environ.get("OPEN3D_ML_ROOT")
    if (
        controlled_root is None
        or Path(controlled_root).resolve(strict=True) != declared_open3dml_root.parent
    ):
        raise ValueError("OPEN3D_ML_ROOT is not bound to the declared source snapshot")

    config_path, config_hash = require_pinned_file(args.config, args.config_sha256)
    checkpoint_path, checkpoint_hash = require_pinned_file(
        args.checkpoint, args.checkpoint_sha256
    )
    root = args.preprocessed_root
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"invalid preprocessed root: {root}")
    root = root.resolve(strict=True)
    preprocess_manifest_path = root / "preprocess_manifest.json"
    if preprocess_manifest_path.is_symlink() or not preprocess_manifest_path.is_file():
        raise ValueError("missing preprocess_manifest.json")
    preprocess_manifest = json.loads(preprocess_manifest_path.read_text(encoding="utf-8"))
    sequence_name = preprocess_manifest["sequence_name"]
    available_frames = preprocess_manifest["frame_count"]
    if type(available_frames) is not int or available_frames <= 0:
        raise ValueError("invalid preprocessed frame count")
    frame_count = args.max_frames or available_frames
    if frame_count > available_frames:
        raise ValueError("requested frame prefix exceeds preprocessed frame count")
    sequence_dir = root / "waymo_processed_data" / f"segment-{sequence_name}"
    point_paths = [sequence_dir / f"{frame_id:04d}.npy" for frame_id in range(frame_count)]
    if any(path.is_symlink() or not path.is_file() for path in point_paths):
        raise ValueError("preprocessed point frame coverage is incomplete")

    import open3d
    import open3d.ml as ml3d_base
    import open3d.ml.torch  # noqa: F401
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot use it")

    cfg = ml3d_base.utils.Config.load_from_file(str(config_path))
    PointPillars = ml3d_base.utils.get_module("model", "PointPillars", "torch")
    ObjectDetection = ml3d_base.utils.get_module(
        "pipeline", "ObjectDetection", "torch"
    )
    for runtime_class in (PointPillars, ObjectDetection):
        require_runtime_module_source(
            importlib.import_module(runtime_class.__module__),
            declared_open3dml_root,
        )

    stage = args.output_dir.with_name(
        f".{args.output_dir.name}.tmp-{uuid.uuid4().hex}"
    )
    stage.mkdir(parents=True)
    started = time.monotonic()
    try:
        model = PointPillars(device=args.device, seed=0, **cfg.model)
        pipeline = ObjectDetection(
            model,
            device=args.device,
            main_log_dir=str(stage / "open3dml-logs"),
        )
        pipeline.load_ckpt(str(checkpoint_path))

        frames = []
        class_counts = {name: 0 for name in ("VEHICLE", "PEDESTRIAN", "CYCLIST")}
        for frame_id, point_path in enumerate(point_paths):
            points = np.load(point_path, allow_pickle=False)
            if (
                points.dtype != np.float32
                or points.ndim != 2
                or points.shape[1] != 6
                or len(points) == 0
                or not np.isfinite(points).all()
            ):
                raise ValueError(f"invalid point array: {point_path}")
            detector_points = points[points[:, 5] != 1.0, :4]
            if len(detector_points) == 0:
                raise ValueError(f"NLZ filtering removed every point: {point_path}")
            attr = {"split": "test"}
            preprocessed = model.preprocess(
                {"point": detector_points, "calib": None}, attr
            )
            transformed = model.transform(preprocessed, attr)
            predictions = pipeline.run_inference(transformed)[0]
            boxes = []
            for prediction in predictions:
                label = str(prediction.label_class).upper()
                if label not in class_counts:
                    raise ValueError(f"unexpected Open3D-ML class: {label}")
                score = float(prediction.confidence)
                if not np.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("Open3D-ML emitted an invalid score")
                boxes.append(
                    {
                        "center": np.asarray(prediction.center, dtype=np.float32),
                        "size": np.asarray(prediction.size, dtype=np.float32),
                        "yaw": np.float32(prediction.yaw),
                        "label": label,
                        "score": np.float32(score),
                    }
                )
                class_counts[label] += 1
            frames.append(
                {
                    "frame_id": frame_id,
                    "point_count": len(points),
                    "model_point_count": len(preprocessed["point"]),
                    "boxes": boxes,
                }
            )
            print(
                f"frame={frame_id:04d} points={len(points)} "
                f"model_points={len(preprocessed['point'])} boxes={len(boxes)}",
                flush=True,
            )

        raw_path = stage / "raw_predictions.npz"
        save_raw_predictions(raw_path, sequence_name, frames)
        shutil.rmtree(stage / "open3dml-logs", ignore_errors=True)
        if sha256_file(config_path) != config_hash:
            raise RuntimeError("Open3D-ML config changed during inference")
        if sha256_file(checkpoint_path) != checkpoint_hash:
            raise RuntimeError("PointPillars checkpoint changed during inference")
        manifest = {
            "schema_version": "open3dml-waymo-detector-manifest-v1",
            "backend": "Open3D-ML PointPillars Waymo",
            "device": args.device,
            "sequence_name": sequence_name,
            "frame_count": frame_count,
            "box_count": sum(class_counts.values()),
            "class_counts": class_counts,
            "point_features": ["x", "y", "z", "intensity"],
            "nlz_filter": "points[:, 5] != 1.0",
            "config": str(args.config_identity.absolute()),
            "config_sha256": config_hash,
            "checkpoint": str(args.checkpoint_identity.absolute()),
            "checkpoint_sha256": checkpoint_hash,
            "open3d_version": open3d.__version__,
            "open3dml_commit": args.open3dml_commit,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "preprocess_manifest": generation_relative_provenance(
                preprocess_manifest_path, args.output_dir
            ),
            "preprocess_manifest_sha256": sha256_file(preprocess_manifest_path),
            "raw_predictions": "raw_predictions.npz",
            "raw_predictions_sha256": sha256_file(raw_path),
            "elapsed_seconds": time.monotonic() - started,
            "checkpoint_license_status": (
                "upstream-model-zoo-source-recorded; no separate weight license found"
            ),
        }
        (stage / "detector_manifest.json").write_text(
            json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        rename_noreplace(stage, args.output_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    print(json.dumps(manifest, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
