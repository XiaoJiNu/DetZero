#!/usr/bin/env python3
"""Run Stage A GRM/PRM inference on real DetZero tracks."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import pickle
import shutil
import sys
from typing import Any
import uuid

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.external_detector.safe_io import safe_load_pickle


CLASS_NAMES = ("Vehicle", "Pedestrian", "Cyclist")


def class_input_status(objects: dict[Any, dict[str, Any]]) -> dict[str, Any]:
    track_count = len(objects)
    box_count = sum(len(track["boxes_global"]) for track in objects.values())
    status = "PENDING" if track_count else "NOT_EXECUTED_NO_INPUT"
    return {
        "input_track_count": track_count,
        "input_box_count": box_count,
        "geometry": {"status": status, "forward_count": 0},
        "position": {"status": status, "forward_count": 0},
    }


def model_output_record(
    track: dict[str, Any], predictions: np.ndarray
) -> dict[str, Any]:
    boxes = np.asarray(predictions, dtype=np.float32)
    count = len(track["sample_idx"])
    if boxes.shape != (count, 7):
        raise ValueError(f"prediction shape mismatch: {boxes.shape} != {(count, 7)}")
    if not np.isfinite(boxes).all() or np.any(boxes[:, 3:6] <= 0):
        raise ValueError("refining predictions contain invalid boxes")
    raw_object_ids = track.get("obj_ids", [track.get("obj_id")])
    object_ids = np.unique(raw_object_ids)
    if len(object_ids) != 1:
        raise ValueError("track must contain exactly one object id")
    for key in ("score", "pose"):
        if len(track[key]) != count:
            raise ValueError(f"track field length mismatch: {key}")
    names = np.asarray(track["name"])
    if names.ndim == 0:
        names = np.full(count, str(names))
    if len(names) != count:
        raise ValueError("track field length mismatch: name")
    return {
        "sequence_name": str(track["sequence_name"]),
        "obj_id": int(object_ids[0]),
        "sample_idx": np.asarray(track["sample_idx"]).copy(),
        "boxes_global": boxes.copy(),
        "score": np.asarray(track["score"], dtype=np.float32).copy(),
        "name": names.copy(),
        "pose": np.asarray(track["pose"], dtype=np.float64).copy(),
    }


def _save_pickle(path: Path, value: Any) -> None:
    with path.open("xb") as stream:
        pickle.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())


def _load_class_objects(class_dir: Path) -> dict[tuple[str, Any], dict[str, Any]]:
    objects = {}
    for path in sorted(class_dir.iterdir()):
        if path.is_symlink() or not path.is_file() or path.suffix != ".pkl":
            raise RuntimeError(f"unexpected object-data entry: {path}")
        sequence_objects = safe_load_pickle(path)
        for object_id, track in sequence_objects.items():
            key = (str(track["sequence_name"]), object_id)
            if key in objects:
                raise RuntimeError(f"duplicate object track: {key}")
            objects[key] = track
    return objects


def run_stage_a_refining(
    waymo_root: Path,
    tracking_path: Path,
    output_dir: Path,
    checkpoint_root: Path = REPO_ROOT / "checkpoints",
    checkpoint_identity_root: Path | None = None,
    device: str = "cuda",
    workers: int = 0,
) -> dict[str, Any]:
    for local_path in (
        REPO_ROOT,
        REPO_ROOT / "utils",
        REPO_ROOT / "daemon",
        REPO_ROOT / "refining",
        REPO_ROOT / "refining" / "tools",
    ):
        sys.path.insert(0, str(local_path))
    from prepare_object_data import WaymoObjectDataPrepare
    from reproduce_inference import run_checkpoint
    from tools.external_detector.pipeline import (
        _sha256_file,
        generation_relative_provenance,
        rename_noreplace,
    )

    waymo_root = Path(waymo_root).resolve(strict=True)
    checkpoint_root = Path(checkpoint_root).resolve(strict=True)
    checkpoint_identity_root = Path(
        checkpoint_identity_root or checkpoint_root
    ).resolve(strict=True)
    if checkpoint_root.is_symlink() or not checkpoint_root.is_dir():
        raise ValueError(f"invalid checkpoint root: {checkpoint_root}")
    tracking_path = Path(tracking_path)
    output_dir = Path(output_dir).absolute()
    if tracking_path.is_symlink() or not tracking_path.is_file():
        raise FileNotFoundError(tracking_path)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    if workers < 0:
        raise ValueError("workers must be non-negative")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = output_dir.with_name(f".{output_dir.name}.tmp-{uuid.uuid4().hex}")
    stage.mkdir()
    result_dir = stage / "result"
    result_dir.mkdir()
    logger = logging.getLogger("stage-a-refining")

    try:
        for class_name in CLASS_NAMES:
            WaymoObjectDataPrepare(
                class_name=class_name,
                root_path=str(waymo_root),
                output_root=str(stage),
                split="test",
                track_data_path=str(tracking_path),
                workers=workers,
                logger=logger,
            ).init_infos_from_tracking()

        class_manifest = {}
        output_hashes = {}
        total_forwards = 0
        for class_name in CLASS_NAMES:
            objects = _load_class_objects(stage / class_name)
            class_status = class_input_status(objects)
            if objects:
                for model_kind in ("geometry", "position"):
                    model_output = {}
                    model_metadata = None
                    for (sequence_name, object_id), track in objects.items():
                        inference = run_checkpoint(
                            track,
                            class_name,
                            model_kind,
                            device=device,
                            checkpoint_root=checkpoint_root,
                        )
                        model_output.setdefault(sequence_name, {})[object_id] = (
                            model_output_record(track, inference.pred_boxes_world)
                        )
                        metadata = {
                            "checkpoint": str(
                                checkpoint_identity_root / inference.checkpoint_path.name
                            ),
                            "checkpoint_sha256": inference.checkpoint_sha256,
                            "loaded_tensors": inference.loaded_tensors,
                            "model_tensors": inference.model_tensors,
                        }
                        if model_metadata is not None and metadata != model_metadata:
                            raise RuntimeError("checkpoint metadata changed between tracks")
                        model_metadata = metadata
                    result_path = result_dir / f"{class_name}_{model_kind}.pkl"
                    _save_pickle(result_path, model_output)
                    assert model_metadata is not None
                    class_status[model_kind] = {
                        "status": "EXECUTED",
                        "forward_count": len(objects),
                        **model_metadata,
                        "output": str(result_path.relative_to(stage)),
                        "output_sha256": _sha256_file(result_path),
                    }
                    output_hashes[str(result_path.relative_to(stage))] = _sha256_file(
                        result_path
                    )
                    total_forwards += len(objects)
            class_manifest[class_name] = class_status
            shutil.rmtree(stage / class_name)

        manifest = {
            "schema_version": "detzero-stage-a-refining-v1",
            "tracking": {
                "path": generation_relative_provenance(tracking_path, output_dir),
                "sha256": _sha256_file(tracking_path),
            },
            "waymo_root": generation_relative_provenance(waymo_root, output_dir),
            "classes": class_manifest,
            "model_forward_count": total_forwards,
            "crm": {"status": "NOT_EXECUTED_NO_CRM_BY_DESIGN"},
            "score_policy": "TRACKING_SCORE_PASSTHROUGH",
            "outputs": output_hashes,
        }
        manifest_path = stage / "refining_manifest.json"
        with manifest_path.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        rename_noreplace(stage, output_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    with (output_dir / "refining_manifest.json").open(encoding="utf-8") as stream:
        published_manifest = json.load(stream)
    if published_manifest != manifest:
        raise RuntimeError("published refining manifest read-back mismatch")
    return published_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waymo-root", type=Path, required=True)
    parser.add_argument("--tracking", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-root", type=Path, default=REPO_ROOT / "checkpoints"
    )
    parser.add_argument("--checkpoint-identity-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    manifest = run_stage_a_refining(
        args.waymo_root,
        args.tracking,
        args.output_dir,
        checkpoint_root=args.checkpoint_root,
        checkpoint_identity_root=args.checkpoint_identity_root,
        device=args.device,
        workers=args.workers,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
