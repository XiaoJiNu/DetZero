#!/usr/bin/env python3
"""Combine DetZero GRM and PRM outputs without CRM."""

from __future__ import annotations

import argparse
import copy
import json
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


def _frame_ids(values: Any) -> np.ndarray:
    normalized = []
    for value in np.asarray(values).tolist():
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("frame ids must be non-negative integers")
        if isinstance(value, str):
            if not value.isdigit():
                raise ValueError(f"invalid frame id: {value!r}")
            value = int(value)
        if not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"invalid frame id: {value!r}")
        normalized.append(int(value))
    return np.asarray(normalized, dtype=np.int64)


def combine_no_crm_track(
    tracking: dict[str, Any],
    geometry: dict[str, Any],
    position: dict[str, Any],
) -> dict[str, Any]:
    tracking_boxes = np.asarray(tracking["boxes_global"], dtype=np.float32)
    geometry_boxes = np.asarray(geometry["boxes_global"], dtype=np.float32)
    position_boxes = np.asarray(position["boxes_global"], dtype=np.float32)
    count = len(tracking["sample_idx"])
    if tracking_boxes.shape != (count, 9):
        raise ValueError(f"tracking boxes must have shape {(count, 9)}")
    if geometry_boxes.shape != (count, 7) or position_boxes.shape != (count, 7):
        raise ValueError("GRM/PRM boxes are not track-aligned")
    tracking_frame_ids = _frame_ids(tracking["sample_idx"])
    for model_record in (geometry, position):
        if not np.array_equal(_frame_ids(model_record["sample_idx"]), tracking_frame_ids):
            raise ValueError("GRM/PRM frame identity mismatch")
        if not np.array_equal(model_record["score"], tracking["score"]):
            raise ValueError("GRM/PRM score is not a tracking-score passthrough")
    final_boxes = tracking_boxes.copy()
    final_boxes[:, :3] = position_boxes[:, :3]
    final_boxes[:, 3:6] = geometry_boxes[:, 3:6]
    final_boxes[:, 6] = position_boxes[:, 6]
    if not np.isfinite(final_boxes).all() or np.any(final_boxes[:, 3:6] <= 0):
        raise ValueError("combined boxes are invalid")
    final = copy.deepcopy(tracking)
    final["boxes_global"] = final_boxes
    final["score"] = np.asarray(tracking["score"], dtype=np.float32).copy()
    return final


def _global_to_lidar(boxes_global: np.ndarray, pose: np.ndarray) -> np.ndarray:
    from tools.external_detector.pipeline import _validated_pose

    pose = _validated_pose(np.asarray(pose))
    boxes_global = np.asarray(boxes_global, dtype=np.float32)
    inverse = np.linalg.inv(pose)
    centers = np.concatenate(
        (boxes_global[:, :3], np.ones((len(boxes_global), 1))), axis=1
    )
    boxes_lidar = boxes_global.copy()
    boxes_lidar[:, :3] = (centers @ inverse.T)[:, :3]
    boxes_lidar[:, 6] = (
        boxes_global[:, 6]
        + np.arctan2(inverse[1, 0], inverse[0, 0])
        + np.pi
    ) % (2 * np.pi) - np.pi
    velocity = np.column_stack(
        (boxes_global[:, 7:9], np.zeros(len(boxes_global), dtype=np.float32))
    )
    boxes_lidar[:, 7:9] = (velocity @ inverse[:3, :3].T)[:, :2]
    return boxes_lidar


def tracks_to_frames(
    tracks_by_sequence: dict[str, dict[Any, dict[str, Any]]],
    detector_frames: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    observations: dict[tuple[str, int], list[tuple[int, dict[str, Any], int]]] = {}
    for sequence_name, tracks in tracks_by_sequence.items():
        for object_id, track in tracks.items():
            count = len(track["sample_idx"])
            for index in range(count):
                key = (sequence_name, int(track["sample_idx"][index]))
                observations.setdefault(key, []).append((int(object_id), track, index))

    output = []
    seen_frames = set()
    for source_frame in detector_frames:
        sequence_name = str(source_frame["sequence_name"])
        frame_id = int(source_frame["frame_id"])
        frame_key = (sequence_name, frame_id)
        if frame_key in seen_frames:
            raise ValueError(f"duplicate detector frame: {frame_key}")
        seen_frames.add(frame_key)
        frame_observations = sorted(observations.pop(frame_key, []), key=lambda item: item[0])
        if frame_observations:
            object_ids = np.asarray([item[0] for item in frame_observations], dtype=np.int64)
            boxes_global = np.stack(
                [item[1]["boxes_global"][item[2]] for item in frame_observations]
            ).astype(np.float32)
            names = np.asarray(
                [item[1]["name"][item[2]] for item in frame_observations]
            )
            scores = np.asarray(
                [item[1]["score"][item[2]] for item in frame_observations],
                dtype=np.float32,
            )
        else:
            object_ids = np.empty(0, dtype=np.int64)
            boxes_global = np.empty((0, 9), dtype=np.float32)
            names = np.empty(0, dtype="<U1")
            scores = np.empty(0, dtype=np.float32)
        pose = np.asarray(source_frame["pose"], dtype=np.float64)
        frame = {
            "sequence_name": sequence_name,
            "frame_id": frame_id,
            "pose": pose.copy(),
            "obj_ids": object_ids,
            "name": names,
            "score": scores,
            "boxes_global": boxes_global,
            "boxes_lidar": _global_to_lidar(boxes_global, pose),
        }
        if "timestamp" in source_frame:
            frame["timestamp"] = source_frame["timestamp"]
        output.append(frame)
    if observations:
        raise ValueError(f"track observations have no detector frame: {sorted(observations)}")
    return output


def _load_pickle(path: Path) -> Any:
    return safe_load_pickle(path)


def _save_pickle(path: Path, value: Any) -> None:
    with path.open("xb") as stream:
        pickle.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())


def _save_final_arrays(path: Path, frames: list[dict[str, Any]]) -> None:
    sequence_names = {str(frame["sequence_name"]) for frame in frames}
    if len(sequence_names) != 1:
        raise ValueError("Stage A final arrays require exactly one sequence")
    counts = np.asarray([len(frame["score"]) for frame in frames], dtype=np.int64)
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)))
    label_by_name = {name: index for index, name in enumerate(CLASS_NAMES)}

    def concatenate(key: str, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
        if int(offsets[-1]) == 0:
            return np.empty(shape, dtype=dtype)
        return np.concatenate(
            [np.asarray(frame[key], dtype=dtype) for frame in frames], axis=0
        )

    names = concatenate("name", (0,), "<U10")
    try:
        labels = np.asarray([label_by_name[str(name)] for name in names], dtype=np.int8)
    except KeyError as error:
        raise ValueError(f"unknown final class: {error.args[0]}") from error
    with path.open("xb") as stream:
        np.savez_compressed(
            stream,
            sequence_name=np.asarray(next(iter(sequence_names))),
            class_names=np.asarray(CLASS_NAMES),
            frame_ids=np.asarray([frame["frame_id"] for frame in frames], dtype=np.int64),
            frame_offsets=offsets,
            poses=np.stack([frame["pose"] for frame in frames]).astype(np.float64),
            boxes_lidar=concatenate("boxes_lidar", (0, 9), np.float32),
            boxes_global=concatenate("boxes_global", (0, 9), np.float32),
            scores=concatenate("score", (0,), np.float32),
            labels=labels,
            object_ids=concatenate("obj_ids", (0,), np.int64),
        )
        stream.flush()
        os.fsync(stream.fileno())


def combine_stage_a(
    tracking_path: Path,
    detector_frames_path: Path,
    refining_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    from tools.external_detector.pipeline import _sha256_file, rename_noreplace

    tracking_path = Path(tracking_path)
    detector_frames_path = Path(detector_frames_path)
    refining_dir = Path(refining_dir)
    output_dir = Path(output_dir).absolute()
    if refining_dir.is_symlink() or not refining_dir.is_dir():
        raise FileNotFoundError(refining_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)

    tracking = _load_pickle(tracking_path)
    detector_frames = _load_pickle(detector_frames_path)
    with (refining_dir / "refining_manifest.json").open(encoding="utf-8") as stream:
        refining_manifest = json.load(stream)
    if set(refining_manifest["classes"]) != set(CLASS_NAMES):
        raise ValueError("refining manifest class set mismatch")
    if refining_manifest.get("crm") != {
        "status": "NOT_EXECUTED_NO_CRM_BY_DESIGN"
    }:
        raise ValueError("CRM must not be executed")
    if refining_manifest.get("score_policy") != "TRACKING_SCORE_PASSTHROUGH":
        raise ValueError("refining score policy mismatch")

    final_tracks = {sequence: {} for sequence in tracking}
    for class_name in CLASS_NAMES:
        class_status = refining_manifest["classes"][class_name]
        statuses = {
            class_status[kind]["status"] for kind in ("geometry", "position")
        }
        class_tracking = {
            (sequence, object_id): track
            for sequence, tracks in tracking.items()
            for object_id, track in tracks.items()
            if set(np.asarray(track["name"]).tolist()) == {class_name}
        }
        if statuses == {"NOT_EXECUTED_NO_INPUT"}:
            if class_tracking or class_status["input_track_count"] != 0:
                raise ValueError(f"false empty-class claim: {class_name}")
            for kind in ("geometry", "position"):
                if (refining_dir / "result" / f"{class_name}_{kind}.pkl").exists():
                    raise ValueError(f"unexpected empty-class model output: {class_name}")
            continue
        if statuses != {"EXECUTED"}:
            raise ValueError(f"incomplete refining status: {class_name}")
        geometry = _load_pickle(
            refining_dir / "result" / f"{class_name}_geometry.pkl"
        )
        position = _load_pickle(
            refining_dir / "result" / f"{class_name}_position.pkl"
        )
        expected_keys = set(class_tracking)
        geometry_keys = {
            (sequence, object_id)
            for sequence, records in geometry.items()
            for object_id in records
        }
        position_keys = {
            (sequence, object_id)
            for sequence, records in position.items()
            for object_id in records
        }
        if geometry_keys != expected_keys or position_keys != expected_keys:
            raise ValueError(f"refining/track key mismatch: {class_name}")
        for (sequence, object_id), track in class_tracking.items():
            final_tracks[sequence][object_id] = combine_no_crm_track(
                track, geometry[sequence][object_id], position[sequence][object_id]
            )

    frames = tracks_to_frames(final_tracks, detector_frames)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = output_dir.with_name(f".{output_dir.name}.tmp-{uuid.uuid4().hex}")
    stage.mkdir()
    try:
        track_output = stage / "final_track.pkl"
        frame_output = stage / "final_frame_grm_prm_score_passthrough.pkl"
        array_output = stage / "final_arrays.npz"
        _save_pickle(track_output, final_tracks)
        _save_pickle(frame_output, frames)
        _save_final_arrays(array_output, frames)
        manifest = {
            "schema_version": "detzero-stage-a-final-v1",
            "frame_count": len(frames),
            "track_count": sum(len(tracks) for tracks in final_tracks.values()),
            "classes": refining_manifest["classes"],
            "crm": {"status": "NOT_EXECUTED_NO_CRM_BY_DESIGN"},
            "score_policy": "TRACKING_SCORE_PASSTHROUGH",
            "inputs": {
                "tracking_sha256": _sha256_file(tracking_path),
                "detector_frames_sha256": _sha256_file(detector_frames_path),
                "refining_manifest_sha256": _sha256_file(
                    refining_dir / "refining_manifest.json"
                ),
            },
            "outputs": {
                track_output.name: _sha256_file(track_output),
                frame_output.name: _sha256_file(frame_output),
                array_output.name: _sha256_file(array_output),
            },
        }
        manifest_path = stage / "final_manifest.json"
        with manifest_path.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        rename_noreplace(stage, output_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    with (output_dir / "final_manifest.json").open(encoding="utf-8") as stream:
        published = json.load(stream)
    if published != manifest:
        raise RuntimeError("published final manifest read-back mismatch")
    return published


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking", type=Path, required=True)
    parser.add_argument("--detector-frames", type=Path, required=True)
    parser.add_argument("--refining-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = combine_stage_a(
        args.tracking,
        args.detector_frames,
        args.refining_dir,
        args.output_dir,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
