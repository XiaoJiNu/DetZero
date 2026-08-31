#!/usr/bin/env python3
"""Shared contracts for the Waymo Open3D-ML stage-A pipeline."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import shutil
import uuid

import numpy as np


_SEQUENCE_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_RAW_CLASSES = ("VEHICLE", "PEDESTRIAN", "CYCLIST")
_RAW_SCHEMA = "open3dml-waymo-raw-v1"


def rename_noreplace(source: str | Path, target: str | Path) -> None:
    """Atomically publish one directory without replacing an existing target."""
    source = Path(source)
    target = Path(target)
    renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), target)
    parent_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def generation_relative_provenance(
    path: str | Path, manifest_dir: str | Path
) -> str:
    """Return one manifest-relative path contained by its generation root."""
    manifest_dir = Path(manifest_dir).absolute()
    generation_root = manifest_dir.parent.resolve(strict=True)
    target = Path(path).resolve(strict=True)
    try:
        target.relative_to(generation_root)
    except ValueError as error:
        raise ValueError(f"provenance path escapes generation root: {path}") from error
    return Path(os.path.relpath(target, manifest_dir)).as_posix()


def sequence_name_from_tfrecord(path: str | Path) -> str:
    """Return the exact DetZero sequence identity for one Waymo TFRecord."""
    name = Path(path).name
    suffix = "_with_camera_labels.tfrecord"
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    elif name.endswith(".tfrecord"):
        name = name[: -len(".tfrecord")]
    else:
        raise ValueError(f"not a TFRecord path: {path}")
    if name.startswith("segment-"):
        name = name[len("segment-") :]
    if not name or not _SEQUENCE_RE.fullmatch(name):
        raise ValueError(f"invalid sequence name: {name!r}")
    return name


def _validated_pose(value: object) -> np.ndarray:
    pose = np.asarray(value)
    if pose.shape != (4, 4) or pose.dtype != np.float64 or not np.isfinite(pose).all():
        raise ValueError("pose must be a finite float64 (4, 4) array")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], rtol=0, atol=1e-9):
        raise ValueError("pose must be homogeneous")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-5):
        raise ValueError("pose rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, rtol=0, atol=1e-5):
        raise ValueError("pose rotation determinant must be one")
    return pose


def _validated_points(value: object) -> np.ndarray:
    points = np.asarray(value)
    if points.ndim != 2 or points.shape[1] != 6 or points.dtype != np.float32:
        raise ValueError("points must be a float32 (N, 6) array")
    if len(points) == 0 or not np.isfinite(points).all():
        raise ValueError("points must be non-empty and finite")
    return points


def records_from_waymo_frames(frames, point_extractor):
    """Lazily adapt decoded Waymo frames to the preprocessing writer contract."""
    for frame_id, frame in enumerate(frames):
        timestamp = frame.timestamp_micros
        if (
            isinstance(timestamp, (bool, np.bool_))
            or not isinstance(timestamp, (int, np.integer))
            or timestamp < 0
            or timestamp > np.iinfo(np.int64).max
        ):
            raise ValueError("frame timestamp_micros must fit a non-negative int64")
        points, lidar_counts = point_extractor(frame)
        yield {
            "frame_id": frame_id,
            "timestamp": np.int64(timestamp),
            "pose": np.asarray(frame.pose.transform, dtype=np.float64).reshape(4, 4),
            "points": points,
            "num_points_of_each_lidar": lidar_counts,
        }


def parse_waymo_frame(serialized, frame_factory):
    """Parse one eager TFRecord tensor with protobuf's immutable-byte contract."""
    payload = serialized.numpy()
    if not isinstance(payload, (bytes, bytearray, memoryview, np.bytes_)):
        raise TypeError("serialized TFRecord payload must be bytes-like")
    frame = frame_factory()
    frame.ParseFromString(bytes(payload))
    return frame


def merge_waymo_lidar_returns(point_returns, nlz_returns):
    """Merge two official Waymo polar/cartesian returns into DetZero point rows."""
    if len(point_returns) != 2 or len(nlz_returns) != 2:
        raise ValueError("Waymo preprocessing requires exactly two returns")
    if any(len(values) != 5 for values in (*point_returns, *nlz_returns)):
        raise ValueError("Waymo preprocessing requires exactly five lidars")

    lidar_points = []
    counts = []
    for lidar_id in range(5):
        returns = []
        for return_id in range(2):
            polar_xyz = np.asarray(point_returns[return_id][lidar_id])
            nlz = np.asarray(nlz_returns[return_id][lidar_id])
            if (
                polar_xyz.ndim != 2
                or polar_xyz.shape[1] != 6
                or polar_xyz.dtype != np.float32
                or nlz.shape != (len(polar_xyz),)
                or nlz.dtype != np.float32
                or not np.isfinite(polar_xyz).all()
                or not np.isfinite(nlz).all()
            ):
                raise ValueError("invalid Waymo return array contract")
            returns.append(
                np.concatenate(
                    [polar_xyz[:, 3:6], polar_xyz[:, 1:3], nlz[:, None]], axis=1
                )
            )
        merged = np.concatenate(returns, axis=0).astype(np.float32, copy=False)
        lidar_points.append(merged)
        counts.append(len(merged))
    points = np.concatenate(lidar_points, axis=0)
    if len(points) == 0:
        raise ValueError("decoded frame has no lidar points")
    return points, counts


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish_preprocessed_records(
    records,
    root_path: str | Path,
    *,
    sequence_name: str,
    expected_frame_count: int,
    manifest_fields: dict[str, object] | None = None,
) -> dict[str, object]:
    """Stream validated records into a fresh DetZero Waymo data root."""
    if not _SEQUENCE_RE.fullmatch(sequence_name):
        raise ValueError(f"invalid sequence name: {sequence_name!r}")
    if isinstance(expected_frame_count, bool) or expected_frame_count <= 0:
        raise ValueError("expected_frame_count must be a positive integer")

    root_path = Path(root_path)
    if root_path.exists() or root_path.is_symlink():
        raise FileExistsError(root_path)
    root_path.parent.mkdir(parents=True, exist_ok=True)
    stage = root_path.with_name(f".{root_path.name}.tmp-{uuid.uuid4().hex}")
    sequence_dir = stage / "waymo_processed_data" / f"segment-{sequence_name}"
    sequence_dir.mkdir(parents=True)

    infos = []
    frame_ids = []
    previous_timestamp = None
    try:
        for expected_frame_id, record in enumerate(records):
            if not isinstance(record, dict):
                raise TypeError("each preprocessed record must be a dict")
            frame_id = record.get("frame_id")
            timestamp = record.get("timestamp")
            if isinstance(frame_id, bool) or frame_id != expected_frame_id:
                raise ValueError(
                    f"frame IDs must be contiguous from zero: expected {expected_frame_id}, got {frame_id}"
                )
            if isinstance(timestamp, (bool, np.bool_)) or not isinstance(
                timestamp, (int, np.integer)
            ):
                raise ValueError("timestamp must be an integer")
            timestamp = int(timestamp)
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                raise ValueError("timestamps must be strictly increasing")

            pose = _validated_pose(record.get("pose"))
            points = _validated_points(record.get("points"))
            lidar_counts = record.get("num_points_of_each_lidar")
            if (
                not isinstance(lidar_counts, (list, tuple))
                or len(lidar_counts) != 5
                or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 0 for v in lidar_counts)
                or sum(int(v) for v in lidar_counts) != len(points)
            ):
                raise ValueError("num_points_of_each_lidar must contain five counts summing to N")

            lidar_name = f"{frame_id:04d}.npy"
            np.save(sequence_dir / lidar_name, points, allow_pickle=False)
            infos.append(
                {
                    "time_stamp": timestamp,
                    "sample_idx": frame_id,
                    "sequence_name": sequence_name,
                    "pose": pose.copy(),
                    "num_points_of_each_lidar": [int(v) for v in lidar_counts],
                    "lidar_path": str(
                        Path("waymo_processed_data")
                        / f"segment-{sequence_name}"
                        / lidar_name
                    ),
                }
            )
            frame_ids.append(frame_id)
            previous_timestamp = timestamp

        if len(infos) != expected_frame_count:
            raise ValueError(
                f"frame count mismatch: expected {expected_frame_count}, got {len(infos)}"
            )

        with (sequence_dir / f"{sequence_name}.pkl").open("xb") as stream:
            pickle.dump(infos, stream, protocol=pickle.HIGHEST_PROTOCOL)
        image_sets = stage / "ImageSets"
        image_sets.mkdir()
        (image_sets / "test.txt").write_text(f"{sequence_name}\n", encoding="utf-8")
        manifest = {
            "sequence_name": sequence_name,
            "frame_count": len(infos),
            "frame_ids": frame_ids,
            "timestamp_first": infos[0]["time_stamp"],
            "timestamp_last": infos[-1]["time_stamp"],
        }
        manifest_fields = manifest_fields or {}
        overlap = set(manifest).intersection(manifest_fields)
        if overlap:
            raise ValueError(f"reserved manifest fields: {sorted(overlap)}")
        manifest.update(manifest_fields)
        (stage / "preprocess_manifest.json").write_text(
            json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        rename_noreplace(stage, root_path)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    return manifest


def preprocess_tfrecord(
    input_path: str | Path,
    root_path: str | Path,
    *,
    expected_frame_count: int,
    frame_loader,
    point_extractor,
    non_commercial_research: bool,
    waymo_terms_accepted: bool,
    source_identity_path: str | Path | None = None,
) -> dict[str, object]:
    """Decode and atomically publish one authorized, source-bound TFRecord."""
    if non_commercial_research is not True or waymo_terms_accepted is not True:
        raise PermissionError(
            "stage A requires non-commercial research use and accepted Waymo terms"
        )
    input_path = Path(input_path)
    if input_path.is_symlink() or not input_path.is_file():
        raise ValueError(f"input must be a regular non-symlink file: {input_path}")
    input_path = input_path.resolve(strict=True)
    source_identity_path = Path(source_identity_path or input_path).absolute()
    source_hash = _sha256_file(input_path)

    records = records_from_waymo_frames(frame_loader(input_path), point_extractor)

    def source_verified_records():
        yield from records
        if _sha256_file(input_path) != source_hash:
            raise RuntimeError("input TFRecord changed while it was being decoded")

    return publish_preprocessed_records(
        source_verified_records(),
        root_path,
        sequence_name=sequence_name_from_tfrecord(source_identity_path),
        expected_frame_count=expected_frame_count,
        manifest_fields={
            "input_tfrecord": str(source_identity_path),
            "input_tfrecord_bytes": input_path.stat().st_size,
            "input_tfrecord_sha256": source_hash,
            "usage": "non-commercial-research",
            "waymo_terms_accepted": True,
        },
    )


def save_raw_predictions(path: str | Path, sequence_name: str, frames) -> None:
    """Write Open3D-ML predictions as a bounded, object-free NPZ."""
    if not _SEQUENCE_RE.fullmatch(sequence_name):
        raise ValueError(f"invalid sequence name: {sequence_name!r}")
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    frame_ids = []
    point_counts = []
    model_point_counts = []
    offsets = [0]
    centers = []
    sizes = []
    yaws = []
    scores = []
    labels = []
    class_to_id = {name: index for index, name in enumerate(_RAW_CLASSES)}
    for expected_frame_id, frame in enumerate(frames):
        frame_id = frame.get("frame_id")
        point_count = frame.get("point_count")
        model_point_count = frame.get("model_point_count")
        if frame_id != expected_frame_id or isinstance(frame_id, bool):
            raise ValueError("raw prediction frame IDs must be contiguous from zero")
        if (
            isinstance(point_count, bool)
            or isinstance(model_point_count, bool)
            or not isinstance(point_count, (int, np.integer))
            or not isinstance(model_point_count, (int, np.integer))
            or not 0 <= model_point_count <= point_count
        ):
            raise ValueError("invalid raw/model point counts")
        frame_ids.append(frame_id)
        point_counts.append(point_count)
        model_point_counts.append(model_point_count)
        for box in frame.get("boxes", []):
            center = np.asarray(box.get("center"))
            size = np.asarray(box.get("size"))
            yaw = box.get("yaw")
            score = box.get("score")
            label = box.get("label")
            if (
                center.shape != (3,)
                or size.shape != (3,)
                or center.dtype != np.float32
                or size.dtype != np.float32
                or not np.isfinite(center).all()
                or not np.isfinite(size).all()
                or np.any(size <= 0)
                or not np.isscalar(yaw)
                or not np.isscalar(score)
                or isinstance(yaw, (bool, np.bool_))
                or isinstance(score, (bool, np.bool_))
                or not np.isfinite(yaw)
                or not np.isfinite(score)
                or not 0 <= float(score) <= 1
                or label not in class_to_id
            ):
                raise ValueError("invalid raw Open3D-ML box")
            centers.append(center)
            sizes.append(size)
            yaws.append(yaw)
            scores.append(score)
            labels.append(class_to_id[label])
        offsets.append(len(centers))

    if not frame_ids:
        raise ValueError("raw predictions must contain at least one frame")
    arrays = {
        "schema_version": np.asarray(_RAW_SCHEMA),
        "sequence_name": np.asarray(sequence_name),
        "class_names": np.asarray(_RAW_CLASSES),
        "frame_ids": np.asarray(frame_ids, dtype=np.int64),
        "frame_offsets": np.asarray(offsets, dtype=np.int64),
        "point_counts": np.asarray(point_counts, dtype=np.int64),
        "model_point_counts": np.asarray(model_point_counts, dtype=np.int64),
        "centers": np.asarray(centers, dtype=np.float32).reshape(-1, 3),
        "sizes": np.asarray(sizes, dtype=np.float32).reshape(-1, 3),
        "yaws": np.asarray(yaws, dtype=np.float32),
        "scores": np.asarray(scores, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int8),
    }
    stage = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with stage.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        rename_noreplace(stage, path)
    except BaseException:
        stage.unlink(missing_ok=True)
        raise


def load_raw_predictions(path: str | Path) -> dict[str, object]:
    """Strictly load the object-free Open3D-ML prediction boundary."""
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024**3:
        raise ValueError(f"invalid raw prediction file: {path}")
    required = {
        "schema_version",
        "sequence_name",
        "class_names",
        "frame_ids",
        "frame_offsets",
        "point_counts",
        "model_point_counts",
        "centers",
        "sizes",
        "yaws",
        "scores",
        "labels",
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise ValueError("raw prediction NPZ field set mismatch")
        values = {name: archive[name].copy() for name in required}

    if values["schema_version"].shape != () or str(values["schema_version"].item()) != _RAW_SCHEMA:
        raise ValueError("raw prediction schema version mismatch")
    sequence_name = str(values["sequence_name"].item())
    if values["sequence_name"].shape != () or not _SEQUENCE_RE.fullmatch(sequence_name):
        raise ValueError("invalid raw prediction sequence name")
    if values["class_names"].tolist() != list(_RAW_CLASSES):
        raise ValueError("raw prediction class order mismatch")

    frame_ids = values["frame_ids"]
    offsets = values["frame_offsets"]
    point_counts = values["point_counts"]
    model_point_counts = values["model_point_counts"]
    centers = values["centers"]
    sizes = values["sizes"]
    yaws = values["yaws"]
    scores = values["scores"]
    labels = values["labels"]
    frame_count = len(frame_ids)
    box_count = len(centers)
    if frame_count == 0 or frame_count > 10000 or box_count > 1000000:
        raise ValueError("raw prediction counts are outside bounds")
    if (
        frame_ids.dtype != np.int64
        or offsets.dtype != np.int64
        or point_counts.dtype != np.int64
        or model_point_counts.dtype != np.int64
        or centers.dtype != np.float32
        or sizes.dtype != np.float32
        or yaws.dtype != np.float32
        or scores.dtype != np.float32
        or labels.dtype != np.int8
        or frame_ids.shape != (frame_count,)
        or offsets.shape != (frame_count + 1,)
        or point_counts.shape != (frame_count,)
        or model_point_counts.shape != (frame_count,)
        or centers.shape != (box_count, 3)
        or sizes.shape != (box_count, 3)
        or yaws.shape != (box_count,)
        or scores.shape != (box_count,)
        or labels.shape != (box_count,)
    ):
        raise ValueError("raw prediction dtype or shape mismatch")
    if (
        not np.array_equal(frame_ids, np.arange(frame_count, dtype=np.int64))
        or offsets[0] != 0
        or offsets[-1] != box_count
        or np.any(np.diff(offsets) < 0)
        or np.any(point_counts < model_point_counts)
        or np.any(model_point_counts < 0)
        or not np.isfinite(centers).all()
        or not np.isfinite(sizes).all()
        or not np.isfinite(yaws).all()
        or not np.isfinite(scores).all()
        or np.any(sizes <= 0)
        or np.any((scores < 0) | (scores > 1))
        or np.any((labels < 0) | (labels >= len(_RAW_CLASSES)))
    ):
        raise ValueError("raw prediction semantic validation failed")

    frames = []
    for frame_id in range(frame_count):
        start, end = int(offsets[frame_id]), int(offsets[frame_id + 1])
        boxes = [
            {
                "center": centers[index].copy(),
                "size": sizes[index].copy(),
                "yaw": float(yaws[index]),
                "label": _RAW_CLASSES[int(labels[index])],
                "score": float(scores[index]),
            }
            for index in range(start, end)
        ]
        frames.append(
            {
                "frame_id": frame_id,
                "point_count": int(point_counts[frame_id]),
                "model_point_count": int(model_point_counts[frame_id]),
                "boxes": boxes,
            }
        )
    return {"sequence_name": sequence_name, "frames": frames}


def adapt_raw_predictions(
    raw_path: str | Path,
    infos: list[dict],
    *,
    expected_frame_count: int | None = None,
) -> list[dict]:
    """Convert validated Open3D-ML boxes and Waymo poses to DetZero frames."""
    raw = load_raw_predictions(raw_path)
    sequence_name = raw["sequence_name"]
    raw_frames = raw["frames"]
    if expected_frame_count is None:
        expected_frame_count = len(infos)
    if (
        isinstance(expected_frame_count, bool)
        or not isinstance(expected_frame_count, int)
        or expected_frame_count <= 0
        or len(raw_frames) != expected_frame_count
        or len(infos) < expected_frame_count
    ):
        raise ValueError("raw prediction and Waymo info frame counts differ")
    infos = infos[:expected_frame_count]

    output = []
    previous_timestamp = None
    class_map = {
        "VEHICLE": "Vehicle",
        "PEDESTRIAN": "Pedestrian",
        "CYCLIST": "Cyclist",
    }
    for frame_id, (raw_frame, info) in enumerate(zip(raw_frames, infos)):
        sample_idx = info.get("sample_idx")
        timestamp = info.get("time_stamp")
        if (
            isinstance(timestamp, (bool, np.bool_))
            or not isinstance(timestamp, (int, np.integer))
            or timestamp < 0
            or timestamp > np.iinfo(np.int64).max
        ):
            raise ValueError(f"invalid Waymo timestamp at frame {frame_id}")
        timestamp = int(timestamp)
        pose = _validated_pose(info.get("pose"))
        if (
            raw_frame["frame_id"] != frame_id
            or info.get("sequence_name") != sequence_name
            or isinstance(sample_idx, (bool, np.bool_))
            or not isinstance(sample_idx, (int, np.integer))
            or int(sample_idx) != frame_id
            or (previous_timestamp is not None and timestamp <= previous_timestamp)
        ):
            raise ValueError(f"invalid Waymo frame info at frame {frame_id}")
        previous_timestamp = timestamp

        boxes = raw_frame["boxes"]
        boxes_lidar = np.zeros((len(boxes), 9), dtype=np.float32)
        names = []
        scores = np.zeros(len(boxes), dtype=np.float32)
        for box_id, box in enumerate(boxes):
            boxes_lidar[box_id] = open3dml_box_to_detzero(
                box["center"], box["size"], box["yaw"]
            )
            names.append(class_map[box["label"]])
            scores[box_id] = box["score"]
        output.append(
            {
                "sequence_name": sequence_name,
                "sample_idx": frame_id,
                "frame_id": frame_id,
                "timestamp": timestamp,
                "pose": pose.copy(),
                "name": np.asarray(names, dtype="<U10"),
                "score": scores,
                "boxes_lidar": boxes_lidar,
            }
        )
    return output


def _wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def open3dml_box_to_detzero(center, size, yaw: float) -> np.ndarray:
    """Convert one Open3D BEVBox3D geometry to DetZero Waymo [xyz,lwh,yaw,vx,vy]."""
    center = np.asarray(center)
    size = np.asarray(size)
    if center.shape != (3,) or size.shape != (3,):
        raise ValueError("center and size must each have shape (3,)")
    if not np.isfinite(center).all() or not np.isfinite(size).all() or not np.isfinite(yaw):
        raise ValueError("box values must be finite")
    if np.any(size <= 0):
        raise ValueError("box sizes must be strictly positive")

    heading = _wrap_to_pi(-float(yaw) - np.pi / 2.0)
    return np.asarray(
        [
            center[0],
            center[1],
            center[2],
            size[2],
            size[0],
            size[1],
            heading,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )
