#!/usr/bin/env python3
"""Independently validate a DetZero Waymo Stage A generation."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Any
import uuid
import zipfile

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.external_detector.safe_io import (
    open_bounded_regular as _open_bounded_regular,
    safe_load_pickle,
)
from tools.external_detector.open3dml_provenance import require_locked_open3dml_source


RAW_CLASSES = ("VEHICLE", "PEDESTRIAN", "CYCLIST")
CLASSES = ("Vehicle", "Pedestrian", "Cyclist")
BUNDLE_REPLAY_VERIFIER = (
    Path.home()
    / ".hermes/skills/software-development/artifact-pipeline-verification/scripts/verify-python-bundle-replay.py"
)
RAW_TO_DETZERO = dict(zip(RAW_CLASSES, CLASSES))
SOURCE_PREFIXES = (
    ".gitignore",
    "LICENSE",
    "daemon/prepare_object_data.py",
    "detection/detzero_det/__init__.py",
    "docs/handoff/detzero-waymo-stage-a-handoff-",
    "pytest.ini",
    "refining/",
    "requirements.txt",
    "tests/test_waymo_external_detector.py",
    "tools/external_detector/",
    "tracking/",
    "utils/",
)
SOURCE_LITERAL_FILES = {".gitignore", "LICENSE"}
SOURCE_SUFFIXES = {".cfg", ".ini", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}


def load_json_strict(
    path: str | Path,
    *,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> Any:
    """Load bounded JSON while rejecting duplicate keys and non-finite numbers."""
    descriptor = _open_bounded_regular(path, 16 * 1024 * 1024)

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    def finite_float(text: str) -> float:
        value = float(text)
        if not math.isfinite(value):
            raise ValueError(f"non-finite JSON number: {text}")
        return value

    def invalid_constant(text: str) -> None:
        raise ValueError(f"invalid JSON constant: {text}")

    with os.fdopen(descriptor, "rb") as stream:
        payload = stream.read()
    if (expected_bytes is None) != (expected_sha256 is None):
        raise ValueError("JSON identity requires both bytes and SHA-256")
    if expected_bytes is not None and (
        len(payload) != expected_bytes
        or hashlib.sha256(payload).hexdigest() != expected_sha256
    ):
        raise ValueError(f"JSON identity mismatch: {path}")
    return json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=object_pairs,
        parse_float=finite_float,
        parse_constant=invalid_constant,
    )


def _regular_file(
    path: str | Path, maximum_bytes: int, *, allow_empty: bool = False
) -> Path:
    path = Path(path)
    if (
        path.is_symlink()
        or not path.is_file()
        or (not allow_empty and path.stat().st_size == 0)
        or path.stat().st_size > maximum_bytes
    ):
        raise ValueError(f"invalid bounded regular file: {path}")
    return path.resolve(strict=True)


def resolve_generation_provenance(
    manifest_dir: str | Path, relative: Any
) -> Path:
    """Resolve one internal provenance path without accepting absolute/escape paths."""
    if not isinstance(relative, str):
        raise ValueError("generation provenance path must be relative")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or pure.as_posix() != relative or relative in {"", "."}:
        raise ValueError("generation provenance path must be relative POSIX syntax")
    manifest_dir = Path(manifest_dir).resolve(strict=True)
    run_root = manifest_dir.parent
    target = (manifest_dir / Path(*pure.parts)).resolve(strict=True)
    try:
        target.relative_to(run_root)
    except ValueError as error:
        raise ValueError("generation provenance path escapes run root") from error
    return target


def _load_pickle(path: str | Path) -> Any:
    return safe_load_pickle(path)


def _validate_pose(value: Any) -> np.ndarray:
    pose = np.asarray(value)
    if pose.shape != (4, 4) or pose.dtype != np.float64 or not np.isfinite(pose).all():
        raise ValueError("pose must be a finite float64 (4, 4) array")
    rotation = pose[:3, :3]
    if (
        not np.allclose(pose[3], [0, 0, 0, 1], rtol=0, atol=1e-9)
        or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1, rtol=0, atol=1e-5)
    ):
        raise ValueError("pose must be rigid")
    return pose


def _load_raw_predictions(path: str | Path, expected_frames: int) -> dict[str, np.ndarray]:
    path = _regular_file(path, 1024**3)
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
    descriptor = _open_bounded_regular(path, 1024**3)
    with os.fdopen(descriptor, "rb") as stream:
        archive_bytes = stream.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        if (
            len(members) != len(required)
            or len(set(names)) != len(names)
            or set(names) != {f"{name}.npy" for name in required}
            or sum(member.file_size for member in members) > 2 * 1024**3
        ):
            raise ValueError("raw prediction NPZ member set or size mismatch")
    with np.load(io.BytesIO(archive_bytes), allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise ValueError("raw prediction NPZ field set mismatch")
        values = {name: archive[name].copy() for name in required}

    if (
        values["schema_version"].shape != ()
        or str(values["schema_version"].item()) != "open3dml-waymo-raw-v1"
        or values["sequence_name"].shape != ()
        or not str(values["sequence_name"].item())
        or values["class_names"].tolist() != list(RAW_CLASSES)
    ):
        raise ValueError("raw prediction identity mismatch")
    frame_ids = values["frame_ids"]
    offsets = values["frame_offsets"]
    point_counts = values["point_counts"]
    model_point_counts = values["model_point_counts"]
    centers = values["centers"]
    sizes = values["sizes"]
    yaws = values["yaws"]
    scores = values["scores"]
    labels = values["labels"]
    box_count = len(centers)
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
        or frame_ids.shape != (expected_frames,)
        or offsets.shape != (expected_frames + 1,)
        or point_counts.shape != (expected_frames,)
        or model_point_counts.shape != (expected_frames,)
        or centers.shape != (box_count, 3)
        or sizes.shape != (box_count, 3)
        or yaws.shape != scores.shape != labels.shape
        or yaws.shape != (box_count,)
    ):
        raise ValueError("raw prediction dtype or shape mismatch")
    if (
        not np.array_equal(frame_ids, np.arange(expected_frames, dtype=np.int64))
        or offsets[0] != 0
        or offsets[-1] != box_count
        or np.any(np.diff(offsets) < 0)
        or np.any(point_counts <= 0)
        or np.any(model_point_counts <= 0)
        or np.any(model_point_counts > point_counts)
        or not all(np.isfinite(value).all() for value in (centers, sizes, yaws, scores))
        or np.any(sizes <= 0)
        or np.any((scores < 0) | (scores > 1))
        or np.any((labels < 0) | (labels >= len(RAW_CLASSES)))
    ):
        raise ValueError("raw prediction semantic mismatch")
    return values


def validate_detector_adapter(
    raw_predictions_path: str | Path,
    adapter_frames_path: str | Path,
    *,
    expected_frames: int,
) -> dict[str, Any]:
    """Independently replay the object-free detector-to-adapter boundary."""
    if type(expected_frames) is not int or expected_frames <= 0:
        raise ValueError("expected_frames must be a positive integer")
    raw = _load_raw_predictions(raw_predictions_path, expected_frames)
    frames = _load_pickle(adapter_frames_path)
    if not isinstance(frames, list) or len(frames) != expected_frames:
        raise ValueError("adapter frame count mismatch")

    sequence_name = str(raw["sequence_name"].item())
    offsets = raw["frame_offsets"]
    class_counts = {name: 0 for name in CLASSES}
    previous_timestamp = None
    for frame_id, frame in enumerate(frames):
        if not isinstance(frame, dict) or set(frame) != {
            "sequence_name",
            "sample_idx",
            "frame_id",
            "timestamp",
            "pose",
            "name",
            "score",
            "boxes_lidar",
        }:
            raise ValueError("adapter frame field set mismatch")
        timestamp = frame["timestamp"]
        if (
            frame["sequence_name"] != sequence_name
            or isinstance(frame["frame_id"], (bool, np.bool_))
            or int(frame["frame_id"]) != frame_id
            or isinstance(frame["sample_idx"], (bool, np.bool_))
            or int(frame["sample_idx"]) != frame_id
            or isinstance(timestamp, (bool, np.bool_))
            or not isinstance(timestamp, (int, np.integer))
            or int(timestamp) < 0
            or (previous_timestamp is not None and int(timestamp) <= previous_timestamp)
        ):
            raise ValueError("adapter frame identity mismatch")
        previous_timestamp = int(timestamp)
        _validate_pose(frame["pose"])

        start, end = int(offsets[frame_id]), int(offsets[frame_id + 1])
        count = end - start
        boxes = np.asarray(frame["boxes_lidar"])
        names = np.asarray(frame["name"])
        scores = np.asarray(frame["score"])
        if (
            boxes.shape != (count, 9)
            or boxes.dtype != np.float32
            or names.shape != (count,)
            or names.dtype.kind not in "US"
            or scores.shape != (count,)
            or scores.dtype != np.float32
            or not np.isfinite(boxes).all()
            or not np.isfinite(scores).all()
            or np.any(boxes[:, 3:6] <= 0)
            or np.any((scores < 0) | (scores > 1))
        ):
            raise ValueError("adapter box schema mismatch")

        raw_labels = raw["labels"][start:end]
        expected_names = np.asarray(
            [RAW_TO_DETZERO[RAW_CLASSES[int(label)]] for label in raw_labels],
            dtype="<U10",
        )
        expected_boxes = np.zeros((count, 9), dtype=np.float32)
        expected_boxes[:, :3] = raw["centers"][start:end]
        expected_boxes[:, 3:6] = raw["sizes"][start:end][:, [2, 0, 1]]
        expected_boxes[:, 6] = (
            (-raw["yaws"][start:end].astype(np.float64) - np.pi / 2 + np.pi)
            % (2 * np.pi)
            - np.pi
        ).astype(np.float32)
        if (
            not np.array_equal(names, expected_names)
            or not np.array_equal(scores, raw["scores"][start:end])
            or not np.allclose(boxes, expected_boxes, rtol=0, atol=1e-6)
        ):
            raise ValueError("detector-to-adapter geometry mismatch")
        for name in names.tolist():
            class_counts[str(name)] += 1

    return {
        "sequence_name": sequence_name,
        "frame_count": expected_frames,
        "box_count": int(len(raw["centers"])),
        "class_counts": class_counts,
    }


def _sha256_file(path: str | Path, *, allow_empty: bool = False) -> str:
    descriptor = _open_bounded_regular(path, 8 * 1024**3, allow_empty=allow_empty)
    digest = hashlib.sha256()
    with os.fdopen(descriptor, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_tree_hash(rows: dict[str, dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for relative, row in sorted(rows.items()):
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(row["sha256"].encode("ascii") + b"\n")
    return digest.hexdigest()


def _physical_source_tree(root: str | Path) -> dict[str, dict[str, Any]]:
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"invalid Open3D-ML source root: {root}")
    rows = {}
    for node in root.rglob("*"):
        if node.is_symlink() or (not node.is_dir() and not node.is_file()):
            raise ValueError(f"invalid Open3D-ML source node: {node}")
        if not node.is_file() or node.suffix not in SOURCE_SUFFIXES:
            continue
        descriptor = _open_bounded_regular(node, 1024**3, allow_empty=True)
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        rows[node.relative_to(root).as_posix()] = {
            "bytes": size,
            "sha256": digest.hexdigest(),
        }
    if not rows:
        raise ValueError("Open3D-ML source tree is empty")
    return rows


def validate_open3dml_provenance(
    run_root: str | Path, detector_dir: str | Path
) -> dict[str, Any]:
    """Bind the declared Open3D-ML commit to the installed source bytes."""
    metadata = load_json_strict(Path(run_root) / "run_metadata.json")
    detector = load_json_strict(Path(detector_dir) / "detector_manifest.json")
    preflight = metadata.get("preflight") if isinstance(metadata, dict) else None
    identity = preflight.get("open3dml") if isinstance(preflight, dict) else None
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema_version") != "detzero-stage-a-run-v1"
        or not isinstance(identity, dict)
        or set(identity)
        != {
            "root",
            "declared_commit",
            "source_file_count",
            "source_tree_sha256",
            "source_lock",
            "source_files",
        }
        or not isinstance(identity["root"], str)
        or not re.fullmatch(r"[0-9a-f]{40}", identity["declared_commit"])
        or type(identity["source_file_count"]) is not int
        or identity["source_file_count"] <= 0
        or not re.fullmatch(r"[0-9a-f]{64}", identity["source_tree_sha256"])
        or not isinstance(identity["source_files"], dict)
        or not isinstance(detector, dict)
        or detector.get("open3dml_commit") != identity["declared_commit"]
    ):
        raise ValueError("Open3D-ML provenance schema mismatch")
    source_lock = require_locked_open3dml_source(
        identity["declared_commit"],
        identity["source_file_count"],
        identity["source_tree_sha256"],
    )
    if identity["source_lock"] != source_lock:
        raise ValueError("Open3D-ML source lock mismatch")
    rows = _physical_source_tree(identity["root"])
    if (
        len(rows) != identity["source_file_count"]
        or _source_tree_hash(rows) != identity["source_tree_sha256"]
        or rows != identity["source_files"]
    ):
        raise ValueError("Open3D-ML source identity mismatch")
    return {
        "declared_commit": identity["declared_commit"],
        "source_file_count": len(rows),
        "source_tree_sha256": identity["source_tree_sha256"],
        "source_lock": source_lock,
    }


def _source_paths(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    paths = []
    for raw_path in result.stdout.split(b"\0"):
        if not raw_path:
            continue
        relative = raw_path.decode("utf-8")
        pure = PurePosixPath(relative)
        if (
            not pure.is_absolute()
            and ".." not in pure.parts
            and pure.as_posix() == relative
            and any(
                relative == prefix or relative.startswith(prefix)
                for prefix in SOURCE_PREFIXES
            )
            and (
                pure.suffix in SOURCE_SUFFIXES
                or relative in SOURCE_LITERAL_FILES
            )
        ):
            paths.append(relative)
    if len(paths) != len(set(paths)):
        raise ValueError("source path roster is duplicated")
    native_root = repo_root / "utils/detzero_utils/ops"
    if native_root.is_dir() and not native_root.is_symlink():
        for path in native_root.rglob("*.so"):
            if path.is_symlink() or not path.is_file():
                raise ValueError("native source must be a regular non-symlink file")
            paths.append(path.relative_to(repo_root).as_posix())
    paths = sorted(set(paths))
    if not paths:
        raise ValueError("source path roster is empty or duplicated")
    return paths


def validate_source_provenance(
    bundle: str | Path, repo_root: str | Path
) -> dict[str, Any]:
    """Independently verify embedded source bytes and live-tree currency."""
    bundle = Path(bundle)
    repo_root = Path(repo_root).resolve(strict=True)
    if bundle.is_symlink() or not bundle.is_dir():
        raise ValueError("invalid source bundle")
    manifest = load_json_strict(bundle / "source_manifest.json")
    rows = manifest.get("source_files") if isinstance(manifest, dict) else None
    if (
        not isinstance(rows, dict)
        or set(manifest)
        != {
            "schema_version",
            "origin_project_root",
            "source_file_count",
            "source_tree_sha256",
            "source_files",
        }
        or manifest["schema_version"] != "detzero-stage-a-source-v1"
        or not rows
    ):
        raise ValueError("source manifest schema mismatch")
    files_root = bundle / "files"
    actual_paths = _closed_regular_files(files_root)
    if actual_paths != set(rows):
        raise ValueError("source bundle path set mismatch")
    actual_rows = {}
    for relative, row in rows.items():
        pure = PurePosixPath(relative)
        if (
            not isinstance(relative, str)
            or pure.is_absolute()
            or ".." in pure.parts
            or pure.as_posix() != relative
            or not isinstance(row, dict)
            or set(row) != {"bytes", "sha256"}
            or type(row["bytes"]) is not int
            or row["bytes"] < 0
            or not isinstance(row["sha256"], str)
            or len(row["sha256"]) != 64
        ):
            raise ValueError("source manifest row mismatch")
        path = files_root / relative
        actual_rows[relative] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path, allow_empty=True),
        }
    if (
        actual_rows != rows
        or manifest["source_file_count"] != len(rows)
        or manifest["source_tree_sha256"] != _source_tree_hash(actual_rows)
    ):
        raise ValueError("embedded source bundle mismatch")
    live_paths = _source_paths(repo_root)
    live_rows = {
        relative: {
            "bytes": (repo_root / relative).stat().st_size,
            "sha256": _sha256_file(repo_root / relative, allow_empty=True),
        }
        for relative in live_paths
    }
    if live_rows != rows:
        raise ValueError("live source drift")
    return {
        "historical_provenance_verified": True,
        "matches_current_workspace": True,
        "source_file_count": len(rows),
        "source_tree_sha256": manifest["source_tree_sha256"],
    }


def validate_bundle_replay(
    bundle: str | Path,
    repo_root: str | Path,
    entries: list[tuple[str, Path]],
) -> dict[str, Any]:
    bundle = Path(bundle).resolve(strict=True)
    repo_root = Path(repo_root).resolve(strict=True)
    verifier = _regular_file(BUNDLE_REPLAY_VERIFIER, 1024 * 1024)
    verifier_sha256 = _sha256_file(verifier)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    replayed = []
    for entry, interpreter in entries:
        result = subprocess.run(
            [
                str(Path(interpreter).absolute()),
                "-B",
                str(verifier),
                "--bundle",
                str(bundle / "files"),
                "--entry",
                entry,
                "--python",
                str(Path(interpreter).absolute()),
                "--forbid-root",
                str(repo_root),
                "--",
                "--help",
            ],
            cwd=bundle.parent,
            env=environment,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise ValueError(f"bundle replay failed for {entry}: {result.stderr[-8192:]}")
        replayed.append(entry)
    if _sha256_file(verifier) != verifier_sha256:
        raise ValueError("bundle replay verifier drift")
    return {
        "bundle_replay_complete": True,
        "entries": replayed,
        "verifier_sha256": verifier_sha256,
    }


def validate_run_ledger(run_root: str | Path) -> dict[str, Any]:
    run_root = Path(run_root)
    if run_root.is_symlink() or not run_root.is_dir():
        raise ValueError("invalid run root")
    ledger = load_json_strict(run_root / "run_ledger.json")
    if (
        not isinstance(ledger, dict)
        or set(ledger)
        != {"schema_version", "source_tree_sha256", "file_count", "files"}
        or ledger["schema_version"] != "detzero-stage-a-run-ledger-v1"
        or not isinstance(ledger["source_tree_sha256"], str)
        or len(ledger["source_tree_sha256"]) != 64
        or type(ledger["file_count"]) is not int
        or not isinstance(ledger["files"], dict)
    ):
        raise ValueError("run ledger schema mismatch")
    expected = ledger["files"]
    actual_paths = _closed_regular_files(run_root) - {
        ".unaccepted",
        "run_ledger.json",
    }
    if actual_paths != set(expected):
        raise ValueError("run ledger path set mismatch")
    for relative, row in expected.items():
        pure = PurePosixPath(relative)
        if (
            not isinstance(relative, str)
            or pure.is_absolute()
            or ".." in pure.parts
            or pure.as_posix() != relative
            or not isinstance(row, dict)
            or set(row) != {"bytes", "sha256"}
            or type(row["bytes"]) is not int
            or row["bytes"] < 0
            or not isinstance(row["sha256"], str)
            or len(row["sha256"]) != 64
        ):
            raise ValueError("run ledger row mismatch")
        path = run_root / relative
        if path.stat().st_size != row["bytes"] or _sha256_file(
            path, allow_empty=True
        ) != row["sha256"]:
            raise ValueError(f"run ledger payload mismatch: {relative}")
    if ledger["file_count"] != len(expected):
        raise ValueError("run ledger count mismatch")
    return {
        "file_count": len(expected),
        "source_tree_sha256": ledger["source_tree_sha256"],
    }


def _frame_ids(values: Any) -> np.ndarray:
    result = []
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("frame IDs must be one-dimensional")
    for value in array.tolist():
        if isinstance(value, str):
            if not value.isdigit():
                raise ValueError("frame IDs must be non-negative integers")
            value = int(value)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError("frame IDs must be non-negative integers")
        result.append(int(value))
    return np.asarray(result, dtype=np.int64)


def _global_to_lidar(boxes_global: np.ndarray, pose: np.ndarray) -> np.ndarray:
    pose = _validate_pose(pose)
    boxes_global = np.asarray(boxes_global, dtype=np.float32)
    inverse = np.linalg.inv(pose)
    centers = np.column_stack((boxes_global[:, :3], np.ones(len(boxes_global))))
    output = boxes_global.copy()
    output[:, :3] = (centers @ inverse.T)[:, :3]
    output[:, 6] = (
        boxes_global[:, 6] + np.arctan2(inverse[1, 0], inverse[0, 0]) + np.pi
    ) % (2 * np.pi) - np.pi
    velocity = np.column_stack((boxes_global[:, 7:9], np.zeros(len(boxes_global))))
    output[:, 7:9] = (velocity @ inverse[:3, :3].T)[:, :2]
    return output


def _validate_track(
    track: Any,
    sequence_name: str,
    object_id: int,
    detector_poses: list[np.ndarray],
) -> dict[str, Any]:
    required = {
        "boxes_global",
        "name",
        "score",
        "sample_idx",
        "hit",
        "num_points",
        "obj_ids",
        "pose",
        "state",
    }
    if not isinstance(track, dict) or set(track) != required:
        raise ValueError("tracking field set mismatch")
    frame_ids = _frame_ids(track["sample_idx"])
    count = len(frame_ids)
    boxes = np.asarray(track["boxes_global"])
    names = np.asarray(track["name"])
    scores = np.asarray(track["score"])
    hits = np.asarray(track["hit"])
    point_counts = np.asarray(track["num_points"])
    object_ids = np.asarray(track["obj_ids"])
    poses = np.asarray(track["pose"])
    if (
        not isinstance(track["state"], str)
        or not track["state"]
        or count == 0
        or len(np.unique(frame_ids)) != count
        or np.any(frame_ids >= len(detector_poses))
        or boxes.shape != (count, 9)
        or boxes.dtype not in (np.dtype(np.float32), np.dtype(np.float64))
        or names.shape != (count,)
        or names.dtype.kind not in "US"
        or scores.shape != (count,)
        or scores.dtype != np.float32
        or hits.shape != (count,)
        or hits.dtype.kind not in "iu"
        or point_counts.shape != (count,)
        or point_counts.dtype.kind not in "fiu"
        or object_ids.shape != (count,)
        or object_ids.dtype.kind not in "iu"
        or poses.shape != (count, 4, 4)
        or poses.dtype != np.float64
        or not np.isfinite(boxes).all()
        or not np.isfinite(scores).all()
        or not np.isfinite(point_counts).all()
        or np.any(boxes[:, 3:6] <= 0)
        or np.any((scores < 0) | (scores > 1))
        or np.any(object_ids != object_id)
        or len(set(names.tolist())) != 1
        or str(names[0]) not in CLASSES
    ):
        raise ValueError("tracking schema or semantics mismatch")
    for index, frame_id in enumerate(frame_ids):
        pose = _validate_pose(poses[index])
        if not np.array_equal(pose, detector_poses[int(frame_id)]):
            raise ValueError("tracking pose/frame mismatch")
    return {
        "frame_ids": frame_ids,
        "boxes": boxes,
        "names": names,
        "scores": scores,
        "poses": poses,
        "class_name": str(names[0]),
    }


def _flatten_model_records(value: Any) -> dict[tuple[str, int], dict[str, Any]]:
    if not isinstance(value, dict):
        raise ValueError("refining output must be nested dictionaries")
    result = {}
    for sequence_name, records in value.items():
        if not isinstance(sequence_name, str) or not isinstance(records, dict):
            raise ValueError("invalid refining output identity")
        for object_id, record in records.items():
            if isinstance(object_id, (bool, np.bool_)) or not isinstance(object_id, (int, np.integer)):
                raise ValueError("invalid refining object ID")
            key = (sequence_name, int(object_id))
            if key in result or not isinstance(record, dict):
                raise ValueError("duplicate or invalid refining record")
            result[key] = record
    return result


def _validate_model_record(
    record: dict[str, Any],
    track: dict[str, Any],
    sequence_name: str,
    object_id: int,
) -> np.ndarray:
    if set(record) != {
        "sequence_name",
        "obj_id",
        "sample_idx",
        "boxes_global",
        "score",
        "name",
        "pose",
    }:
        raise ValueError("refining record field set mismatch")
    count = len(track["frame_ids"])
    boxes = np.asarray(record["boxes_global"])
    if (
        record["sequence_name"] != sequence_name
        or int(record["obj_id"]) != object_id
        or boxes.shape != (count, 7)
        or boxes.dtype != np.float32
        or not np.isfinite(boxes).all()
        or np.any(boxes[:, 3:6] <= 0)
        or not np.array_equal(_frame_ids(record["sample_idx"]), track["frame_ids"])
        or not np.array_equal(np.asarray(record["score"]), track["scores"])
        or not np.array_equal(np.asarray(record["name"]), track["names"])
        or not np.array_equal(np.asarray(record["pose"]), track["poses"])
    ):
        raise ValueError("refining output is not track-aligned")
    return boxes


def _load_final_npz(path: Path, expected_frames: int) -> dict[str, np.ndarray]:
    path = _regular_file(path, 2 * 1024**3)
    required = {
        "sequence_name",
        "class_names",
        "frame_ids",
        "frame_offsets",
        "poses",
        "boxes_lidar",
        "boxes_global",
        "scores",
        "labels",
        "object_ids",
    }
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        if (
            len(members) != len(required)
            or len(set(names)) != len(names)
            or set(names) != {f"{name}.npy" for name in required}
            or sum(member.file_size for member in members) > 4 * 1024**3
        ):
            raise ValueError("final NPZ member set or size mismatch")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise ValueError("final NPZ field set mismatch")
        values = {name: archive[name].copy() for name in required}
    count = len(values["boxes_global"])
    if (
        values["sequence_name"].shape != ()
        or values["class_names"].tolist() != list(CLASSES)
        or values["frame_ids"].dtype != np.int64
        or values["frame_ids"].shape != (expected_frames,)
        or not np.array_equal(values["frame_ids"], np.arange(expected_frames))
        or values["frame_offsets"].dtype != np.int64
        or values["frame_offsets"].shape != (expected_frames + 1,)
        or values["frame_offsets"][0] != 0
        or values["frame_offsets"][-1] != count
        or np.any(np.diff(values["frame_offsets"]) < 0)
        or values["poses"].shape != (expected_frames, 4, 4)
        or values["poses"].dtype != np.float64
        or values["boxes_lidar"].shape != (count, 9)
        or values["boxes_lidar"].dtype != np.float32
        or values["boxes_global"].shape != (count, 9)
        or values["boxes_global"].dtype != np.float32
        or values["scores"].shape != (count,)
        or values["scores"].dtype != np.float32
        or values["labels"].shape != (count,)
        or values["labels"].dtype != np.int8
        or values["object_ids"].shape != (count,)
        or values["object_ids"].dtype != np.int64
        or not all(
            np.isfinite(values[name]).all()
            for name in ("poses", "boxes_lidar", "boxes_global", "scores")
        )
        or np.any(values["boxes_global"][:, 3:6] <= 0)
        or np.any((values["scores"] < 0) | (values["scores"] > 1))
        or np.any((values["labels"] < 0) | (values["labels"] >= len(CLASSES)))
    ):
        raise ValueError("final NPZ schema or semantics mismatch")
    return values


def validate_tracking_refining_final(
    tracking_path: str | Path,
    detector_frames_path: str | Path,
    refining_dir: str | Path,
    final_dir: str | Path,
    *,
    expected_frames: int,
) -> dict[str, Any]:
    """Replay track/refiner alignment, no-CRM algebra, transforms, and final NPZ."""
    if type(expected_frames) is not int or expected_frames <= 0:
        raise ValueError("expected_frames must be a positive integer")
    refining_dir = Path(refining_dir).resolve(strict=True)
    final_dir = Path(final_dir).resolve(strict=True)
    if refining_dir.is_symlink() or final_dir.is_symlink():
        raise ValueError("artifact directories must not be symlinks")

    detector_frames = _load_pickle(detector_frames_path)
    if not isinstance(detector_frames, list) or len(detector_frames) != expected_frames:
        raise ValueError("detector frame roster mismatch")
    sequences = {frame.get("sequence_name") for frame in detector_frames if isinstance(frame, dict)}
    if len(sequences) != 1 or not isinstance(next(iter(sequences)), str):
        raise ValueError("detector sequence identity mismatch")
    sequence_name = next(iter(sequences))
    detector_poses = []
    for frame_id, frame in enumerate(detector_frames):
        if int(frame.get("frame_id", -1)) != frame_id:
            raise ValueError("detector frame IDs must be contiguous")
        detector_poses.append(_validate_pose(frame.get("pose")))

    tracking = _load_pickle(tracking_path)
    if not isinstance(tracking, dict) or set(tracking) != {sequence_name} or not isinstance(tracking[sequence_name], dict):
        raise ValueError("tracking sequence identity mismatch")
    tracks = {}
    class_track_counts = {name: 0 for name in CLASSES}
    for object_id, track in tracking[sequence_name].items():
        if isinstance(object_id, (bool, np.bool_)) or not isinstance(object_id, (int, np.integer)):
            raise ValueError("tracking object IDs must be integers")
        normalized_id = int(object_id)
        validated = _validate_track(track, sequence_name, normalized_id, detector_poses)
        tracks[(sequence_name, normalized_id)] = validated
        class_track_counts[validated["class_name"]] += 1

    refining_manifest = load_json_strict(refining_dir / "refining_manifest.json")
    if not isinstance(refining_manifest, dict) or set(refining_manifest) != {
        "schema_version",
        "tracking",
        "waymo_root",
        "classes",
        "model_forward_count",
        "crm",
        "score_policy",
        "outputs",
    }:
        raise ValueError("refining manifest field set mismatch")
    if (
        not isinstance(refining_manifest["tracking"], dict)
        or set(refining_manifest["tracking"]) != {"path", "sha256"}
    ):
        raise ValueError("refining manifest tracking provenance mismatch")
    tracking_manifest_path = resolve_generation_provenance(
        refining_dir, refining_manifest["tracking"]["path"]
    )
    waymo_manifest_root = resolve_generation_provenance(
        refining_dir, refining_manifest["waymo_root"]
    )
    if (
        refining_manifest["schema_version"] != "detzero-stage-a-refining-v1"
        or tracking_manifest_path != Path(tracking_path).resolve(strict=True)
        or refining_manifest["tracking"]["sha256"] != _sha256_file(tracking_path)
        or waymo_manifest_root.is_symlink()
        or not waymo_manifest_root.is_dir()
        or set(refining_manifest["classes"]) != set(CLASSES)
        or refining_manifest.get("crm") != {"status": "NOT_EXECUTED_NO_CRM_BY_DESIGN"}
        or refining_manifest.get("score_policy") != "TRACKING_SCORE_PASSTHROUGH"
    ):
        raise ValueError("refining manifest policy mismatch")

    model_boxes = {}
    expected_refining_files = {"refining_manifest.json"}
    expected_output_hashes = {}
    expected_forward_count = 0
    for class_name in CLASSES:
        class_tracks = {key: value for key, value in tracks.items() if value["class_name"] == class_name}
        box_count = sum(len(value["frame_ids"]) for value in class_tracks.values())
        status = refining_manifest["classes"][class_name]
        if (
            not isinstance(status, dict)
            or set(status) != {"input_track_count", "input_box_count", "geometry", "position"}
            or
            status.get("input_track_count") != len(class_tracks)
            or status.get("input_box_count") != box_count
        ):
            raise ValueError("refining input counts do not match tracking")
        geometry_path = refining_dir / "result" / f"{class_name}_geometry.pkl"
        position_path = refining_dir / "result" / f"{class_name}_position.pkl"
        if not class_tracks:
            expected = {"status": "NOT_EXECUTED_NO_INPUT", "forward_count": 0}
            if any(status.get(kind) != expected for kind in ("geometry", "position")):
                raise ValueError("empty refining class status mismatch")
            if geometry_path.exists() or position_path.exists():
                raise ValueError("empty refining class has model outputs")
            continue
        for kind, path in (("geometry", geometry_path), ("position", position_path)):
            model_status = status.get(kind, {})
            relative_output = f"result/{class_name}_{kind}.pkl"
            if (
                not isinstance(model_status, dict)
                or set(model_status)
                != {
                    "status",
                    "checkpoint",
                    "checkpoint_sha256",
                    "loaded_tensors",
                    "model_tensors",
                    "forward_count",
                    "output",
                    "output_sha256",
                }
                or model_status.get("status") != "EXECUTED"
                or model_status.get("forward_count") != len(class_tracks)
                or type(model_status.get("loaded_tensors")) is not int
                or model_status.get("loaded_tensors", 0) <= 0
                or model_status.get("loaded_tensors")
                != model_status.get("model_tensors")
                or model_status.get("checkpoint_sha256")
                != _sha256_file(
                    Path(model_status.get("checkpoint"))
                    if Path(model_status.get("checkpoint")).is_absolute()
                    else REPO_ROOT / model_status.get("checkpoint")
                )
                or model_status.get("output") != relative_output
                or model_status.get("output_sha256") != _sha256_file(path)
            ):
                raise ValueError("executed refining class status mismatch")
            expected_refining_files.add(relative_output)
            expected_output_hashes[relative_output] = model_status["output_sha256"]
            expected_forward_count += len(class_tracks)
            records = _flatten_model_records(_load_pickle(path))
            if set(records) != set(class_tracks):
                raise ValueError("refining/track key mismatch")
            for key, track in class_tracks.items():
                model_boxes[(key, kind)] = _validate_model_record(
                    records[key], track, key[0], key[1]
                )
    if (
        _closed_regular_files(refining_dir) != expected_refining_files
        or refining_manifest["outputs"] != expected_output_hashes
        or refining_manifest["model_forward_count"] != expected_forward_count
    ):
        raise ValueError("refining output is not closed-world")

    final_tracks_nested = _load_pickle(final_dir / "final_track.pkl")
    if not isinstance(final_tracks_nested, dict) or set(final_tracks_nested) != {sequence_name}:
        raise ValueError("final track sequence mismatch")
    final_tracks = {
        (sequence_name, int(object_id)): track
        for object_id, track in final_tracks_nested[sequence_name].items()
    }
    if set(final_tracks) != set(tracks):
        raise ValueError("final/tracking key mismatch")
    for key, track in tracks.items():
        final_track = final_tracks[key]
        validated_final = _validate_track(
            final_track, sequence_name, key[1], detector_poses
        )
        for field in ("name", "score", "sample_idx", "hit", "num_points", "obj_ids", "pose"):
            if not np.array_equal(np.asarray(final_track[field]), np.asarray(tracking[sequence_name][key[1]][field])):
                raise ValueError("final track metadata is not a tracking passthrough")
        if final_track["state"] != tracking[sequence_name][key[1]]["state"]:
            raise ValueError("final track state is not a tracking passthrough")
        expected_boxes = np.asarray(track["boxes"], dtype=np.float32).copy()
        expected_boxes[:, :3] = model_boxes[(key, "position")][:, :3]
        expected_boxes[:, 3:6] = model_boxes[(key, "geometry")][:, 3:6]
        expected_boxes[:, 6] = model_boxes[(key, "position")][:, 6]
        if not np.array_equal(validated_final["scores"], track["scores"]) or not np.array_equal(
            np.asarray(final_track["boxes_global"]), expected_boxes
        ):
            raise ValueError("no-CRM final algebra mismatch")

    final_frames = _load_pickle(final_dir / "final_frame_grm_prm_score_passthrough.pkl")
    if not isinstance(final_frames, list) or len(final_frames) != expected_frames:
        raise ValueError("final frame roster mismatch")
    flattened = {name: [] for name in ("boxes_lidar", "boxes_global", "scores", "labels", "object_ids")}
    offsets = [0]
    for frame_id, frame in enumerate(final_frames):
        observations = []
        for (_, object_id), track in final_tracks.items():
            ids = _frame_ids(track["sample_idx"])
            matches = np.flatnonzero(ids == frame_id)
            if len(matches) > 1:
                raise ValueError("duplicate object observation in one frame")
            if len(matches) == 1:
                observations.append((object_id, track, int(matches[0])))
        observations.sort(key=lambda item: item[0])
        count = len(observations)
        object_ids = np.asarray([item[0] for item in observations], dtype=np.int64)
        boxes_global = np.asarray(
            [item[1]["boxes_global"][item[2]] for item in observations], dtype=np.float32
        ).reshape(count, 9)
        names = np.asarray(
            [item[1]["name"][item[2]] for item in observations], dtype="<U10"
        )
        scores = np.asarray(
            [item[1]["score"][item[2]] for item in observations], dtype=np.float32
        )
        boxes_lidar = _global_to_lidar(boxes_global, detector_poses[frame_id])
        required_frame_fields = {
            "sequence_name",
            "frame_id",
            "pose",
            "obj_ids",
            "name",
            "score",
            "boxes_global",
            "boxes_lidar",
        }
        if "timestamp" in detector_frames[frame_id]:
            required_frame_fields.add("timestamp")
        if not isinstance(frame, dict) or set(frame) != required_frame_fields:
            raise ValueError("final frame field set mismatch")
        if (
            frame["sequence_name"] != sequence_name
            or int(frame["frame_id"]) != frame_id
            or not np.array_equal(frame["pose"], detector_poses[frame_id])
            or not np.array_equal(frame["obj_ids"], object_ids)
            or not np.array_equal(np.asarray(frame["name"], dtype="<U10"), names)
            or not np.array_equal(frame["score"], scores)
            or not np.array_equal(frame["boxes_global"], boxes_global)
            or not np.allclose(frame["boxes_lidar"], boxes_lidar, rtol=0, atol=1e-5)
            or (
                "timestamp" in required_frame_fields
                and frame["timestamp"] != detector_frames[frame_id]["timestamp"]
            )
        ):
            raise ValueError("final frame reconstruction mismatch")
        labels = np.asarray([CLASSES.index(str(name)) for name in names], dtype=np.int8)
        for name, value in (
            ("boxes_lidar", boxes_lidar),
            ("boxes_global", boxes_global),
            ("scores", scores),
            ("labels", labels),
            ("object_ids", object_ids),
        ):
            flattened[name].append(value)
        offsets.append(offsets[-1] + count)

    final_npz = _load_final_npz(final_dir / "final_arrays.npz", expected_frames)
    if str(final_npz["sequence_name"].item()) != sequence_name:
        raise ValueError("final NPZ sequence mismatch")
    if not np.array_equal(final_npz["frame_offsets"], np.asarray(offsets, dtype=np.int64)) or not np.array_equal(
        final_npz["poses"], np.stack(detector_poses)
    ):
        raise ValueError("final NPZ frame identity mismatch")
    for name, parts in flattened.items():
        expected = np.concatenate(parts, axis=0)
        if name == "boxes_lidar":
            equal = np.allclose(final_npz[name], expected, rtol=0, atol=1e-5)
        else:
            equal = np.array_equal(final_npz[name], expected)
        if not equal:
            raise ValueError(f"final NPZ {name} replay mismatch")

    final_manifest = load_json_strict(final_dir / "final_manifest.json")
    if not isinstance(final_manifest, dict) or set(final_manifest) != {
        "schema_version",
        "inputs",
        "frame_count",
        "track_count",
        "classes",
        "crm",
        "score_policy",
        "outputs",
    }:
        raise ValueError("final manifest field set mismatch")
    expected_outputs = {
        "final_track.pkl": _sha256_file(final_dir / "final_track.pkl"),
        "final_frame_grm_prm_score_passthrough.pkl": _sha256_file(
            final_dir / "final_frame_grm_prm_score_passthrough.pkl"
        ),
        "final_arrays.npz": _sha256_file(final_dir / "final_arrays.npz"),
    }
    if (
        _closed_regular_files(final_dir)
        != {
            "final_manifest.json",
            "final_track.pkl",
            "final_frame_grm_prm_score_passthrough.pkl",
            "final_arrays.npz",
        }
        or final_manifest["schema_version"] != "detzero-stage-a-final-v1"
        or final_manifest["frame_count"] != expected_frames
        or final_manifest["track_count"] != len(tracks)
        or final_manifest["classes"] != refining_manifest["classes"]
        or final_manifest["crm"] != {"status": "NOT_EXECUTED_NO_CRM_BY_DESIGN"}
        or final_manifest["score_policy"] != "TRACKING_SCORE_PASSTHROUGH"
        or final_manifest["outputs"] != expected_outputs
        or final_manifest["inputs"]
        != {
            "tracking_sha256": _sha256_file(tracking_path),
            "detector_frames_sha256": _sha256_file(detector_frames_path),
            "refining_manifest_sha256": _sha256_file(
                refining_dir / "refining_manifest.json"
            ),
        }
    ):
        raise ValueError("final manifest mismatch")

    return {
        "sequence_name": sequence_name,
        "frame_count": expected_frames,
        "track_count": len(tracks),
        "observation_count": sum(len(track["frame_ids"]) for track in tracks.values()),
        "class_track_counts": class_track_counts,
    }


def _validate_visual_boxes(frame: Any) -> int:
    if not isinstance(frame, dict):
        raise ValueError("visual frame source must be a dictionary")
    boxes = np.asarray(frame.get("boxes_lidar"))
    names = np.asarray(frame.get("name"))
    if (
        boxes.ndim != 2
        or boxes.shape[1] != 9
        or boxes.dtype != np.float32
        or names.shape != (len(boxes),)
        or names.dtype.kind not in "US"
        or not np.isfinite(boxes).all()
        or np.any(boxes[:, 3:6] <= 0)
        or not set(names.tolist()) <= set(CLASSES)
    ):
        raise ValueError("visual source box schema mismatch")
    return len(boxes)


def _point_layer_matches(pixels: np.ndarray, points: np.ndarray, panel_left: int) -> int:
    visible = points[
        (points[:, 0] >= -80)
        & (points[:, 0] <= 80)
        & (points[:, 1] >= -80)
        & (points[:, 1] <= 80)
    ]
    if len(visible) == 0:
        return 0
    columns = panel_left + np.rint((80 - visible[:, 1]) / 160 * 599).astype(int)
    rows = 40 + np.rint((80 - visible[:, 0]) / 160 * 579).astype(int)
    shades = np.clip(105 + 100 * visible[:, 3], 65, 230).astype(np.uint8)
    expected = {}
    for row, column, shade in zip(rows.tolist(), columns.tolist(), shades.tolist()):
        expected[(row, column)] = max(expected.get((row, column), 0), shade)
    return sum(
        np.array_equal(pixels[row, column], [shade, shade, shade])
        for (row, column), shade in expected.items()
    )


def validate_visuals(
    waymo_root: str | Path,
    detector_frames_path: str | Path,
    final_frames_path: str | Path,
    visuals_dir: str | Path,
    *,
    expected_frames: int,
) -> dict[str, Any]:
    """Validate exhaustive PNG coverage, hashes, decoding, and point-layer binding."""
    if type(expected_frames) is not int or expected_frames <= 0:
        raise ValueError("expected_frames must be a positive integer")
    waymo_root = Path(waymo_root)
    visuals_dir = Path(visuals_dir)
    if waymo_root.is_symlink() or not waymo_root.is_dir():
        raise ValueError("invalid Waymo root")
    if visuals_dir.is_symlink() or not visuals_dir.is_dir():
        raise ValueError("invalid visuals directory")
    waymo_root = waymo_root.resolve(strict=True)
    visuals_dir = visuals_dir.resolve(strict=True)

    detector_frames = _load_pickle(detector_frames_path)
    final_frames = _load_pickle(final_frames_path)
    if (
        not isinstance(detector_frames, list)
        or not isinstance(final_frames, list)
        or len(detector_frames) != expected_frames
        or len(final_frames) != expected_frames
    ):
        raise ValueError("visual source frame count mismatch")
    sequence_names = {
        frame.get("sequence_name")
        for frame in detector_frames + final_frames
        if isinstance(frame, dict)
    }
    if len(sequence_names) != 1:
        raise ValueError("visual source sequence mismatch")
    sequence_name = next(iter(sequence_names))
    if not isinstance(sequence_name, str) or not sequence_name:
        raise ValueError("invalid visual sequence identity")

    manifest = load_json_strict(visuals_dir / "render_manifest.json")
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version",
        "sequence_name",
        "frame_count",
        "inputs",
        "frames",
    }:
        raise ValueError("render manifest field set mismatch")
    inputs = manifest["inputs"]
    if not isinstance(inputs, dict) or set(inputs) != {
        "waymo_root",
        "detector_frames",
        "detector_frames_sha256",
        "final_frames",
        "final_frames_sha256",
    }:
        raise ValueError("render manifest input field set mismatch")
    manifest_waymo_root = resolve_generation_provenance(
        visuals_dir, inputs["waymo_root"]
    )
    manifest_detector_frames = resolve_generation_provenance(
        visuals_dir, inputs["detector_frames"]
    )
    manifest_final_frames = resolve_generation_provenance(
        visuals_dir, inputs["final_frames"]
    )
    if (
        manifest["schema_version"] != "detzero-stage-a-render-v1"
        or manifest["sequence_name"] != sequence_name
        or manifest["frame_count"] != expected_frames
        or inputs["detector_frames_sha256"] != _sha256_file(detector_frames_path)
        or inputs["final_frames_sha256"] != _sha256_file(final_frames_path)
        or manifest_waymo_root != waymo_root
        or manifest_detector_frames != Path(detector_frames_path).resolve(strict=True)
        or manifest_final_frames != Path(final_frames_path).resolve(strict=True)
        or not isinstance(manifest["frames"], list)
        or len(manifest["frames"]) != expected_frames
    ):
        raise ValueError("render manifest identity or input hash mismatch")

    expected_entries = {"render_manifest.json"} | {
        f"{frame_id:04d}.png" for frame_id in range(expected_frames)
    }
    entries = list(visuals_dir.iterdir())
    if (
        {entry.name for entry in entries} != expected_entries
        or any(entry.is_symlink() or not entry.is_file() for entry in entries)
        or any(entry.suffix.lower() in {".html", ".htm"} for entry in entries)
    ):
        raise ValueError("visual directory is not closed-world PNG output")

    image_hashes = set()
    point_hashes = set()
    frame_fields = {
        "frame_id",
        "path",
        "bytes",
        "sha256",
        "width",
        "height",
        "point_count",
        "point_path",
        "point_sha256",
        "detector_box_count",
        "final_box_count",
    }
    for frame_id, frame_manifest in enumerate(manifest["frames"]):
        if not isinstance(frame_manifest, dict) or set(frame_manifest) != frame_fields:
            raise ValueError("render frame manifest field set mismatch")
        if (
            detector_frames[frame_id].get("sequence_name") != sequence_name
            or final_frames[frame_id].get("sequence_name") != sequence_name
            or int(detector_frames[frame_id].get("frame_id", -1)) != frame_id
            or int(final_frames[frame_id].get("frame_id", -1)) != frame_id
            or frame_manifest["frame_id"] != frame_id
            or frame_manifest["path"] != f"{frame_id:04d}.png"
        ):
            raise ValueError("visual frame identity mismatch")
        detector_count = _validate_visual_boxes(detector_frames[frame_id])
        final_count = _validate_visual_boxes(final_frames[frame_id])

        relative_point_path = Path(frame_manifest["point_path"])
        expected_point_path = (
            Path("waymo_processed_data")
            / f"segment-{sequence_name}"
            / f"{frame_id:04d}.npy"
        )
        if relative_point_path != expected_point_path or relative_point_path.is_absolute():
            raise ValueError("render point path mismatch")
        point_path = (waymo_root / relative_point_path).resolve(strict=True)
        if waymo_root not in point_path.parents or point_path.is_symlink():
            raise ValueError("render point path escapes Waymo root")
        points = np.load(_regular_file(point_path, 1024**3), allow_pickle=False)
        if (
            points.ndim != 2
            or points.shape[1] != 6
            or points.dtype != np.float32
            or len(points) == 0
            or not np.isfinite(points).all()
        ):
            raise ValueError("render point array mismatch")
        point_hash = _sha256_file(point_path)

        image_path = _regular_file(visuals_dir / frame_manifest["path"], 64 * 1024**2)
        image_hash = _sha256_file(image_path)
        with Image.open(image_path) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (1280, 640):
                raise ValueError("rendered image format mismatch")
            pixels = np.asarray(image).copy()
        if (
            pixels.shape != (640, 1280, 3)
            or float(pixels.std()) <= 10
            or _point_layer_matches(pixels, points, 20) == 0
            or _point_layer_matches(pixels, points, 660) == 0
            or frame_manifest["bytes"] != image_path.stat().st_size
            or frame_manifest["sha256"] != image_hash
            or frame_manifest["width"] != 1280
            or frame_manifest["height"] != 640
            or frame_manifest["point_count"] != len(points)
            or frame_manifest["point_sha256"] != point_hash
            or frame_manifest["detector_box_count"] != detector_count
            or frame_manifest["final_box_count"] != final_count
        ):
            raise ValueError("rendered image payload or data binding mismatch")
        image_hashes.add(image_hash)
        point_hashes.add(point_hash)

    if len(point_hashes) > 1 and len(image_hashes) <= 1:
        raise ValueError("different point inputs produced identical visual payloads")
    return {
        "sequence_name": sequence_name,
        "frame_count": expected_frames,
        "unique_image_count": len(image_hashes),
    }


def _closed_regular_files(root: Path) -> set[str]:
    files = set()
    for path in root.rglob("*"):
        if path.is_symlink() or (not path.is_dir() and not path.is_file()):
            raise ValueError(f"undeclared special filesystem node: {path}")
        if path.is_file():
            files.add(path.relative_to(root).as_posix())
    return files


_SEQUENCE_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_SOURCE_POLICY_RELATIVE = "tools/external_detector/stage_a_source_policy.json"


def _formal_source_policy(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    policy = load_json_strict(Path(repo_root) / _SOURCE_POLICY_RELATIVE)
    if (
        not isinstance(policy, dict)
        or set(policy) != {"schema_version", "expected_frames", "source_paths"}
        or policy["schema_version"] != "detzero-stage-a-source-policy-v1"
        or policy["expected_frames"] != 199
        or not isinstance(policy["source_paths"], list)
        or policy["source_paths"] != sorted(set(policy["source_paths"]))
        or _SOURCE_POLICY_RELATIVE not in policy["source_paths"]
        or any(
            not isinstance(relative, str)
            or PurePosixPath(relative).is_absolute()
            or ".." in PurePosixPath(relative).parts
            or PurePosixPath(relative).as_posix() != relative
            for relative in policy["source_paths"]
        )
    ):
        raise ValueError("formal Stage-A source policy mismatch")
    return policy


def expected_formal_source_paths(repo_root: str | Path = REPO_ROOT) -> set[str]:
    return set(_formal_source_policy(repo_root)["source_paths"])


def expected_stage_a_paths(
    run_root: str | Path, *, allow_unaccepted_marker: bool = False
) -> set[str]:
    """Derive the exact required Stage-A file set from run-local manifests.

    Dynamic sequence identity comes from the run, while formal frame count and
    source membership come from the independently reviewed local source policy.
    """
    root = Path(run_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"invalid run root: {root}")
    preprocess_path = root / "data/waymo/preprocess_manifest.json"
    try:
        preprocess = load_json_strict(preprocess_path)
    except (ValueError, OSError) as error:
        raise ValueError(
            f"run lacks a valid Stage-A profile manifest: {preprocess_path}"
        ) from error
    if not isinstance(preprocess, dict):
        raise ValueError("preprocess manifest must be an object")
    required = {"frame_count", "frame_ids", "sequence_name"}
    if not required <= set(preprocess):
        raise ValueError("preprocess manifest lacks profile identity fields")
    sequence_name = preprocess["sequence_name"]
    frame_count = preprocess["frame_count"]
    frame_ids = preprocess["frame_ids"]
    policy = _formal_source_policy()
    if (
        not isinstance(sequence_name, str)
        or not _SEQUENCE_RE.fullmatch(sequence_name)
        or type(frame_count) is not int
        or frame_count != policy["expected_frames"]
        or not isinstance(frame_ids, list)
        or frame_ids != list(range(frame_count))
    ):
        raise ValueError("formal Stage-A profile frame identity mismatch")
    source_manifest_path = root / "source_bundle/source_manifest.json"
    source_manifest = load_json_strict(source_manifest_path)
    if not isinstance(source_manifest, dict):
        raise ValueError("source manifest must be an object")
    source_files = source_manifest.get("source_files")
    if not isinstance(source_files, dict):
        raise ValueError("source manifest lacks source_files roster")
    if any(
        not isinstance(relative, str)
        or PurePosixPath(relative).is_absolute()
        or ".." in PurePosixPath(relative).parts
        for relative in source_files
    ):
        raise ValueError("source manifest contains an unsafe path")
    if set(source_files) != set(policy["source_paths"]):
        raise ValueError("formal Stage-A profile source roster mismatch")
    expected = {
        "command.sh",
        "run_ledger.json",
        "run_metadata.json",
        "data/waymo/preprocess_manifest.json",
        "data/waymo/ImageSets/test.txt",
        f"data/waymo/waymo_processed_data/segment-{sequence_name}/{sequence_name}.pkl",
        "detector/detector_manifest.json",
        "detector/raw_predictions.npz",
        "adapter/adapter_manifest.json",
        "adapter/detzero_result.pkl",
        "tracking/tracking.pkl",
        "tracking/dropped.pkl",
        "refining/refining_manifest.json",
        "final/final_arrays.npz",
        "final/final_frame_grm_prm_score_passthrough.pkl",
        "final/final_manifest.json",
        "final/final_track.pkl",
        "visuals/render_manifest.json",
        "source_bundle/source_manifest.json",
    }
    expected |= {
        f"data/waymo/waymo_processed_data/segment-{sequence_name}/{frame_id:04d}.npy"
        for frame_id in range(frame_count)
    }
    expected |= {
        f"refining/result/{class_name}_{kind}.pkl"
        for class_name in CLASSES
        for kind in ("geometry", "position")
    }
    expected |= {f"visuals/{frame_id:04d}.png" for frame_id in range(frame_count)}
    expected |= {
        f"logs/{name}.log"
        for name in (
            "01-preprocess",
            "02-detector",
            "03-adapter",
            "04-tracking",
            "05-refining",
            "06-final",
            "07-visuals",
        )
    }
    expected |= {
        f"source_bundle/files/{relative}" for relative in source_files
    }
    actual = _closed_regular_files(root)
    if ".unaccepted" in actual and allow_unaccepted_marker:
        marker = root / ".unaccepted"
        descriptor = _open_bounded_regular(marker, 32)
        with os.fdopen(descriptor, "rb") as stream:
            marker_bytes = stream.read()
        if marker_bytes != b"UNACCEPTED\n":
            raise ValueError("invalid acceptance marker")
        actual.remove(".unaccepted")
    if actual != expected:
        missing = sorted(expected - actual)
        undeclared = sorted(actual - expected)
        raise ValueError(
            "run does not match the expected Stage-A profile: "
            f"missing={missing} undeclared={undeclared}"
        )
    expected_dirs = set()
    for relative in expected:
        parts = Path(relative).parts[:-1]
        for index in range(1, len(parts) + 1):
            expected_dirs.add("/".join(parts[:index]))
    for path in root.rglob("*"):
        if not path.is_dir() or path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        if relative not in expected_dirs:
            raise ValueError(f"undeclared empty directory: {relative}")
    return expected


def validate_preprocess_detector_adapter(
    waymo_root: str | Path,
    detector_dir: str | Path,
    adapter_dir: str | Path,
    *,
    expected_frames: int,
) -> dict[str, Any]:
    """Replay preprocessing, detector manifest, and adapter boundary facts."""
    if type(expected_frames) is not int or expected_frames <= 0:
        raise ValueError("expected_frames must be a positive integer")
    roots = [Path(value) for value in (waymo_root, detector_dir, adapter_dir)]
    if any(path.is_symlink() or not path.is_dir() for path in roots):
        raise ValueError("stage roots must be regular directories")
    waymo_root, detector_dir, adapter_dir = [
        path.resolve(strict=True) for path in roots
    ]

    preprocess_path = waymo_root / "preprocess_manifest.json"
    preprocess = load_json_strict(preprocess_path)
    preprocess_fields = {
        "sequence_name",
        "frame_count",
        "frame_ids",
        "timestamp_first",
        "timestamp_last",
        "input_tfrecord",
        "input_tfrecord_bytes",
        "input_tfrecord_sha256",
        "usage",
        "waymo_terms_accepted",
    }
    if not isinstance(preprocess, dict) or set(preprocess) != preprocess_fields:
        raise ValueError("preprocess manifest field set mismatch")
    sequence_name = preprocess["sequence_name"]
    source_path = Path(preprocess["input_tfrecord"])
    if (
        not isinstance(sequence_name, str)
        or not sequence_name
        or preprocess["frame_count"] != expected_frames
        or preprocess["frame_ids"] != list(range(expected_frames))
        or preprocess["usage"] != "non-commercial-research"
        or preprocess["waymo_terms_accepted"] is not True
        or preprocess["input_tfrecord_bytes"]
        != _regular_file(source_path, 1024**3).stat().st_size
        or preprocess["input_tfrecord_sha256"] != _sha256_file(source_path)
    ):
        raise ValueError("preprocess manifest identity or source mismatch")

    sequence_dir = (
        waymo_root / "waymo_processed_data" / f"segment-{sequence_name}"
    )
    info_path = sequence_dir / f"{sequence_name}.pkl"
    infos = _load_pickle(info_path)
    if not isinstance(infos, list) or len(infos) != expected_frames:
        raise ValueError("preprocess info roster mismatch")
    expected_waymo_files = {
        "preprocess_manifest.json",
        "ImageSets/test.txt",
        (
            f"waymo_processed_data/segment-{sequence_name}/{sequence_name}.pkl"
        ),
    } | {
        f"waymo_processed_data/segment-{sequence_name}/{frame_id:04d}.npy"
        for frame_id in range(expected_frames)
    }
    if _closed_regular_files(waymo_root) != expected_waymo_files:
        raise ValueError("preprocess output is not closed-world")
    if (waymo_root / "ImageSets" / "test.txt").read_text(encoding="utf-8") != f"{sequence_name}\n":
        raise ValueError("Waymo split identity mismatch")

    timestamps = []
    point_counts = []
    info_fields = {
        "time_stamp",
        "sample_idx",
        "sequence_name",
        "pose",
        "num_points_of_each_lidar",
        "lidar_path",
    }
    adapter_poses = []
    for frame_id, info in enumerate(infos):
        expected_relative_path = (
            Path("waymo_processed_data")
            / f"segment-{sequence_name}"
            / f"{frame_id:04d}.npy"
        )
        if not isinstance(info, dict) or set(info) != info_fields:
            raise ValueError("preprocess info field set mismatch")
        timestamp = info["time_stamp"]
        lidar_counts = info["num_points_of_each_lidar"]
        if (
            isinstance(timestamp, (bool, np.bool_))
            or not isinstance(timestamp, (int, np.integer))
            or (timestamps and int(timestamp) <= timestamps[-1])
            or info["sample_idx"] != frame_id
            or info["sequence_name"] != sequence_name
            or Path(info["lidar_path"]) != expected_relative_path
            or not isinstance(lidar_counts, list)
            or len(lidar_counts) != 5
            or any(type(value) is not int or value < 0 for value in lidar_counts)
        ):
            raise ValueError("preprocess info identity or count mismatch")
        pose = _validate_pose(info["pose"])
        point_path = _regular_file(waymo_root / expected_relative_path, 1024**3)
        points = np.load(point_path, mmap_mode="r", allow_pickle=False)
        if (
            points.ndim != 2
            or points.shape[1] != 6
            or points.dtype != np.float32
            or len(points) == 0
            or not np.isfinite(points).all()
            or not set(np.unique(points[:, 5]).tolist()) <= {-1.0, 1.0}
            or sum(lidar_counts) != len(points)
        ):
            raise ValueError("preprocess point array mismatch")
        timestamps.append(int(timestamp))
        point_counts.append(len(points))
        adapter_poses.append(pose)
    if (
        preprocess["timestamp_first"] != timestamps[0]
        or preprocess["timestamp_last"] != timestamps[-1]
    ):
        raise ValueError("preprocess timestamp range mismatch")

    detector_manifest_path = detector_dir / "detector_manifest.json"
    detector = load_json_strict(detector_manifest_path)
    detector_fields = {
        "schema_version",
        "backend",
        "device",
        "config",
        "config_sha256",
        "checkpoint",
        "checkpoint_sha256",
        "preprocess_manifest",
        "preprocess_manifest_sha256",
        "sequence_name",
        "frame_count",
        "box_count",
        "class_counts",
        "point_features",
        "nlz_filter",
        "open3d_version",
        "open3dml_commit",
        "torch_version",
        "numpy_version",
        "raw_predictions",
        "raw_predictions_sha256",
        "elapsed_seconds",
        "checkpoint_license_status",
    }
    if not isinstance(detector, dict) or set(detector) != detector_fields:
        raise ValueError("detector manifest field set mismatch")
    raw_path = detector_dir / "raw_predictions.npz"
    raw = _load_raw_predictions(raw_path, expected_frames)
    raw_class_counts = {
        class_name: int(np.count_nonzero(raw["labels"] == index))
        for index, class_name in enumerate(RAW_CLASSES)
    }
    config_path = Path(detector["config"])
    checkpoint_path = Path(detector["checkpoint"])
    detector_preprocess_path = resolve_generation_provenance(
        detector_dir, detector["preprocess_manifest"]
    )
    if (
        _closed_regular_files(detector_dir)
        != {"detector_manifest.json", "raw_predictions.npz"}
        or detector["schema_version"] != "open3dml-waymo-detector-manifest-v1"
        or detector["backend"] != "Open3D-ML PointPillars Waymo"
        or detector["device"] not in {"cpu", "cuda"}
        or detector["config_sha256"] != _sha256_file(config_path)
        or detector["checkpoint_sha256"] != _sha256_file(checkpoint_path)
        or detector_preprocess_path != preprocess_path.resolve(strict=True)
        or detector["preprocess_manifest_sha256"] != _sha256_file(preprocess_path)
        or detector["sequence_name"] != sequence_name
        or detector["frame_count"] != expected_frames
        or detector["box_count"] != len(raw["centers"])
        or detector["class_counts"] != raw_class_counts
        or detector["point_features"] != ["x", "y", "z", "intensity"]
        or detector["nlz_filter"] != "points[:, 5] != 1.0"
        or any(
            not isinstance(detector[field], str) or not detector[field]
            for field in (
                "open3d_version",
                "open3dml_commit",
                "torch_version",
                "numpy_version",
            )
        )
        or detector["raw_predictions"] != "raw_predictions.npz"
        or detector["raw_predictions_sha256"] != _sha256_file(raw_path)
        or isinstance(detector["elapsed_seconds"], bool)
        or not isinstance(detector["elapsed_seconds"], (int, float))
        or not math.isfinite(detector["elapsed_seconds"])
        or detector["elapsed_seconds"] < 0
        or detector["checkpoint_license_status"]
        != "upstream-model-zoo-source-recorded; no separate weight license found"
        or not np.array_equal(raw["point_counts"], np.asarray(point_counts, dtype=np.int64))
    ):
        raise ValueError("detector manifest or preprocess binding mismatch")

    adapter_manifest_path = adapter_dir / "adapter_manifest.json"
    adapter = load_json_strict(adapter_manifest_path)
    adapter_fields = {
        "schema_version",
        "sequence_name",
        "frame_count",
        "box_count",
        "class_counts",
        "input_raw_predictions",
        "input_raw_predictions_sha256",
        "input_waymo_info",
        "input_waymo_info_sha256",
        "output_pickle",
        "output_pickle_sha256",
        "box_schema",
        "center_definition",
        "yaw_conversion",
        "size_conversion",
        "velocity_status",
    }
    if not isinstance(adapter, dict) or set(adapter) != adapter_fields:
        raise ValueError("adapter manifest field set mismatch")
    adapter_path = adapter_dir / "detzero_result.pkl"
    adapter_summary = validate_detector_adapter(
        raw_path, adapter_path, expected_frames=expected_frames
    )
    adapter_raw_path = resolve_generation_provenance(
        adapter_dir, adapter["input_raw_predictions"]
    )
    adapter_info_path = resolve_generation_provenance(
        adapter_dir, adapter["input_waymo_info"]
    )
    frames = _load_pickle(adapter_path)
    for frame_id, frame in enumerate(frames):
        if (
            frame["timestamp"] != timestamps[frame_id]
            or not np.array_equal(frame["pose"], adapter_poses[frame_id])
        ):
            raise ValueError("adapter/preprocess pose or timestamp mismatch")
    if (
        _closed_regular_files(adapter_dir)
        != {"adapter_manifest.json", "detzero_result.pkl"}
        or adapter["schema_version"] != "open3dml-to-detzero-adapter-manifest-v1"
        or adapter["sequence_name"] != sequence_name
        or adapter["frame_count"] != expected_frames
        or adapter["box_count"] != adapter_summary["box_count"]
        or adapter["class_counts"] != adapter_summary["class_counts"]
        or adapter_raw_path != raw_path.resolve(strict=True)
        or adapter["input_raw_predictions_sha256"] != _sha256_file(raw_path)
        or adapter_info_path != info_path.resolve(strict=True)
        or adapter["input_waymo_info_sha256"] != _sha256_file(info_path)
        or adapter["output_pickle"] != "detzero_result.pkl"
        or adapter["output_pickle_sha256"] != _sha256_file(adapter_path)
        or adapter["box_schema"]
        != ["x", "y", "z", "length", "width", "height", "heading", "vx", "vy"]
        or adapter["center_definition"] != "geometric box center"
        or adapter["yaw_conversion"]
        != "detzero_heading = wrap(-open3d_yaw - pi/2)"
        or adapter["size_conversion"]
        != "open3d [width,height,length] -> detzero [length,width,height]"
        or adapter["velocity_status"] != "unavailable; vx=vy=0"
    ):
        raise ValueError("adapter manifest or upstream binding mismatch")

    return {
        "sequence_name": sequence_name,
        "frame_count": expected_frames,
        "point_count": sum(point_counts),
        "box_count": adapter_summary["box_count"],
        "class_counts": adapter_summary["class_counts"],
    }


def validate_dropped_frames(
    dropped_path: str | Path,
    adapter_frames_path: str | Path,
    *,
    expected_frames: int,
) -> dict[str, Any]:
    """Validate every dropped detection as an exact adapter-frame multiset subset."""
    if type(expected_frames) is not int or expected_frames <= 0:
        raise ValueError("expected_frames must be a positive integer")
    adapter_frames = _load_pickle(adapter_frames_path)
    dropped = _load_pickle(dropped_path)
    if not isinstance(adapter_frames, list) or len(adapter_frames) != expected_frames:
        raise ValueError("adapter frame roster mismatch")
    sequence_names = {
        frame.get("sequence_name") for frame in adapter_frames if isinstance(frame, dict)
    }
    if len(sequence_names) != 1:
        raise ValueError("adapter sequence identity mismatch")
    sequence_name = next(iter(sequence_names))
    if (
        not isinstance(sequence_name, str)
        or not isinstance(dropped, dict)
        or set(dropped) != {sequence_name}
        or not isinstance(dropped[sequence_name], dict)
        or set(dropped[sequence_name])
        != {str(frame_id) for frame_id in range(expected_frames)}
    ):
        raise ValueError("dropped frame roster mismatch")

    frame_fields = {
        "sequence_name",
        "sample_idx",
        "frame_id",
        "timestamp",
        "pose",
        "name",
        "score",
        "boxes_lidar",
    }
    dropped_count = 0
    for frame_id, adapter_frame in enumerate(adapter_frames):
        frame = dropped[sequence_name][str(frame_id)]
        if not isinstance(adapter_frame, dict) or set(adapter_frame) != frame_fields:
            raise ValueError("adapter frame field set mismatch")
        if not isinstance(frame, dict) or set(frame) != frame_fields:
            raise ValueError("dropped frame field set mismatch")
        boxes = np.asarray(frame["boxes_lidar"])
        names = np.asarray(frame["name"])
        scores = np.asarray(frame["score"])
        count = len(boxes)
        if (
            frame["sequence_name"] != sequence_name
            or frame["sample_idx"] != frame_id
            or frame["frame_id"] != frame_id
            or frame["timestamp"] != adapter_frame["timestamp"]
            or not np.array_equal(frame["pose"], adapter_frame["pose"])
            or boxes.shape != (count, 9)
            or boxes.dtype != np.float32
            or names.shape != (count,)
            or names.dtype.kind not in "US"
            or scores.shape != (count,)
            or scores.dtype != np.float32
            or not np.isfinite(boxes).all()
            or not np.isfinite(scores).all()
            or np.any(boxes[:, 3:6] <= 0)
            or np.any((scores < 0) | (scores > 1))
            or not set(names.tolist()) <= set(CLASSES)
        ):
            raise ValueError("dropped frame schema or identity mismatch")

        def observations(source: dict[str, Any]) -> Counter:
            source_boxes = np.asarray(source["boxes_lidar"])
            source_names = np.asarray(source["name"])
            source_scores = np.asarray(source["score"])
            return Counter(
                (
                    str(source_names[index]),
                    source_scores[index].tobytes(),
                    source_boxes[index].tobytes(),
                )
                for index in range(len(source_boxes))
            )

        adapter_observations = observations(adapter_frame)
        dropped_observations = observations(frame)
        if any(
            multiplicity > adapter_observations[observation]
            for observation, multiplicity in dropped_observations.items()
        ):
            raise ValueError("dropped detections are not an adapter subset")
        dropped_count += count

    return {
        "sequence_name": sequence_name,
        "frame_count": expected_frames,
        "dropped_box_count": dropped_count,
    }


def validate_stage_a_artifacts(
    *,
    run_root: str | Path,
    live_repo_root: str | Path,
    pipeline_python: str | Path,
    detector_python: str | Path,
    preprocess_python: str | Path,
    waymo_root: str | Path,
    detector_dir: str | Path,
    adapter_dir: str | Path,
    tracking_path: str | Path,
    dropped_path: str | Path,
    refining_dir: str | Path,
    final_dir: str | Path,
    visuals_dir: str | Path,
    expected_frames: int,
) -> dict[str, Any]:
    """Run all independent Stage A artifact checks and separate release blockers."""
    inputs = {
        "run_root": str(Path(run_root).absolute()),
        "live_repo_root": str(Path(live_repo_root).absolute()),
        "pipeline_python": str(Path(pipeline_python).absolute()),
        "detector_python": str(Path(detector_python).absolute()),
        "preprocess_python": str(Path(preprocess_python).absolute()),
        "waymo_root": str(Path(waymo_root).absolute()),
        "detector_dir": str(Path(detector_dir).absolute()),
        "adapter_dir": str(Path(adapter_dir).absolute()),
        "tracking": str(Path(tracking_path).absolute()),
        "dropped": str(Path(dropped_path).absolute()),
        "refining_dir": str(Path(refining_dir).absolute()),
        "final_dir": str(Path(final_dir).absolute()),
        "visuals_dir": str(Path(visuals_dir).absolute()),
    }
    checks = {}
    report = {
        "schema_version": "detzero-stage-a-validation-v1",
        "expected_frames": expected_frames,
        "inputs": inputs,
        "checks": checks,
        "artifact_validation_passed": False,
        "passed": False,
        "release_eligible": False,
        "release_blockers": [
            "NOT_EVALUATED_NO_GROUND_TRUTH",
            "POINTPILLARS_WEIGHT_LICENSE_UNRESOLVED",
            "GENERATION_SOURCE_BUNDLE_NOT_VALIDATED",
        ],
        "error": None,
    }
    try:
        ledger_start = validate_run_ledger(run_root)
        checks["run_ledger"] = ledger_start
        expected_paths_set = expected_stage_a_paths(
            run_root, allow_unaccepted_marker=True
        )
        checks["stage_a_profile"] = {"expected_file_count": len(expected_paths_set)}
        checks["acceptance_marker"] = {
            "state": (
                "UNACCEPTED"
                if (Path(run_root) / ".unaccepted").exists()
                else "CANONICAL"
            )
        }
        run_root_path = Path(run_root).resolve(strict=True)
        expected_paths = {
            "waymo_root": run_root_path / "data/waymo",
            "detector_dir": run_root_path / "detector",
            "adapter_dir": run_root_path / "adapter",
            "tracking": run_root_path / "tracking/tracking.pkl",
            "dropped": run_root_path / "tracking/dropped.pkl",
            "refining_dir": run_root_path / "refining",
            "final_dir": run_root_path / "final",
            "visuals_dir": run_root_path / "visuals",
        }
        if any(
            Path(inputs[name]).resolve(strict=True) != expected
            for name, expected in expected_paths.items()
        ):
            raise ValueError("stage path does not match run-root contract")
        source_start = validate_source_provenance(
            run_root_path / "source_bundle", live_repo_root
        )
        checks["source_provenance"] = source_start
        if ledger_start["source_tree_sha256"] != source_start["source_tree_sha256"]:
            raise ValueError("run ledger/source tree mismatch")
        open3dml_start = validate_open3dml_provenance(run_root_path, detector_dir)
        checks["open3dml_provenance"] = open3dml_start
        pipeline_interpreter = Path(pipeline_python).absolute()
        detector_interpreter = Path(detector_python).absolute()
        preprocess_interpreter = Path(preprocess_python).absolute()
        pipeline_target = pipeline_interpreter.resolve(strict=True)
        detector_target = detector_interpreter.resolve(strict=True)
        preprocess_target = preprocess_interpreter.resolve(strict=True)
        interpreter_start = {
            "pipeline": {
                "bytes": _regular_file(pipeline_target, 1024**3).stat().st_size,
                "sha256": _sha256_file(pipeline_target),
            },
            "detector": {
                "bytes": _regular_file(detector_target, 1024**3).stat().st_size,
                "sha256": _sha256_file(detector_target),
            },
            "preprocess": {
                "bytes": _regular_file(preprocess_target, 1024**3).stat().st_size,
                "sha256": _sha256_file(preprocess_target),
            },
        }
        checks["bundle_replay"] = validate_bundle_replay(
            run_root_path / "source_bundle",
            live_repo_root,
            [
                ("tools/external_detector/run_stage_a.py", pipeline_interpreter),
                ("tools/external_detector/preprocess_waymo_test_segment.py", preprocess_interpreter),
                ("tools/external_detector/run_open3dml_waymo_pointpillars.py", detector_interpreter),
                ("tools/external_detector/adapt_open3dml_to_detzero.py", pipeline_interpreter),
                ("tracking/tools/run_track.py", pipeline_interpreter),
                ("tools/external_detector/run_stage_a_refining.py", pipeline_interpreter),
                ("tools/external_detector/combine_grm_prm_no_crm.py", pipeline_interpreter),
                ("tools/external_detector/render_waymo_sequence.py", pipeline_interpreter),
                ("tools/external_detector/validate_stage_a.py", pipeline_interpreter),
                ("tools/external_detector/compare_stage_a_runs.py", pipeline_interpreter),
            ],
        )
        report["release_blockers"].remove(
            "GENERATION_SOURCE_BUNDLE_NOT_VALIDATED"
        )
        checks["preprocess_detector_adapter"] = validate_preprocess_detector_adapter(
            waymo_root, detector_dir, adapter_dir, expected_frames=expected_frames
        )
        checks["dropped_frames"] = validate_dropped_frames(
            dropped_path,
            Path(adapter_dir) / "detzero_result.pkl",
            expected_frames=expected_frames,
        )
        checks["tracking_refining_final"] = validate_tracking_refining_final(
            tracking_path,
            Path(adapter_dir) / "detzero_result.pkl",
            refining_dir,
            final_dir,
            expected_frames=expected_frames,
        )
        checks["visuals"] = validate_visuals(
            waymo_root,
            Path(adapter_dir) / "detzero_result.pkl",
            Path(final_dir) / "final_frame_grm_prm_score_passthrough.pkl",
            visuals_dir,
            expected_frames=expected_frames,
        )
        identities = {
            (checks[name]["sequence_name"], checks[name]["frame_count"])
            for name in (
                "preprocess_detector_adapter",
                "dropped_frames",
                "tracking_refining_final",
                "visuals",
            )
        }
        if len(identities) != 1:
            raise ValueError("cross-stage sequence/frame identity mismatch")
        if validate_source_provenance(
            run_root_path / "source_bundle", live_repo_root
        ) != source_start:
            raise ValueError("source provenance drift during validation")
        if validate_run_ledger(run_root_path) != ledger_start:
            raise ValueError("run ledger drift during validation")
        if validate_open3dml_provenance(run_root_path, detector_dir) != open3dml_start:
            raise ValueError("Open3D-ML provenance drift during validation")
        if {
            "pipeline": {
                "bytes": _regular_file(pipeline_target, 1024**3).stat().st_size,
                "sha256": _sha256_file(pipeline_target),
            },
            "detector": {
                "bytes": _regular_file(detector_target, 1024**3).stat().st_size,
                "sha256": _sha256_file(detector_target),
            },
            "preprocess": {
                "bytes": _regular_file(preprocess_target, 1024**3).stat().st_size,
                "sha256": _sha256_file(preprocess_target),
            },
        } != interpreter_start:
            raise ValueError("interpreter drift during validation")
        report["artifact_validation_passed"] = True
        report["passed"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    return report


def write_report_noreplace(path: str | Path, report: dict[str, Any]) -> None:
    """Atomically publish one immutable JSON report without replacing a path."""
    path = Path(path).absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    if load_json_strict(path) != report:
        raise RuntimeError("validation report read-back mismatch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--live-repo-root", type=Path, required=True)
    parser.add_argument("--pipeline-python", type=Path, required=True)
    parser.add_argument("--detector-python", type=Path, required=True)
    parser.add_argument("--preprocess-python", type=Path, required=True)
    parser.add_argument("--waymo-root", type=Path, required=True)
    parser.add_argument("--detector-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--tracking", type=Path, required=True)
    parser.add_argument("--dropped", type=Path, required=True)
    parser.add_argument("--refining-dir", type=Path, required=True)
    parser.add_argument("--final-dir", type=Path, required=True)
    parser.add_argument("--visuals-dir", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate_stage_a_artifacts(
        run_root=args.run_root,
        live_repo_root=args.live_repo_root,
        pipeline_python=args.pipeline_python,
        detector_python=args.detector_python,
        preprocess_python=args.preprocess_python,
        waymo_root=args.waymo_root,
        detector_dir=args.detector_dir,
        adapter_dir=args.adapter_dir,
        tracking_path=args.tracking,
        dropped_path=args.dropped,
        refining_dir=args.refining_dir,
        final_dir=args.final_dir,
        visuals_dir=args.visuals_dir,
        expected_frames=args.expected_frames,
    )
    write_report_noreplace(args.report, report)
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
