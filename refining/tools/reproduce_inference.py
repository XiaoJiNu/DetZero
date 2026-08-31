#!/usr/bin/env python3
"""Run DetZero GRM/PRM checkpoints on deterministic minimal tracks."""

from __future__ import annotations

import argparse
import copy
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import stat
import sys
import tempfile
import time
from typing import Any

import numpy as np
import yaml
from easydict import EasyDict


REPO_ROOT = Path(__file__).resolve().parents[2]

_CLASS_SPECS = {
    "Vehicle": {"size": (4.8, 1.9, 1.6), "speed": 1.0, "seed": 3101},
    "Pedestrian": {"size": (0.8, 0.8, 1.7), "speed": 0.35, "seed": 3102},
    "Cyclist": {"size": (1.8, 0.8, 1.7), "speed": 0.65, "seed": 3103},
}

_CONFIG_NAMES = {
    ("Vehicle", "geometry"): "vehicle_grm_model.yaml",
    ("Vehicle", "position"): "vehicle_prm_model.yaml",
    ("Pedestrian", "geometry"): "pedestrian_grm_model.yaml",
    ("Pedestrian", "position"): "pedestrian_prm_model.yaml",
    ("Cyclist", "geometry"): "cyclist_grm_model.yaml",
    ("Cyclist", "position"): "cyclist_prm_model.yaml",
}

_CHECKPOINT_NAMES = {
    ("Vehicle", "geometry"): "vehicle_grm_model.pth",
    ("Vehicle", "position"): "vehicle_prm_model.pth",
    ("Pedestrian", "geometry"): "pedestrian_grm_model.pth",
    ("Pedestrian", "position"): "pedestrian_prm_model.pth",
    ("Cyclist", "geometry"): "cyclist_grm_model.pth",
    ("Cyclist", "position"): "cyclist_prm_model.pth",
}


@dataclass
class PreparedInput:
    config: EasyDict
    dataset: Any
    batch: dict[str, Any]


@dataclass
class InferenceResult:
    class_name: str
    model_kind: str
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_epoch: Any
    checkpoint_version: str
    loaded_tensors: int
    model_tensors: int
    elapsed_seconds: float
    pred_boxes_world: np.ndarray


def _load_config(class_name: str, model_kind: str) -> EasyDict:
    try:
        config_name = _CONFIG_NAMES[(class_name, model_kind)]
    except KeyError as error:
        raise ValueError(
            f"unsupported class/model pair: {class_name}/{model_kind}"
        ) from error

    tools_dir = Path(__file__).resolve().parent
    config_path = tools_dir / "cfgs" / "ref_model_cfgs" / config_name
    with config_path.open("r", encoding="utf-8") as stream:
        config_data = yaml.safe_load(stream)

    data_config = dict(config_data["DATA_CONFIG"])
    base_path = tools_dir / data_config.pop("_BASE_CONFIG_")
    with base_path.open("r", encoding="utf-8") as stream:
        merged_data_config = yaml.safe_load(stream)
    merged_data_config.update(data_config)
    config_data["DATA_CONFIG"] = merged_data_config
    return EasyDict(config_data)


def _add_project_paths() -> None:
    for path in (REPO_ROOT / "utils", REPO_ROOT / "refining"):
        path_string = str(path)
        if path_string not in sys.path:
            sys.path.insert(0, path_string)


def prepare_model_input(
    track: dict[str, Any],
    class_name: str,
    model_kind: str,
) -> PreparedInput:
    """Run the repository's official GRM/PRM feature extraction in memory."""
    _add_project_paths()
    config = _load_config(class_name, model_kind)

    if model_kind == "geometry":
        from detzero_refine.datasets.waymo.waymo_geometry_dataset import (
            WaymoGeometryDataset,
        )

        dataset = WaymoGeometryDataset.__new__(WaymoGeometryDataset)
    elif model_kind == "position":
        from detzero_refine.datasets.waymo.waymo_position_dataset import (
            WaymoPositionDataset,
        )

        dataset = WaymoPositionDataset.__new__(WaymoPositionDataset)
    else:
        raise ValueError(f"unsupported model kind: {model_kind}")

    dataset.dataset_cfg = config.DATA_CONFIG
    dataset.class_names = config.CLASS_NAMES
    dataset.training = False
    dataset.tta = False
    dataset.encoding = config.DATA_CONFIG.ENCODING
    dataset.class_map = {
        "Vehicle": 1,
        "Pedestrian": 2,
        "Cyclist": 3,
        1: "Vehicle",
        2: "Pedestrian",
        3: "Cyclist",
    }
    dataset.query_num = config.DATA_CONFIG.QUERY_NUM
    dataset.query_pts_num = config.DATA_CONFIG.QUERY_POINTS_NUM
    dataset.memory_pts_num = config.DATA_CONFIG.MEMORY_POINTS_NUM

    class_index = ("Vehicle", "Pedestrian", "Cyclist").index(class_name)
    kind_index = ("geometry", "position").index(model_kind)
    sampling_seed = 20260825 + class_index * 10 + kind_index
    numpy_random_state = np.random.get_state()
    python_random_state = random.getstate()
    try:
        np.random.seed(sampling_seed)
        random.seed(sampling_seed)
        feature_dict = dataset.extract_track_feature(copy.deepcopy(track))
    finally:
        np.random.set_state(numpy_random_state)
        random.setstate(python_random_state)
    batch = dataset.collate_batch([feature_dict])
    return PreparedInput(config=config, dataset=dataset, batch=batch)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with Path(path).open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _save_npz_fsync(path: Path, payload: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **payload)
    _fsync_file(path)


def _reject_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for key, value in pairs:
        if key in parsed:
            raise ValueError(f"duplicate JSON key: {key}")
        parsed[key] = value
    return parsed


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def load_json_strict(path: Path) -> Any:
    """Read JSON while rejecting duplicate keys and non-finite constants."""
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(
            stream,
            object_pairs_hook=_reject_duplicate_json_pairs,
            parse_constant=_reject_json_constant,
        )


def validate_reproduction_artifacts(
    artifact_dir: Path,
    *,
    expect_unaccepted_marker: bool,
) -> dict[str, Any]:
    """Validate a staged or canonical reproduction artifact tree."""
    artifact_dir = Path(artifact_dir)
    if artifact_dir.is_symlink() or not artifact_dir.is_dir():
        raise RuntimeError(f"artifact directory is not a regular directory: {artifact_dir}")

    expected_files = {"results.json", "results.npz", "visualization.png"}
    if expect_unaccepted_marker:
        expected_files.add(".UNACCEPTED")
    actual_files = {entry.name for entry in artifact_dir.iterdir()}
    if actual_files != expected_files:
        raise RuntimeError(
            "artifact file set mismatch: "
            f"missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}"
        )
    for name in sorted(expected_files):
        path = artifact_dir / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"artifact entry is not a regular file: {name}")

    manifest = load_json_strict(artifact_dir / "results.json")
    if not isinstance(manifest, dict):
        raise RuntimeError("results.json must contain a JSON object")
    expected_manifest_keys = {
        "artifacts",
        "classes",
        "input",
        "model_forward_count",
        "models",
        "producer",
        "runtime",
        "schema_version",
        "source",
        "status",
    }
    if set(manifest) != expected_manifest_keys:
        raise RuntimeError("results.json top-level key set mismatch")
    if manifest["schema_version"] != 1:
        raise RuntimeError("schema_version must be exactly 1")
    if manifest["producer"] != "refining/tools/reproduce_inference.py":
        raise RuntimeError("producer path mismatch")
    if manifest["status"] != "passed":
        raise RuntimeError("manifest status must be passed")

    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "results_npz",
        "visualization",
    }:
        raise RuntimeError("artifact manifest key set mismatch")
    artifact_specs = {
        "results_npz": "results.npz",
        "visualization": "visualization.png",
    }
    for artifact_key, filename in artifact_specs.items():
        artifact = artifacts.get(artifact_key)
        if not isinstance(artifact, dict) or set(artifact) != {"path", "validation"}:
            raise RuntimeError(f"invalid artifact declaration: {artifact_key}")
        if artifact["path"] != filename:
            raise RuntimeError(f"artifact path mismatch: {artifact_key}")
        validation = artifact["validation"]
        if not isinstance(validation, dict):
            raise RuntimeError(f"artifact validation must be an object: {artifact_key}")
        recorded_hash = validation.get("sha256")
        actual_hash = _sha256(artifact_dir / filename)
        if recorded_hash != actual_hash:
            raise RuntimeError(f"{filename} SHA-256 mismatch")

    source = manifest["source"]
    if not isinstance(source, dict) or set(source) != {
        "script_sha256",
        "checkpoint_loader_sha256",
    }:
        raise RuntimeError("source declaration key set mismatch")
    if source["script_sha256"] != _sha256(Path(__file__).resolve()):
        raise RuntimeError("producer script SHA-256 mismatch")
    checkpoint_loader = REPO_ROOT / "utils" / "detzero_utils" / "model_utils.py"
    if source["checkpoint_loader_sha256"] != _sha256(checkpoint_loader):
        raise RuntimeError("checkpoint loader SHA-256 mismatch")

    if (
        isinstance(manifest["model_forward_count"], bool)
        or manifest["model_forward_count"] != 6
    ):
        raise RuntimeError("model_forward_count must be exactly 6")
    if not isinstance(manifest["classes"], dict) or set(manifest["classes"]) != set(
        _CLASS_SPECS
    ):
        raise RuntimeError("class manifest must contain exactly three classes")

    models = manifest["models"]
    expected_pairs = [
        (class_name, model_kind)
        for class_name in ("Vehicle", "Pedestrian", "Cyclist")
        for model_kind in ("geometry", "position")
    ]
    if not isinstance(models, list) or len(models) != 6:
        raise RuntimeError("model list must contain exactly six entries")
    actual_pairs = [
        (model.get("class_name"), model.get("model_kind"))
        if isinstance(model, dict)
        else (None, None)
        for model in models
    ]
    if actual_pairs != expected_pairs or len(set(actual_pairs)) != 6:
        raise RuntimeError(
            f"model pair list mismatch: expected={expected_pairs}, actual={actual_pairs}"
        )

    expected_model_keys = {
        "checkpoint",
        "checkpoint_epoch",
        "checkpoint_sha256",
        "checkpoint_version",
        "class_name",
        "elapsed_seconds",
        "loaded_tensors",
        "model_kind",
        "model_tensors",
    }
    for model, pair in zip(models, expected_pairs):
        if set(model) != expected_model_keys:
            raise RuntimeError(f"model record key set mismatch: {pair}")
        expected_checkpoint = Path("checkpoints") / _CHECKPOINT_NAMES[pair]
        if model["checkpoint"] != expected_checkpoint.as_posix():
            raise RuntimeError(f"checkpoint path mismatch: {pair}")
        checkpoint_path = REPO_ROOT / expected_checkpoint
        if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
            raise RuntimeError(f"checkpoint is not a regular file: {expected_checkpoint}")
        if model["checkpoint_sha256"] != _sha256(checkpoint_path):
            raise RuntimeError(f"checkpoint SHA-256 mismatch: {expected_checkpoint}")
        loaded_tensors = model["loaded_tensors"]
        model_tensors = model["model_tensors"]
        if (
            isinstance(loaded_tensors, bool)
            or isinstance(model_tensors, bool)
            or not isinstance(loaded_tensors, int)
            or not isinstance(model_tensors, int)
            or loaded_tensors <= 0
            or loaded_tensors != model_tensors
        ):
            raise RuntimeError(f"loaded_tensors/model_tensors mismatch: {pair}")
        elapsed_seconds = model["elapsed_seconds"]
        if (
            isinstance(elapsed_seconds, bool)
            or not isinstance(elapsed_seconds, (int, float))
            or not np.isfinite(elapsed_seconds)
            or elapsed_seconds < 0
        ):
            raise RuntimeError(f"invalid elapsed_seconds: {pair}")
        if not isinstance(model["checkpoint_version"], str):
            raise RuntimeError(f"invalid checkpoint_version: {pair}")

    input_manifest = manifest["input"]
    expected_input = {
        "kind": "deterministic_synthetic_minimal_track",
        "frames_per_class": 5,
        "points_per_frame": 320,
        "classes": ["Vehicle", "Pedestrian", "Cyclist"],
        "scope": "checkpoint execution smoke test only",
        "benchmark_eligible": False,
        "note": "Synthetic input is not Waymo accuracy evidence.",
    }
    if input_manifest != expected_input:
        raise RuntimeError("input manifest contract mismatch")

    npz_path = artifact_dir / "results.npz"
    npz_arrays = _load_validated_npz_arrays(npz_path)
    npz_validation = artifacts["results_npz"]["validation"]
    if set(npz_validation) != {"arrays", "bytes", "sha256"}:
        raise RuntimeError("results.npz validation key set mismatch")
    if (
        isinstance(npz_validation["arrays"], bool)
        or not isinstance(npz_validation["arrays"], int)
        or npz_validation["arrays"] != 18
    ):
        raise RuntimeError("results.npz validation array count mismatch")
    if (
        isinstance(npz_validation["bytes"], bool)
        or not isinstance(npz_validation["bytes"], int)
        or npz_validation["bytes"] != npz_path.stat().st_size
    ):
        raise RuntimeError("results.npz validation byte count mismatch")

    class_record_keys = {
        "input_boxes",
        "geometry_boxes",
        "position_boxes",
        "combined_boxes",
        "reference_boxes",
        "input_metrics_against_synthetic_reference",
        "combined_metrics_against_synthetic_reference",
    }
    box_fields = (
        "input_boxes",
        "geometry_boxes",
        "position_boxes",
        "combined_boxes",
        "reference_boxes",
    )
    pose_columns = (0, 1, 2, 6)
    for class_name in ("Vehicle", "Pedestrian", "Cyclist"):
        record = manifest["classes"][class_name]
        if not isinstance(record, dict) or set(record) != class_record_keys:
            raise RuntimeError(f"class record key set mismatch: {class_name}")
        prefix = class_name.lower()
        for field in box_fields:
            json_value = record[field]
            if (
                not isinstance(json_value, list)
                or len(json_value) != 5
                or any(not isinstance(row, list) or len(row) != 7 for row in json_value)
                or any(
                    isinstance(item, bool) or not isinstance(item, (int, float))
                    for row in json_value
                    for item in row
                )
            ):
                raise RuntimeError(f"invalid JSON box array: {class_name}/{field}")
            json_array = np.asarray(json_value, dtype=np.float64)
            if not np.isfinite(json_array).all():
                raise RuntimeError(f"non-finite JSON box array: {class_name}/{field}")
            npz_array = npz_arrays[f"{prefix}_{field}"]
            if not np.array_equal(json_array, npz_array):
                raise RuntimeError(f"JSON/NPZ array mismatch: {class_name}/{field}")

        input_boxes = npz_arrays[f"{prefix}_input_boxes"]
        geometry_boxes = npz_arrays[f"{prefix}_geometry_boxes"]
        position_boxes = npz_arrays[f"{prefix}_position_boxes"]
        combined_boxes = npz_arrays[f"{prefix}_combined_boxes"]
        reference_boxes = npz_arrays[f"{prefix}_reference_boxes"]
        if not np.array_equal(
            geometry_boxes[:, pose_columns], input_boxes[:, pose_columns]
        ):
            raise RuntimeError(f"geometry boxes must retain input pose: {class_name}")
        if not np.array_equal(
            combined_boxes[:, pose_columns], position_boxes[:, pose_columns]
        ):
            raise RuntimeError(f"combined boxes must use PRM pose: {class_name}")
        if not np.array_equal(combined_boxes[:, 3:6], geometry_boxes[:, 3:6]):
            raise RuntimeError(f"combined boxes must use GRM size: {class_name}")

        expected_metrics = {
            "input_metrics_against_synthetic_reference": _box_metrics(
                input_boxes, reference_boxes
            ),
            "combined_metrics_against_synthetic_reference": _box_metrics(
                combined_boxes, reference_boxes
            ),
        }
        for metric_name, expected_values in expected_metrics.items():
            actual_values = record[metric_name]
            if actual_values != expected_values:
                raise RuntimeError(f"class metric mismatch: {class_name}/{metric_name}")

    visualization_path = artifact_dir / "visualization.png"
    visualization_validation = artifacts["visualization"]["validation"]
    expected_visualization_keys = {
        "width",
        "height",
        "channel_std_min",
        "non_background_fraction",
        "sha256",
    }
    if set(visualization_validation) != expected_visualization_keys:
        raise RuntimeError("visualization validation key set mismatch")
    actual_visualization_validation = validate_visualization(visualization_path)
    if visualization_validation != actual_visualization_validation:
        raise RuntimeError("visualization validation metadata mismatch")

    visualization_records = {
        class_name: {
            "points": npz_arrays[f"{class_name.lower()}_points"],
            "input_boxes": npz_arrays[f"{class_name.lower()}_input_boxes"],
            "reference_boxes": npz_arrays[f"{class_name.lower()}_reference_boxes"],
            "refined_boxes": npz_arrays[f"{class_name.lower()}_combined_boxes"],
        }
        for class_name in ("Vehicle", "Pedestrian", "Cyclist")
    }
    artifact_resolved = artifact_dir.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="detzero-reproduction-validator-") as scratch:
        scratch_path = Path(scratch).resolve(strict=True)
        if (
            artifact_resolved == scratch_path
            or artifact_resolved in scratch_path.parents
            or scratch_path in artifact_resolved.parents
        ):
            raise RuntimeError("validator render scratch overlaps artifact tree")
        rerendered_path = scratch_path / "visualization.png"
        render_visualization(visualization_records, rerendered_path)
        validate_visualization(rerendered_path)
        if rerendered_path.read_bytes() != visualization_path.read_bytes():
            raise RuntimeError(
                "visualization.png does not match deterministic NPZ re-render"
            )

    return manifest


def _device_metadata(device: str, torch_module: Any) -> dict[str, Any]:
    parsed_device = torch_module.device(device)
    if parsed_device.type != "cuda":
        return {
            "device": str(parsed_device),
            "device_index": None,
            "gpu": None,
            "compute_capability": None,
        }

    device_index = parsed_device.index
    if device_index is None:
        device_index = torch_module.cuda.current_device()
    indexed_device = torch_module.device("cuda", device_index)
    return {
        "device": str(indexed_device),
        "device_index": device_index,
        "gpu": torch_module.cuda.get_device_name(indexed_device),
        "compute_capability": list(
            torch_module.cuda.get_device_capability(indexed_device)
        ),
    }


def _move_batch_to_device(batch: dict[str, Any], device: str) -> None:
    import torch

    for key, value in batch.items():
        if not isinstance(value, np.ndarray):
            continue
        if key in {"frame_id", "sequence_name", "poses"}:
            continue
        batch[key] = torch.as_tensor(value, device=device).float()


def _revert_position_predictions(dataset: Any, prediction: dict[str, Any]) -> Any:
    return dataset.revert_to_each_frame(prediction)


def run_checkpoint(
    track: dict[str, Any],
    class_name: str,
    model_kind: str,
    device: str = "cuda",
    checkpoint_root: str | Path | None = None,
) -> InferenceResult:
    """Load every checkpoint tensor and execute one real model forward pass."""
    import torch

    _add_project_paths()
    from detzero_refine.models import build_network
    from detzero_utils.model_utils import _load_checkpoint

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but is not available")

    prepared = prepare_model_input(track, class_name, model_kind)
    prepared.config.MODEL.POST_PROCESSING.GENERATE_RECALL = False
    model = build_network(
        model_cfg=prepared.config.MODEL,
        dataset=prepared.dataset,
    )

    checkpoint_root = Path(checkpoint_root or REPO_ROOT / "checkpoints").resolve(
        strict=True
    )
    checkpoint_path = checkpoint_root / _CHECKPOINT_NAMES[(class_name, model_kind)]
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = _load_checkpoint(checkpoint_path, map_location=torch.device("cpu"))
    checkpoint_state = checkpoint["model_state"]
    model_state = model.state_dict()

    missing = sorted(set(model_state) - set(checkpoint_state))
    unexpected = sorted(set(checkpoint_state) - set(model_state))
    shape_mismatch = sorted(
        key
        for key in set(model_state) & set(checkpoint_state)
        if model_state[key].shape != checkpoint_state[key].shape
    )
    if missing or unexpected or shape_mismatch:
        raise RuntimeError(
            "checkpoint/model mismatch: "
            f"missing={missing}, unexpected={unexpected}, shape_mismatch={shape_mismatch}"
        )

    model.load_state_dict(checkpoint_state, strict=True)
    model.to(device)
    model.eval()
    _move_batch_to_device(prepared.batch, device)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        pred_dicts, _, _ = model(prepared.batch)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - started

    if model_kind == "geometry":
        pred_boxes_world = np.asarray(track["boxes_global"], dtype=np.float32).copy()
        predicted_size = np.asarray(pred_dicts["pred_boxes"][0, 3:6], dtype=np.float32)
        pred_boxes_world[:, 3:6] = predicted_size
    else:
        _, _, world_boxes, _ = _revert_position_predictions(
            prepared.dataset, pred_dicts
        )
        pred_boxes_world = np.asarray(world_boxes[0], dtype=np.float32)

    return InferenceResult(
        class_name=class_name,
        model_kind=model_kind,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=_sha256(checkpoint_path),
        checkpoint_epoch=checkpoint.get("epoch"),
        checkpoint_version=str(checkpoint.get("version", "unknown")),
        loaded_tensors=len(checkpoint_state),
        model_tensors=len(model_state),
        elapsed_seconds=elapsed_seconds,
        pred_boxes_world=pred_boxes_world,
    )


def _box_corners_bev(box: np.ndarray) -> np.ndarray:
    half_length, half_width = box[3] / 2, box[4] / 2
    corners = np.asarray(
        [
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, -half_width],
            [-half_length, half_width],
            [half_length, half_width],
        ],
        dtype=np.float32,
    )
    cosine, sine = np.cos(box[6]), np.sin(box[6])
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float32)
    return corners @ rotation.T + box[None, :2]


def _plot_box(axis, box: np.ndarray, color: str, label: str | None, style: str) -> None:
    corners = _box_corners_bev(box)
    axis.plot(
        corners[:, 0],
        corners[:, 1],
        color=color,
        linestyle=style,
        linewidth=2.0,
        label=label,
    )
    front = (corners[0] + corners[1]) / 2
    axis.plot(
        [box[0], front[0]],
        [box[1], front[1]],
        color=color,
        linewidth=1.5,
    )


def render_visualization(
    records: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """Render first-frame and whole-track BEV comparisons for every class."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    class_order = ("Vehicle", "Pedestrian", "Cyclist")
    missing = [name for name in class_order if name not in records]
    if missing:
        raise ValueError(f"visualization records missing classes: {missing}")

    figure, axes = plt.subplots(3, 2, figsize=(16, 12), dpi=140)
    figure.suptitle(
        "DetZero GRM + PRM checkpoint inference — minimal deterministic tracks",
        fontsize=16,
        fontweight="bold",
    )

    for row, class_name in enumerate(class_order):
        record = records[class_name]
        points = record["points"]
        input_boxes = np.asarray(record["input_boxes"])
        reference_boxes = np.asarray(record["reference_boxes"])
        refined_boxes = np.asarray(record["refined_boxes"])
        frame_count = len(input_boxes)

        first_axis = axes[row, 0]
        first_points = np.asarray(points[0])
        first_axis.scatter(
            first_points[:, 0],
            first_points[:, 1],
            c=first_points[:, 3],
            cmap="Greys",
            s=5,
            alpha=0.65,
            linewidths=0,
        )
        _plot_box(first_axis, input_boxes[0], "#d62728", "Input track box", "--")
        _plot_box(first_axis, refined_boxes[0], "#2ca02c", "GRM + PRM output", "-")
        _plot_box(first_axis, reference_boxes[0], "#1f77b4", "Synthetic reference", ":")
        first_axis.set_title(f"{class_name}: frame 0")
        first_axis.legend(loc="upper right", fontsize=8)

        track_axis = axes[row, 1]
        all_points = np.concatenate(points, axis=0)
        track_axis.scatter(
            all_points[:, 0],
            all_points[:, 1],
            c=all_points[:, 3],
            cmap="Greys",
            s=2,
            alpha=0.22,
            linewidths=0,
        )
        track_axis.plot(
            input_boxes[:, 0],
            input_boxes[:, 1],
            "o--",
            color="#d62728",
            markersize=4,
            label="Input centers",
        )
        track_axis.plot(
            refined_boxes[:, 0],
            refined_boxes[:, 1],
            "o-",
            color="#2ca02c",
            markersize=4,
            label="Refined centers",
        )
        track_axis.plot(
            reference_boxes[:, 0],
            reference_boxes[:, 1],
            "o:",
            color="#1f77b4",
            markersize=4,
            label="Reference centers",
        )
        for frame_index in range(frame_count):
            _plot_box(track_axis, input_boxes[frame_index], "#d62728", None, "--")
            _plot_box(track_axis, refined_boxes[frame_index], "#2ca02c", None, "-")
        track_axis.set_title(f"{class_name}: all {frame_count} frames")
        track_axis.legend(loc="upper left", fontsize=8)

        for axis in (first_axis, track_axis):
            axis.set_aspect("equal", adjustable="datalim")
            axis.set_xlabel("world x (m)")
            axis.set_ylabel("world y (m)")
            axis.grid(True, alpha=0.2)

    figure.text(
        0.5,
        0.01,
        "Real checkpoint forwards; deterministic synthetic input is a smoke test, not Waymo benchmark evidence.",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.96))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(output_path, facecolor="white")
    finally:
        plt.close(figure)
    _fsync_file(output_path)


def validate_visualization(path: Path) -> dict[str, Any]:
    """Decode the PNG and reject blank or undersized visual output."""
    from PIL import Image

    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"))

    height, width, _ = pixels.shape
    channel_std = pixels.reshape(-1, 3).std(axis=0)
    background = np.all(pixels >= 245, axis=2)
    result = {
        "width": int(width),
        "height": int(height),
        "channel_std_min": float(channel_std.min()),
        "non_background_fraction": float((~background).mean()),
        "sha256": _sha256(Path(path)),
    }
    if width < 1800 or height < 1200:
        raise RuntimeError(f"visualization is undersized: {width}x{height}")
    if result["channel_std_min"] <= 10 or result["non_background_fraction"] <= 0.01:
        raise RuntimeError(f"visualization appears blank: {result}")
    return result


def create_minimal_track(
    class_name: str,
    frame_count: int = 5,
    points_per_frame: int = 320,
) -> dict[str, Any]:
    """Create a deterministic, physically plausible object-centric track."""
    if class_name not in _CLASS_SPECS:
        raise ValueError(f"unsupported class: {class_name}")
    if frame_count < 3:
        raise ValueError("frame_count must be at least 3")
    if points_per_frame < 32:
        raise ValueError("points_per_frame must be at least 32")

    spec = _CLASS_SPECS[class_name]
    rng = np.random.default_rng(spec["seed"])
    size = np.asarray(spec["size"], dtype=np.float32)
    frame_index = np.arange(frame_count, dtype=np.float32)

    gt_boxes = np.zeros((frame_count, 7), dtype=np.float32)
    gt_boxes[:, 0] = frame_index * np.float32(spec["speed"])
    gt_boxes[:, 1] = 0.35 * np.sin(frame_index * 0.45)
    gt_boxes[:, 2] = size[2] / 2
    gt_boxes[:, 3:6] = size
    gt_boxes[:, 6] = 0.06 * frame_index

    boxes = gt_boxes.copy()
    boxes[:, 0] += 0.22 * np.sin(frame_index * 0.7 + 0.2)
    boxes[:, 1] -= 0.16 * np.cos(frame_index * 0.5)
    boxes[:, 2] += 0.04
    boxes[:, 3:6] *= np.asarray((1.10, 0.90, 1.06), dtype=np.float32)
    boxes[:, 6] += 0.10 * np.cos(frame_index * 0.4)

    points = []
    for box in gt_boxes:
        local = rng.uniform(-0.45, 0.45, size=(points_per_frame, 3)).astype(np.float32)
        local *= box[3:6]
        cosine = np.cos(box[6])
        sine = np.sin(box[6])
        rotated = local.copy()
        rotated[:, 0] = local[:, 0] * cosine - local[:, 1] * sine
        rotated[:, 1] = local[:, 0] * sine + local[:, 1] * cosine
        rotated += box[:3]
        intensity = rng.uniform(0.05, 0.95, size=(points_per_frame, 1)).astype(np.float32)
        points.append(np.concatenate((rotated, intensity), axis=1).astype(np.float32))

    sequence_name = "10231929575853664160_1160_000_1180_000"
    return {
        "sequence_name": sequence_name,
        "obj_id": f"minimal-{class_name.lower()}",
        "name": class_name,
        "boxes_global": boxes,
        "score": np.linspace(0.96, 0.82, frame_count, dtype=np.float32),
        "sample_idx": np.asarray([f"{index:04d}" for index in range(frame_count)]),
        "hit": np.ones(frame_count, dtype=np.int64),
        "pose": np.repeat(np.eye(4, dtype=np.float32)[None, ...], frame_count, axis=0),
        "state": "dynamic",
        "matched": np.ones(frame_count, dtype=np.bool_),
        "matched_tracklet": True,
        "pts": points,
        "gt_boxes_global": gt_boxes,
    }


def _box_metrics(predicted: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    center_error = predicted[:, :3] - reference[:, :3]
    heading_error = np.arctan2(
        np.sin(predicted[:, 6] - reference[:, 6]),
        np.cos(predicted[:, 6] - reference[:, 6]),
    )
    return {
        "center_rmse_m": float(np.sqrt(np.mean(np.square(center_error)))),
        "size_mae_m": float(np.mean(np.abs(predicted[:, 3:6] - reference[:, 3:6]))),
        "heading_mae_rad": float(np.mean(np.abs(heading_error))),
    }


def _load_validated_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    expected_keys = {
        f"{class_name}_{field}"
        for class_name in ("vehicle", "pedestrian", "cyclist")
        for field in (
            "input_boxes",
            "geometry_boxes",
            "position_boxes",
            "combined_boxes",
            "reference_boxes",
            "points",
        )
    }
    arrays: dict[str, np.ndarray] = {}
    try:
        with np.load(path, allow_pickle=False) as archive:
            actual_key_list = list(archive.files)
            actual_keys = set(actual_key_list)
            if len(actual_key_list) != 18 or actual_keys != expected_keys:
                raise RuntimeError(
                    f"NPZ key mismatch: missing={sorted(expected_keys - actual_keys)}, "
                    f"extra={sorted(actual_keys - expected_keys)}, "
                    f"entry_count={len(actual_key_list)}"
                )
            for key in sorted(expected_keys):
                value = archive[key]
                expected_shape = (5, 320, 4) if key.endswith("_points") else (5, 7)
                if value.dtype != np.dtype(np.float32):
                    raise RuntimeError(f"invalid NPZ dtype for {key}: {value.dtype}")
                if value.shape != expected_shape:
                    raise RuntimeError(f"invalid NPZ shape for {key}: {value.shape}")
                if not np.isfinite(value).all():
                    raise RuntimeError(f"non-finite NPZ payload: {key}")
                if key.endswith("_boxes") and (value[:, 3:6] <= 0).any():
                    raise RuntimeError(f"non-positive box size: {key}")
                arrays[key] = value.copy()
    except RuntimeError:
        raise
    except Exception as error:
        raise RuntimeError(f"invalid NPZ archive: {path}") from error
    return arrays


def _validate_npz(path: Path) -> dict[str, Any]:
    _load_validated_npz_arrays(path)
    return {"sha256": _sha256(path), "bytes": path.stat().st_size, "arrays": 18}


def _native_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _paths_overlap(first: Path, second: Path) -> bool:
    return (
        first == second
        or first in second.parents
        or second in first.parents
    )


def _validate_output_path(output_dir: Path) -> Path:
    """Reject symlinked, source-overlapping, or overly broad outputs."""
    output_dir = Path(os.path.abspath(os.fspath(output_dir)))

    current = Path(output_dir.anchor)
    for component in output_dir.parts[1:]:
        current /= component
        if os.path.lexists(current) and current.is_symlink():
            raise RuntimeError(f"output path contains symlink: {current}")

    repo_root = REPO_ROOT.resolve(strict=True)
    repo_output = repo_root / "output"
    under_repo_output = output_dir != repo_output and repo_output in output_dir.parents
    protected_trees = [
        repo_root,
        repo_root / "checkpoints",
        repo_root / "refining",
        repo_root / "utils",
        repo_root / "tests",
        repo_root / "docs",
        repo_root / "detection",
    ]
    for protected in protected_trees:
        if protected == repo_root and under_repo_output:
            continue
        if _paths_overlap(output_dir, protected):
            raise RuntimeError(
                f"output conflicts with protected repository tree: {protected}"
            )
    return output_dir


def _directory_identity(path: Path) -> tuple[int, int]:
    file_status = os.lstat(path)
    if not stat.S_ISDIR(file_status.st_mode):
        raise RuntimeError(f"stage is not a directory: {path}")
    return file_status.st_dev, file_status.st_ino


def _cleanup_owned_stage(stage: Path, identity: tuple[int, int]) -> bool:
    try:
        current_identity = _directory_identity(stage)
    except FileNotFoundError:
        return False
    except RuntimeError:
        return False
    if current_identity != identity:
        return False
    shutil.rmtree(stage)
    return True


def _publish_no_replace(stage: Path, destination: Path) -> None:
    """Atomically rename a stage directory without replacing any destination."""
    if os.name != "posix":
        raise RuntimeError("atomic no-replace publication requires a POSIX host")

    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as error:
        raise RuntimeError("renameat2 is unavailable; refusing fail-open publication") from error

    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(stage),
        at_fdcwd,
        os.fsencode(destination),
        rename_noreplace,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(
                error_number,
                os.strerror(error_number),
                destination,
            )
        raise OSError(error_number, os.strerror(error_number), destination)

    _fsync_directory(destination.parent)


def run_reproduction(
    output_dir: Path,
    device: str = "cuda",
) -> dict[str, Path]:
    """Execute all six checkpoints and atomically publish verified artifacts."""
    import torch

    output_dir = _validate_output_path(Path(output_dir))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise FileExistsError(f"refusing to replace existing output: {output_dir}")

    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-",
            dir=output_dir.parent,
        )
    )
    stage_identity = _directory_identity(stage)
    class_manifest: dict[str, Any] = {}
    model_manifest: list[dict[str, Any]] = []
    npz_payload: dict[str, np.ndarray] = {}
    visualization_records: dict[str, dict[str, Any]] = {}

    try:
        for class_name in ("Vehicle", "Pedestrian", "Cyclist"):
            track = create_minimal_track(class_name)
            geometry = run_checkpoint(track, class_name, "geometry", device=device)
            position = run_checkpoint(track, class_name, "position", device=device)

            input_boxes = np.asarray(track["boxes_global"], dtype=np.float32)
            reference_boxes = np.asarray(track["gt_boxes_global"], dtype=np.float32)
            geometry_boxes = geometry.pred_boxes_world.astype(np.float32, copy=True)
            position_boxes = position.pred_boxes_world.astype(np.float32, copy=True)
            combined_boxes = position_boxes.copy()
            combined_boxes[:, 3:6] = geometry_boxes[:, 3:6]
            if not np.isfinite(combined_boxes).all() or (combined_boxes[:, 3:6] <= 0).any():
                raise RuntimeError(f"invalid combined predictions for {class_name}")

            key_prefix = class_name.lower()
            npz_payload.update(
                {
                    f"{key_prefix}_input_boxes": input_boxes,
                    f"{key_prefix}_geometry_boxes": geometry_boxes,
                    f"{key_prefix}_position_boxes": position_boxes,
                    f"{key_prefix}_combined_boxes": combined_boxes,
                    f"{key_prefix}_reference_boxes": reference_boxes,
                    f"{key_prefix}_points": np.stack(track["pts"], axis=0),
                }
            )
            visualization_records[class_name] = {
                "points": track["pts"],
                "input_boxes": input_boxes,
                "reference_boxes": reference_boxes,
                "refined_boxes": combined_boxes,
            }
            class_manifest[class_name] = {
                "input_boxes": input_boxes.tolist(),
                "geometry_boxes": geometry_boxes.tolist(),
                "position_boxes": position_boxes.tolist(),
                "combined_boxes": combined_boxes.tolist(),
                "reference_boxes": reference_boxes.tolist(),
                "input_metrics_against_synthetic_reference": _box_metrics(
                    input_boxes, reference_boxes
                ),
                "combined_metrics_against_synthetic_reference": _box_metrics(
                    combined_boxes, reference_boxes
                ),
            }

            for inference in (geometry, position):
                model_manifest.append(
                    {
                        "class_name": inference.class_name,
                        "model_kind": inference.model_kind,
                        "checkpoint": str(
                            inference.checkpoint_path.relative_to(REPO_ROOT)
                        ),
                        "checkpoint_sha256": inference.checkpoint_sha256,
                        "checkpoint_epoch": _native_scalar(inference.checkpoint_epoch),
                        "checkpoint_version": inference.checkpoint_version,
                        "loaded_tensors": inference.loaded_tensors,
                        "model_tensors": inference.model_tensors,
                        "elapsed_seconds": inference.elapsed_seconds,
                    }
                )

        npz_path = stage / "results.npz"
        _save_npz_fsync(npz_path, npz_payload)
        npz_validation = _validate_npz(npz_path)

        visualization_path = stage / "visualization.png"
        render_visualization(visualization_records, visualization_path)
        visualization_validation = validate_visualization(visualization_path)

        manifest = {
            "schema_version": 1,
            "producer": "refining/tools/reproduce_inference.py",
            "status": "passed",
            "model_forward_count": len(model_manifest),
            "input": {
                "kind": "deterministic_synthetic_minimal_track",
                "frames_per_class": 5,
                "points_per_frame": 320,
                "classes": ["Vehicle", "Pedestrian", "Cyclist"],
                "scope": "checkpoint execution smoke test only",
                "benchmark_eligible": False,
                "note": "Synthetic input is not Waymo accuracy evidence.",
            },
            "runtime": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                **_device_metadata(device, torch),
            },
            "source": {
                "script_sha256": _sha256(Path(__file__).resolve()),
                "checkpoint_loader_sha256": _sha256(
                    REPO_ROOT / "utils" / "detzero_utils" / "model_utils.py"
                ),
            },
            "models": model_manifest,
            "classes": class_manifest,
            "artifacts": {
                "results_npz": {
                    "path": "results.npz",
                    "validation": npz_validation,
                },
                "visualization": {
                    "path": "visualization.png",
                    "validation": visualization_validation,
                },
            },
        }
        json_path = stage / "results.json"
        json_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
        with json_path.open("xb") as stream:
            stream.write(json_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        parsed_manifest = load_json_strict(json_path)
        if parsed_manifest != manifest:
            raise RuntimeError("results.json read-back mismatch")

        _fsync_directory(stage)
        validate_reproduction_artifacts(
            stage, expect_unaccepted_marker=False
        )

        unaccepted_marker = stage / ".UNACCEPTED"
        with unaccepted_marker.open("xb") as stream:
            stream.write(b"canonical post-publication validation pending\n")
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(stage)

        _publish_no_replace(stage, output_dir)
    except Exception:
        _cleanup_owned_stage(stage, stage_identity)
        raise

    canonical_json = output_dir / "results.json"
    canonical_npz = output_dir / "results.npz"
    canonical_visualization = output_dir / "visualization.png"
    validate_reproduction_artifacts(
        output_dir, expect_unaccepted_marker=True
    )

    canonical_marker = output_dir / ".UNACCEPTED"
    if not canonical_marker.is_file():
        raise RuntimeError("canonical unaccepted marker is missing")
    canonical_marker.unlink()
    _fsync_directory(output_dir)

    return {
        "output_dir": output_dir,
        "results_json": canonical_json,
        "results_npz": canonical_npz,
        "visualization": canonical_visualization,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all DetZero GRM/PRM checkpoints on minimal deterministic tracks."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output" / "inference_reproduction",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    artifacts = run_reproduction(args.output_dir, device=args.device)
    print(json.dumps({key: str(value.resolve()) for key, value in artifacts.items()}, indent=2))


if __name__ == "__main__":
    main()
