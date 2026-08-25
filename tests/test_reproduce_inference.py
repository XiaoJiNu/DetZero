from pathlib import Path
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "refining" / "tools"))

import reproduce_inference as reproduction  # noqa: E402
from reproduce_inference import (  # noqa: E402
    _publish_no_replace,
    create_minimal_track,
    prepare_model_input,
    render_visualization,
    run_checkpoint,
    run_reproduction,
    validate_visualization,
)


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_valid_artifact(root, *, with_marker=False):
    root.mkdir()
    npz_payload = {}
    class_manifest = {}
    visualization_records = {}

    for class_name in ("Vehicle", "Pedestrian", "Cyclist"):
        track = create_minimal_track(class_name)
        prefix = class_name.lower()
        input_boxes = track["boxes_global"].astype(np.float32, copy=True)
        reference_boxes = track["gt_boxes_global"].astype(np.float32, copy=True)
        geometry_boxes = input_boxes.copy()
        geometry_boxes[:, 3:6] *= np.float32(0.95)
        position_boxes = input_boxes.copy()
        position_boxes[:, :3] += np.asarray((0.03, -0.02, 0.01), dtype=np.float32)
        position_boxes[:, 6] += np.float32(0.04)
        combined_boxes = position_boxes.copy()
        combined_boxes[:, 3:6] = geometry_boxes[:, 3:6]
        points = np.stack(track["pts"]).astype(np.float32, copy=False)

        arrays = {
            "input_boxes": input_boxes,
            "geometry_boxes": geometry_boxes,
            "position_boxes": position_boxes,
            "combined_boxes": combined_boxes,
            "reference_boxes": reference_boxes,
            "points": points,
        }
        npz_payload.update({f"{prefix}_{key}": value for key, value in arrays.items()})
        class_manifest[class_name] = {
            key: arrays[key].tolist()
            for key in (
                "input_boxes",
                "geometry_boxes",
                "position_boxes",
                "combined_boxes",
                "reference_boxes",
            )
        }
        class_manifest[class_name].update(
            {
                "input_metrics_against_synthetic_reference": reproduction._box_metrics(
                    input_boxes, reference_boxes
                ),
                "combined_metrics_against_synthetic_reference": reproduction._box_metrics(
                    combined_boxes, reference_boxes
                ),
            }
        )
        visualization_records[class_name] = {
            "points": points,
            "input_boxes": input_boxes,
            "reference_boxes": reference_boxes,
            "refined_boxes": combined_boxes,
        }

    npz_path = root / "results.npz"
    np.savez_compressed(npz_path, **npz_payload)
    visualization_path = root / "visualization.png"
    render_visualization(visualization_records, visualization_path)

    models = []
    tensor_counts = {"geometry": 104, "position": 112}
    for class_name in ("Vehicle", "Pedestrian", "Cyclist"):
        for model_kind in ("geometry", "position"):
            checkpoint = Path("checkpoints") / reproduction._CHECKPOINT_NAMES[
                (class_name, model_kind)
            ]
            checkpoint_path = REPO_ROOT / checkpoint
            models.append(
                {
                    "class_name": class_name,
                    "model_kind": model_kind,
                    "checkpoint": checkpoint.as_posix(),
                    "checkpoint_sha256": _file_sha256(checkpoint_path),
                    "checkpoint_epoch": 30,
                    "checkpoint_version": "fixture",
                    "loaded_tensors": tensor_counts[model_kind],
                    "model_tensors": tensor_counts[model_kind],
                    "elapsed_seconds": 0.01,
                }
            )

    manifest = {
        "schema_version": 1,
        "producer": "refining/tools/reproduce_inference.py",
        "status": "passed",
        "model_forward_count": 6,
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
            "torch": "fixture",
            "torch_cuda": None,
            "device": "cpu",
            "device_index": None,
            "gpu": None,
            "compute_capability": None,
        },
        "source": {
            "script_sha256": _file_sha256(reproduction.__file__),
            "checkpoint_loader_sha256": _file_sha256(
                REPO_ROOT / "utils" / "detzero_utils" / "model_utils.py"
            ),
        },
        "models": models,
        "classes": class_manifest,
        "artifacts": {
            "results_npz": {
                "path": "results.npz",
                "validation": reproduction._validate_npz(npz_path),
            },
            "visualization": {
                "path": "visualization.png",
                "validation": validate_visualization(visualization_path),
            },
        },
    }
    (root / "results.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if with_marker:
        (root / ".UNACCEPTED").write_text("pending\n", encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def strict_artifact_template(tmp_path_factory):
    return _write_valid_artifact(tmp_path_factory.mktemp("strict-artifact") / "artifact")


def test_strict_json_loader_rejects_duplicate_keys(tmp_path):
    manifest_path = tmp_path / "results.json"
    manifest_path.write_text(
        '{"status": "failed", "status": "passed"}\n', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="duplicate JSON key.*status"):
        reproduction.load_json_strict(manifest_path)


def test_cuda_device_metadata_uses_explicit_torch_device_index():
    import torch

    calls = []

    class FakeCuda:
        @staticmethod
        def current_device():  # pragma: no cover - explicit cuda:1 must bypass this
            raise AssertionError("current_device must not be queried for cuda:1")

        @staticmethod
        def get_device_name(device):
            calls.append(("name", device))
            return "GPU number one"

        @staticmethod
        def get_device_capability(device):
            calls.append(("capability", device))
            return (9, 1)

    class FakeTorch:
        device = torch.device
        cuda = FakeCuda()

    metadata = reproduction._device_metadata("cuda:1", FakeTorch)

    assert metadata == {
        "device": "cuda:1",
        "device_index": 1,
        "gpu": "GPU number one",
        "compute_capability": [9, 1],
    }
    assert calls == [
        ("name", torch.device("cuda:1")),
        ("capability", torch.device("cuda:1")),
    ]


def test_strict_artifact_validator_accepts_complete_stage(strict_artifact_template):
    manifest = reproduction.validate_reproduction_artifacts(
        strict_artifact_template, expect_unaccepted_marker=False
    )

    assert manifest["status"] == "passed"
    assert manifest["model_forward_count"] == 6


def _copy_artifact(template, destination):
    return Path(shutil.copytree(template, destination))


def _read_manifest(artifact):
    return json.loads((Path(artifact) / "results.json").read_text(encoding="utf-8"))


def _write_manifest(artifact, manifest):
    (Path(artifact) / "results.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _rewrite_npz_and_reseal(artifact, mutate):
    artifact = Path(artifact)
    npz_path = artifact / "results.npz"
    with np.load(npz_path, allow_pickle=False) as archive:
        payload = {key: archive[key].copy() for key in archive.files}
    mutate(payload)
    np.savez_compressed(npz_path, **payload)
    manifest = _read_manifest(artifact)
    validation = manifest["artifacts"]["results_npz"]["validation"]
    validation.update(
        {
            "arrays": len(payload),
            "bytes": npz_path.stat().st_size,
            "sha256": _file_sha256(npz_path),
        }
    )
    _write_manifest(artifact, manifest)
    return payload, manifest


def test_strict_artifact_validator_rejects_extra_file(strict_artifact_template, tmp_path):
    artifact = _copy_artifact(strict_artifact_template, tmp_path / "extra-file")
    (artifact / "extra.txt").write_text("undeclared\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="artifact file set mismatch.*extra.txt"):
        reproduction.validate_reproduction_artifacts(
            artifact, expect_unaccepted_marker=False
        )


def test_strict_artifact_validator_enforces_canonical_marker(
    strict_artifact_template, tmp_path
):
    artifact = _copy_artifact(strict_artifact_template, tmp_path / "missing-marker")

    with pytest.raises(RuntimeError, match="artifact file set mismatch.*UNACCEPTED"):
        reproduction.validate_reproduction_artifacts(
            artifact, expect_unaccepted_marker=True
        )


def test_strict_artifact_validator_rejects_manifest_artifact_hash_drift(
    strict_artifact_template, tmp_path
):
    artifact = _copy_artifact(strict_artifact_template, tmp_path / "artifact-hash")
    manifest = _read_manifest(artifact)
    manifest["artifacts"]["results_npz"]["validation"]["sha256"] = "0" * 64
    _write_manifest(artifact, manifest)

    with pytest.raises(RuntimeError, match="results.npz SHA-256 mismatch"):
        reproduction.validate_reproduction_artifacts(
            artifact, expect_unaccepted_marker=False
        )


def test_strict_artifact_validator_rejects_source_hash_drift(
    strict_artifact_template, tmp_path
):
    artifact = _copy_artifact(strict_artifact_template, tmp_path / "source-hash")
    manifest = _read_manifest(artifact)
    manifest["source"]["script_sha256"] = "0" * 64
    _write_manifest(artifact, manifest)

    with pytest.raises(RuntimeError, match="producer script SHA-256 mismatch"):
        reproduction.validate_reproduction_artifacts(
            artifact, expect_unaccepted_marker=False
        )


def test_strict_artifact_validator_rejects_empty_execution_claims(
    strict_artifact_template, tmp_path
):
    artifact = _copy_artifact(strict_artifact_template, tmp_path / "empty-claims")
    manifest = _read_manifest(artifact)
    manifest["models"] = []
    manifest["classes"] = {}
    manifest["model_forward_count"] = 0
    _write_manifest(artifact, manifest)

    with pytest.raises(RuntimeError, match="model_forward_count must be exactly 6"):
        reproduction.validate_reproduction_artifacts(
            artifact, expect_unaccepted_marker=False
        )


def test_strict_artifact_validator_rejects_duplicate_model_pair(
    strict_artifact_template, tmp_path
):
    artifact = _copy_artifact(strict_artifact_template, tmp_path / "duplicate-model")
    manifest = _read_manifest(artifact)
    manifest["models"][-1] = dict(manifest["models"][0])
    _write_manifest(artifact, manifest)

    with pytest.raises(RuntimeError, match="model pair list mismatch"):
        reproduction.validate_reproduction_artifacts(
            artifact, expect_unaccepted_marker=False
        )


def test_strict_artifact_validator_rejects_checkpoint_hash_drift(
    strict_artifact_template, tmp_path
):
    artifact = _copy_artifact(strict_artifact_template, tmp_path / "checkpoint-hash")
    manifest = _read_manifest(artifact)
    manifest["models"][0]["checkpoint_sha256"] = "0" * 64
    _write_manifest(artifact, manifest)

    with pytest.raises(RuntimeError, match="checkpoint SHA-256 mismatch"):
        reproduction.validate_reproduction_artifacts(
            artifact, expect_unaccepted_marker=False
        )


@pytest.mark.parametrize(
    ("case", "expected_error"),
    [
        ("float64_points", "invalid NPZ dtype for vehicle_points"),
        ("extra_key", "NPZ key mismatch"),
        ("wrong_shape", "invalid NPZ shape for vehicle_input_boxes"),
        ("non_finite", "non-finite NPZ payload: vehicle_points"),
        ("non_positive_size", "non-positive box size: vehicle_input_boxes"),
    ],
)
def test_strict_artifact_validator_rejects_invalid_npz_contract(
    strict_artifact_template, tmp_path, case, expected_error
):
    artifact = _copy_artifact(strict_artifact_template, tmp_path / case)

    def mutate(payload):
        if case == "float64_points":
            payload["vehicle_points"] = payload["vehicle_points"].astype(np.float64)
        elif case == "extra_key":
            payload["extra"] = np.zeros((1,), dtype=np.float32)
        elif case == "wrong_shape":
            payload["vehicle_input_boxes"] = payload["vehicle_input_boxes"][:4]
        elif case == "non_finite":
            payload["vehicle_points"][0, 0, 0] = np.nan
        elif case == "non_positive_size":
            payload["vehicle_input_boxes"][0, 3] = 0.0
        else:  # pragma: no cover - parametrization is closed above
            raise AssertionError(case)

    _rewrite_npz_and_reseal(artifact, mutate)

    with pytest.raises(RuntimeError, match=expected_error):
        reproduction.validate_reproduction_artifacts(
            artifact, expect_unaccepted_marker=False
        )


def test_strict_artifact_validator_binds_json_arrays_to_npz(
    strict_artifact_template, tmp_path
):
    artifact = _copy_artifact(strict_artifact_template, tmp_path / "json-array-drift")
    manifest = _read_manifest(artifact)
    manifest["classes"]["Vehicle"]["input_boxes"][0][0] += 12345.0
    _write_manifest(artifact, manifest)

    with pytest.raises(RuntimeError, match="JSON/NPZ array mismatch: Vehicle/input_boxes"):
        reproduction.validate_reproduction_artifacts(
            artifact, expect_unaccepted_marker=False
        )


def test_strict_artifact_validator_rejects_geometry_pose_drift_even_when_json_matches(
    strict_artifact_template, tmp_path
):
    artifact = _copy_artifact(strict_artifact_template, tmp_path / "geometry-pose")

    def mutate(payload):
        payload["vehicle_geometry_boxes"][:, 0] += np.float32(12345.0)

    payload, manifest = _rewrite_npz_and_reseal(artifact, mutate)
    manifest["classes"]["Vehicle"]["geometry_boxes"] = payload[
        "vehicle_geometry_boxes"
    ].tolist()
    _write_manifest(artifact, manifest)

    with pytest.raises(RuntimeError, match="geometry boxes must retain input pose: Vehicle"):
        reproduction.validate_reproduction_artifacts(
            artifact, expect_unaccepted_marker=False
        )


def test_strict_artifact_validator_rejects_combined_center_tamper_when_json_matches(
    strict_artifact_template, tmp_path
):
    artifact = _copy_artifact(strict_artifact_template, tmp_path / "combined-center")

    def mutate(payload):
        payload["vehicle_combined_boxes"][:, 0] += np.float32(12345.0)

    payload, manifest = _rewrite_npz_and_reseal(artifact, mutate)
    manifest["classes"]["Vehicle"]["combined_boxes"] = payload[
        "vehicle_combined_boxes"
    ].tolist()
    _write_manifest(artifact, manifest)

    with pytest.raises(RuntimeError, match="combined boxes must use PRM pose: Vehicle"):
        reproduction.validate_reproduction_artifacts(
            artifact, expect_unaccepted_marker=False
        )


def test_strict_artifact_validator_rejects_duplicate_status_in_artifact(
    strict_artifact_template, tmp_path
):
    artifact = _copy_artifact(strict_artifact_template, tmp_path / "duplicate-status")
    json_path = artifact / "results.json"
    contents = json_path.read_text(encoding="utf-8")
    contents = contents.replace(
        '  "status": "passed"\n}',
        '  "status": "failed",\n  "status": "passed"\n}',
        1,
    )
    assert contents.count('"status"') == 2
    json_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key.*status"):
        reproduction.validate_reproduction_artifacts(
            artifact, expect_unaccepted_marker=False
        )


def test_strict_artifact_validator_rejects_non_data_png(
    strict_artifact_template, tmp_path
):
    from PIL import Image

    artifact = _copy_artifact(strict_artifact_template, tmp_path / "checkerboard")
    rows, columns = np.indices((1680, 2240))
    checkerboard = ((rows // 40 + columns // 40) % 2 * 255).astype(np.uint8)
    rgb = np.repeat(checkerboard[:, :, None], 3, axis=2)
    png_path = artifact / "visualization.png"
    Image.fromarray(rgb, mode="RGB").save(png_path)
    manifest = _read_manifest(artifact)
    manifest["artifacts"]["visualization"]["validation"] = validate_visualization(
        png_path
    )
    _write_manifest(artifact, manifest)

    with pytest.raises(RuntimeError, match="does not match deterministic NPZ re-render"):
        reproduction.validate_reproduction_artifacts(
            artifact, expect_unaccepted_marker=False
        )


def test_strict_artifact_validator_rerenders_in_private_external_scratch(
    strict_artifact_template, monkeypatch
):
    observed = []
    real_renderer = reproduction.render_visualization

    def recording_renderer(records, output_path):
        output_path = Path(output_path)
        observed.append(
            {
                "path": output_path,
                "parent_mode": os.stat(output_path.parent).st_mode & 0o777,
            }
        )
        return real_renderer(records, output_path)

    monkeypatch.setattr(reproduction, "render_visualization", recording_renderer)
    reproduction.validate_reproduction_artifacts(
        strict_artifact_template, expect_unaccepted_marker=False
    )

    assert len(observed) == 1
    scratch_path = observed[0]["path"].parent.resolve()
    artifact_path = strict_artifact_template.resolve()
    assert artifact_path not in (scratch_path, *scratch_path.parents)
    assert observed[0]["parent_mode"] & 0o077 == 0
    assert not scratch_path.exists()


@pytest.mark.parametrize("class_name", ["Vehicle", "Pedestrian", "Cyclist"])
def test_create_minimal_track_is_deterministic_and_schema_complete(class_name):
    first = create_minimal_track(class_name, frame_count=5, points_per_frame=320)
    second = create_minimal_track(class_name, frame_count=5, points_per_frame=320)

    expected_keys = {
        "sequence_name",
        "obj_id",
        "name",
        "boxes_global",
        "score",
        "sample_idx",
        "hit",
        "pose",
        "state",
        "matched",
        "matched_tracklet",
        "pts",
        "gt_boxes_global",
    }
    assert set(first) == expected_keys
    assert first["name"] == class_name
    assert first["boxes_global"].shape == (5, 7)
    assert first["gt_boxes_global"].shape == (5, 7)
    assert first["pose"].shape == (5, 4, 4)
    assert first["sample_idx"].tolist() == ["0000", "0001", "0002", "0003", "0004"]
    assert len(first["pts"]) == 5

    for first_points, second_points in zip(first["pts"], second["pts"]):
        assert first_points.shape == (320, 4)
        assert first_points.dtype == np.float32
        assert np.isfinite(first_points).all()
        assert np.array_equal(first_points, second_points)

    for key in ("boxes_global", "gt_boxes_global", "score", "pose"):
        assert np.array_equal(first[key], second[key])


@pytest.mark.parametrize(
    ("class_name", "model_kind", "expected_shape", "expected_channels"),
    [
        ("Vehicle", "geometry", (1, 3, 256, 4), 11),
        ("Pedestrian", "position", (1, 200, 256, 32), 32),
        ("Cyclist", "position", (1, 200, 256, 35), 35),
    ],
)
def test_prepare_model_input_uses_official_feature_shapes(
    class_name,
    model_kind,
    expected_shape,
    expected_channels,
):
    track = create_minimal_track(class_name)
    prepared = prepare_model_input(track, class_name, model_kind)

    if model_kind == "geometry":
        assert prepared.batch["geo_query_points"].shape == expected_shape
        assert prepared.batch["geo_memory_points"].shape == (1, 4096, expected_channels)
        assert prepared.config.MODEL.QUERY_POINT_DIMS == expected_channels
    else:
        assert prepared.batch["pos_query_points"].shape == expected_shape
        assert prepared.batch["pos_memory_points"].shape == (1, 200, 48, expected_channels)
        assert prepared.config.MODEL.QUERY_POINT_DIMS == expected_channels

    assert prepared.dataset.training is False
    assert prepared.dataset.tta is False


def test_position_world_reversion_matches_non_identity_pose_heading_oracle():
    reproduction._add_project_paths()
    from detzero_refine.datasets.waymo.waymo_position_dataset import (
        WaymoPositionDataset,
    )

    local_boxes = np.asarray(
        [
            [1.2, -0.4, 0.3, 4.2, 1.7, 1.5, 0.25],
            [-0.6, 2.1, -0.2, 3.9, 1.5, 1.4, -2.8],
        ],
        dtype=np.float64,
    )
    init_box = np.asarray(
        [10.0, -4.0, 1.2, 4.0, 1.6, 1.5, 0.7], dtype=np.float64
    )
    poses = []
    for yaw, translation in (
        (-0.35, (3.0, 5.0, -1.0)),
        (1.1, (-2.0, 7.0, 0.5)),
    ):
        cosine, sine = np.cos(yaw), np.sin(yaw)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = np.asarray(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
        )
        pose[:3, 3] = translation
        poses.append(pose)

    prediction = {
        "pred_boxes": local_boxes[None, ...],
        "pos_init_box": init_box[None, ...],
        "pose": np.asarray(poses)[None, ...],
        "gt_pos_trajectory": local_boxes[None, ...],
    }
    dataset = WaymoPositionDataset.__new__(WaymoPositionDataset)

    lidar_boxes, _, world_boxes, _ = reproduction._revert_position_predictions(
        dataset, prediction
    )

    expected_world = np.asarray(
        [
            [
                11.175497722625732,
                -3.532875680923462,
                1.5,
                4.2,
                1.7,
                1.5,
                0.95,
            ],
            [
                8.188237565755845,
                -2.7803619563579556,
                1.0,
                3.9,
                1.5,
                1.4,
                -2.1,
            ],
        ]
    )
    expected_lidar = np.asarray(
        [
            [
                10.605743836859098,
                -5.212190332833099,
                2.5,
                4.2,
                1.7,
                1.5,
                1.3,
            ],
            [
                -4.094985515581854,
                -13.51616655419825,
                0.5,
                3.9,
                1.5,
                1.4,
                -3.2,
            ],
        ]
    )
    np.testing.assert_allclose(world_boxes[0], expected_world, rtol=0, atol=1e-12)
    np.testing.assert_allclose(lidar_boxes[0], expected_lidar, rtol=0, atol=1e-12)


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("class_name", ["Vehicle", "Pedestrian", "Cyclist"])
@pytest.mark.parametrize("model_kind", ["geometry", "position"])
def test_run_checkpoint_loads_all_weights_and_returns_finite_world_boxes(
    class_name,
    model_kind,
):
    track = create_minimal_track(class_name)
    result = run_checkpoint(track, class_name, model_kind)

    assert result.loaded_tensors == result.model_tensors
    assert result.loaded_tensors > 100
    assert len(result.checkpoint_sha256) == 64
    assert result.pred_boxes_world.shape == (5, 7)
    assert np.isfinite(result.pred_boxes_world).all()
    assert (result.pred_boxes_world[:, 3:6] > 0).all()
    assert np.isfinite(result.elapsed_seconds)
    assert result.elapsed_seconds >= 0


def test_render_visualization_writes_a_nontrivial_source_bound_png(tmp_path):
    records = {}
    for class_name in ("Vehicle", "Pedestrian", "Cyclist"):
        track = create_minimal_track(class_name)
        records[class_name] = {
            "points": track["pts"],
            "input_boxes": track["boxes_global"],
            "reference_boxes": track["gt_boxes_global"],
            "refined_boxes": track["gt_boxes_global"],
        }

    output_path = tmp_path / "visualization.png"
    render_visualization(records, output_path)
    validation = validate_visualization(output_path)

    assert validation["width"] >= 1800
    assert validation["height"] >= 1200
    assert validation["channel_std_min"] > 10
    assert validation["non_background_fraction"] > 0.01


def test_npz_writer_explicitly_fsyncs_file(tmp_path, monkeypatch):
    observed = []
    real_fsync = os.fsync

    def recording_fsync(file_descriptor):
        observed.append(Path(os.readlink(f"/proc/self/fd/{file_descriptor}")).name)
        return real_fsync(file_descriptor)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    output_path = tmp_path / "payload.npz"
    reproduction._save_npz_fsync(
        output_path, {"array": np.ones((2, 3), dtype=np.float32)}
    )

    assert output_path.is_file()
    assert "payload.npz" in observed


def test_visualization_writer_explicitly_fsyncs_file(tmp_path, monkeypatch):
    records = {}
    for class_name in ("Vehicle", "Pedestrian", "Cyclist"):
        track = create_minimal_track(class_name)
        records[class_name] = {
            "points": track["pts"],
            "input_boxes": track["boxes_global"],
            "reference_boxes": track["gt_boxes_global"],
            "refined_boxes": track["gt_boxes_global"],
        }
    observed = []
    real_fsync = os.fsync

    def recording_fsync(file_descriptor):
        observed.append(Path(os.readlink(f"/proc/self/fd/{file_descriptor}")).name)
        return real_fsync(file_descriptor)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    output_path = tmp_path / "visualization.png"
    render_visualization(records, output_path)

    assert "visualization.png" in observed


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="CUDA required")
def test_run_reproduction_publishes_all_six_forwards_and_valid_artifacts(
    tmp_path, monkeypatch
):
    fsynced_files = []
    fsynced_directories = []
    real_fsync = os.fsync

    def recording_fsync(file_descriptor):
        target = Path(os.readlink(f"/proc/self/fd/{file_descriptor}"))
        if stat.S_ISDIR(os.fstat(file_descriptor).st_mode):
            fsynced_directories.append(target.name)
        else:
            fsynced_files.append(target.name)
        return real_fsync(file_descriptor)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    output_dir = tmp_path / "reproduction"
    artifacts = run_reproduction(output_dir)

    assert artifacts["output_dir"] == output_dir
    assert artifacts["results_json"].is_file()
    assert artifacts["results_npz"].is_file()
    assert artifacts["visualization"].is_file()
    assert not (output_dir / ".UNACCEPTED").exists()
    assert {
        "results.npz",
        "visualization.png",
        "results.json",
        ".UNACCEPTED",
    }.issubset(fsynced_files)
    assert output_dir.name in fsynced_directories

    manifest = reproduction.load_json_strict(artifacts["results_json"])
    assert manifest["schema_version"] == 1
    assert manifest["input"]["kind"] == "deterministic_synthetic_minimal_track"
    assert manifest["model_forward_count"] == 6
    assert set(manifest["classes"]) == {"Vehicle", "Pedestrian", "Cyclist"}
    assert all(len(item["checkpoint_sha256"]) == 64 for item in manifest["models"])
    assert all(item["loaded_tensors"] == item["model_tensors"] for item in manifest["models"])

    with np.load(artifacts["results_npz"], allow_pickle=False) as archive:
        assert len(archive.files) == 18
        for class_name in ("vehicle", "pedestrian", "cyclist"):
            assert archive[f"{class_name}_input_boxes"].shape == (5, 7)
            assert archive[f"{class_name}_geometry_boxes"].shape == (5, 7)
            assert archive[f"{class_name}_position_boxes"].shape == (5, 7)
            assert archive[f"{class_name}_combined_boxes"].shape == (5, 7)
            assert archive[f"{class_name}_reference_boxes"].shape == (5, 7)
            assert archive[f"{class_name}_points"].shape == (5, 320, 4)

    validation = validate_visualization(artifacts["visualization"])
    assert validation == manifest["artifacts"]["visualization"]["validation"]


def test_one_command_wrapper_exposes_cli_help():
    wrapper = REPO_ROOT / "reproduce_inference.sh"
    completed = subprocess.run(
        [str(wrapper), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--output-dir" in completed.stdout
    assert "--device" in completed.stdout


def test_output_path_safety_rejects_existing_parent_symlink(tmp_path):
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlink.*linked-parent"):
        reproduction._validate_output_path(linked_parent / "new-output")

    assert not (real_parent / "new-output").exists()


def test_output_path_safety_rejects_source_tree_and_repo_ancestor(tmp_path):
    with pytest.raises(RuntimeError, match="protected repository tree"):
        reproduction._validate_output_path(REPO_ROOT / "tests" / "new-output")
    with pytest.raises(RuntimeError, match="protected repository tree"):
        reproduction._validate_output_path(REPO_ROOT.parent)

    allowed_repo_output = REPO_ROOT / "output" / f"unit-{tmp_path.name}"
    assert reproduction._validate_output_path(allowed_repo_output) == allowed_repo_output
    allowed_tmp_output = tmp_path / "new-output"
    assert reproduction._validate_output_path(allowed_tmp_output) == allowed_tmp_output


def test_output_path_safety_rejects_output_symlink(tmp_path):
    target = tmp_path / "target"
    output_link = tmp_path / "output-link"
    output_link.symlink_to(target, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlink.*output-link"):
        reproduction._validate_output_path(output_link)


def test_owned_stage_cleanup_refuses_replaced_inode(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    original_identity = reproduction._directory_identity(stage)
    preserved_original = tmp_path / "preserved-original"
    stage.rename(preserved_original)
    stage.mkdir()
    caller_file = stage / "caller.txt"
    caller_file.write_text("do not delete\n", encoding="utf-8")

    assert reproduction._cleanup_owned_stage(stage, original_identity) is False
    assert caller_file.read_text(encoding="utf-8") == "do not delete\n"
    assert preserved_original.is_dir()


def test_publish_no_replace_preserves_a_racing_empty_destination(tmp_path):
    stage = tmp_path / "stage"
    destination = tmp_path / "destination"
    stage.mkdir()
    (stage / "payload.txt").write_text("owned stage", encoding="utf-8")
    destination.mkdir()
    destination_inode = destination.stat().st_ino

    with pytest.raises(FileExistsError):
        _publish_no_replace(stage, destination)

    assert stage.is_dir()
    assert (stage / "payload.txt").read_text(encoding="utf-8") == "owned stage"
    assert destination.is_dir()
    assert destination.stat().st_ino == destination_inode
    assert list(destination.iterdir()) == []


def test_publish_no_replace_moves_stage_when_destination_is_absent(tmp_path):
    stage = tmp_path / "stage"
    destination = tmp_path / "destination"
    stage.mkdir()
    (stage / "payload.txt").write_text("published", encoding="utf-8")

    _publish_no_replace(stage, destination)

    assert not stage.exists()
    assert (destination / "payload.txt").read_text(encoding="utf-8") == "published"


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="CUDA required")
def test_post_publish_validation_failure_leaves_unaccepted_marker(tmp_path, monkeypatch):
    output_dir = tmp_path / "reproduction"
    real_validator = reproduction.validate_reproduction_artifacts
    invocations = []

    def fail_only_at_canonical_path(path, *, expect_unaccepted_marker):
        invocations.append((expect_unaccepted_marker, {item.name for item in Path(path).iterdir()}))
        if expect_unaccepted_marker:
            raise RuntimeError("injected canonical validation failure")
        return real_validator(
            path, expect_unaccepted_marker=expect_unaccepted_marker
        )

    monkeypatch.setattr(
        reproduction, "validate_reproduction_artifacts", fail_only_at_canonical_path
    )

    with pytest.raises(RuntimeError, match="injected canonical validation failure"):
        reproduction.run_reproduction(output_dir)

    assert invocations == [
        (False, {"results.json", "results.npz", "visualization.png"}),
        (
            True,
            {"results.json", "results.npz", "visualization.png", ".UNACCEPTED"},
        ),
    ]
    assert output_dir.is_dir()
    assert (output_dir / ".UNACCEPTED").is_file()
    assert (output_dir / "results.json").is_file()


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="CUDA required")
def test_run_reproduction_is_deterministic_across_two_clean_subprocesses(tmp_path):
    script = REPO_ROOT / "refining" / "tools" / "reproduce_inference.py"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    outputs = [tmp_path / "first", tmp_path / "second"]
    completed_runs = []

    for output in outputs:
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--output-dir",
                str(output),
                "--device",
                "cuda",
            ],
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        completed_runs.append(completed)

    assert len(completed_runs) == 2
    assert (outputs[0] / "results.npz").read_bytes() == (
        outputs[1] / "results.npz"
    ).read_bytes()
    assert (outputs[0] / "visualization.png").read_bytes() == (
        outputs[1] / "visualization.png"
    ).read_bytes()

    manifests = [
        reproduction.load_json_strict(output / "results.json") for output in outputs
    ]
    for manifest in manifests:
        for model in manifest["models"]:
            model.pop("elapsed_seconds")
    assert manifests[0] == manifests[1]
