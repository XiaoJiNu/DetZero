from pathlib import Path
import hashlib
import json
import logging
import os
import pickle
import shutil
import subprocess
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.external_detector.pipeline import (  # noqa: E402
    adapt_raw_predictions,
    merge_waymo_lidar_returns,
    open3dml_box_to_detzero,
    parse_waymo_frame,
    preprocess_tfrecord,
    publish_preprocessed_records,
    records_from_waymo_frames,
    rename_noreplace,
    load_raw_predictions,
    save_raw_predictions,
    sequence_name_from_tfrecord,
)


@pytest.mark.parametrize(
    ("package_name", "source"),
    [
        ("detzero_track", REPO_ROOT / "tracking" / "detzero_track" / "__init__.py"),
        ("detzero_refine", REPO_ROOT / "refining" / "detzero_refine" / "__init__.py"),
        ("detzero_det", REPO_ROOT / "detection" / "detzero_det" / "__init__.py"),
    ],
)
def test_source_packages_import_without_generated_version_file(
    tmp_path, package_name, source
):
    package = tmp_path / package_name
    package.mkdir()
    (package / "__init__.py").write_bytes(source.read_bytes())
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(tmp_path)!r}); "
        f"import {package_name} as package; "
        "print(package.__version__)"
    )

    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0.1.0+py0000000"


def test_source_bundle_binds_embedded_and_live_bytes(tmp_path):
    from tools.external_detector.run_stage_a import (
        capture_source_bundle,
        verify_source_bundle,
    )

    repo = tmp_path / "repo"
    source = repo / "tools" / "external_detector" / "worker.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    pytest_config = repo / "pytest.ini"
    pytest_config.write_text("[pytest]\nnorecursedirs = output\n", encoding="utf-8")
    gitignore = repo / ".gitignore"
    gitignore.write_text("*.so\n", encoding="utf-8")
    native = repo / "utils/detzero_utils/ops/example/native.so"
    native.parent.mkdir(parents=True)
    native.write_bytes(b"native-extension")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "add",
            str(source.relative_to(repo)),
            str(pytest_config.relative_to(repo)),
            str(gitignore.relative_to(repo)),
        ],
        cwd=repo,
        check=True,
    )
    bundle = tmp_path / "bundle"

    manifest = capture_source_bundle(repo, bundle)

    assert manifest["source_file_count"] == 4
    assert ".gitignore" in manifest["source_files"]
    assert "pytest.ini" in manifest["source_files"]
    assert "utils/detzero_utils/ops/example/native.so" in manifest["source_files"]
    assert verify_source_bundle(repo, bundle)["matches_current_workspace"] is True
    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="live source drift"):
        verify_source_bundle(repo, bundle)


def test_source_bundle_rejects_symlink_manifest(tmp_path):
    from tools.external_detector.run_stage_a import (
        capture_source_bundle,
        verify_source_bundle,
    )

    repo = tmp_path / "repo"
    source = repo / "tools" / "external_detector" / "worker.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", str(source.relative_to(repo))], cwd=repo, check=True)
    bundle = tmp_path / "bundle"
    capture_source_bundle(repo, bundle)
    manifest = bundle / "source_manifest.json"
    outside = tmp_path / "outside.json"
    outside.write_bytes(manifest.read_bytes())
    manifest.unlink()
    manifest.symlink_to(outside)

    with pytest.raises(ValueError, match="invalid source manifest"):
        verify_source_bundle(repo, bundle)


def test_validator_independently_verifies_source_bundle(tmp_path):
    from tools.external_detector.run_stage_a import capture_source_bundle
    from tools.external_detector.validate_stage_a import validate_source_provenance

    repo = tmp_path / "repo"
    source = repo / "tools" / "external_detector" / "worker.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    empty_package = source.parent / "__init__.py"
    empty_package.touch()
    pytest_config = repo / "pytest.ini"
    pytest_config.write_text("[pytest]\nnorecursedirs = output\n", encoding="utf-8")
    gitignore = repo / ".gitignore"
    gitignore.write_text("*.so\n", encoding="utf-8")
    native = repo / "utils/detzero_utils/ops/example/native.so"
    native.parent.mkdir(parents=True)
    native.write_bytes(b"native-extension")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "add",
            str(source.relative_to(repo)),
            str(empty_package.relative_to(repo)),
            str(pytest_config.relative_to(repo)),
            str(gitignore.relative_to(repo)),
        ],
        cwd=repo,
        check=True,
    )
    bundle = tmp_path / "bundle"
    capture_source_bundle(repo, bundle)

    assert validate_source_provenance(bundle, repo) == {
        "historical_provenance_verified": True,
        "matches_current_workspace": True,
        "source_file_count": 5,
        "source_tree_sha256": json.loads(
            (bundle / "source_manifest.json").read_text(encoding="utf-8")
        )["source_tree_sha256"],
    }


def test_source_bundle_replays_without_live_repo_borrowing(tmp_path):
    from tools.external_detector.run_stage_a import (
        capture_source_bundle,
        replay_source_bundle,
    )

    repo = tmp_path / "repo"
    package = repo / "tools" / "external_detector"
    package.mkdir(parents=True)
    (package / "worker.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "entry.py").write_text(
        "import argparse\nfrom tools.external_detector.worker import VALUE\n"
        "argparse.ArgumentParser().parse_args()\nassert VALUE == 1\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    bundle = tmp_path / "bundle"
    capture_source_bundle(repo, bundle)

    result = replay_source_bundle(
        bundle,
        repo,
        [("entry", "tools/external_detector/entry.py", Path(sys.executable))],
    )

    assert result["bundle_replay_complete"] is True
    assert result["entries"] == ["tools/external_detector/entry.py"]
    assert len(result["verifier_sha256"]) == 64


def test_stage_a_validator_independently_replays_source_bundle(tmp_path):
    from tools.external_detector.run_stage_a import capture_source_bundle
    from tools.external_detector.validate_stage_a import validate_bundle_replay

    repo = tmp_path / "repo"
    package = repo / "tools" / "external_detector"
    package.mkdir(parents=True)
    (package / "worker.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "entry.py").write_text(
        "from tools.external_detector.worker import VALUE\n"
        "assert VALUE == 1\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    bundle = tmp_path / "bundle"
    capture_source_bundle(repo, bundle)

    result = validate_bundle_replay(
        bundle,
        repo,
        [("tools/external_detector/entry.py", Path(sys.executable))],
    )

    assert result["bundle_replay_complete"] is True
    assert result["entries"] == ["tools/external_detector/entry.py"]


def test_stage_a_launcher_bootstrap_has_no_first_party_imports():
    import ast

    script = REPO_ROOT / "tools" / "external_detector" / "run_stage_a.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))
    first_party = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "tools" or node.module.startswith("tools."):
                first_party.append(node.module)
        elif isinstance(node, ast.Import):
            first_party.extend(
                alias.name
                for alias in node.names
                if alias.name == "tools" or alias.name.startswith("tools.")
            )

    assert first_party == []


def test_stage_a_launcher_executes_stage_scripts_from_bundle():
    import inspect

    from tools.external_detector.run_stage_a import preflight, run_generation

    source = inspect.getsource(run_generation)
    preflight_source = inspect.getsource(preflight)

    assert 'execution_root = run_dir / "source_bundle/files"' in source
    assert 'str(REPO_ROOT / "tools/external_detector/' not in source
    assert '"--run-root"' in source
    assert '"--live-repo-root"' in source
    assert '"--pipeline-python"' in source
    assert '"--detector-python"' in source
    assert '"--preprocess-python"' in source
    assert '"--checkpoint-root"' in source
    assert '"--root_path"' in source
    assert '"--split"' in source and '"test"' in source
    assert "tools/external_detector/compare_stage_a_runs.py" in source
    assert "tools/external_detector/compare_stage_a_runs.py" in preflight_source


def test_stage_a_launcher_preserves_virtualenv_entrypoints():
    import inspect

    from tools.external_detector.run_stage_a import preflight, run_generation

    preflight_source = inspect.getsource(preflight)
    generation_source = inspect.getsource(run_generation)

    assert "pipeline_executable = Path(args.pipeline_python).absolute()" in preflight_source
    assert "detector_executable = Path(args.detector_python).absolute()" in preflight_source
    assert "preprocess_executable = Path(args.preprocess_python).absolute()" in preflight_source
    assert "pipeline_python = str(Path(args.pipeline_python).absolute())" in generation_source
    assert "detector_python = str(Path(args.detector_python).absolute())" in generation_source
    assert "preprocess_python = str(Path(args.preprocess_python).absolute())" in generation_source


def test_stage_a_launcher_exposes_one_command_contract():
    script = REPO_ROOT / "tools" / "external_detector" / "run_stage_a.py"

    result = subprocess.run(
        [sys.executable, "-B", str(script), "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    for option in (
        "--input-tfrecord",
        "--run-dir",
        "--receipt",
        "--detector-config",
        "--detector-checkpoint",
        "--detector-python",
        "--preprocess-python",
        "--tracking-config",
        "--refining-checkpoint-root",
        "--open3dml-root",
        "--refining-device",
        "--role",
        "--preflight",
    ):
        assert option in result.stdout


def test_stage_a_launcher_preflight_is_read_only(tmp_path):
    from tools.external_detector.run_stage_a import capture_source_bundle

    historical = REPO_ROOT / "output" / "waymo-stage-a-20260825-181832-CST"
    detector_manifest_path = historical / "detector-full-0199" / "detector_manifest.json"
    preprocess_manifest_path = historical / "data" / "waymo" / "preprocess_manifest.json"
    detector_python = Path("/data/software/venvs/detzero-open3dml/bin/python")
    preprocess_python = Path("/data/software/venvs/detzero-waymo-tf213/bin/python")
    open3dml_root = Path(
        "/data/software/venvs/detzero-open3dml/lib/python3.10/site-packages/open3d/_ml3d"
    )
    required = [
        detector_manifest_path,
        preprocess_manifest_path,
        detector_python,
        preprocess_python,
        open3dml_root,
        *(REPO_ROOT / "checkpoints").glob("*.pth"),
    ]
    if len(required) != 11 or not all(path.exists() for path in required):
        pytest.skip("local Stage A preflight assets are unavailable")
    detector = json.loads(detector_manifest_path.read_text(encoding="utf-8"))
    preprocess = json.loads(preprocess_manifest_path.read_text(encoding="utf-8"))
    run_dir = tmp_path / "run"
    receipt = tmp_path / "receipt.json"
    script = REPO_ROOT / "tools" / "external_detector" / "run_stage_a.py"

    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(script),
            "--input-tfrecord",
            preprocess["input_tfrecord"],
            "--run-dir",
            str(run_dir),
            "--receipt",
            str(receipt),
            "--detector-config",
            detector["config"],
            "--detector-config-sha256",
            detector["config_sha256"],
            "--detector-checkpoint",
            detector["checkpoint"],
            "--detector-checkpoint-sha256",
            detector["checkpoint_sha256"],
            "--tracking-config",
            str(REPO_ROOT / "tracking/tools/cfgs/tk_model_cfgs/waymo_detzero_track.yaml"),
            "--refining-checkpoint-root",
            str(REPO_ROOT / "checkpoints"),
            "--open3dml-root",
            str(open3dml_root),
            "--open3dml-commit",
            detector["open3dml_commit"],
            "--pipeline-python",
            sys.executable,
            "--detector-python",
            str(detector_python),
            "--preprocess-python",
            str(preprocess_python),
            "--role",
            "A",
            "--non-commercial-research",
            "--waymo-terms-accepted",
            "--preflight",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    preflight = json.loads(result.stdout)
    assert preflight["status"] == "PREFLIGHT_PASS"
    verifier = preflight["bundle_replay_verifier"]
    assert Path(verifier["path"]).name == "verify-python-bundle-replay.py"
    assert len(verifier["sha256"]) == 64
    assert Path(preflight["pipeline_python"]["executable"]) == Path(sys.executable).absolute()
    assert Path(preflight["detector_python"]["executable"]) == detector_python.absolute()
    assert Path(preflight["preprocess_python"]["executable"]) == preprocess_python.absolute()
    assert preflight["runtime_probes"] == {
        "detector": True,
        "pipeline": True,
        "preprocess": True,
    }
    assert not run_dir.exists()
    assert not receipt.exists()
    bundle = tmp_path / "source-bundle"
    capture_source_bundle(REPO_ROOT, bundle)
    bundle_command = list(result.args)
    bundle_command[2] = str(bundle / "files/tools/external_detector/run_stage_a.py")
    bundle_result = subprocess.run(
        bundle_command,
        cwd=bundle / "files",
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert bundle_result.returncode == 0, bundle_result.stderr
    rebundle = tmp_path / "source-rebundle"
    rebundle_manifest = capture_source_bundle(bundle / "files", rebundle)
    assert rebundle_manifest["source_file_count"] == preflight["source_file_count"]
    assert rebundle_manifest["source_tree_sha256"] == preflight["source_tree_sha256"]
    assert json.loads(bundle_result.stdout)["status"] == "PREFLIGHT_PASS"
    assert not run_dir.exists()
    assert not receipt.exists()


def test_stage_a_launcher_refuses_existing_run_before_inputs(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    sentinel = run_dir / "foreign.txt"
    sentinel.write_text("foreign\n", encoding="utf-8")
    missing = tmp_path / "missing"
    script = REPO_ROOT / "tools" / "external_detector" / "run_stage_a.py"

    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(script),
            "--input-tfrecord",
            str(missing),
            "--run-dir",
            str(run_dir),
            "--receipt",
            str(tmp_path / "receipt.json"),
            "--detector-config",
            str(missing),
            "--detector-config-sha256",
            "0" * 64,
            "--detector-checkpoint",
            str(missing),
            "--detector-checkpoint-sha256",
            "0" * 64,
            "--tracking-config",
            str(missing),
            "--refining-checkpoint-root",
            str(missing),
            "--open3dml-root",
            str(missing),
            "--open3dml-commit",
            "0" * 40,
            "--detector-python",
            sys.executable,
            "--preprocess-python",
            sys.executable,
            "--role",
            "A",
            "--non-commercial-research",
            "--waymo-terms-accepted",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "FileExistsError" in result.stderr
    assert sentinel.read_text(encoding="utf-8") == "foreign\n"


def test_stage_a_launcher_records_stage_log(tmp_path):
    from tools.external_detector.run_stage_a import run_command

    log = tmp_path / "stage.log"

    run_command(
        [sys.executable, "-B", "-c", "print('stage-ok')"],
        log,
        os.environ.copy(),
    )

    assert log.read_text(encoding="utf-8") == "stage-ok\n"


def _seal_full_stage_a_run(
    root,
    role,
    payload=b"release-payload",
    *,
    frame_count=199,
    source_paths=None,
):
    """Build a structurally complete Stage-A run tree plus a matching ledger."""
    import shutil

    if root.exists() or root.is_symlink():
        shutil.rmtree(root)
    root.mkdir()
    sequence = "seq"
    if source_paths is None:
        from tools.external_detector.run_stage_a import _source_paths

        source_paths = _source_paths(REPO_ROOT)
    (root / "logs").mkdir(parents=True)
    (root / "data/waymo/waymo_processed_data" / f"segment-{sequence}").mkdir(
        parents=True
    )
    (root / "detector").mkdir()
    (root / "adapter").mkdir()
    (root / "tracking").mkdir()
    (root / "refining/result").mkdir(parents=True)
    (root / "final").mkdir()
    (root / "visuals").mkdir()
    (root / "source_bundle/files").mkdir(parents=True)
    (root / "command.sh").write_text(f"run {role}\n", encoding="utf-8")
    for name in (
        "01-preprocess",
        "02-detector",
        "03-adapter",
        "04-tracking",
        "05-refining",
        "06-final",
        "07-visuals",
    ):
        (root / "logs" / f"{name}.log").write_text(
            f"volatile {role} {name}\n", encoding="utf-8"
        )
    preprocess_manifest = {
        "schema_version": "detzero-stage-a-preprocess-v1",
        "sequence_name": sequence,
        "frame_count": frame_count,
        "frame_ids": list(range(frame_count)),
        "timestamp_first": 1000,
        "timestamp_last": 1001,
        "input_tfrecord": str(root / "data/waymo/segment.tfrecord"),
        "input_tfrecord_bytes": 0,
        "input_tfrecord_sha256": "0" * 64,
        "usage": "non-commercial-research",
        "waymo_terms_accepted": True,
    }
    (root / "data/waymo/preprocess_manifest.json").write_text(
        json.dumps(preprocess_manifest), encoding="utf-8"
    )
    (root / "data/waymo/ImageSets").mkdir(parents=True)
    (root / "data/waymo/ImageSets/test.txt").write_text(
        f"{sequence}\n", encoding="utf-8"
    )
    with (root / "data/waymo/waymo_processed_data/segment-seq/seq.pkl").open(
        "wb"
    ) as stream:
        pickle.dump([], stream)
    for frame_id in range(frame_count):
        (root / f"data/waymo/waymo_processed_data/segment-seq/{frame_id:04d}.npy").write_bytes(
            b"x"
        )
    (root / "detector/detector_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "detzero-stage-a-detector-v1",
                "backend": "open3dml",
                "sequence_name": sequence,
                "frame_count": frame_count,
                "box_count": 0,
                "class_counts": {},
                "device": "cpu",
            }
        ),
        encoding="utf-8",
    )
    (root / "detector/raw_predictions.npz").write_bytes(b"x")
    (root / "adapter/adapter_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "detzero-stage-a-adapter-v1",
                "sequence_name": sequence,
                "frame_count": frame_count,
                "box_count": 0,
                "class_counts": {},
            }
        ),
        encoding="utf-8",
    )
    (root / "adapter/detzero_result.pkl").write_bytes(b"x")
    (root / "tracking/tracking.pkl").write_bytes(b"x")
    (root / "tracking/dropped.pkl").write_bytes(b"x")
    for kind in ("geometry", "position"):
        for class_name in ("Vehicle", "Pedestrian", "Cyclist"):
            (root / f"refining/result/{class_name}_{kind}.pkl").write_bytes(b"x")
    (root / "refining/refining_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "detzero-stage-a-refining-v1",
                "tracking": {"path": "../tracking/tracking.pkl", "sha256": "0" * 64},
                "waymo_root": "../data/waymo",
                "classes": {},
                "model_forward_count": 0,
                "crm": {"status": "NOT_EXECUTED_NO_CRM_BY_DESIGN"},
                "score_policy": "TRACKING_SCORE_PASSTHROUGH",
                "outputs": {},
            }
        ),
        encoding="utf-8",
    )
    (root / "final/final_arrays.npz").write_bytes(b"x")
    (root / "final/final_frame_grm_prm_score_passthrough.pkl").write_bytes(b"x")
    (root / "final/final_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "detzero-stage-a-final-v1",
                "frame_count": frame_count,
                "track_count": 0,
                "score_policy": "TRACKING_SCORE_PASSTHROUGH",
                "crm": {"status": "NOT_EXECUTED_NO_CRM_BY_DESIGN"},
                "inputs": {},
                "outputs": {},
                "classes": {},
            }
        ),
        encoding="utf-8",
    )
    (root / "final/final_track.pkl").write_bytes(payload)
    for frame_id in range(frame_count):
        (root / f"visuals/{frame_id:04d}.png").write_bytes(b"x")
    (root / "visuals/render_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "detzero-stage-a-render-v1",
                "sequence_name": sequence,
                "frame_count": frame_count,
                "frames": {},
                "inputs": {},
            }
        ),
        encoding="utf-8",
    )
    source_rows = {}
    for relative in source_paths:
        source_payload = b"{}\n" if relative.endswith(".json") else (relative + "\n").encode()
        source_path = root / "source_bundle/files" / relative
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(source_payload)
        source_rows[relative] = {
            "bytes": len(source_payload),
            "sha256": hashlib.sha256(source_payload).hexdigest(),
        }
    (root / "source_bundle/source_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "detzero-stage-a-source-v1",
                "origin_project_root": str(root),
                "source_file_count": len(source_rows),
                "source_tree_sha256": "a" * 64,
                "source_files": source_rows,
            }
        ),
        encoding="utf-8",
    )
    metadata = {
        "schema_version": "detzero-stage-a-run-v1",
        "role": role,
        "expected_frames": frame_count,
        "source": {"source_tree_sha256": "a" * 64},
        "bundle_replay": {"bundle_replay_complete": True},
        "preflight": {"role": role, "input": "same"},
        "runtime": {"python": "same"},
        "command": {
            "argv": [
                "runner",
                "--run-dir",
                str(root),
                "--receipt",
                str(root.parent / f"{root.name}-receipt.json"),
                "--role",
                role,
            ],
            "script": "command.sh",
            "script_sha256": "a" * 64,
        },
        "commands": [[str(root)]],
        "logs": {
            f"0{index}-stage": {
                "path": f"logs/0{index}-preprocess.log",
                "bytes": 1 if role == "A" else 2,
                "sha256": "0" * 64,
            }
            for index in range(1, 8)
        },
    }
    (root / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    rows = {}
    for path in root.rglob("*"):
        if path.is_file() and path.name != "run_ledger.json":
            payload_bytes = path.read_bytes()
            rows[path.relative_to(root).as_posix()] = {
                "bytes": len(payload_bytes),
                "sha256": hashlib.sha256(payload_bytes).hexdigest(),
            }
    (root / "run_ledger.json").write_text(
        json.dumps(
            {
                "schema_version": "detzero-stage-a-run-ledger-v1",
                "source_tree_sha256": "a" * 64,
                "file_count": len(rows),
                "files": rows,
            }
        ),
        encoding="utf-8",
    )


def test_stage_a_comparator_covers_every_ledger_path(tmp_path):
    from tools.external_detector.compare_stage_a_runs import compare_stage_a_runs

    run_a = tmp_path / "a"
    run_b = tmp_path / "b"
    _seal_full_stage_a_run(run_a, "A")
    _seal_full_stage_a_run(run_b, "B")
    _publish_pass_receipt(run_a, "A")
    _publish_pass_receipt(run_b, "B")

    report = compare_stage_a_runs(run_a, run_b)
    assert report["status"] == "PASS"
    assert report["classified_file_count"] == report["expected_file_count"] - 1
    assert report["exact_file_count"] == report["classified_file_count"] - 17
    assert report["semantic_json_file_count"] == 8
    assert report["stable_metadata_file_count"] == 1
    assert report["volatile_execution_file_count"] == 8
    assert report["classified_file_count"] == sum(
        report[key]
        for key in (
            "exact_file_count",
            "semantic_json_file_count",
            "stable_metadata_file_count",
            "volatile_execution_file_count",
        )
    )

    metadata_path = run_b / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["commands"].append(["unexpected-command"])
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    ledger_path = run_b / "run_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    payload_bytes = metadata_path.read_bytes()
    ledger["files"]["run_metadata.json"] = {
        "bytes": len(payload_bytes),
        "sha256": hashlib.sha256(payload_bytes).hexdigest(),
    }
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    _publish_pass_receipt(run_b, "B")
    with pytest.raises(ValueError, match="stable run metadata differs"):
        compare_stage_a_runs(run_a, run_b)
    _seal_full_stage_a_run(run_b, "B")
    _publish_pass_receipt(run_b, "B")

    (run_b / "run_ledger.json").unlink()
    _seal_full_stage_a_run(run_b, "B", b"changed-payload")
    _publish_pass_receipt(run_b, "B")
    with pytest.raises(ValueError, match="exact artifact mismatch: final/final_track.pkl"):
        compare_stage_a_runs(run_a, run_b)


def test_stage_a_comparator_rejects_self_described_nonformal_profiles(tmp_path):
    from tools.external_detector.compare_stage_a_runs import compare_stage_a_runs

    for name, options in (
        ("two-frame", {"frame_count": 2}),
        ("shrunk-source", {"source_paths": ["__init__.py"]}),
    ):
        run_a = tmp_path / f"{name}-a"
        run_b = tmp_path / f"{name}-b"
        _seal_full_stage_a_run(run_a, "A", **options)
        _seal_full_stage_a_run(run_b, "B", **options)
        with pytest.raises(ValueError, match="formal Stage-A profile"):
            compare_stage_a_runs(run_a, run_b)


def test_stage_a_profile_allows_only_valid_unaccepted_marker_during_audit(tmp_path):
    from tools.external_detector.validate_stage_a import expected_stage_a_paths

    run = tmp_path / "run"
    _seal_full_stage_a_run(run, "A")
    marker = run / ".unaccepted"
    marker.write_bytes(b"UNACCEPTED\n")
    with pytest.raises(ValueError, match="expected Stage-A profile"):
        expected_stage_a_paths(run)
    assert expected_stage_a_paths(run, allow_unaccepted_marker=True)
    marker.write_bytes(b"REJECTED\n")
    with pytest.raises(ValueError, match="invalid acceptance marker"):
        expected_stage_a_paths(run, allow_unaccepted_marker=True)


def test_stage_a_comparator_rejects_incomplete_and_undeclared_runs(tmp_path):
    from tools.external_detector.compare_stage_a_runs import compare_stage_a_runs
    from tools.external_detector.validate_stage_a import expected_stage_a_paths

    complete = tmp_path / "complete"
    _seal_full_stage_a_run(complete, "A")
    _publish_pass_receipt(complete, "A")
    assert len(expected_stage_a_paths(complete)) == 559

    incomplete = tmp_path / "incomplete"
    _seal_full_stage_a_run(incomplete, "B")
    (incomplete / "detector" / "raw_predictions.npz").unlink()
    (incomplete / "run_ledger.json").unlink()
    _rebuild_ledger(incomplete)
    with pytest.raises(ValueError, match="does not match the expected Stage-A profile"):
        compare_stage_a_runs(complete, incomplete)

    undeclared = tmp_path / "undeclared"
    _seal_full_stage_a_run(undeclared, "B")
    (undeclared / "undeclared-empty-directory").mkdir()
    (undeclared / "run_ledger.json").unlink()
    _rebuild_ledger(undeclared)
    with pytest.raises(ValueError, match="undeclared empty directory"):
        compare_stage_a_runs(complete, undeclared)


def _rebuild_ledger(root):
    rows = {}
    for path in root.rglob("*"):
        if path.is_file() and path.name != "run_ledger.json":
            payload_bytes = path.read_bytes()
            rows[path.relative_to(root).as_posix()] = {
                "bytes": len(payload_bytes),
                "sha256": hashlib.sha256(payload_bytes).hexdigest(),
            }
    (root / "run_ledger.json").write_text(
        json.dumps(
            {
                "schema_version": "detzero-stage-a-run-ledger-v1",
                "source_tree_sha256": "a" * 64,
                "file_count": len(rows),
                "files": rows,
            }
        ),
        encoding="utf-8",
    )


def _publish_pass_receipt(root, role):
    """Write a valid external run receipt and PASS audits beside a sealed run."""
    from tools.external_detector.validate_stage_a import expected_stage_a_paths

    base = root.parent / f"{root.name}-receipt"
    receipt_path = base.with_suffix(".json")
    pre_audit = base.with_name(f"{base.name}-pre-audit.json")
    post_audit = base.with_name(f"{base.name}-post-audit.json")
    pre_log = base.with_name(f"{base.name}-pre-audit.log")
    post_log = base.with_name(f"{base.name}-post-audit.log")
    ledger = json.loads((root / "run_ledger.json").read_text(encoding="utf-8"))
    audit = {
        "schema_version": "detzero-stage-a-validation-v1",
        "passed": True,
        "artifact_validation_passed": True,
        "release_eligible": False,
        "release_blockers": ["NOT_EVALUATED_NO_GROUND_TRUTH"],
        "checks": {
            "run_ledger": {
                "file_count": ledger["file_count"],
                "source_tree_sha256": ledger["source_tree_sha256"],
            },
            "stage_a_profile": {
                "expected_file_count": len(expected_stage_a_paths(root))
            },
            "acceptance_marker": {"state": "UNACCEPTED"},
        },
        "inputs": {"run_root": str(root)},
        "expected_frames": 199,
        "error": None,
    }
    pre_audit.write_text(json.dumps(audit), encoding="utf-8")
    post_audit.write_text(json.dumps(audit), encoding="utf-8")
    pre_log.write_text("pre-ok\n", encoding="utf-8")
    post_log.write_text("post-ok\n", encoding="utf-8")
    ledger_path = root / "run_ledger.json"
    ledger_hash = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    receipt = {
        "schema_version": "detzero-stage-a-run-receipt-v1",
        "status": "PASS",
        "role": role,
        "run_dir": str(root),
        "destination_absent_before_launch": True,
        "source_tree_sha256": "a" * 64,
        "run_ledger_sha256": ledger_hash,
        "pre_audit_sha256": hashlib.sha256(pre_audit.read_bytes()).hexdigest(),
        "post_audit_sha256": hashlib.sha256(post_audit.read_bytes()).hexdigest(),
        "pre_audit_log_sha256": hashlib.sha256(pre_log.read_bytes()).hexdigest(),
        "post_audit_log_sha256": hashlib.sha256(post_log.read_bytes()).hexdigest(),
        "logs": {},
        "command": ["runner", "--role", role],
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")


def test_stage_a_comparator_requires_pass_receipts_for_both_runs(tmp_path):
    from tools.external_detector.compare_stage_a_runs import compare_stage_a_runs

    run_a = tmp_path / "a"
    run_b = tmp_path / "b"
    _seal_full_stage_a_run(run_a, "A")
    _seal_full_stage_a_run(run_b, "B")

    with pytest.raises(ValueError, match="missing run receipt"):
        compare_stage_a_runs(run_a, run_b)

    _publish_pass_receipt(run_a, "A")
    _publish_pass_receipt(run_b, "B")
    report = compare_stage_a_runs(run_a, run_b)
    assert report["status"] == "PASS"
    assert [entry["receipt"] for entry in report["receipts"]] == [
        str(run_a.parent / "a-receipt.json"),
        str(run_b.parent / "b-receipt.json"),
    ]

    _publish_pass_receipt(run_a, "A")
    _publish_pass_receipt(run_b, "B")
    (run_b.parent / "b-receipt.json").unlink()
    with pytest.raises(ValueError, match="missing run receipt"):
        compare_stage_a_runs(run_a, run_b)
    _publish_pass_receipt(run_a, "A")
    _publish_pass_receipt(run_b, "B")
    receipt_b = run_b.parent / "b-receipt.json"
    receipt_b.write_text(
        receipt_b.read_text(encoding="utf-8").replace('"role": "B"', '"role": "A"'),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not a PASS for role B"):
        compare_stage_a_runs(run_a, run_b)
    _publish_pass_receipt(run_a, "A")
    _publish_pass_receipt(run_b, "B")
    receipt_b = run_b.parent / "b-receipt.json"
    receipt = json.loads(receipt_b.read_text(encoding="utf-8"))
    receipt["destination_absent_before_launch"] = False
    receipt_b.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="destination was not absent before launch"):
        compare_stage_a_runs(run_a, run_b)


def test_stage_a_comparator_rejects_resealed_unbound_audit(tmp_path):
    from tools.external_detector.compare_stage_a_runs import compare_stage_a_runs

    run_a = tmp_path / "a"
    run_b = tmp_path / "b"
    _seal_full_stage_a_run(run_a, "A")
    _seal_full_stage_a_run(run_b, "B")
    _publish_pass_receipt(run_a, "A")
    _publish_pass_receipt(run_b, "B")
    audit_path = tmp_path / "b-receipt-pre-audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["inputs"]["run_root"] = str(tmp_path / "other-run")
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    receipt_path = tmp_path / "b-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["pre_audit_sha256"] = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="audit identity mismatch"):
        compare_stage_a_runs(run_a, run_b)


def test_stage_a_comparator_rejects_mid_compare_mutation(tmp_path, monkeypatch):
    import tools.external_detector.compare_stage_a_runs as comparator

    run_a = tmp_path / "a"
    run_b = tmp_path / "b"
    _seal_full_stage_a_run(run_a, "A")
    _seal_full_stage_a_run(run_b, "B")
    _publish_pass_receipt(run_a, "A")
    _publish_pass_receipt(run_b, "B")
    canonical = comparator._canonical
    mutated = False

    def mutate_after_payload_comparison(value, root):
        nonlocal mutated
        if (
            not mutated
            and root == run_b
            and isinstance(value, dict)
            and value.get("schema_version") == "detzero-stage-a-source-v1"
        ):
            (run_b / "final/final_track.pkl").write_bytes(b"mid-compare-tamper")
            mutated = True
        return canonical(value, root)

    monkeypatch.setattr(comparator, "_canonical", mutate_after_payload_comparison)
    with pytest.raises(ValueError, match="run changed during comparison"):
        comparator.compare_stage_a_runs(run_a, run_b)
    assert mutated


def test_stage_a_comparator_direct_cli_resolves_repository_modules():
    script = REPO_ROOT / "tools" / "external_detector" / "compare_stage_a_runs.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "ModuleNotFoundError" not in result.stderr


def test_restricted_pickle_loader_rejects_arbitrary_globals(tmp_path):
    from tools.external_detector.validate_stage_a import safe_load_pickle

    safe_path = tmp_path / "safe.pkl"
    with safe_path.open("wb") as stream:
        pickle.dump({"values": np.asarray([1, 2], dtype=np.float32)}, stream)
    assert safe_load_pickle(safe_path)["values"].tolist() == [1.0, 2.0]

    class Arbitrary:
        def __reduce__(self):
            return (eval, ("__import__('os').getcwd()",))

    unsafe_path = tmp_path / "unsafe.pkl"
    with unsafe_path.open("wb") as stream:
        pickle.dump(Arbitrary(), stream)
    with pytest.raises(ValueError, match="forbidden pickle global"):
        safe_load_pickle(unsafe_path)


def test_stage_a_formal_pickle_consumers_use_restricted_loader():
    consumers = (
        "tools/external_detector/adapt_open3dml_to_detzero.py",
        "tracking/detzero_track/datasets/waymo_dataset.py",
        "daemon/prepare_object_data.py",
        "tools/external_detector/run_stage_a_refining.py",
        "tools/external_detector/combine_grm_prm_no_crm.py",
        "tools/external_detector/render_waymo_sequence.py",
    )

    for relative in consumers:
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        assert "pickle.load(" not in source, relative
        assert "safe_load_pickle" in source, relative


def test_validator_binds_open3dml_commit_to_physical_source_tree(tmp_path, monkeypatch):
    from tools.external_detector.open3dml_provenance import OPEN3DML_SOURCE_LOCKS
    from tools.external_detector.validate_stage_a import (
        validate_open3dml_provenance,
    )

    source_root = tmp_path / "open3dml"
    source_root.mkdir()
    (source_root / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    digest = hashlib.sha256(
        b"model.py\0"
        + hashlib.sha256(b"VALUE = 1\n").hexdigest().encode("ascii")
        + b"\n"
    ).hexdigest()
    run_root = tmp_path / "run"
    detector_dir = run_root / "detector"
    detector_dir.mkdir(parents=True)
    commit = "f" * 40
    source_lock = {"source_file_count": 1, "source_tree_sha256": digest}
    source_files = {
        "model.py": {
            "bytes": len(b"VALUE = 1\n"),
            "sha256": hashlib.sha256(b"VALUE = 1\n").hexdigest(),
        }
    }
    monkeypatch.setitem(OPEN3DML_SOURCE_LOCKS, commit, source_lock)
    (run_root / "run_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "detzero-stage-a-run-v1",
                "preflight": {
                    "open3dml": {
                        "root": str(source_root),
                        "declared_commit": commit,
                        "source_file_count": 1,
                        "source_tree_sha256": digest,
                        "source_lock": source_lock,
                        "source_files": source_files,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (detector_dir / "detector_manifest.json").write_text(
        json.dumps({"open3dml_commit": commit}), encoding="utf-8"
    )

    assert validate_open3dml_provenance(run_root, detector_dir) == {
        "declared_commit": commit,
        "source_file_count": 1,
        "source_tree_sha256": digest,
        "source_lock": source_lock,
    }
    (source_root / "model.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Open3D-ML source identity mismatch"):
        validate_open3dml_provenance(run_root, detector_dir)


def test_validator_rejects_unlocked_open3dml_commit(tmp_path):
    from tools.external_detector.validate_stage_a import validate_open3dml_provenance

    source_root = tmp_path / "open3dml"
    source_root.mkdir()
    content = b"VALUE = 1\n"
    (source_root / "model.py").write_bytes(content)
    digest = hashlib.sha256(
        b"model.py\0" + hashlib.sha256(content).hexdigest().encode("ascii") + b"\n"
    ).hexdigest()
    run_root = tmp_path / "run"
    detector_dir = run_root / "detector"
    detector_dir.mkdir(parents=True)
    commit = "e" * 40
    (run_root / "run_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "detzero-stage-a-run-v1",
                "preflight": {
                    "open3dml": {
                        "root": str(source_root),
                        "declared_commit": commit,
                        "source_file_count": 1,
                        "source_tree_sha256": digest,
                        "source_lock": {
                            "source_file_count": 1,
                            "source_tree_sha256": digest,
                        },
                        "source_files": {
                            "model.py": {
                                "bytes": len(content),
                                "sha256": hashlib.sha256(content).hexdigest(),
                            }
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (detector_dir / "detector_manifest.json").write_text(
        json.dumps({"open3dml_commit": commit}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="unlocked Open3D-ML commit"):
        validate_open3dml_provenance(run_root, detector_dir)


def test_stage_a_snapshots_external_inputs_before_use(tmp_path):
    from tools.external_detector.run_stage_a import _snapshot_inputs

    source_root = tmp_path / "source"
    source_root.mkdir()
    rows = {}
    for name in (
        "input_tfrecord",
        "detector_config",
        "detector_checkpoint",
        "tracking_config",
    ):
        path = source_root / name
        payload = f"{name}\n".encode()
        path.write_bytes(payload)
        rows[name] = {
            "path": str(path),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    checkpoint = source_root / "vehicle_grm_model.pth"
    checkpoint.write_bytes(b"checkpoint\n")
    rows["refining_checkpoints"] = [
        {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        }
    ]
    open3dml_root = source_root / "open3dml"
    open3dml_rows = {}
    for relative, payload in {
        "__init__.py": b"",
        "torch/model.py": b"MODEL = True\n",
    }.items():
        path = open3dml_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        open3dml_rows[relative] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    preflight = {
        "inputs": rows,
        "open3dml": {"root": str(open3dml_root), "source_files": open3dml_rows},
    }

    snapshot = _snapshot_inputs(tmp_path / "run", preflight)
    (source_root / "input_tfrecord").write_bytes(b"changed\n")
    assert snapshot["input_tfrecord"].read_bytes() == b"input_tfrecord\n"
    assert snapshot["refining_checkpoint_root"].joinpath(
        "vehicle_grm_model.pth"
    ).read_bytes() == b"checkpoint\n"
    assert snapshot["open3dml_root"].joinpath("torch/model.py").read_bytes() == b"MODEL = True\n"
    (source_root / "input_tfrecord").write_bytes(b"input_tfrecord\n")

    outside = source_root / "outside"
    outside.write_bytes(b"outside\n")
    symlink_parent = tmp_path / "symlink-parent"
    symlink_parent.symlink_to(source_root, target_is_directory=True)
    bad = dict(rows["detector_config"])
    bad.update(
        path=str(symlink_parent / "outside"),
        bytes=outside.stat().st_size,
        sha256=hashlib.sha256(outside.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="invalid bounded regular file"):
        _snapshot_inputs(
            tmp_path / "bad-run",
            {**preflight, "inputs": {**rows, "detector_config": bad}},
        )

    invalid_run = tmp_path / "invalid-run"
    with pytest.raises(ValueError, match="preflight input schema"):
        _snapshot_inputs(invalid_run, {"inputs": {}})
    assert not (invalid_run / ".input-snapshot").exists()


def test_stage_a_runtime_environment_disables_unbound_open3dml_and_pyc(tmp_path, monkeypatch):
    from tools.external_detector.run_stage_a import _stage_environment

    run_dir = tmp_path / "run"
    monkeypatch.setenv("OPEN3D_ML_ROOT", str(tmp_path / "attacker"))
    with pytest.raises(ValueError, match="OPEN3D_ML_ROOT"):
        _stage_environment(run_dir)
    monkeypatch.delenv("OPEN3D_ML_ROOT")
    environment = _stage_environment(run_dir)
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert environment["PYTHONPYCACHEPREFIX"] == str(run_dir / ".python-cache")
    assert not (run_dir / ".python-cache").exists()


def test_generation_internal_provenance_is_relative_and_contained(tmp_path):
    from tools.external_detector.pipeline import generation_relative_provenance
    from tools.external_detector.validate_stage_a import (
        resolve_generation_provenance,
    )

    run_root = tmp_path / "run"
    manifest_dir = run_root / "detector"
    internal = run_root / "data" / "manifest.json"
    manifest_dir.mkdir(parents=True)
    internal.parent.mkdir()
    internal.write_text("{}\n", encoding="utf-8")

    assert generation_relative_provenance(internal, manifest_dir) == (
        "../data/manifest.json"
    )
    assert resolve_generation_provenance(
        manifest_dir, "../data/manifest.json"
    ) == internal.resolve()
    with pytest.raises(ValueError, match="must be relative"):
        resolve_generation_provenance(manifest_dir, str(internal.resolve()))
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="escapes generation root"):
        generation_relative_provenance(outside, manifest_dir)


def test_stage_a_acceptance_marker_removed_only_after_receipt_publish():
    import inspect

    from tools.external_detector.run_stage_a import run_generation

    source = inspect.getsource(run_generation)
    receipt_line = next(
        line
        for line in source.splitlines()
        if "_write_json_noreplace(receipt_path, receipt)" in line
    )
    readback_line = next(
        line
        for line in source.splitlines()
        if "load_json_strict(receipt_path) != receipt" in line
    )
    marker_line = next(
        line for line in source.splitlines() if "marker.unlink()" in line
    )
    assert source.index(receipt_line) < source.index(readback_line) < source.index(marker_line)
    assert "validate_run_ledger" not in source.split(marker_line, 1)[1]


def test_stage_a_validator_independently_checks_run_ledger(tmp_path):
    from tools.external_detector.run_stage_a import _build_run_ledger
    from tools.external_detector.validate_stage_a import validate_run_ledger

    run = tmp_path / "run"
    run.mkdir()
    payload = run / "payload.txt"
    payload.write_text("payload\n", encoding="utf-8")
    (run / "empty-package-marker.py").touch()
    _build_run_ledger(run, "a" * 64)

    assert validate_run_ledger(run) == {
        "file_count": 2,
        "source_tree_sha256": "a" * 64,
    }
    (run / "undeclared.txt").write_text("extra\n", encoding="utf-8")
    with pytest.raises(ValueError, match="run ledger path set mismatch"):
        validate_run_ledger(run)


def test_canonical_pytest_ignores_generation_source_bundles():
    probe_root = REPO_ROOT / "output" / ".pytest-source-bundle-probe"
    probe = probe_root / "source_bundle" / "files" / "tests" / "test_shadow_probe.py"
    if probe_root.exists() or probe_root.is_symlink():
        pytest.fail(f"foreign probe path already exists: {probe_root}")
    try:
        probe.parent.mkdir(parents=True)
        probe.write_text("def test_shadow_probe():\n    pass\n", encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-B", "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
            cwd=REPO_ROOT,
            env={
                **os.environ,
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": os.pathsep.join(
                    str(REPO_ROOT / path) for path in ("tracking", "refining", "utils")
                ),
            },
            capture_output=True,
            text=True,
        )
    finally:
        shutil.rmtree(probe_root, ignore_errors=True)

    assert result.returncode == 0, result.stderr
    assert "test_shadow_probe" not in result.stdout


def test_publish_preprocessed_records_preserves_stream_contract(tmp_path):
    sequence_name = sequence_name_from_tfrecord(
        "segment-123_456_with_camera_labels.tfrecord"
    )
    assert sequence_name == "123_456"

    records = (
        {
            "frame_id": frame_id,
            "timestamp": np.int64(1000 + frame_id),
            "pose": np.eye(4, dtype=np.float64),
            "points": np.asarray(
                [[frame_id + 0.5, 1.0, 2.0, 0.25, 0.5, -1.0]],
                dtype=np.float32,
            ),
            "num_points_of_each_lidar": [1, 0, 0, 0, 0],
        }
        for frame_id in range(2)
    )

    manifest = publish_preprocessed_records(
        records,
        tmp_path / "waymo",
        sequence_name=sequence_name,
        expected_frame_count=2,
    )

    sequence_dir = tmp_path / "waymo" / "waymo_processed_data" / "segment-123_456"
    assert sorted(path.name for path in sequence_dir.glob("*.npy")) == [
        "0000.npy",
        "0001.npy",
    ]
    for frame_id in range(2):
        points = np.load(sequence_dir / f"{frame_id:04d}.npy", allow_pickle=False)
        assert points.shape == (1, 6)
        assert points.dtype == np.float32
        assert np.isfinite(points).all()

    info_path = sequence_dir / "123_456.pkl"
    with info_path.open("rb") as stream:
        infos = pickle.load(stream)
    assert [info["sample_idx"] for info in infos] == [0, 1]
    assert [info["time_stamp"] for info in infos] == [1000, 1001]
    assert all(info["pose"].dtype == np.float64 for info in infos)
    assert all("annos" not in info for info in infos)
    assert (tmp_path / "waymo" / "ImageSets" / "test.txt").read_text() == "123_456\n"
    assert manifest["frame_count"] == 2
    assert manifest["frame_ids"] == [0, 1]
    assert json.loads(
        (tmp_path / "waymo" / "preprocess_manifest.json").read_text()
    ) == manifest

    with pytest.raises(FileExistsError):
        publish_preprocessed_records(
            (),
            tmp_path / "waymo",
            sequence_name=sequence_name,
            expected_frame_count=1,
        )


def test_records_from_waymo_frames_is_lazy_and_preserves_exact_types():
    observed = []

    class Pose:
        transform = np.eye(4, dtype=np.float64).reshape(-1).tolist()

    class Frame:
        def __init__(self, timestamp):
            self.timestamp_micros = timestamp
            self.pose = Pose()

    def extract(frame):
        observed.append(frame.timestamp_micros)
        return (
            np.asarray([[1, 2, 3, 0.5, 0.25, -1]], dtype=np.float32),
            [1, 0, 0, 0, 0],
        )

    records = records_from_waymo_frames((Frame(10), Frame(20)), extract)
    assert observed == []
    first = next(records)
    assert observed == [10]
    assert first["frame_id"] == 0
    assert first["timestamp"] == np.int64(10)
    assert first["pose"].dtype == np.float64
    assert first["points"].dtype == np.float32
    assert [record["frame_id"] for record in records] == [1]


def test_rename_noreplace_preserves_foreign_destination(tmp_path):
    source = tmp_path / "stage"
    target = tmp_path / "published"
    source.mkdir()
    (source / "ours").write_text("ours")
    target.mkdir()
    (target / "foreign").write_text("foreign")
    target_identity = (target.stat().st_dev, target.stat().st_ino)

    with pytest.raises(FileExistsError):
        rename_noreplace(source, target)

    assert source.is_dir()
    assert (target.stat().st_dev, target.stat().st_ino) == target_identity
    assert (target / "foreign").read_text() == "foreign"


def test_preprocess_tfrecord_binds_source_and_user_authorization(tmp_path):
    input_path = tmp_path / "segment-abc_with_camera_labels.tfrecord"
    source_identity_path = tmp_path / "logical" / input_path.name
    input_path.write_bytes(b"source-bytes")

    class Pose:
        transform = np.eye(4, dtype=np.float64).reshape(-1).tolist()

    class Frame:
        timestamp_micros = 123
        pose = Pose()

    def load_frames(path):
        assert path == input_path
        return iter([Frame()])

    def extract(_frame):
        return np.ones((1, 6), dtype=np.float32), [1, 0, 0, 0, 0]

    manifest = preprocess_tfrecord(
        input_path,
        tmp_path / "waymo",
        expected_frame_count=1,
        frame_loader=load_frames,
        point_extractor=extract,
        non_commercial_research=True,
        waymo_terms_accepted=True,
        source_identity_path=source_identity_path,
    )

    assert manifest["input_tfrecord_sha256"] == hashlib.sha256(b"source-bytes").hexdigest()
    assert manifest["input_tfrecord"] == str(source_identity_path.absolute())
    assert manifest["usage"] == "non-commercial-research"
    assert manifest["waymo_terms_accepted"] is True
    assert json.loads(
        (tmp_path / "waymo" / "preprocess_manifest.json").read_text()
    ) == manifest


def test_preprocess_cli_exposes_authorization_and_frame_scope():
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "external_detector" / "preprocess_waymo_test_segment.py"),
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--expected-frames" in result.stdout
    assert "--non-commercial-research" in result.stdout
    assert "--waymo-terms-accepted" in result.stdout


def test_open3dml_runner_cli_exposes_pinned_runtime_inputs():
    result = subprocess.run(
        [
            sys.executable,
            str(
                REPO_ROOT
                / "tools"
                / "external_detector"
                / "run_open3dml_waymo_pointpillars.py"
            ),
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    for option in (
        "--preprocessed-root",
        "--config",
        "--config-identity",
        "--checkpoint",
        "--checkpoint-identity",
        "--open3dml-root",
        "--output-dir",
        "--max-frames",
        "--device",
    ):
        assert option in result.stdout


def test_open3dml_runner_binds_runtime_module_sources(tmp_path):
    from types import SimpleNamespace

    from tools.external_detector.run_open3dml_waymo_pointpillars import (
        require_runtime_module_source,
    )

    root = tmp_path / "ml3d"
    source = root / "torch/model.py"
    source.parent.mkdir(parents=True)
    source.write_text("MODEL = True\n", encoding="utf-8")
    assert require_runtime_module_source(SimpleNamespace(__file__=str(source)), root) == source
    outside = tmp_path / "outside.py"
    outside.write_text("MODEL = False\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside declared Open3D-ML root"):
        require_runtime_module_source(SimpleNamespace(__file__=str(outside)), root)


def test_detector_adapter_cli_exposes_explicit_boundary_files():
    result = subprocess.run(
        [
            sys.executable,
            str(
                REPO_ROOT
                / "tools"
                / "external_detector"
                / "adapt_open3dml_to_detzero.py"
            ),
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--raw-predictions" in result.stdout
    assert "--waymo-info" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--expected-frames" in result.stdout


def test_tracking_cli_exposes_fresh_output_directory():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "utils"), str(REPO_ROOT / "tracking")]
    )
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tracking" / "tools" / "run_track.py"),
            "--help",
        ],
        cwd=REPO_ROOT / "tracking" / "tools",
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--output_path" in result.stdout
    assert "--root_path" in result.stdout


def test_merge_waymo_lidar_returns_keeps_all_lidars_and_nlz():
    point_returns = []
    nlz_returns = []
    for return_id in range(2):
        point_returns.append([])
        nlz_returns.append([])
        for lidar_id in range(5):
            point_returns[-1].append(
                np.asarray(
                    [[10, 0.1 + return_id, 0.2 + lidar_id, lidar_id, return_id, 3]],
                    dtype=np.float32,
                )
            )
            nlz_returns[-1].append(np.asarray([-1 + 2 * return_id], dtype=np.float32))

    points, counts = merge_waymo_lidar_returns(point_returns, nlz_returns)

    assert counts == [2, 2, 2, 2, 2]
    assert points.shape == (10, 6)
    assert points.dtype == np.float32
    np.testing.assert_allclose(points[0], [0, 0, 3, 0.1, 0.2, -1])
    np.testing.assert_allclose(points[1], [0, 1, 3, 1.1, 0.2, 1])
    np.testing.assert_allclose(points[-2], [4, 0, 3, 0.1, 4.2, -1])
    np.testing.assert_allclose(points[-1], [4, 1, 3, 1.1, 4.2, 1])


def test_parse_waymo_frame_passes_immutable_bytes_to_protobuf():
    class Serialized:
        def numpy(self):
            return b"frame-bytes"

    class Frame:
        def ParseFromString(self, payload):
            assert type(payload) is bytes
            assert payload == b"frame-bytes"

    frame = parse_waymo_frame(Serialized(), Frame)
    assert isinstance(frame, Frame)


def test_raw_prediction_npz_roundtrip_preserves_empty_frames(tmp_path):
    path = tmp_path / "raw_predictions.npz"
    frames = [
        {"frame_id": 0, "point_count": 100, "model_point_count": 90, "boxes": []},
        {
            "frame_id": 1,
            "point_count": 120,
            "model_point_count": 110,
            "boxes": [
                {
                    "center": np.asarray([1, 2, 3], dtype=np.float32),
                    "size": np.asarray([1.8, 1.6, 4.7], dtype=np.float32),
                    "yaw": np.float32(-0.5),
                    "label": "VEHICLE",
                    "score": np.float32(0.75),
                }
            ],
        },
    ]

    save_raw_predictions(path, "seq", frames)
    raw = load_raw_predictions(path)

    assert raw["sequence_name"] == "seq"
    assert [frame["frame_id"] for frame in raw["frames"]] == [0, 1]
    assert raw["frames"][0]["boxes"] == []
    box = raw["frames"][1]["boxes"][0]
    np.testing.assert_array_equal(box["center"], np.asarray([1, 2, 3], dtype=np.float32))
    np.testing.assert_array_equal(
        box["size"], np.asarray([1.8, 1.6, 4.7], dtype=np.float32)
    )
    assert box["label"] == "VEHICLE"
    assert box["score"] == pytest.approx(0.75)
    with pytest.raises(FileExistsError):
        save_raw_predictions(path, "seq", frames)


def test_adapt_raw_predictions_emits_detzero_frame_schema(tmp_path):
    path = tmp_path / "raw_predictions.npz"
    save_raw_predictions(
        path,
        "seq",
        [
            {
                "frame_id": 0,
                "point_count": 100,
                "model_point_count": 90,
                "boxes": [
                    {
                        "center": np.asarray([10, -2, 1.5], dtype=np.float32),
                        "size": np.asarray([2, 3, 4], dtype=np.float32),
                        "yaw": np.float32(0.25),
                        "label": "VEHICLE",
                        "score": np.float32(0.8),
                    }
                ],
            },
            {"frame_id": 1, "point_count": 80, "model_point_count": 70, "boxes": []},
        ],
    )
    infos = [
        {
            "sequence_name": "seq",
            "sample_idx": frame_id,
            "pose": np.eye(4, dtype=np.float64),
            "time_stamp": 1000 + frame_id,
        }
        for frame_id in range(2)
    ]

    frames = adapt_raw_predictions(path, infos)

    assert len(frames) == 2
    assert frames[0]["sequence_name"] == "seq"
    assert frames[0]["sample_idx"] == frames[0]["frame_id"] == 0
    assert frames[0]["timestamp"] == 1000
    assert frames[0]["name"].tolist() == ["Vehicle"]
    assert frames[0]["score"].dtype == np.float32
    np.testing.assert_allclose(
        frames[0]["boxes_lidar"][0],
        [10, -2, 1.5, 4, 2, 3, -0.25 - np.pi / 2, 0, 0],
        atol=1e-6,
    )
    assert frames[1]["boxes_lidar"].shape == (0, 9)
    assert frames[1]["name"].shape == (0,)
    assert frames[1]["name"].dtype == np.dtype("<U10")


def test_stage_a_validator_replays_detector_adapter_boundary(tmp_path):
    from tools.external_detector.validate_stage_a import validate_detector_adapter

    raw_path = tmp_path / "raw_predictions.npz"
    save_raw_predictions(
        raw_path,
        "seq",
        [
            {
                "frame_id": 0,
                "point_count": 100,
                "model_point_count": 90,
                "boxes": [
                    {
                        "center": np.asarray([10, -2, 1.5], dtype=np.float32),
                        "size": np.asarray([2, 3, 4], dtype=np.float32),
                        "yaw": np.float32(0.25),
                        "label": "VEHICLE",
                        "score": np.float32(0.8),
                    }
                ],
            },
            {"frame_id": 1, "point_count": 80, "model_point_count": 70, "boxes": []},
        ],
    )
    infos = [
        {
            "sequence_name": "seq",
            "sample_idx": frame_id,
            "pose": np.eye(4, dtype=np.float64),
            "time_stamp": 1000 + frame_id,
        }
        for frame_id in range(2)
    ]
    frames = adapt_raw_predictions(raw_path, infos)
    adapter_path = tmp_path / "detzero_result.pkl"
    with adapter_path.open("wb") as stream:
        pickle.dump(frames, stream)

    summary = validate_detector_adapter(raw_path, adapter_path, expected_frames=2)

    assert summary == {
        "sequence_name": "seq",
        "frame_count": 2,
        "box_count": 1,
        "class_counts": {"Vehicle": 1, "Pedestrian": 0, "Cyclist": 0},
    }
    frames[0]["boxes_lidar"][0, 0] += 1
    tampered_path = tmp_path / "tampered.pkl"
    with tampered_path.open("wb") as stream:
        pickle.dump(frames, stream)
    with pytest.raises(ValueError, match="detector-to-adapter geometry mismatch"):
        validate_detector_adapter(raw_path, tampered_path, expected_frames=2)


def test_adapt_raw_predictions_accepts_only_an_explicit_info_prefix(tmp_path):
    path = tmp_path / "raw_predictions.npz"
    save_raw_predictions(
        path,
        "seq",
        [{"frame_id": 0, "point_count": 10, "model_point_count": 9, "boxes": []}],
    )
    infos = [
        {
            "sequence_name": "seq",
            "sample_idx": frame_id,
            "pose": np.eye(4, dtype=np.float64),
            "time_stamp": 1000 + frame_id,
        }
        for frame_id in range(2)
    ]

    frames = adapt_raw_predictions(path, infos, expected_frame_count=1)

    assert [frame["frame_id"] for frame in frames] == [0]


def test_reverse_tracking_handles_a_leading_empty_detection_frame(monkeypatch):
    tracking_root = REPO_ROOT / "tracking"
    sys.path.insert(0, str(REPO_ROOT / "utils"))
    sys.path.insert(0, str(tracking_root))
    from easydict import EasyDict
    from detzero_utils.config_utils import cfg_from_yaml_file
    from detzero_track.models.tracking_modules.track_manager import TrackManager

    monkeypatch.chdir(tracking_root / "tools")
    config = EasyDict()
    cfg_from_yaml_file(
        "cfgs/tk_model_cfgs/waymo_detzero_track.yaml", config
    )
    manager = TrackManager(config.MODEL.TRACKING)
    empty = {
        "boxes_global": np.zeros((0, 9), dtype=np.float32),
        "name": np.asarray([], dtype="<U10"),
        "score": np.zeros(0, dtype=np.float32),
        "pose": np.eye(4, dtype=np.float64),
    }
    detected = {
        "boxes_global": np.asarray(
            [[1, 2, 0, 4, 2, 1.5, 0, 0, 0]], dtype=np.float32
        ),
        "name": np.asarray(["Vehicle"]),
        "score": np.asarray([0.9], dtype=np.float32),
        "pose": np.eye(4, dtype=np.float64),
    }

    result = manager.forward({"0": empty, "1": detected})

    assert result
    assert any("0" in track["sample_idx"] for track in result.values())


def test_tracking_processor_preserves_scalar_sample_idx():
    tracking_root = REPO_ROOT / "tracking"
    sys.path.insert(0, str(REPO_ROOT / "utils"))
    sys.path.insert(0, str(tracking_root))
    from easydict import EasyDict
    from detzero_track.datasets.data_processor import DataProcessor

    processor = DataProcessor([])
    frame = {
        "sequence_name": "seq",
        "sample_idx": 0,
        "frame_id": 0,
        "timestamp": 1000,
        "pose": np.eye(4, dtype=np.float64),
        "boxes_lidar": np.asarray(
            [[1, 2, 0, 4, 2, 1.5, 0, 0, 0]], dtype=np.float32
        ),
        "name": np.asarray(["Vehicle"]),
        "score": np.asarray([0.9], dtype=np.float32),
    }
    config = EasyDict(
        METHOD="max_score",
        CLASS_THRESHOLD={"Vehicle": 0.3, "Pedestrian": 0.2, "Cyclist": 0.2},
    )

    kept, dropped = processor.overlap_box_filter(frame, config)

    assert kept["sample_idx"] == dropped["sample_idx"] == 0


def test_object_crop_writes_only_to_explicit_output_root(tmp_path):
    sys.path.insert(0, str(REPO_ROOT / "utils"))
    sys.path.insert(0, str(REPO_ROOT / "daemon"))
    from prepare_object_data import WaymoObjectDataPrepare

    sequence = "123"
    data_root = tmp_path / "waymo"
    lidar_dir = data_root / "waymo_processed_data" / f"segment-{sequence}"
    lidar_dir.mkdir(parents=True)
    points = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.4, 0.0, -1.0],
            [0.5, 0.2, 0.1, 0.6, 0.0, -1.0],
            [8.0, 8.0, 0.0, 0.3, 0.0, -1.0],
        ],
        dtype=np.float32,
    )
    np.save(lidar_dir / "0000.npy", points)
    track = {
        "sequence_name": sequence,
        "obj_ids": np.asarray([7]),
        "name": np.asarray(["Vehicle"]),
        "boxes_global": np.asarray(
            [[0.0, 0.0, 0.0, 4.0, 2.0, 2.0, 0.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        "score": np.asarray([0.9], dtype=np.float32),
        "hit": np.asarray([1], dtype=np.int64),
        "sample_idx": np.asarray(["0"]),
        "state": "static",
        "pose": np.eye(4, dtype=np.float64)[None, ...],
    }
    output_root = tmp_path / "objects"
    preparer = WaymoObjectDataPrepare(
        class_name="Vehicle",
        root_path=str(data_root),
        output_root=str(output_root),
        split="test",
        track_data_path="unused.pkl",
        workers=0,
        logger=logging.getLogger("object-crop-test"),
    )

    preparer.prepare_data_worker({sequence: {7: track}})

    object_path = output_root / "Vehicle" / f"{sequence}.pkl"
    assert object_path.is_file()
    assert not (data_root / "refining").exists()
    with object_path.open("rb") as stream:
        objects = pickle.load(stream)
    assert set(objects) == {7}
    assert objects[7]["pts"][0].shape == (2, 4)


def test_object_crop_cli_exposes_explicit_input_and_output_roots():
    script = REPO_ROOT / "daemon" / "prepare_object_data.py"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(REPO_ROOT / "utils"), str(REPO_ROOT / "daemon"))
    )
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert "--root_path" in result.stdout
    assert "--output_root" in result.stdout


def test_empty_refining_class_is_not_executed():
    from tools.external_detector.run_stage_a_refining import class_input_status

    status = class_input_status({})

    assert status == {
        "input_track_count": 0,
        "input_box_count": 0,
        "geometry": {"status": "NOT_EXECUTED_NO_INPUT", "forward_count": 0},
        "position": {"status": "NOT_EXECUTED_NO_INPUT", "forward_count": 0},
    }


def test_refining_output_record_stays_aligned_with_track():
    from tools.external_detector.run_stage_a_refining import model_output_record

    track = {
        "sequence_name": "seq",
        "obj_ids": np.asarray([7, 7]),
        "sample_idx": np.asarray(["0", "2"]),
        "boxes_global": np.asarray(
            [
                [0, 0, 0, 4, 2, 1.5, 0, 0, 0],
                [1, 0, 0, 4, 2, 1.5, 0, 1, 0],
            ],
            dtype=np.float32,
        ),
        "score": np.asarray([0.8, 0.7], dtype=np.float32),
        "name": np.asarray(["Vehicle", "Vehicle"]),
        "pose": np.repeat(np.eye(4)[None, ...], 2, axis=0),
    }
    predictions = track["boxes_global"][:, :7].copy()
    predictions[:, 3:6] += 0.1

    record = model_output_record(track, predictions)

    assert record["sequence_name"] == "seq"
    assert record["obj_id"] == 7
    np.testing.assert_array_equal(record["sample_idx"], ["0", "2"])
    np.testing.assert_array_equal(record["score"], track["score"])
    np.testing.assert_array_equal(record["boxes_global"], predictions)


def test_stage_a_refining_cli_pins_real_inputs_and_output():
    script = REPO_ROOT / "tools" / "external_detector" / "run_stage_a_refining.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    for option in (
        "--waymo-root",
        "--tracking",
        "--output-dir",
        "--checkpoint-root",
        "--checkpoint-identity-root",
        "--device",
    ):
        assert option in result.stdout


def test_stage_a_refining_direct_cli_resolves_repository_modules(tmp_path):
    script = REPO_ROOT / "tools" / "external_detector" / "run_stage_a_refining.py"
    missing = tmp_path / "missing"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--waymo-root",
            str(missing),
            "--tracking",
            str(missing),
            "--output-dir",
            str(tmp_path / "output"),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "ModuleNotFoundError" not in result.stderr
    assert str(missing) in result.stderr


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="CUDA required")
def test_stage_a_refining_runs_real_models_only_for_nonempty_classes(tmp_path):
    from tools.external_detector.run_stage_a_refining import run_stage_a_refining

    sequence = "456"
    waymo_root = tmp_path / "waymo"
    lidar_root = waymo_root / "waymo_processed_data" / f"segment-{sequence}"
    lidar_root.mkdir(parents=True)
    frame_count = 5
    boxes = np.zeros((frame_count, 9), dtype=np.float32)
    boxes[:, 0] = np.arange(frame_count, dtype=np.float32) * 0.2
    boxes[:, 2] = 1.0
    boxes[:, 3:6] = (4.0, 2.0, 2.0)
    rng = np.random.default_rng(8)
    for frame_id, box in enumerate(boxes):
        xyz = rng.uniform((-1.5, -0.7, -0.7), (1.5, 0.7, 0.7), (64, 3))
        xyz += box[:3]
        points = np.zeros((64, 6), dtype=np.float32)
        points[:, :3] = xyz
        points[:, 3] = 0.5
        points[:, 5] = -1
        np.save(lidar_root / f"{frame_id:04d}.npy", points)
    track = {
        "sequence_name": sequence,
        "obj_ids": np.full(frame_count, 7, dtype=np.int64),
        "name": np.full(frame_count, "Vehicle"),
        "boxes_global": boxes,
        "score": np.linspace(0.9, 0.8, frame_count, dtype=np.float32),
        "sample_idx": np.asarray([str(index) for index in range(frame_count)]),
        "hit": np.ones(frame_count, dtype=np.int64),
        "pose": np.repeat(np.eye(4)[None, ...], frame_count, axis=0),
        "state": "dynamic",
    }
    tracking_path = tmp_path / "tracking.pkl"
    with tracking_path.open("wb") as stream:
        pickle.dump({sequence: {7: track}}, stream)
    output_dir = tmp_path / "refining"

    manifest = run_stage_a_refining(
        waymo_root=waymo_root,
        tracking_path=tracking_path,
        output_dir=output_dir,
        checkpoint_root=REPO_ROOT / "checkpoints",
        device="cuda",
    )

    assert manifest["classes"]["Vehicle"]["geometry"]["status"] == "EXECUTED"
    assert manifest["classes"]["Vehicle"]["position"]["status"] == "EXECUTED"
    for class_name in ("Pedestrian", "Cyclist"):
        assert manifest["classes"][class_name]["geometry"]["status"] == "NOT_EXECUTED_NO_INPUT"
        assert manifest["classes"][class_name]["position"]["status"] == "NOT_EXECUTED_NO_INPUT"
    assert manifest["model_forward_count"] == 2
    assert manifest["crm"]["status"] == "NOT_EXECUTED_NO_CRM_BY_DESIGN"
    assert manifest["tracking"]["path"] == "../tracking.pkl"
    assert manifest["waymo_root"] == "../waymo"
    assert (output_dir / "result" / "Vehicle_geometry.pkl").is_file()
    assert (output_dir / "result" / "Vehicle_position.pkl").is_file()
    assert not (output_dir / "result" / "Cyclist_geometry.pkl").exists()
    assert not (output_dir / "Vehicle").exists()
    assert not (output_dir / "Pedestrian").exists()
    assert not (output_dir / "Cyclist").exists()


def test_no_crm_combination_uses_prm_pose_grm_size_and_tracking_score():
    from tools.external_detector.combine_grm_prm_no_crm import combine_no_crm_track

    tracking = {
        "boxes_global": np.asarray(
            [[1, 2, 3, 4, 2, 1.5, 0.1, 0.3, -0.2]], dtype=np.float32
        ),
        "score": np.asarray([0.73], dtype=np.float32),
        "sample_idx": np.asarray(["4"]),
        "name": np.asarray(["Vehicle"]),
    }
    geometry = {
        "boxes_global": np.asarray(
            [[10, 20, 30, 4.4, 2.2, 1.7, 1.0]], dtype=np.float32
        ),
        "score": tracking["score"].copy(),
        "sample_idx": tracking["sample_idx"].copy(),
    }
    position = {
        "boxes_global": np.asarray(
            [[1.2, 2.3, 3.4, 8, 8, 8, 0.25]], dtype=np.float32
        ),
        "score": tracking["score"].copy(),
        "sample_idx": tracking["sample_idx"].copy(),
    }

    final = combine_no_crm_track(tracking, geometry, position)

    np.testing.assert_allclose(
        final["boxes_global"],
        [[1.2, 2.3, 3.4, 4.4, 2.2, 1.7, 0.25, 0.3, -0.2]],
    )
    np.testing.assert_array_equal(final["score"], tracking["score"])
    np.testing.assert_array_equal(final["sample_idx"], tracking["sample_idx"])


def test_no_crm_combination_accepts_equivalent_zero_padded_frame_ids():
    from tools.external_detector.combine_grm_prm_no_crm import combine_no_crm_track

    tracking = {
        "boxes_global": np.asarray(
            [[1, 2, 3, 4, 2, 1.5, 0.1, 0.3, -0.2]], dtype=np.float32
        ),
        "score": np.asarray([0.73], dtype=np.float32),
        "sample_idx": np.asarray(["4"]),
    }
    model_record = {
        "boxes_global": tracking["boxes_global"][:, :7].copy(),
        "score": tracking["score"].copy(),
        "sample_idx": np.asarray(["0004"]),
    }

    final = combine_no_crm_track(tracking, model_record, model_record)

    np.testing.assert_array_equal(final["sample_idx"], ["4"])


def test_final_frame_output_keeps_explicit_empty_detector_frames():
    from tools.external_detector.combine_grm_prm_no_crm import tracks_to_frames

    detector_frames = [
        {
            "sequence_name": "seq",
            "frame_id": frame_id,
            "pose": np.eye(4, dtype=np.float64),
        }
        for frame_id in range(2)
    ]
    track = {
        "sequence_name": "seq",
        "boxes_global": np.asarray(
            [[1, 2, 3, 4, 2, 1.5, 0.1, 0.3, -0.2]], dtype=np.float32
        ),
        "score": np.asarray([0.73], dtype=np.float32),
        "sample_idx": np.asarray(["1"]),
        "name": np.asarray(["Vehicle"]),
        "obj_ids": np.asarray([7]),
    }

    frames = tracks_to_frames({"seq": {7: track}}, detector_frames)

    assert [frame["frame_id"] for frame in frames] == [0, 1]
    assert frames[0]["boxes_lidar"].shape == (0, 9)
    assert frames[0]["score"].shape == (0,)
    assert frames[1]["boxes_lidar"].shape == (1, 9)
    np.testing.assert_array_equal(frames[1]["obj_ids"], [7])
    np.testing.assert_array_equal(frames[1]["score"], [np.float32(0.73)])


def test_no_crm_combine_cli_pins_all_inputs_and_output():
    script = REPO_ROOT / "tools" / "external_detector" / "combine_grm_prm_no_crm.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    for option in ("--tracking", "--detector-frames", "--refining-dir", "--output-dir"):
        assert option in result.stdout


def test_no_crm_combiner_publishes_track_aligned_full_frame_output(tmp_path):
    from tools.external_detector.combine_grm_prm_no_crm import combine_stage_a
    from tools.external_detector.validate_stage_a import (
        validate_tracking_refining_final,
    )

    sequence = "seq"
    tracking_track = {
        "boxes_global": np.asarray(
            [[1, 2, 3, 4, 2, 1.5, 0.1, 0.3, -0.2]], dtype=np.float32
        ),
        "score": np.asarray([0.73], dtype=np.float32),
        "sample_idx": np.asarray(["1"]),
        "name": np.asarray(["Vehicle"]),
        "obj_ids": np.asarray([7], dtype=np.int64),
        "hit": np.asarray([1], dtype=np.int64),
        "num_points": np.asarray([12], dtype=np.float32),
        "pose": np.eye(4, dtype=np.float64)[None, ...],
        "state": "dynamic",
    }
    tracking_path = tmp_path / "tracking.pkl"
    with tracking_path.open("wb") as stream:
        pickle.dump({sequence: {7: tracking_track}}, stream)
    detector_path = tmp_path / "detector.pkl"
    detector_frames = [
        {
            "sequence_name": sequence,
            "frame_id": frame_id,
            "pose": np.eye(4, dtype=np.float64),
            "timestamp": 1000 + frame_id,
        }
        for frame_id in range(2)
    ]
    with detector_path.open("wb") as stream:
        pickle.dump(detector_frames, stream)
    refining_dir = tmp_path / "refining"
    result_dir = refining_dir / "result"
    result_dir.mkdir(parents=True)
    common = {
        "sequence_name": sequence,
        "obj_id": 7,
        "sample_idx": np.asarray(["1"]),
        "score": tracking_track["score"].copy(),
        "name": tracking_track["name"].copy(),
        "pose": np.eye(4, dtype=np.float64)[None, ...],
    }
    geometry = dict(common, boxes_global=np.asarray([[9, 9, 9, 4.4, 2.2, 1.7, 1]], dtype=np.float32))
    position = dict(common, boxes_global=np.asarray([[1.2, 2.3, 3.4, 8, 8, 8, 0.25]], dtype=np.float32))
    for kind, record in (("geometry", geometry), ("position", position)):
        with (result_dir / f"Vehicle_{kind}.pkl").open("wb") as stream:
            pickle.dump({sequence: {7: record}}, stream)
    geometry_checkpoint = tmp_path / "geometry.ckpt"
    position_checkpoint = tmp_path / "position.ckpt"
    geometry_checkpoint.write_bytes(b"geometry")
    position_checkpoint.write_bytes(b"position")
    class_status = {
        "Vehicle": {
            "input_track_count": 1,
            "input_box_count": 1,
            "geometry": {
                "status": "EXECUTED",
                "checkpoint": str(geometry_checkpoint.resolve()),
                "checkpoint_sha256": hashlib.sha256(
                    geometry_checkpoint.read_bytes()
                ).hexdigest(),
                "loaded_tensors": 1,
                "model_tensors": 1,
                "forward_count": 1,
                "output": "result/Vehicle_geometry.pkl",
                "output_sha256": hashlib.sha256(
                    (result_dir / "Vehicle_geometry.pkl").read_bytes()
                ).hexdigest(),
            },
            "position": {
                "status": "EXECUTED",
                "checkpoint": str(position_checkpoint.resolve()),
                "checkpoint_sha256": hashlib.sha256(
                    position_checkpoint.read_bytes()
                ).hexdigest(),
                "loaded_tensors": 1,
                "model_tensors": 1,
                "forward_count": 1,
                "output": "result/Vehicle_position.pkl",
                "output_sha256": hashlib.sha256(
                    (result_dir / "Vehicle_position.pkl").read_bytes()
                ).hexdigest(),
            },
        },
        **{
            name: {
                "input_track_count": 0,
                "input_box_count": 0,
                "geometry": {
                    "status": "NOT_EXECUTED_NO_INPUT",
                    "forward_count": 0,
                },
                "position": {
                    "status": "NOT_EXECUTED_NO_INPUT",
                    "forward_count": 0,
                },
            }
            for name in ("Pedestrian", "Cyclist")
        },
    }
    (refining_dir / "refining_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "detzero-stage-a-refining-v1",
                "tracking": {
                    "path": os.path.relpath(tracking_path, refining_dir),
                    "sha256": hashlib.sha256(tracking_path.read_bytes()).hexdigest(),
                },
                "waymo_root": os.path.relpath(tmp_path, refining_dir),
                "classes": class_status,
                "model_forward_count": 2,
                "crm": {"status": "NOT_EXECUTED_NO_CRM_BY_DESIGN"},
                "score_policy": "TRACKING_SCORE_PASSTHROUGH",
                "outputs": {
                    "result/Vehicle_geometry.pkl": hashlib.sha256(
                        (result_dir / "Vehicle_geometry.pkl").read_bytes()
                    ).hexdigest(),
                    "result/Vehicle_position.pkl": hashlib.sha256(
                        (result_dir / "Vehicle_position.pkl").read_bytes()
                    ).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "final"

    manifest = combine_stage_a(
        tracking_path, detector_path, refining_dir, output_dir
    )

    assert manifest["frame_count"] == 2
    assert manifest["track_count"] == 1
    assert manifest["crm"]["status"] == "NOT_EXECUTED_NO_CRM_BY_DESIGN"
    with (output_dir / "final_frame_grm_prm_score_passthrough.pkl").open("rb") as stream:
        frames = pickle.load(stream)
    assert frames[0]["boxes_lidar"].shape == (0, 9)
    np.testing.assert_array_equal(frames[1]["score"], tracking_track["score"])
    with np.load(output_dir / "final_arrays.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {
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
        np.testing.assert_array_equal(archive["frame_ids"], [0, 1])
        np.testing.assert_array_equal(archive["frame_offsets"], [0, 0, 1])
        assert archive["boxes_lidar"].shape == (1, 9)

    summary = validate_tracking_refining_final(
        tracking_path,
        detector_path,
        refining_dir,
        output_dir,
        expected_frames=2,
    )
    assert summary == {
        "sequence_name": sequence,
        "frame_count": 2,
        "track_count": 1,
        "observation_count": 1,
        "class_track_counts": {"Vehicle": 1, "Pedestrian": 0, "Cyclist": 0},
    }

    refining_manifest_path = refining_dir / "refining_manifest.json"
    refining_manifest = json.loads(refining_manifest_path.read_text(encoding="utf-8"))
    refining_manifest["unexpected"] = True
    refining_manifest_path.write_text(
        json.dumps(refining_manifest, allow_nan=False, sort_keys=True), encoding="utf-8"
    )
    final_manifest_path = output_dir / "final_manifest.json"
    final_manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))
    final_manifest["inputs"]["refining_manifest_sha256"] = hashlib.sha256(
        refining_manifest_path.read_bytes()
    ).hexdigest()
    final_manifest_path.write_text(
        json.dumps(final_manifest, allow_nan=False, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="refining manifest field set mismatch"):
        validate_tracking_refining_final(
            tracking_path,
            detector_path,
            refining_dir,
            output_dir,
            expected_frames=2,
        )

    del refining_manifest["unexpected"]
    refining_manifest_path.write_text(
        json.dumps(refining_manifest, allow_nan=False, sort_keys=True), encoding="utf-8"
    )
    final_manifest["inputs"]["refining_manifest_sha256"] = hashlib.sha256(
        refining_manifest_path.read_bytes()
    ).hexdigest()
    final_manifest["unexpected"] = True
    final_manifest_path.write_text(
        json.dumps(final_manifest, allow_nan=False, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="final manifest field set mismatch"):
        validate_tracking_refining_final(
            tracking_path,
            detector_path,
            refining_dir,
            output_dir,
            expected_frames=2,
        )


def test_static_renderer_is_bound_to_points_and_predictions():
    from tools.external_detector.render_waymo_sequence import render_frame

    points = np.asarray(
        [[0, 0, 0, 0.2], [10, 4, 0, 0.8], [-8, -3, 0, -0.1]],
        dtype=np.float32,
    )
    detector_boxes = np.asarray([[10, 4, 1, 4, 2, 1.5, 0.2, 0, 0]], dtype=np.float32)
    final_boxes = np.asarray([[-8, -3, 1, 1, 0.8, 1.7, -0.4, 0, 0]], dtype=np.float32)

    image = np.asarray(
        render_frame(
            points,
            detector_boxes,
            np.asarray(["Vehicle"]),
            final_boxes,
            np.asarray(["Pedestrian"]),
            frame_id=4,
        )
    )
    changed = np.asarray(
        render_frame(
            points + np.asarray([20, 0, 0, 0], dtype=np.float32),
            detector_boxes,
            np.asarray(["Vehicle"]),
            final_boxes,
            np.asarray(["Pedestrian"]),
            frame_id=4,
        )
    )

    assert image.shape == changed.shape == (640, 1280, 3)
    assert image.std() > 10
    assert not np.array_equal(image, changed)


def test_static_renderer_cli_pins_inputs_output_and_frame_count():
    script = REPO_ROOT / "tools" / "external_detector" / "render_waymo_sequence.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    for option in (
        "--waymo-root",
        "--detector-frames",
        "--final-frames",
        "--output-dir",
        "--expected-frames",
    ):
        assert option in result.stdout


def test_static_renderer_batch_publishes_two_frames_and_manifest(tmp_path):
    sequence = "seq"
    waymo_root = tmp_path / "waymo"
    point_root = waymo_root / "waymo_processed_data" / f"segment-{sequence}"
    point_root.mkdir(parents=True)
    detector_frames = []
    final_frames = []
    for frame_id in range(2):
        points = np.asarray(
            [[frame_id, 0, 0, 0.2, 0, -1], [10, 4, 0, 0.8, 0, -1]],
            dtype=np.float32,
        )
        np.save(point_root / f"{frame_id:04d}.npy", points, allow_pickle=False)
        detector_frames.append(
            {
                "sequence_name": sequence,
                "frame_id": frame_id,
                "boxes_lidar": np.asarray(
                    [[10, 4, 1, 4, 2, 1.5, 0.2, 0, 0]], dtype=np.float32
                ),
                "name": np.asarray(["Vehicle"]),
            }
        )
        final_frames.append(
            {
                "sequence_name": sequence,
                "frame_id": frame_id,
                "boxes_lidar": np.asarray(
                    [[-8, -3, 1, 1, 0.8, 1.7, -0.4, 0, 0]], dtype=np.float32
                ),
                "name": np.asarray(["Pedestrian"]),
            }
        )
    detector_path = tmp_path / "detector.pkl"
    final_path = tmp_path / "final.pkl"
    for path, frames in ((detector_path, detector_frames), (final_path, final_frames)):
        with path.open("wb") as stream:
            pickle.dump(frames, stream)
    output_dir = tmp_path / "visuals"
    script = REPO_ROOT / "tools" / "external_detector" / "render_waymo_sequence.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--waymo-root",
            str(waymo_root),
            "--detector-frames",
            str(detector_path),
            "--final-frames",
            str(final_path),
            "--output-dir",
            str(output_dir),
            "--expected-frames",
            "2",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "0000.png",
        "0001.png",
        "render_manifest.json",
    ]
    manifest = json.loads((output_dir / "render_manifest.json").read_text())
    assert manifest["schema_version"] == "detzero-stage-a-render-v1"
    assert manifest["sequence_name"] == sequence
    assert manifest["frame_count"] == 2
    assert [frame["frame_id"] for frame in manifest["frames"]] == [0, 1]
    for frame in manifest["frames"]:
        image_path = output_dir / frame["path"]
        assert frame["path"] == f"{frame['frame_id']:04d}.png"
        assert frame["bytes"] == image_path.stat().st_size
        assert frame["sha256"] == hashlib.sha256(image_path.read_bytes()).hexdigest()
        assert [frame["width"], frame["height"]] == [1280, 640]
        assert frame["point_count"] == 2
        assert frame["detector_box_count"] == 1
        assert frame["final_box_count"] == 1

    from tools.external_detector.validate_stage_a import validate_visuals

    assert validate_visuals(
        waymo_root,
        detector_path,
        final_path,
        output_dir,
        expected_frames=2,
    ) == {
        "sequence_name": sequence,
        "frame_count": 2,
        "unique_image_count": 2,
    }


def test_stage_a_validator_rejects_duplicate_json_keys(tmp_path):
    from tools.external_detector.validate_stage_a import load_json_strict

    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"frame_count": 2, "frame_count": 2}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key: frame_count"):
        load_json_strict(manifest)


def test_stage_a_validator_cli_publishes_transactional_failure_report(tmp_path):
    script = REPO_ROOT / "tools" / "external_detector" / "validate_stage_a.py"
    missing = tmp_path / "missing"
    report_path = tmp_path / "audit" / "validation_report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--run-root",
            str(missing),
            "--live-repo-root",
            str(REPO_ROOT),
            "--pipeline-python",
            sys.executable,
            "--detector-python",
            sys.executable,
            "--preprocess-python",
            sys.executable,
            "--waymo-root",
            str(missing),
            "--detector-dir",
            str(missing),
            "--adapter-dir",
            str(missing),
            "--tracking",
            str(missing),
            "--dropped",
            str(missing),
            "--refining-dir",
            str(missing),
            "--final-dir",
            str(missing),
            "--visuals-dir",
            str(missing),
            "--expected-frames",
            "2",
            "--report",
            str(report_path),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == "detzero-stage-a-validation-v1"
    assert report["passed"] is False
    assert report["artifact_validation_passed"] is False
    assert report["release_eligible"] is False
    assert report["checks"] == {}
    assert "invalid run root" in report["error"]


def test_stage_a_validator_replays_preprocess_detector_adapter_manifests(tmp_path):
    from tools.external_detector.validate_stage_a import (
        validate_dropped_frames,
        validate_preprocess_detector_adapter,
    )

    sequence = "seq"
    input_path = tmp_path / "segment-seq.tfrecord"
    input_path.write_bytes(b"two-frame-source")
    waymo_root = tmp_path / "waymo"
    input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    publish_preprocessed_records(
        (
            {
                "frame_id": frame_id,
                "timestamp": np.int64(1000 + frame_id),
                "pose": np.eye(4, dtype=np.float64),
                "points": np.asarray(
                    [[frame_id, 1, 2, 0.25, 0.5, -1]], dtype=np.float32
                ),
                "num_points_of_each_lidar": [1, 0, 0, 0, 0],
            }
            for frame_id in range(2)
        ),
        waymo_root,
        sequence_name=sequence,
        expected_frame_count=2,
        manifest_fields={
            "input_tfrecord": str(input_path.resolve()),
            "input_tfrecord_bytes": input_path.stat().st_size,
            "input_tfrecord_sha256": input_hash,
            "usage": "non-commercial-research",
            "waymo_terms_accepted": True,
        },
    )
    preprocess_manifest_path = waymo_root / "preprocess_manifest.json"

    detector_dir = tmp_path / "detector"
    detector_dir.mkdir()
    raw_path = detector_dir / "raw_predictions.npz"
    save_raw_predictions(
        raw_path,
        sequence,
        [
            {
                "frame_id": 0,
                "point_count": 1,
                "model_point_count": 1,
                "boxes": [
                    {
                        "center": np.asarray([1, 2, 3], dtype=np.float32),
                        "size": np.asarray([2, 1.5, 4], dtype=np.float32),
                        "yaw": np.float32(0.2),
                        "score": np.float32(0.8),
                        "label": "VEHICLE",
                    }
                ],
            },
            {
                "frame_id": 1,
                "point_count": 1,
                "model_point_count": 1,
                "boxes": [],
            },
        ],
    )
    config_path = tmp_path / "pointpillars.yml"
    checkpoint_path = tmp_path / "pointpillars.pth"
    config_path.write_text("model: PointPillars\n", encoding="utf-8")
    checkpoint_path.write_bytes(b"checkpoint")
    raw_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    (detector_dir / "detector_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "open3dml-waymo-detector-manifest-v1",
                "backend": "Open3D-ML PointPillars Waymo",
                "device": "cpu",
                "config": str(config_path.resolve()),
                "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_sha256": hashlib.sha256(
                    checkpoint_path.read_bytes()
                ).hexdigest(),
                "preprocess_manifest": os.path.relpath(
                    preprocess_manifest_path, detector_dir
                ),
                "preprocess_manifest_sha256": hashlib.sha256(
                    preprocess_manifest_path.read_bytes()
                ).hexdigest(),
                "sequence_name": sequence,
                "frame_count": 2,
                "box_count": 1,
                "class_counts": {"VEHICLE": 1, "PEDESTRIAN": 0, "CYCLIST": 0},
                "point_features": ["x", "y", "z", "intensity"],
                "nlz_filter": "points[:, 5] != 1.0",
                "open3d_version": "0.test",
                "open3dml_commit": "0d9e52d",
                "torch_version": "2.test",
                "numpy_version": np.__version__,
                "raw_predictions": "raw_predictions.npz",
                "raw_predictions_sha256": raw_hash,
                "elapsed_seconds": 1.25,
                "checkpoint_license_status": (
                    "upstream-model-zoo-source-recorded; no separate weight license found"
                ),
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    info_path = (
        waymo_root
        / "waymo_processed_data"
        / "segment-seq"
        / "seq.pkl"
    )
    with info_path.open("rb") as stream:
        infos = pickle.load(stream)
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    adapter_path = adapter_dir / "detzero_result.pkl"
    adapter_frames = adapt_raw_predictions(raw_path, infos)
    with adapter_path.open("xb") as stream:
        pickle.dump(adapter_frames, stream)
    (adapter_dir / "adapter_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "open3dml-to-detzero-adapter-manifest-v1",
                "sequence_name": sequence,
                "frame_count": 2,
                "box_count": 1,
                "class_counts": {"Vehicle": 1, "Pedestrian": 0, "Cyclist": 0},
                "input_raw_predictions": os.path.relpath(raw_path, adapter_dir),
                "input_raw_predictions_sha256": raw_hash,
                "input_waymo_info": os.path.relpath(info_path, adapter_dir),
                "input_waymo_info_sha256": hashlib.sha256(
                    info_path.read_bytes()
                ).hexdigest(),
                "output_pickle": "detzero_result.pkl",
                "output_pickle_sha256": hashlib.sha256(
                    adapter_path.read_bytes()
                ).hexdigest(),
                "box_schema": [
                    "x", "y", "z", "length", "width", "height", "heading", "vx", "vy"
                ],
                "center_definition": "geometric box center",
                "yaw_conversion": "detzero_heading = wrap(-open3d_yaw - pi/2)",
                "size_conversion": (
                    "open3d [width,height,length] -> detzero [length,width,height]"
                ),
                "velocity_status": "unavailable; vx=vy=0",
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    assert validate_preprocess_detector_adapter(
        waymo_root,
        detector_dir,
        adapter_dir,
        expected_frames=2,
    ) == {
        "sequence_name": sequence,
        "frame_count": 2,
        "point_count": 2,
        "box_count": 1,
        "class_counts": {"Vehicle": 1, "Pedestrian": 0, "Cyclist": 0},
    }

    dropped_path = tmp_path / "dropped.pkl"
    dropped = {
        sequence: {
            str(frame_id): {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in frame.items()
            }
            for frame_id, frame in enumerate(adapter_frames)
        }
    }
    with dropped_path.open("wb") as stream:
        pickle.dump(dropped, stream)
    assert validate_dropped_frames(
        dropped_path, adapter_path, expected_frames=2
    ) == {
        "sequence_name": sequence,
        "frame_count": 2,
        "dropped_box_count": 1,
    }
    dropped[sequence]["0"]["score"][0] = np.float32(0.7)
    with dropped_path.open("wb") as stream:
        pickle.dump(dropped, stream)
    with pytest.raises(ValueError, match="dropped detections are not an adapter subset"):
        validate_dropped_frames(dropped_path, adapter_path, expected_frames=2)


@pytest.mark.parametrize("heading", [0.0, np.pi / 2, -np.pi / 4])
def test_open3dml_box_to_detzero_golden_geometry(heading):
    center = np.asarray([10.0, -2.0, 1.5], dtype=np.float32)
    size_whl = np.asarray([1.8, 1.6, 4.7], dtype=np.float32)
    open3d_yaw = -heading - np.pi / 2

    box = open3dml_box_to_detzero(center, size_whl, open3d_yaw)

    np.testing.assert_allclose(box[:3], center, rtol=0, atol=0)
    np.testing.assert_allclose(box[3:6], [4.7, 1.8, 1.6], rtol=0, atol=1e-6)
    np.testing.assert_allclose(box[6], heading, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(box[7:9], np.zeros(2, dtype=np.float32))
    assert box.shape == (9,)
    assert box.dtype == np.float32


