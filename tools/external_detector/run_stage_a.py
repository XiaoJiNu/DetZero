#!/usr/bin/env python3
"""Run one source-bound DetZero Waymo Stage A generation."""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any
import uuid

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

BUNDLE_REPLAY_VERIFIER = (
    Path.home()
    / ".hermes/skills/software-development/artifact-pipeline-verification/scripts/verify-python-bundle-replay.py"
)


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
SOURCE_POLICY_RELATIVE = "tools/external_detector/stage_a_source_policy.json"
SOURCE_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".so",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
REFINING_CHECKPOINT_NAMES = tuple(
    f"{class_name}_{kind}_model.pth"
    for class_name in ("vehicle", "pedestrian", "cyclist")
    for kind in ("grm", "prm")
)
_SEQUENCE_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def rename_noreplace(source: str | Path, target: str | Path) -> None:
    """Atomically publish one directory without replacing a racing destination."""
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


def sequence_name_from_tfrecord(path: str | Path) -> str:
    name = Path(path).name
    for suffix in ("_with_camera_labels.tfrecord", ".tfrecord"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    else:
        raise ValueError(f"not a TFRecord path: {path}")
    if name.startswith("segment-"):
        name = name[len("segment-") :]
    if not name or not _SEQUENCE_RE.fullmatch(name):
        raise ValueError(f"invalid sequence name: {name!r}")
    return name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        path = raw_path.decode("utf-8")
        pure = PurePosixPath(path)
        if (
            pure.is_absolute()
            or ".." in pure.parts
            or pure.as_posix() != path
            or not any(
                path == prefix or path.startswith(prefix) for prefix in SOURCE_PREFIXES
            )
            or (
                pure.suffix not in SOURCE_SUFFIXES
                and path not in SOURCE_LITERAL_FILES
            )
        ):
            continue
        paths.append(path)
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


def _source_policy(repo_root: Path) -> dict[str, Any]:
    policy = json.loads((repo_root / SOURCE_POLICY_RELATIVE).read_text(encoding="utf-8"))
    paths = policy.get("source_paths") if isinstance(policy, dict) else None
    if (
        not isinstance(policy, dict)
        or set(policy) != {"schema_version", "expected_frames", "source_paths"}
        or policy["schema_version"] != "detzero-stage-a-source-policy-v1"
        or policy["expected_frames"] != 199
        or not isinstance(paths, list)
        or any(not isinstance(relative, str) for relative in paths)
        or paths != sorted(set(paths))
        or SOURCE_POLICY_RELATIVE not in paths
    ):
        raise ValueError("formal Stage-A source policy mismatch")
    return policy


def _preflight_source_paths(repo_root: Path, policy: dict[str, Any]) -> list[str]:
    if (repo_root / ".git").exists() or (repo_root / ".git").is_symlink():
        return _source_paths(repo_root)
    actual = set()
    for node in repo_root.rglob("*"):
        if node.is_symlink() or (not node.is_dir() and not node.is_file()):
            raise ValueError(f"source bundle contains a symlink or special file: {node}")
        if node.is_file():
            actual.add(node.relative_to(repo_root).as_posix())
    expected = set(policy["source_paths"])
    if actual != expected:
        raise ValueError("source bundle file set does not match formal Stage-A policy")
    return policy["source_paths"]


def _portable_source_paths(repo_root: Path) -> list[str]:
    if (repo_root / ".git").exists():
        return _source_paths(repo_root)
    return _preflight_source_paths(repo_root, _source_policy(repo_root))


def _tree_hash(rows: dict[str, dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for path, row in sorted(rows.items()):
        digest.update(path.encode("utf-8") + b"\0")
        digest.update(row["sha256"].encode("ascii") + b"\n")
    return digest.hexdigest()


def _pinned_file(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    from tools.external_detector.safe_io import open_bounded_regular

    path = Path(path).absolute()
    descriptor = open_bounded_regular(path, 8 * 1024**3)
    digest_builder = hashlib.sha256()
    with os.fdopen(descriptor, "rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest_builder.update(block)
    digest = digest_builder.hexdigest()
    if expected_sha256 is not None and digest != expected_sha256.lower():
        raise ValueError(f"SHA-256 mismatch for {path}")
    return {"path": str(path), "bytes": size, "sha256": digest}


def _snapshot_inputs(run_dir: Path, preflight_result: dict[str, Any]) -> dict[str, Path]:
    """Copy every external byte input once into a private per-run snapshot."""
    from tools.external_detector.safe_io import open_bounded_regular

    inputs = preflight_result.get("inputs")
    open3dml = preflight_result.get("open3dml")
    if not isinstance(inputs, dict) or set(inputs) != {
        "input_tfrecord",
        "detector_config",
        "detector_checkpoint",
        "tracking_config",
        "refining_checkpoints",
    } or not isinstance(inputs["refining_checkpoints"], list):
        raise ValueError("invalid preflight input schema")
    if (
        not isinstance(open3dml, dict)
        or not isinstance(open3dml.get("root"), str)
        or not isinstance(open3dml.get("source_files"), dict)
        or not open3dml["source_files"]
    ):
        raise ValueError("invalid Open3D-ML source snapshot schema")
    run_dir = Path(run_dir).absolute()
    if run_dir.is_symlink():
        raise ValueError(f"invalid run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    root = run_dir / ".input-snapshot"
    root.mkdir(mode=0o700)

    def copy_row(
        row: dict[str, Any], target: Path, *, allow_empty: bool = False
    ) -> None:
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "bytes", "sha256"}
            or not isinstance(row["path"], str)
            or type(row["bytes"]) is not int
            or row["bytes"] < 0
            or not isinstance(row["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
        ):
            raise ValueError("invalid preflight input row")
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        copied = 0
        source_descriptor = open_bounded_regular(
            row["path"], 8 * 1024**3, allow_empty=allow_empty
        )
        with os.fdopen(source_descriptor, "rb") as source:
            target_descriptor = os.open(
                target,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o400,
            )
            with os.fdopen(target_descriptor, "wb") as destination:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    destination.write(block)
                    digest.update(block)
                    copied += len(block)
                destination.flush()
                os.fsync(destination.fileno())
        if copied != row["bytes"] or digest.hexdigest() != row["sha256"]:
            raise ValueError(f"input changed while snapshotting: {row['path']}")

    paths = {
        name: root / Path(inputs[name]["path"]).name
        for name in (
            "input_tfrecord",
            "detector_config",
            "detector_checkpoint",
            "tracking_config",
        )
    }
    if len(paths.values()) != len(set(paths.values())):
        raise ValueError("external input snapshot names collide")
    try:
        for name, target in paths.items():
            copy_row(inputs[name], target)
        checkpoint_root = root / "checkpoints"
        checkpoint_root.mkdir()
        seen = set()
        for row in inputs["refining_checkpoints"]:
            name = Path(row["path"]).name
            if name in seen or name != Path(name).name or not name.endswith(".pth"):
                raise ValueError("invalid refining checkpoint snapshot name")
            seen.add(name)
            copy_row(row, checkpoint_root / name)
        paths["refining_checkpoint_root"] = checkpoint_root
        open3dml_root = root / "ml3d"
        open3dml_root.mkdir()
        source_root = Path(open3dml["root"])
        for relative, row in open3dml["source_files"].items():
            pure = PurePosixPath(relative) if isinstance(relative, str) else None
            if (
                pure is None
                or pure.is_absolute()
                or ".." in pure.parts
                or pure.as_posix() != relative
                or not isinstance(row, dict)
                or set(row) != {"bytes", "sha256"}
            ):
                raise ValueError("invalid Open3D-ML source snapshot row")
            copy_row(
                {"path": str(source_root / relative), **row},
                open3dml_root / relative,
                allow_empty=True,
            )
        paths["open3dml_root"] = open3dml_root
        return paths
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise


def _regular_tree(root: Path, suffixes: set[str]) -> dict[str, dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"invalid source tree: {root}")
    rows = {}
    for node in root.rglob("*"):
        if node.is_symlink() or (not node.is_dir() and not node.is_file()):
            raise ValueError(f"source tree contains a symlink or special file: {node}")
        if node.is_file() and node.suffix in suffixes:
            relative = node.relative_to(root).as_posix()
            rows[relative] = {
                "bytes": node.stat().st_size,
                "sha256": _sha256(node),
            }
    if not rows:
        raise ValueError(f"source tree has no declared files: {root}")
    return rows


def _stage_environment(run_dir: Path) -> dict[str, str]:
    if "OPEN3D_ML_ROOT" in os.environ:
        raise ValueError("OPEN3D_ML_ROOT must be unset for bundled Open3D-ML")
    cache_root = run_dir.absolute() / ".python-cache"
    if cache_root.exists() or cache_root.is_symlink():
        raise ValueError(f"Python cache root must be absent: {cache_root}")
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPYCACHEPREFIX"] = str(cache_root)
    environment["PYTHONPATH"] = os.pathsep.join(
        str(REPO_ROOT / path) for path in ("tracking", "refining", "utils")
    ) + os.pathsep + str(REPO_ROOT)
    return environment


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Validate formal inputs and every stage entrypoint without writes."""
    from tools.external_detector.open3dml_provenance import require_locked_open3dml_source

    if args.expected_frames != 199:
        raise ValueError("formal Stage A requires exactly 199 frames")
    if args.run_dir.exists() or args.run_dir.is_symlink():
        raise FileExistsError(args.run_dir)
    if args.receipt.exists() or args.receipt.is_symlink():
        raise FileExistsError(args.receipt)
    run_dir = args.run_dir.absolute()
    receipt = args.receipt.absolute()
    if run_dir == receipt or run_dir in receipt.parents:
        raise ValueError("receipt must be outside the run directory")
    inputs = {
        "input_tfrecord": _pinned_file(args.input_tfrecord),
        "detector_config": _pinned_file(
            args.detector_config, args.detector_config_sha256
        ),
        "detector_checkpoint": _pinned_file(
            args.detector_checkpoint, args.detector_checkpoint_sha256
        ),
        "tracking_config": _pinned_file(args.tracking_config),
        "refining_checkpoints": [
            _pinned_file(args.refining_checkpoint_root / name)
            for name in REFINING_CHECKPOINT_NAMES
        ],
    }
    if not re.fullmatch(r"[0-9a-f]{40}", args.open3dml_commit):
        raise ValueError("Open3D-ML commit must be a full lowercase SHA-1")
    open3dml_rows = _regular_tree(args.open3dml_root, SOURCE_SUFFIXES)
    open3dml_source_tree_sha256 = _tree_hash(open3dml_rows)
    open3dml_source_lock = require_locked_open3dml_source(
        args.open3dml_commit,
        len(open3dml_rows),
        open3dml_source_tree_sha256,
    )
    pipeline_executable = Path(args.pipeline_python).absolute()
    detector_executable = Path(args.detector_python).absolute()
    preprocess_executable = Path(args.preprocess_python).absolute()
    pipeline_python = pipeline_executable.resolve(strict=True)
    detector_python = detector_executable.resolve(strict=True)
    preprocess_python = preprocess_executable.resolve(strict=True)
    if not all(
        path.is_file() for path in (pipeline_python, detector_python, preprocess_python)
    ):
        raise ValueError("invalid pipeline, detector, or preprocess interpreter")

    environment = _stage_environment(args.run_dir)
    runtime_probes = {
        "pipeline": (
            pipeline_executable,
            "import numpy, torch; "
            + ("assert torch.cuda.is_available()" if args.refining_device == "cuda" else "assert True"),
        ),
        "detector": (
            detector_executable,
            "import numpy, open3d, torch; "
            + ("assert torch.cuda.is_available()" if args.device == "cuda" else "assert True"),
        ),
        "preprocess": (
            preprocess_executable,
            "import numpy, tensorflow, waymo_open_dataset",
        ),
    }
    for name, (interpreter, code) in runtime_probes.items():
        result = subprocess.run(
            [str(interpreter), "-I", "-B", "-c", code],
            env=environment,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"runtime probe failed: {name}: {result.stderr}")
    stage_entries = [
        (preprocess_executable, "tools/external_detector/preprocess_waymo_test_segment.py"),
        (detector_executable, "tools/external_detector/run_open3dml_waymo_pointpillars.py"),
        (pipeline_executable, "tools/external_detector/adapt_open3dml_to_detzero.py"),
        (pipeline_executable, "tracking/tools/run_track.py"),
        (pipeline_executable, "tools/external_detector/run_stage_a_refining.py"),
        (pipeline_executable, "tools/external_detector/combine_grm_prm_no_crm.py"),
        (pipeline_executable, "tools/external_detector/render_waymo_sequence.py"),
        (pipeline_executable, "tools/external_detector/validate_stage_a.py"),
        (pipeline_executable, "tools/external_detector/compare_stage_a_runs.py"),
    ]
    for interpreter, relative in stage_entries:
        result = subprocess.run(
            [str(interpreter), "-B", str(REPO_ROOT / relative), "--help"],
            cwd=REPO_ROOT,
            env=environment,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"entrypoint preflight failed: {relative}: {result.stderr}"
            )
    source_policy = _source_policy(REPO_ROOT)
    source_paths = _preflight_source_paths(REPO_ROOT, source_policy)
    if (
        source_paths != source_policy["source_paths"]
        or args.expected_frames != source_policy["expected_frames"]
    ):
        raise ValueError("live source roster does not match formal Stage-A policy")
    source_rows = {
        relative: {
            "bytes": (REPO_ROOT / relative).stat().st_size,
            "sha256": _sha256(REPO_ROOT / relative),
        }
        for relative in source_paths
    }
    return {
        "status": "PREFLIGHT_PASS",
        "bundle_replay_verifier": _pinned_file(BUNDLE_REPLAY_VERIFIER),
        "runtime_probes": {name: True for name in runtime_probes},
        "role": args.role,
        "expected_frames": args.expected_frames,
        "inputs": inputs,
        "pipeline_python": {
            **_pinned_file(pipeline_python),
            "executable": str(pipeline_executable),
        },
        "detector_python": {
            **_pinned_file(detector_python),
            "executable": str(detector_executable),
        },
        "preprocess_python": {
            **_pinned_file(preprocess_python),
            "executable": str(preprocess_executable),
        },
        "open3dml": {
            "root": str(args.open3dml_root.resolve(strict=True)),
            "declared_commit": args.open3dml_commit,
            "source_file_count": len(open3dml_rows),
            "source_tree_sha256": open3dml_source_tree_sha256,
            "source_lock": open3dml_source_lock,
            "source_files": open3dml_rows,
        },
        "source_file_count": len(source_rows),
        "source_tree_sha256": _tree_hash(source_rows),
    }


def run_command(
    command: list[str],
    log_path: Path,
    environment: dict[str, str],
    cwd: Path = REPO_ROOT,
) -> None:
    """Run one bounded stage and preserve its combined output in a fresh log."""
    with log_path.open("xb") as stream:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        stream.flush()
        os.fsync(stream.fileno())
    if result.returncode != 0:
        diagnostic = log_path.read_text(encoding="utf-8", errors="replace")[-8192:]
        raise RuntimeError(
            f"stage exited {result.returncode}: {' '.join(command)}\n{diagnostic}"
        )


def capture_source_bundle(repo_root: str | Path, bundle: str | Path) -> dict[str, Any]:
    """Copy the exact declared source roster into one immutable no-replace bundle."""
    repo_root = Path(repo_root).resolve(strict=True)
    bundle = Path(bundle).absolute()
    if bundle.exists() or bundle.is_symlink():
        raise FileExistsError(bundle)
    stage = bundle.with_name(f".{bundle.name}.tmp-{uuid.uuid4().hex}")
    rows: dict[str, dict[str, Any]] = {}
    try:
        files_root = stage / "files"
        files_root.mkdir(parents=True)
        for relative in _portable_source_paths(repo_root):
            source = repo_root / relative
            if source.is_symlink() or not source.is_file():
                raise ValueError(
                    f"source must be a regular non-symlink file: {relative}"
                )
            target = files_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = source.read_bytes()
            with target.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            rows[relative] = {
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        manifest = {
            "schema_version": "detzero-stage-a-source-v1",
            "origin_project_root": str(repo_root),
            "source_file_count": len(rows),
            "source_tree_sha256": _tree_hash(rows),
            "source_files": rows,
        }
        with (stage / "source_manifest.json").open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, allow_nan=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        bundle.parent.mkdir(parents=True, exist_ok=True)
        rename_noreplace(stage, bundle)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    verify_source_bundle(repo_root, bundle)
    return manifest


def _load_manifest(path: Path) -> dict[str, Any]:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > 16 * 1024 * 1024
    ):
        raise ValueError("invalid source manifest")

    def reject_duplicate(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    with path.open(encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=reject_duplicate)
    if not isinstance(value, dict):
        raise ValueError("source manifest must be an object")
    return value


def verify_source_bundle(repo_root: str | Path, bundle: str | Path) -> dict[str, Any]:
    """Verify embedded bytes, manifest, and current live-workspace currency."""
    repo_root = Path(repo_root).resolve(strict=True)
    bundle = Path(bundle)
    if bundle.is_symlink() or not bundle.is_dir():
        raise ValueError("invalid source bundle")
    manifest = _load_manifest(bundle / "source_manifest.json")
    rows = manifest.get("source_files")
    if (
        set(manifest)
        != {
            "schema_version",
            "origin_project_root",
            "source_file_count",
            "source_tree_sha256",
            "source_files",
        }
        or manifest["schema_version"] != "detzero-stage-a-source-v1"
        or not isinstance(rows, dict)
    ):
        raise ValueError("source manifest schema mismatch")
    actual_paths = []
    files_root = bundle / "files"
    for node in files_root.rglob("*"):
        if node.is_symlink() or (not node.is_dir() and not node.is_file()):
            raise ValueError("source bundle contains a symlink or special file")
        if node.is_file():
            actual_paths.append(node.relative_to(files_root).as_posix())
    if set(actual_paths) != set(rows):
        raise ValueError("source bundle path set mismatch")
    actual_rows = {}
    for relative in sorted(rows):
        path = files_root / relative
        actual_rows[relative] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    if (
        actual_rows != rows
        or manifest["source_file_count"] != len(rows)
        or manifest["source_tree_sha256"] != _tree_hash(actual_rows)
    ):
        raise ValueError("embedded source bundle mismatch")
    live_paths = _portable_source_paths(repo_root)
    live_rows = {
        relative: {
            "bytes": (repo_root / relative).stat().st_size,
            "sha256": _sha256(repo_root / relative),
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


def replay_source_bundle(
    bundle: str | Path,
    repo_root: str | Path,
    entries: list[tuple[str, str, Path]],
) -> dict[str, Any]:
    """Replay bundled launchers with the pinned isolated verifier."""
    bundle = Path(bundle).resolve(strict=True)
    repo_root = Path(repo_root).resolve(strict=True)
    verifier = _pinned_file(BUNDLE_REPLAY_VERIFIER)
    replayed = []
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for name, entry, interpreter in entries:
        command = [
            str(Path(interpreter).absolute()),
            "-B",
            str(BUNDLE_REPLAY_VERIFIER),
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
        ]
        result = subprocess.run(
            command,
            cwd=bundle.parent,
            env=environment,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"bundle replay failed for {name}: {result.stderr[-8192:]}"
            )
        replayed.append(entry)
    if _pinned_file(BUNDLE_REPLAY_VERIFIER) != verifier:
        raise ValueError("bundle replay verifier drift")
    verify_source_bundle(repo_root, bundle)
    return {
        "bundle_replay_complete": True,
        "entries": replayed,
        "verifier_sha256": verifier["sha256"],
    }


def _write_json_noreplace(path: Path, value: dict[str, Any]) -> None:
    path = path.absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _runtime_identity(interpreter: Path, distributions: list[str]) -> dict[str, Any]:
    executable = interpreter.absolute()
    target = executable.resolve(strict=True)
    code = """
import importlib.metadata, json, platform, sys
versions = {}
for name in sys.argv[1:]:
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None
print(json.dumps({"python": platform.python_version(), "packages": versions}, sort_keys=True))
"""
    result = subprocess.run(
        [str(executable), "-I", "-B", "-c", code, *distributions],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        **_pinned_file(target),
        "executable": str(executable),
        **json.loads(result.stdout),
    }


def _build_run_ledger(run_dir: Path, source_tree_sha256: str) -> dict[str, Any]:
    rows = {}
    for node in run_dir.rglob("*"):
        if node.is_symlink() or (not node.is_dir() and not node.is_file()):
            raise ValueError(f"run contains a symlink or special file: {node}")
        if node.is_file():
            relative = node.relative_to(run_dir).as_posix()
            if relative in {".unaccepted", "run_ledger.json"}:
                continue
            rows[relative] = {
                "bytes": node.stat().st_size,
                "sha256": _sha256(node),
            }
    ledger = {
        "schema_version": "detzero-stage-a-run-ledger-v1",
        "source_tree_sha256": source_tree_sha256,
        "file_count": len(rows),
        "files": rows,
    }
    _write_json_noreplace(run_dir / "run_ledger.json", ledger)
    return ledger


def run_generation(args: argparse.Namespace) -> dict[str, Any]:
    """Execute all Stage A producers once into one fresh, marked run directory."""
    preflight_result = preflight(args)
    run_dir = args.run_dir.absolute()
    receipt_path = args.receipt.absolute()
    pre_audit = receipt_path.with_name(f"{receipt_path.stem}-pre-audit.json")
    post_audit = receipt_path.with_name(f"{receipt_path.stem}-post-audit.json")
    pre_audit_log = receipt_path.with_name(f"{receipt_path.stem}-pre-audit.log")
    post_audit_log = receipt_path.with_name(f"{receipt_path.stem}-post-audit.log")
    lock_path = run_dir.with_name(f".{run_dir.name}.lock")
    for path in (pre_audit, post_audit, pre_audit_log, post_audit_log, lock_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)

    start_wall = datetime.now().astimezone().isoformat()
    start_monotonic = time.monotonic()
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(lock_descriptor, "w", encoding="utf-8") as stream:
        stream.write(f"role={args.role}\nrun={run_dir}\n")
        stream.flush()
        os.fsync(stream.fileno())
    run_dir.mkdir()
    marker = run_dir / ".unaccepted"
    marker.write_text("UNACCEPTED\n", encoding="utf-8")
    logs_dir = run_dir / "logs"
    logs_dir.mkdir()
    commands: list[list[str]] = []
    logs: dict[str, dict[str, Any]] = {}
    source_result: dict[str, Any] | None = None
    input_snapshot: dict[str, Path] | None = None
    environment = _stage_environment(run_dir)

    def execute(
        name: str,
        command: list[str],
        cwd: Path | None = None,
        stage_environment: dict[str, str] | None = None,
    ) -> None:
        commands.append(command)
        log_path = logs_dir / f"{name}.log"
        run_command(
            command,
            log_path,
            stage_environment or environment,
            cwd or execution_root,
        )
        logs[name] = {
            "path": str(log_path.relative_to(run_dir)),
            "bytes": log_path.stat().st_size,
            "sha256": _sha256(log_path),
        }

    try:
        source_manifest = capture_source_bundle(REPO_ROOT, run_dir / "source_bundle")
        if source_manifest["source_tree_sha256"] != preflight_result["source_tree_sha256"]:
            raise ValueError("preflight/source-capture drift")
        pipeline_interpreter = Path(args.pipeline_python).absolute()
        detector_interpreter = Path(args.detector_python).absolute()
        preprocess_interpreter = Path(args.preprocess_python).absolute()
        bundle_replay = replay_source_bundle(
            run_dir / "source_bundle",
            REPO_ROOT,
            [
                ("launcher", "tools/external_detector/run_stage_a.py", pipeline_interpreter),
                ("preprocess", "tools/external_detector/preprocess_waymo_test_segment.py", preprocess_interpreter),
                ("detector", "tools/external_detector/run_open3dml_waymo_pointpillars.py", detector_interpreter),
                ("adapter", "tools/external_detector/adapt_open3dml_to_detzero.py", pipeline_interpreter),
                ("tracking", "tracking/tools/run_track.py", pipeline_interpreter),
                ("refining", "tools/external_detector/run_stage_a_refining.py", pipeline_interpreter),
                ("final", "tools/external_detector/combine_grm_prm_no_crm.py", pipeline_interpreter),
                ("visuals", "tools/external_detector/render_waymo_sequence.py", pipeline_interpreter),
                ("validator", "tools/external_detector/validate_stage_a.py", pipeline_interpreter),
                ("comparator", "tools/external_detector/compare_stage_a_runs.py", pipeline_interpreter),
            ],
        )
        execution_root = run_dir / "source_bundle/files"
        environment["PYTHONPATH"] = os.pathsep.join(
            str(execution_root / path) for path in ("tracking", "refining", "utils")
        ) + os.pathsep + str(execution_root)
        input_snapshot = _snapshot_inputs(run_dir, preflight_result)
        detector_environment = dict(environment)
        detector_environment["OPEN3D_ML_ROOT"] = str(
            input_snapshot["open3dml_root"].parent
        )

        frozen_command = "#!/usr/bin/env bash\nset -euo pipefail\nexec " + shlex.join(
            [
                str(Path(args.pipeline_python).absolute()),
                "-B",
                str(execution_root / "tools/external_detector/run_stage_a.py"),
                *sys.argv[1:],
            ]
        ) + "\n"
        command_path = run_dir / "command.sh"
        command_path.write_text(frozen_command, encoding="utf-8")
        command_path.chmod(0o555)
        subprocess.run(["bash", "-n", str(command_path)], check=True)

        pipeline_python = str(Path(args.pipeline_python).absolute())
        detector_python = str(Path(args.detector_python).absolute())
        preprocess_python = str(Path(args.preprocess_python).absolute())
        waymo_root = run_dir / "data/waymo"
        detector_dir = run_dir / "detector"
        adapter_dir = run_dir / "adapter"
        tracking_dir = run_dir / "tracking"
        refining_dir = run_dir / "refining"
        final_dir = run_dir / "final"
        visuals_dir = run_dir / "visuals"
        sequence = sequence_name_from_tfrecord(args.input_tfrecord)
        info_path = (
            waymo_root
            / "waymo_processed_data"
            / f"segment-{sequence}"
            / f"{sequence}.pkl"
        )
        execute(
            "01-preprocess",
            [
                preprocess_python,
                "-B",
                str(execution_root / "tools/external_detector/preprocess_waymo_test_segment.py"),
                "--input-tfrecord",
                str(input_snapshot["input_tfrecord"]),
                "--source-identity-tfrecord",
                preflight_result["inputs"]["input_tfrecord"]["path"],
                "--root-path",
                str(waymo_root),
                "--expected-frames",
                str(args.expected_frames),
                "--non-commercial-research",
                "--waymo-terms-accepted",
            ],
        )
        execute(
            "02-detector",
            [
                detector_python,
                "-B",
                str(execution_root / "tools/external_detector/run_open3dml_waymo_pointpillars.py"),
                "--preprocessed-root",
                str(waymo_root),
                "--config",
                str(input_snapshot["detector_config"]),
                "--config-identity",
                preflight_result["inputs"]["detector_config"]["path"],
                "--config-sha256",
                args.detector_config_sha256,
                "--checkpoint",
                str(input_snapshot["detector_checkpoint"]),
                "--checkpoint-identity",
                preflight_result["inputs"]["detector_checkpoint"]["path"],
                "--checkpoint-sha256",
                args.detector_checkpoint_sha256,
                "--open3dml-root",
                str(input_snapshot["open3dml_root"]),
                "--output-dir",
                str(detector_dir),
                "--device",
                args.device,
                "--open3dml-commit",
                args.open3dml_commit,
            ],
            stage_environment=detector_environment,
        )
        execute(
            "03-adapter",
            [
                pipeline_python,
                "-B",
                str(execution_root / "tools/external_detector/adapt_open3dml_to_detzero.py"),
                "--raw-predictions",
                str(detector_dir / "raw_predictions.npz"),
                "--waymo-info",
                str(info_path),
                "--output-dir",
                str(adapter_dir),
                "--expected-frames",
                str(args.expected_frames),
            ],
        )
        execute(
            "04-tracking",
            [
                pipeline_python,
                "-B",
                str(execution_root / "tracking/tools/run_track.py"),
                "--cfg_file",
                str(input_snapshot["tracking_config"]),
                "--data_path",
                str(adapter_dir / "detzero_result.pkl"),
                "--output_path",
                str(tracking_dir),
                "--root_path",
                str(waymo_root),
                "--split",
                "test",
                "--workers",
                "0",
            ],
            execution_root / "tracking/tools",
        )
        execute(
            "05-refining",
            [
                pipeline_python,
                "-B",
                str(execution_root / "tools/external_detector/run_stage_a_refining.py"),
                "--waymo-root",
                str(waymo_root),
                "--tracking",
                str(tracking_dir / "tracking.pkl"),
                "--output-dir",
                str(refining_dir),
                "--checkpoint-root",
                str(input_snapshot["refining_checkpoint_root"]),
                "--checkpoint-identity-root",
                str(Path(preflight_result["inputs"]["refining_checkpoints"][0]["path"]).parent),
                "--device",
                args.refining_device,
                "--workers",
                "0",
            ],
        )
        execute(
            "06-final",
            [
                pipeline_python,
                "-B",
                str(execution_root / "tools/external_detector/combine_grm_prm_no_crm.py"),
                "--tracking",
                str(tracking_dir / "tracking.pkl"),
                "--detector-frames",
                str(adapter_dir / "detzero_result.pkl"),
                "--refining-dir",
                str(refining_dir),
                "--output-dir",
                str(final_dir),
            ],
        )
        execute(
            "07-visuals",
            [
                pipeline_python,
                "-B",
                str(execution_root / "tools/external_detector/render_waymo_sequence.py"),
                "--waymo-root",
                str(waymo_root),
                "--detector-frames",
                str(adapter_dir / "detzero_result.pkl"),
                "--final-frames",
                str(final_dir / "final_frame_grm_prm_score_passthrough.pkl"),
                "--output-dir",
                str(visuals_dir),
                "--expected-frames",
                str(args.expected_frames),
            ],
        )

        for name in (
            "input_tfrecord",
            "detector_config",
            "detector_checkpoint",
            "tracking_config",
        ):
            expected = preflight_result["inputs"][name]
            actual = _pinned_file(input_snapshot[name], expected["sha256"])
            if actual["bytes"] != expected["bytes"]:
                raise ValueError(f"input snapshot size drift: {name}")
        for expected in preflight_result["inputs"]["refining_checkpoints"]:
            actual = _pinned_file(
                input_snapshot["refining_checkpoint_root"] / Path(expected["path"]).name,
                expected["sha256"],
            )
            if actual["bytes"] != expected["bytes"]:
                raise ValueError("refining checkpoint snapshot size drift")
        if _regular_tree(input_snapshot["open3dml_root"], SOURCE_SUFFIXES) != (
            preflight_result["open3dml"]["source_files"]
        ):
            raise ValueError("Open3D-ML source snapshot drift")
        shutil.rmtree(run_dir / ".input-snapshot")
        input_snapshot = None

        source_result = verify_source_bundle(REPO_ROOT, run_dir / "source_bundle")
        for name in (
            "input_tfrecord",
            "detector_config",
            "detector_checkpoint",
            "tracking_config",
        ):
            row = preflight_result["inputs"][name]
            if _pinned_file(Path(row["path"]), row["sha256"]) != row:
                raise ValueError(f"input drift: {name}")
        for row in preflight_result["inputs"]["refining_checkpoints"]:
            if _pinned_file(Path(row["path"]), row["sha256"]) != row:
                raise ValueError("refining checkpoint drift")
        if (
            _tree_hash(_regular_tree(args.open3dml_root, SOURCE_SUFFIXES))
            != preflight_result["open3dml"]["source_tree_sha256"]
        ):
            raise ValueError("Open3D-ML source drift")
        metadata = {
            "schema_version": "detzero-stage-a-run-v1",
            "role": args.role,
            "expected_frames": args.expected_frames,
            "source": source_result,
            "bundle_replay": bundle_replay,
            "preflight": preflight_result,
            "runtime": {
                "pipeline": _runtime_identity(
                    Path(args.pipeline_python),
                    ["numpy", "torch"],
                ),
                "detector": _runtime_identity(
                    Path(args.detector_python),
                    ["numpy", "torch", "open3d"],
                ),
                "preprocess": _runtime_identity(
                    Path(args.preprocess_python),
                    ["numpy", "tensorflow", "waymo-open-dataset-tf-2-13-0"],
                ),
            },
            "command": {
                "argv": sys.argv,
                "script": "command.sh",
                "script_sha256": _sha256(command_path),
            },
            "commands": commands,
            "logs": logs,
        }
        _write_json_noreplace(run_dir / "run_metadata.json", metadata)
        _build_run_ledger(run_dir, source_result["source_tree_sha256"])

        validator_command = [
            pipeline_python,
            "-B",
            str(execution_root / "tools/external_detector/validate_stage_a.py"),
            "--run-root",
            str(run_dir),
            "--live-repo-root",
            str(REPO_ROOT),
            "--pipeline-python",
            str(Path(args.pipeline_python).absolute()),
            "--detector-python",
            str(Path(args.detector_python).absolute()),
            "--preprocess-python",
            str(Path(args.preprocess_python).absolute()),
            "--waymo-root",
            str(waymo_root),
            "--detector-dir",
            str(detector_dir),
            "--adapter-dir",
            str(adapter_dir),
            "--tracking",
            str(tracking_dir / "tracking.pkl"),
            "--dropped",
            str(tracking_dir / "dropped.pkl"),
            "--refining-dir",
            str(refining_dir),
            "--final-dir",
            str(final_dir),
            "--visuals-dir",
            str(visuals_dir),
            "--expected-frames",
            str(args.expected_frames),
        ]
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        run_command(
            [*validator_command, "--report", str(pre_audit)],
            pre_audit_log,
            environment,
        )
        run_command(
            [*validator_command, "--report", str(post_audit)],
            post_audit_log,
            environment,
        )
        end_wall = datetime.now().astimezone().isoformat()
        receipt = {
            "schema_version": "detzero-stage-a-run-receipt-v1",
            "status": "PASS",
            "role": args.role,
            "run_dir": str(run_dir),
            "destination_absent_before_launch": True,
            "start_time": start_wall,
            "end_time": end_wall,
            "elapsed_seconds": time.monotonic() - start_monotonic,
            "source_tree_sha256": source_result["source_tree_sha256"],
            "run_ledger_sha256": _sha256(run_dir / "run_ledger.json"),
            "pre_audit_sha256": _sha256(pre_audit),
            "post_audit_sha256": _sha256(post_audit),
            "pre_audit_log_sha256": _sha256(pre_audit_log),
            "post_audit_log_sha256": _sha256(post_audit_log),
            "logs": logs,
            "command": sys.argv,
        }
        _write_json_noreplace(receipt_path, receipt)
        from tools.external_detector.validate_stage_a import load_json_strict

        if load_json_strict(receipt_path) != receipt:
            marker.write_text("REJECTED\n", encoding="utf-8")
            raise RuntimeError("receipt publication failed")
        marker.unlink()
        return receipt
    except BaseException as error:
        if input_snapshot is not None:
            shutil.rmtree(run_dir / ".input-snapshot", ignore_errors=True)
        if run_dir.is_dir() and not marker.exists():
            marker.write_text("REJECTED\n", encoding="utf-8")
        failure = {
            "schema_version": "detzero-stage-a-run-receipt-v1",
            "status": "FAIL",
            "role": args.role,
            "run_dir": str(run_dir),
            "start_time": start_wall,
            "end_time": datetime.now().astimezone().isoformat(),
            "elapsed_seconds": time.monotonic() - start_monotonic,
            "error": f"{type(error).__name__}: {error}",
            "source": source_result,
            "logs": logs,
            "command": sys.argv,
        }
        if not receipt_path.exists() and not receipt_path.is_symlink():
            _write_json_noreplace(receipt_path, failure)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tfrecord", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--detector-config", type=Path, required=True)
    parser.add_argument("--detector-config-sha256", required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-checkpoint-sha256", required=True)
    parser.add_argument("--tracking-config", type=Path, required=True)
    parser.add_argument("--refining-checkpoint-root", type=Path, required=True)
    parser.add_argument("--open3dml-root", type=Path, required=True)
    parser.add_argument("--open3dml-commit", required=True)
    parser.add_argument("--pipeline-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--detector-python", type=Path, required=True)
    parser.add_argument("--preprocess-python", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--refining-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--expected-frames", type=int, default=199)
    parser.add_argument("--role", choices=("A", "B"), required=True)
    parser.add_argument("--non-commercial-research", action="store_true", required=True)
    parser.add_argument("--waymo-terms-accepted", action="store_true", required=True)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.preflight:
        print(json.dumps(preflight(args), allow_nan=False, sort_keys=True))
        return 0
    print(json.dumps(run_generation(args), allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
