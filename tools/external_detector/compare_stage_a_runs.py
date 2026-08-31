#!/usr/bin/env python3
"""Compare two validated Stage-A runs without conflating volatile execution evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.external_detector.validate_stage_a import (
    _sha256_file,
    expected_stage_a_paths,
    load_json_strict,
    validate_run_ledger,
    write_report_noreplace,
)


_VOLATILE_JSON_KEYS = {"elapsed_seconds"}


def _canonical(value: Any, root: Path) -> Any:
    if isinstance(value, dict):
        return {
            key: _canonical(item, root)
            for key, item in value.items()
            if key not in _VOLATILE_JSON_KEYS
        }
    if isinstance(value, list):
        return [_canonical(item, root) for item in value]
    if isinstance(value, str):
        return value.replace(str(root), "<RUN_ROOT>")
    return value


def _stable_metadata(
    path: Path, root: Path, role: str, ledger_row: dict[str, Any]
) -> dict[str, Any]:
    metadata = load_json_strict(
        path,
        expected_bytes=ledger_row["bytes"],
        expected_sha256=ledger_row["sha256"],
    )
    required = {
        "schema_version",
        "role",
        "expected_frames",
        "source",
        "bundle_replay",
        "preflight",
        "runtime",
        "command",
        "commands",
        "logs",
    }
    if (
        not isinstance(metadata, dict)
        or set(metadata) != required
        or metadata["schema_version"] != "detzero-stage-a-run-v1"
        or metadata["role"] != role
        or not isinstance(metadata["preflight"], dict)
        or metadata["preflight"].get("role") != role
    ):
        raise ValueError(f"run metadata mismatch: {role}")
    command = metadata["command"]
    commands = metadata["commands"]
    logs = metadata["logs"]
    if (
        not isinstance(command, dict)
        or set(command) != {"argv", "script", "script_sha256"}
        or not isinstance(command["argv"], list)
        or not all(isinstance(item, str) for item in command["argv"])
        or command["script"] != "command.sh"
        or not isinstance(command["script_sha256"], str)
        or len(command["script_sha256"]) != 64
        or not isinstance(commands, list)
        or not all(
            isinstance(items, list) and all(isinstance(item, str) for item in items)
            for items in commands
        )
        or not isinstance(logs, dict)
    ):
        raise ValueError(f"run execution metadata mismatch: {role}")
    argv = list(command["argv"])
    for option, expected, replacement in (
        ("--run-dir", str(root), "<RUN_ROOT>"),
        ("--receipt", None, "<RECEIPT>"),
        ("--role", role, "<ROLE>"),
    ):
        indexes = [index for index, value in enumerate(argv) if value == option]
        if (
            len(indexes) != 1
            or indexes[0] + 1 >= len(argv)
            or (expected is not None and argv[indexes[0] + 1] != expected)
        ):
            raise ValueError(f"run command mismatch: {role}")
        argv[indexes[0] + 1] = replacement
    stable_logs = {}
    for name, row in logs.items():
        if (
            not isinstance(name, str)
            or not isinstance(row, dict)
            or set(row) != {"path", "bytes", "sha256"}
            or not isinstance(row["path"], str)
            or type(row["bytes"]) is not int
            or row["bytes"] < 0
            or not isinstance(row["sha256"], str)
            or len(row["sha256"]) != 64
        ):
            raise ValueError(f"run log metadata mismatch: {role}")
        stable_logs[name] = {"path": row["path"]}
    preflight = dict(metadata["preflight"])
    preflight.pop("role")
    return _canonical(
        {
            "schema_version": metadata["schema_version"],
            "expected_frames": metadata["expected_frames"],
            "source": metadata["source"],
            "bundle_replay": metadata["bundle_replay"],
            "preflight": preflight,
            "runtime": metadata["runtime"],
            "command": {"argv": argv, "script": command["script"]},
            "commands": commands,
            "logs": stable_logs,
        },
        root,
    )


def _validate_receipts(
    root: Path,
    role: str,
    source_tree_sha256: str,
    ledger_sha256: str,
    receipt_path: Path,
    expected_file_count: int,
    ledger_file_count: int,
) -> dict[str, Any]:
    """Strictly verify one external run receipt and its pre/post audits."""
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError(f"missing run receipt: {receipt_path}")
    receipt = load_json_strict(receipt_path)
    if not isinstance(receipt, dict) or receipt.get("schema_version") != (
        "detzero-stage-a-run-receipt-v1"
    ):
        raise ValueError(f"invalid run receipt schema: {receipt_path}")
    if receipt.get("status") != "PASS" or receipt.get("role") != role:
        raise ValueError(f"run receipt is not a PASS for role {role}: {receipt_path}")
    if receipt.get("destination_absent_before_launch") is not True:
        raise ValueError(
            f"run receipt destination was not absent before launch: {receipt_path}"
        )
    if receipt.get("run_dir") != str(root) or receipt.get("source_tree_sha256") != (
        source_tree_sha256
    ):
        raise ValueError(f"run receipt identity mismatch: {receipt_path}")
    if receipt.get("run_ledger_sha256") != ledger_sha256:
        raise ValueError(f"run receipt ledger hash mismatch: {receipt_path}")
    pre_audit = receipt_path.with_name(f"{receipt_path.stem}-pre-audit.json")
    post_audit = receipt_path.with_name(f"{receipt_path.stem}-post-audit.json")
    for name, audit_path in (("pre", pre_audit), ("post", post_audit)):
        if audit_path.is_symlink() or not audit_path.is_file():
            raise ValueError(f"missing {name} audit report: {audit_path}")
        audit = load_json_strict(audit_path)
        inputs = audit.get("inputs") if isinstance(audit, dict) else None
        checks = audit.get("checks") if isinstance(audit, dict) else None
        if (
            not isinstance(audit, dict)
            or audit.get("schema_version") != "detzero-stage-a-validation-v1"
            or audit.get("passed") is not True
            or audit.get("artifact_validation_passed") is not True
            or audit.get("expected_frames") != 199
            or audit.get("error") is not None
        ):
            raise ValueError(f"{name} audit did not pass: {audit_path}")
        if (
            not isinstance(inputs, dict)
            or inputs.get("run_root") != str(root)
            or not isinstance(checks, dict)
            or checks.get("run_ledger")
            != {
                "file_count": ledger_file_count,
                "source_tree_sha256": source_tree_sha256,
            }
            or checks.get("stage_a_profile")
            != {"expected_file_count": expected_file_count}
            or checks.get("acceptance_marker") != {"state": "UNACCEPTED"}
        ):
            raise ValueError(f"{name} audit identity mismatch: {audit_path}")
    expected = {
        "pre_audit_sha256": pre_audit,
        "post_audit_sha256": post_audit,
        "pre_audit_log_sha256": receipt_path.with_name(
            f"{receipt_path.stem}-pre-audit.log"
        ),
        "post_audit_log_sha256": receipt_path.with_name(
            f"{receipt_path.stem}-post-audit.log"
        ),
    }
    for field, path in expected.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing receipt-side file: {path}")
        digest = _sha256_file(path)
        if receipt.get(field) != digest:
            raise ValueError(f"receipt {field} mismatch: {path}")
    return {
        "receipt": str(receipt_path),
        "pre_audit": str(pre_audit),
        "post_audit": str(post_audit),
    }


def compare_stage_a_runs(
    run_a: str | Path,
    run_b: str | Path,
) -> dict[str, Any]:
    roots = [Path(run_a).absolute(), Path(run_b).absolute()]
    for root in roots:
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f"invalid run root: {root}")
        marker = root / ".unaccepted"
        if marker.exists() or marker.is_symlink():
            raise ValueError(f"run is not accepted: {root}")

    audits = [validate_run_ledger(root) for root in roots]
    ledgers = [load_json_strict(root / "run_ledger.json") for root in roots]
    ledger_rows = [ledger["files"] for ledger in ledgers]
    if audits[0]["source_tree_sha256"] != audits[1]["source_tree_sha256"]:
        raise ValueError("source trees differ")
    ledger_hashes = [
        _sha256_file(root / "run_ledger.json")
        for root in roots
    ]
    expected_sets = [expected_stage_a_paths(root) for root in roots]
    if expected_sets[0] != expected_sets[1]:
        raise ValueError("expected Stage-A path sets differ between runs")
    metadata_argv = [
        load_json_strict(
            root / "run_metadata.json",
            expected_bytes=rows["run_metadata.json"]["bytes"],
            expected_sha256=rows["run_metadata.json"]["sha256"],
        )["command"]["argv"]
        for root, rows in zip(roots, ledger_rows)
    ]
    receipt_paths = []
    for argv in metadata_argv:
        receipt_indexes = [
            index for index, value in enumerate(argv) if value == "--receipt"
        ]
        if len(receipt_indexes) != 1 or receipt_indexes[0] + 1 >= len(argv):
            raise ValueError("run metadata lacks one --receipt path")
        receipt_paths.append(Path(argv[receipt_indexes[0] + 1]))
    receipts = [
        _validate_receipts(
            root,
            role,
            source_tree["source_tree_sha256"],
            ledger_hash,
            receipt_path,
            len(expected_set),
            source_tree["file_count"],
        )
        for root, role, source_tree, ledger_hash, receipt_path, expected_set in zip(
            roots,
            ("A", "B"),
            audits,
            ledger_hashes,
            receipt_paths,
            expected_sets,
        )
    ]
    paths = expected_sets[0] - {"run_ledger.json"}

    metadata = [
        _stable_metadata(
            roots[0] / "run_metadata.json",
            roots[0],
            "A",
            ledger_rows[0]["run_metadata.json"],
        ),
        _stable_metadata(
            roots[1] / "run_metadata.json",
            roots[1],
            "B",
            ledger_rows[1]["run_metadata.json"],
        ),
    ]
    if metadata[0] != metadata[1]:
        raise ValueError("stable run metadata differs")

    exact_count = 0
    json_count = 0
    volatile_count = 0
    for relative in sorted(paths):
        if relative == "command.sh" or relative.startswith("logs/"):
            volatile_count += 1
            continue
        if relative == "run_metadata.json":
            continue
        if relative.endswith(".json"):
            values = [
                _canonical(
                    load_json_strict(
                        root / relative,
                        expected_bytes=rows[relative]["bytes"],
                        expected_sha256=rows[relative]["sha256"],
                    ),
                    root,
                )
                for root, rows in zip(roots, ledger_rows)
            ]
            if values[0] != values[1]:
                raise ValueError(f"semantic JSON mismatch: {relative}")
            json_count += 1
            continue
        rows = [ledger_rows[index][relative] for index in range(2)]
        if rows[0] != rows[1]:
            raise ValueError(f"exact artifact mismatch: {relative}")
        exact_count += 1

    try:
        end_audits = [validate_run_ledger(root) for root in roots]
        end_ledger_hashes = [
            _sha256_file(root / "run_ledger.json") for root in roots
        ]
        end_receipts = [
            _validate_receipts(
                root,
                role,
                source_tree["source_tree_sha256"],
                ledger_hash,
                receipt_path,
                len(expected_set),
                source_tree["file_count"],
            )
            for root, role, source_tree, ledger_hash, receipt_path, expected_set in zip(
                roots,
                ("A", "B"),
                audits,
                ledger_hashes,
                receipt_paths,
                expected_sets,
            )
        ]
    except Exception as error:
        raise ValueError("run changed during comparison") from error
    if (
        end_audits != audits
        or end_ledger_hashes != ledger_hashes
        or end_receipts != receipts
    ):
        raise ValueError("run changed during comparison")

    return {
        "schema_version": "detzero-stage-a-comparison-v1",
        "status": "PASS",
        "run_a": str(roots[0]),
        "run_b": str(roots[1]),
        "source_tree_sha256": audits[0]["source_tree_sha256"],
        "classified_file_count": len(paths),
        "exact_file_count": exact_count,
        "semantic_json_file_count": json_count,
        "stable_metadata_file_count": 1,
        "volatile_execution_file_count": volatile_count,
        "stable_run_metadata_equal": True,
        "receipts": receipts,
        "expected_file_count": len(expected_sets[0]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = compare_stage_a_runs(args.run_a, args.run_b)
    write_report_noreplace(args.report, report)
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
