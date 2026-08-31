#!/usr/bin/env python3
"""Audited Open3D-ML source locks used by the Stage-A release path."""

from __future__ import annotations

from typing import Any


OPEN3DML_SOURCE_LOCKS: dict[str, dict[str, Any]] = {
    "fcf97c07bf7a113a47d0fcf63760b245c2a2784e": {
        "upstream_url": "https://github.com/isl-org/Open3D-ML.git",
        "git_ref": "refs/tags/v0.18.0",
        "source_subtree": "ml3d",
        "source_file_count": 173,
        "source_tree_sha256": "cc52a3eb417e0fd754d732856fa7eef219985e138e32b3b22a3536b18c8598ca",
        "license_spdx": "MIT",
        "license_sha256": "14f4eb78224ed8fa32d6d9316c059734fffcb74beb84c1984e5c745771c0b15e",
    }
}


def require_locked_open3dml_source(
    commit: str, source_file_count: int, source_tree_sha256: str
) -> dict[str, Any]:
    lock = OPEN3DML_SOURCE_LOCKS.get(commit)
    if lock is None:
        raise ValueError(f"unlocked Open3D-ML commit: {commit}")
    if (
        lock["source_file_count"] != source_file_count
        or lock["source_tree_sha256"] != source_tree_sha256
    ):
        raise ValueError("Open3D-ML source identity mismatch")
    return dict(lock)
