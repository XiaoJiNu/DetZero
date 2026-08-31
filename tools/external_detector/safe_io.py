#!/usr/bin/env python3
"""Descriptor-bound safe readers for Stage-A trust boundaries."""

from __future__ import annotations

import collections
import os
from pathlib import Path
import pickle
import stat
from typing import Any

import numpy as np


_multiarray = getattr(np.core, "multiarray")
_numeric = getattr(np.core, "numeric")
_ALLOWED_PICKLE_GLOBALS = {
    ("builtins", "list"): list,
    ("collections", "defaultdict"): collections.defaultdict,
    ("numpy", "dtype"): np.dtype,
    ("numpy", "ndarray"): np.ndarray,
    ("numpy.core.multiarray", "_reconstruct"): _multiarray._reconstruct,
    ("numpy.core.multiarray", "scalar"): _multiarray.scalar,
    ("numpy._core.multiarray", "_reconstruct"): _multiarray._reconstruct,
    ("numpy._core.multiarray", "scalar"): _multiarray.scalar,
    ("numpy.core.numeric", "_frombuffer"): _numeric._frombuffer,
    ("numpy._core.numeric", "_frombuffer"): _numeric._frombuffer,
}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        try:
            return _ALLOWED_PICKLE_GLOBALS[(module, name)]
        except KeyError as error:
            raise ValueError(f"forbidden pickle global: {module}.{name}") from error


def open_bounded_regular(
    path: str | Path, maximum_bytes: int, *, allow_empty: bool = False
) -> int:
    """Open a bounded regular file without following any path symlink."""
    absolute = Path(os.path.abspath(os.fspath(path)))
    if absolute == Path("/") or maximum_bytes < 0:
        raise ValueError(f"invalid bounded regular file: {path}")
    directory = os.open(
        "/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    )
    try:
        for component in absolute.parts[1:-1]:
            child = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=directory,
            )
            os.close(directory)
            directory = child
        descriptor = os.open(
            absolute.name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=directory,
        )
    except OSError as error:
        raise ValueError(f"invalid bounded regular file: {path}") from error
    finally:
        os.close(directory)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"invalid bounded regular file: {path}")
        if (not allow_empty and info.st_size == 0) or info.st_size > maximum_bytes:
            raise ValueError(f"invalid bounded regular file: {path}")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def safe_load_pickle(path: str | Path, maximum_bytes: int = 1024**3) -> Any:
    """Load NumPy/container-only pickle data from one pinned descriptor."""
    descriptor = open_bounded_regular(path, maximum_bytes)
    with os.fdopen(descriptor, "rb") as stream:
        return _RestrictedUnpickler(stream).load()
