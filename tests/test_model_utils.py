from pathlib import Path
import sys

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "utils"))

from detzero_utils.model_utils import (  # noqa: E402
    _load_checkpoint,
    load_params_from_file,
)


class _Logger:
    def info(self, _message):
        pass


def test_load_params_accepts_legacy_numpy_scalar_metadata(tmp_path):
    source = torch.nn.Linear(2, 1)
    with torch.no_grad():
        source.weight.fill_(2.5)
        source.bias.fill_(-0.5)

    checkpoint_path = tmp_path / "legacy_checkpoint.pth"
    torch.save(
        {
            "model_state": source.state_dict(),
            "epoch": np.float64(3.0),
            "it": np.float64(9.0),
            "version": "legacy-detzero",
        },
        checkpoint_path,
    )

    target = torch.nn.Linear(2, 1)
    load_params_from_file(target, str(checkpoint_path), _Logger(), to_cpu=True)

    assert torch.equal(target.weight, source.weight)
    assert torch.equal(target.bias, source.bias)


def test_checkpoint_loader_fails_closed_without_safe_globals(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pth"
    checkpoint_path.write_bytes(b"must-not-be-loaded")
    load_calls = []

    monkeypatch.delattr(torch.serialization, "safe_globals")

    def unrestricted_load(*args, **kwargs):
        load_calls.append((args, kwargs))
        return {"model_state": {}}

    monkeypatch.setattr(torch, "load", unrestricted_load)

    with pytest.raises(RuntimeError, match="restricted checkpoint loading"):
        _load_checkpoint(checkpoint_path, map_location=torch.device("cpu"))

    assert load_calls == []
