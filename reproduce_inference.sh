#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/data/software/conda/anaconda3/envs/mv2d/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
    printf 'DetZero inference interpreter not found: %s\n' "${PYTHON}" >&2
    exit 2
fi

exec "${PYTHON}" "${REPO_ROOT}/refining/tools/reproduce_inference.py" "$@"
