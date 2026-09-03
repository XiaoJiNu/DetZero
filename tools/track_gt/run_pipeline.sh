#!/usr/bin/env bash
set -euo pipefail
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-/data/software/conda/anaconda3/envs/mv2d/bin/python}"
exec "$PY" "$WT/tools/track_gt/run_track_gt_pipeline.py" "$@"
