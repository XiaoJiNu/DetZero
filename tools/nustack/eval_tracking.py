#!/usr/bin/env python3
"""N4: official nuScenes TrackingEval on mini_val for each comparison run.

Runs devkit TrackingEval (tracking_nips_2019 config, mini_val split) on one
or more tracking submission jsons and dumps a merged metric table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# motmetrics>=1.0 builds `MultiIndex.from_tuples([], names=['FrameId','Event'])`
# for classes with no events (e.g. trailer on mini_val) and pandas>=1.3 raises
# on the level-count mismatch. Empty-arrays keeps the 2 named levels.
_orig_from_tuples = pd.MultiIndex.from_tuples


def _safe_from_tuples(tuples, sortorder=None, names=None):
    try:
        return _orig_from_tuples(tuples, sortorder=sortorder, names=names)
    except ValueError:
        if names and len(list(tuples)) == 0:
            return pd.MultiIndex.from_arrays([[]] * len(names), names=names)
        raise


pd.MultiIndex.from_tuples = _safe_from_tuples

from nuscenes.eval.common.config import config_factory  # noqa: E402
from nuscenes.eval.tracking.evaluate import TrackingEval  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True,
                        help="NAME=path pairs of tracking submission jsons")
    parser.add_argument("--dataroot", type=Path,
                        default=Path("/data/data/automomous/nuscenes/v1.0-mini"))
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--eval-set", default="mini_val")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    merged = {}
    for spec in args.results:
        name, _, path = str(spec).partition("=")
        cfg = config_factory("tracking_nips_2019")
        out_dir = Path(path).parent / f"trackeval_{name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        eval_api = TrackingEval(
            config=cfg,
            result_path=str(path),
            eval_set=args.eval_set,
            output_dir=str(out_dir),
            nusc_version=args.version,
            nusc_dataroot=str(args.dataroot),
            verbose=False,
        )
        summary = eval_api.main(render_curves=True)
        keys = ("amota", "amotp", "mota", "motp", "recalled", "idf1",
                "idsw", "frag", "tid", "lgd", "tenant", "mt", "ml")
        merged[name] = {"overall": {k: summary.get(k) for k in keys if k in summary},
                        "amota_per_class": summary["label_metrics"].get("amota", {}),
                        "runtime_s": summary.get("eval_time")}
        print(name, json.dumps(merged[name]["overall"], sort_keys=True))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
