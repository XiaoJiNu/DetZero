#!/usr/bin/env python3
"""Thin wrapper around official nuScenes TrackingEval."""
import argparse
import json
import os

from nuscenes.eval.common.config import config_factory
from nuscenes.eval.tracking.data_classes import TrackingConfig
from nuscenes.eval.tracking.evaluate import TrackingEval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_path")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--eval_set", default="mini_val")
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--config_path", default="")
    ap.add_argument("--render_curves", type=int, default=0)
    ap.add_argument("--verbose", type=int, default=1)
    args = ap.parse_args()

    if args.config_path:
        with open(args.config_path, "r") as f:
            cfg = TrackingConfig.deserialize(json.load(f))
    else:
        cfg = config_factory("tracking_nips_2019")

    nusc_eval = TrackingEval(
        config=cfg,
        result_path=os.path.expanduser(args.result_path),
        eval_set=args.eval_set,
        output_dir=os.path.expanduser(args.output_dir),
        nusc_version=args.version,
        nusc_dataroot=args.dataroot,
        verbose=bool(args.verbose),
    )
    nusc_eval.main(render_curves=bool(args.render_curves))


if __name__ == "__main__":
    main()
