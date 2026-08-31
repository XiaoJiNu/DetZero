#!/usr/bin/env python3
"""Stream one Waymo testing TFRecord into DetZero's native data layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.external_detector.pipeline import (
    merge_waymo_lidar_returns,
    parse_waymo_frame,
    preprocess_tfrecord,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tfrecord", type=Path, required=True)
    parser.add_argument("--source-identity-tfrecord", type=Path, required=True)
    parser.add_argument("--root-path", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, default=199)
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Process only this prefix for a canary; omit for the complete segment.",
    )
    parser.add_argument("--non-commercial-research", action="store_true", required=True)
    parser.add_argument("--waymo-terms-accepted", action="store_true", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_frames is not None and args.max_frames != args.expected_frames:
        raise ValueError("--max-frames must equal --expected-frames for a prefix canary")

    import tensorflow as tf
    from waymo_open_dataset import dataset_pb2
    from waymo_open_dataset.utils import frame_utils

    def load_frames(path: Path):
        dataset = tf.data.TFRecordDataset(str(path), compression_type="")
        for frame_id, serialized in enumerate(dataset):
            if args.max_frames is not None and frame_id >= args.max_frames:
                break
            yield parse_waymo_frame(serialized, dataset_pb2.Frame)

    def extract_points(frame):
        range_images, camera_projections, _, top_pose = (
            frame_utils.parse_range_image_and_camera_projection(frame)
        )
        point_returns = []
        nlz_returns = []
        calibrations = sorted(frame.context.laser_calibrations, key=lambda item: item.name)
        for return_id in (0, 1):
            points, _ = frame_utils.convert_range_image_to_point_cloud(
                frame,
                range_images,
                camera_projections,
                top_pose,
                ri_index=return_id,
                keep_polar_features=True,
            )
            point_returns.append(points)
            nlz_returns.append([])
            for calibration in calibrations:
                range_image = range_images[calibration.name][return_id]
                tensor = tf.reshape(
                    tf.convert_to_tensor(range_image.data), range_image.shape.dims
                )
                mask = tensor[..., 0] > 0
                nlz_returns[-1].append(tf.boolean_mask(tensor[..., 3], mask).numpy())
        return merge_waymo_lidar_returns(point_returns, nlz_returns)

    manifest = preprocess_tfrecord(
        args.input_tfrecord,
        args.root_path,
        expected_frame_count=args.expected_frames,
        frame_loader=load_frames,
        point_extractor=extract_points,
        non_commercial_research=args.non_commercial_research,
        waymo_terms_accepted=args.waymo_terms_accepted,
        source_identity_path=args.source_identity_tfrecord,
    )
    print(json.dumps(manifest, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
