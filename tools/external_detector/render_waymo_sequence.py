#!/usr/bin/env python3
"""Render every Stage A Waymo frame as a standalone BEV PNG."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any
import uuid

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.external_detector.pipeline import (  # noqa: E402
    _sha256_file,
    _validated_points,
    generation_relative_provenance,
    rename_noreplace,
)
from tools.external_detector.safe_io import safe_load_pickle


IMAGE_SIZE = (1280, 640)
PANEL_TOP = 40
PANEL_SIZE = (600, 580)
PANEL_LEFTS = (20, 660)
XY_LIMIT = 80.0
CLASS_COLORS = {
    "Vehicle": (255, 165, 45),
    "Pedestrian": (80, 220, 120),
    "Cyclist": (50, 210, 235),
}


def _validate_boxes(boxes: Any, names: Any) -> tuple[np.ndarray, np.ndarray]:
    boxes = np.asarray(boxes, dtype=np.float32)
    names = np.asarray(names)
    if boxes.ndim != 2 or boxes.shape[1] != 9 or names.shape != (len(boxes),):
        raise ValueError("boxes/names must have shapes (N, 9)/(N,)")
    if not np.isfinite(boxes).all() or np.any(boxes[:, 3:6] <= 0):
        raise ValueError("boxes must be finite with positive sizes")
    if not set(names.tolist()) <= set(CLASS_COLORS):
        raise ValueError("unsupported class name")
    return boxes, names


def _pixel(x: np.ndarray, y: np.ndarray, left: int) -> tuple[np.ndarray, np.ndarray]:
    width, height = PANEL_SIZE
    columns = left + ((XY_LIMIT - y) / (2 * XY_LIMIT) * (width - 1)).round().astype(int)
    rows = PANEL_TOP + ((XY_LIMIT - x) / (2 * XY_LIMIT) * (height - 1)).round().astype(int)
    return columns, rows


def _draw_boxes(
    draw: ImageDraw.ImageDraw,
    boxes: np.ndarray,
    names: np.ndarray,
    panel_left: int,
) -> None:
    for box, name in zip(boxes, names):
        center = box[:2]
        length, width, heading = float(box[3]), float(box[4]), float(box[6])
        forward = np.asarray([np.cos(heading), np.sin(heading)])
        lateral = np.asarray([-np.sin(heading), np.cos(heading)])
        corners = np.asarray(
            [
                center + sx * length / 2 * forward + sy * width / 2 * lateral
                for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1), (1, 1))
            ]
        )
        columns, rows = _pixel(corners[:, 0], corners[:, 1], panel_left)
        color = CLASS_COLORS[str(name)]
        draw.line(list(zip(columns.tolist(), rows.tolist())), fill=color, width=2)
        front = center + length / 2 * forward
        arrow_x, arrow_y = _pixel(
            np.asarray([center[0], front[0]]),
            np.asarray([center[1], front[1]]),
            panel_left,
        )
        draw.line(list(zip(arrow_x.tolist(), arrow_y.tolist())), fill=color, width=2)


def render_frame(
    points: Any,
    detector_boxes: Any,
    detector_names: Any,
    final_boxes: Any,
    final_names: Any,
    frame_id: int,
) -> Image.Image:
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 4 or not np.isfinite(points[:, :4]).all():
        raise ValueError("points must be finite with shape (N, >=4)")
    detector_boxes, detector_names = _validate_boxes(detector_boxes, detector_names)
    final_boxes, final_names = _validate_boxes(final_boxes, final_names)

    canvas = np.full((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), 246, dtype=np.uint8)
    panel_width, panel_height = PANEL_SIZE
    in_range = (
        (points[:, 0] >= -XY_LIMIT)
        & (points[:, 0] <= XY_LIMIT)
        & (points[:, 1] >= -XY_LIMIT)
        & (points[:, 1] <= XY_LIMIT)
    )
    visible = points[in_range]
    shade = np.clip(105 + 100 * visible[:, 3], 65, 230).astype(np.uint8)
    for left in PANEL_LEFTS:
        panel = canvas[PANEL_TOP : PANEL_TOP + panel_height, left : left + panel_width]
        panel[:] = (18, 24, 32)
        for grid in range(-60, 61, 20):
            grid_columns, _ = _pixel(
                np.asarray([0]), np.asarray([grid]), left
            )
            _, grid_rows = _pixel(np.asarray([grid]), np.asarray([0]), left)
            panel[:, grid_columns[0] - left] = (42, 49, 59)
            panel[grid_rows[0] - PANEL_TOP, :] = (42, 49, 59)
        columns, rows = _pixel(visible[:, 0], visible[:, 1], left)
        local_columns = columns - left
        local_rows = rows - PANEL_TOP
        for channel in range(3):
            np.maximum.at(panel[:, :, channel], (local_rows, local_columns), shade)

    image = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.text((20, 12), f"Frame {int(frame_id):04d} | diagnostic BEV, range +/-{int(XY_LIMIT)} m", fill=(20, 25, 32))
    draw.text((28, 48), f"Detector boxes: {len(detector_boxes)}", fill=(235, 238, 242))
    draw.text((668, 48), f"Tracking + GRM + PRM: {len(final_boxes)} | score passthrough | no CRM", fill=(235, 238, 242))
    _draw_boxes(draw, detector_boxes, detector_names, PANEL_LEFTS[0])
    _draw_boxes(draw, final_boxes, final_names, PANEL_LEFTS[1])
    legend_x = 930
    for index, (name, color) in enumerate(CLASS_COLORS.items()):
        y = 12 + index * 9
        draw.line((legend_x, y + 3, legend_x + 16, y + 3), fill=color, width=3)
        draw.text((legend_x + 20, y), name, fill=(20, 25, 32))
    return image


def _load_frames(path: Path, expected_frames: int) -> tuple[list[dict[str, Any]], str]:
    frames = safe_load_pickle(path)
    if not isinstance(frames, list) or len(frames) != expected_frames:
        raise ValueError(f"frame count mismatch: expected {expected_frames}")
    sequence_name = None
    for expected_frame_id, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise TypeError("each frame must be a dict")
        frame_id = frame.get("frame_id")
        if (
            isinstance(frame_id, (bool, np.bool_))
            or not isinstance(frame_id, (int, np.integer))
            or int(frame_id) != expected_frame_id
        ):
            raise ValueError("frame IDs must be contiguous from zero")
        current_sequence = frame.get("sequence_name")
        if not isinstance(current_sequence, str) or not current_sequence:
            raise ValueError("invalid sequence name")
        if sequence_name is None:
            sequence_name = current_sequence
        elif current_sequence != sequence_name:
            raise ValueError("sequence identity changed between frames")
        _validate_boxes(frame.get("boxes_lidar"), frame.get("name"))
    assert sequence_name is not None
    return frames, sequence_name


def render_sequence(
    waymo_root: Path,
    detector_frames_path: Path,
    final_frames_path: Path,
    output_dir: Path,
    expected_frames: int,
) -> dict[str, Any]:
    if isinstance(expected_frames, bool) or expected_frames <= 0:
        raise ValueError("expected_frames must be a positive integer")
    waymo_root = Path(waymo_root)
    if waymo_root.is_symlink() or not waymo_root.is_dir():
        raise ValueError(f"invalid Waymo root: {waymo_root}")
    waymo_root = waymo_root.resolve(strict=True)
    detector_frames_path = Path(detector_frames_path).resolve(strict=True)
    final_frames_path = Path(final_frames_path).resolve(strict=True)
    output_dir = Path(output_dir).absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)

    detector_frames, detector_sequence = _load_frames(
        detector_frames_path, expected_frames
    )
    final_frames, final_sequence = _load_frames(final_frames_path, expected_frames)
    if detector_sequence != final_sequence:
        raise ValueError("detector/final sequence identity mismatch")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = output_dir.with_name(f".{output_dir.name}.tmp-{uuid.uuid4().hex}")
    stage.mkdir()
    frame_manifest = []
    try:
        point_root = (
            waymo_root
            / "waymo_processed_data"
            / f"segment-{detector_sequence}"
        )
        for frame_id, (detector_frame, final_frame) in enumerate(
            zip(detector_frames, final_frames)
        ):
            point_path = point_root / f"{frame_id:04d}.npy"
            if point_path.is_symlink() or not point_path.is_file():
                raise FileNotFoundError(point_path)
            points = _validated_points(np.load(point_path, allow_pickle=False))
            detector_boxes, detector_names = _validate_boxes(
                detector_frame["boxes_lidar"], detector_frame["name"]
            )
            final_boxes, final_names = _validate_boxes(
                final_frame["boxes_lidar"], final_frame["name"]
            )
            image = render_frame(
                points,
                detector_boxes,
                detector_names,
                final_boxes,
                final_names,
                frame_id,
            )
            image_path = stage / f"{frame_id:04d}.png"
            image.save(image_path, format="PNG")
            with image_path.open("rb") as stream:
                os.fsync(stream.fileno())
            with Image.open(image_path) as published_image:
                if published_image.size != IMAGE_SIZE or published_image.mode != "RGB":
                    raise RuntimeError("rendered PNG read-back mismatch")
                published_image.load()
            frame_manifest.append(
                {
                    "frame_id": frame_id,
                    "path": image_path.name,
                    "bytes": image_path.stat().st_size,
                    "sha256": _sha256_file(image_path),
                    "width": IMAGE_SIZE[0],
                    "height": IMAGE_SIZE[1],
                    "point_count": len(points),
                    "point_path": str(point_path.relative_to(waymo_root)),
                    "point_sha256": _sha256_file(point_path),
                    "detector_box_count": len(detector_boxes),
                    "final_box_count": len(final_boxes),
                }
            )

        manifest = {
            "schema_version": "detzero-stage-a-render-v1",
            "sequence_name": detector_sequence,
            "frame_count": expected_frames,
            "inputs": {
                "waymo_root": generation_relative_provenance(
                    waymo_root, output_dir
                ),
                "detector_frames": generation_relative_provenance(
                    detector_frames_path, output_dir
                ),
                "detector_frames_sha256": _sha256_file(detector_frames_path),
                "final_frames": generation_relative_provenance(
                    final_frames_path, output_dir
                ),
                "final_frames_sha256": _sha256_file(final_frames_path),
            },
            "frames": frame_manifest,
        }
        manifest_path = stage / "render_manifest.json"
        with manifest_path.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, allow_nan=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        rename_noreplace(stage, output_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    published = json.loads(
        (output_dir / "render_manifest.json").read_text(encoding="utf-8")
    )
    if published != manifest:
        raise RuntimeError("published render manifest read-back mismatch")
    return published


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waymo-root", type=Path, required=True)
    parser.add_argument("--detector-frames", type=Path, required=True)
    parser.add_argument("--final-frames", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    args = parser.parse_args()
    manifest = render_sequence(
        args.waymo_root,
        args.detector_frames,
        args.final_frames,
        args.output_dir,
        args.expected_frames,
    )
    print(json.dumps(manifest, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
