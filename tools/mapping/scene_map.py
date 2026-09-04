"""Build static/dynamic map for one nuScenes scene (SurroundOcc-style)."""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion

from geometry import (
    global_to_lidar,
    lidar_to_global,
    object_local_to_world,
    optional_poisson,
    points_in_oriented_boxes,
    quat_to_rot,
    self_range_mask,
    voxel_downsample,
    world_to_object_local,
    write_pcd,
    write_ply,
)


def list_sample_tokens(nusc: NuScenes, scene: dict) -> List[str]:
    tokens = []
    tok = scene["first_sample_token"]
    while tok:
        tokens.append(tok)
        tok = nusc.get("sample", tok)["next"]
    return tokens


def list_lidar_sample_data_tokens(nusc: NuScenes, scene: dict) -> List[str]:
    """Walk LIDAR_TOP sample_data chain (all sweeps + keyframes) for a scene."""
    first_sample = nusc.get("sample", scene["first_sample_token"])
    sd_tok = first_sample["data"]["LIDAR_TOP"]
    # rewind to start of chain
    while True:
        sd = nusc.get("sample_data", sd_tok)
        if not sd["prev"]:
            break
        sd_tok = sd["prev"]
    tokens = []
    while sd_tok:
        tokens.append(sd_tok)
        sd_tok = nusc.get("sample_data", sd_tok)["next"]
    return tokens


def load_lidar_points(nusc: NuScenes, sample_token: str) -> Tuple[np.ndarray, dict, dict, str]:
    sample = nusc.get("sample", sample_token)
    sd_token = sample["data"]["LIDAR_TOP"]
    return load_lidar_points_from_sd(nusc, sd_token)


def load_lidar_points_from_sd(
    nusc: NuScenes, sd_token: str
) -> Tuple[np.ndarray, dict, dict, str]:
    sd = nusc.get("sample_data", sd_token)
    path = nusc.get_sample_data_path(sd_token)
    pc = LidarPointCloud.from_file(path)
    pts = pc.points[:3, :].T.copy()  # (N,3)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    pose = nusc.get("ego_pose", sd["ego_pose_token"])
    return pts, cs, pose, sd_token


def pose_mats(cs: dict, pose: dict):
    return (
        quat_to_rot(cs["rotation"]),
        np.asarray(cs["translation"], dtype=np.float64),
        quat_to_rot(pose["rotation"]),
        np.asarray(pose["translation"], dtype=np.float64),
    )


def _box_by_tid(boxes: List[dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for b in boxes:
        tid = str(b["tracking_id"])
        out[tid] = b
    return out


def interpolate_tracking_boxes(
    boxes_a: List[dict],
    boxes_b: List[dict],
    t: float,
    ta: float,
    tb: float,
) -> List[dict]:
    """Interpolate boxes between keyframes A (ta) and B (tb) at timestamp t.

    Match by tracking_id. translation/size: lerp; rotation: Quaternion.slerp.
    If track only on one side: use that side's box.
    """
    if tb == ta:
        alpha = 0.0
    else:
        alpha = float(np.clip((t - ta) / (tb - ta), 0.0, 1.0))

    map_a = _box_by_tid(boxes_a)
    map_b = _box_by_tid(boxes_b)
    tids = set(map_a) | set(map_b)
    out: List[dict] = []
    for tid in tids:
        ba = map_a.get(tid)
        bb = map_b.get(tid)
        if ba is not None and bb is not None:
            ta_arr = np.asarray(ba["translation"], dtype=np.float64)
            tb_arr = np.asarray(bb["translation"], dtype=np.float64)
            sa = np.asarray(ba["size"], dtype=np.float64)
            sb = np.asarray(bb["size"], dtype=np.float64)
            qa = Quaternion(ba["rotation"])
            qb = Quaternion(bb["rotation"])
            q = Quaternion.slerp(qa, qb, amount=alpha)
            interp = {
                "tracking_id": tid,
                "tracking_name": ba.get("tracking_name") or bb.get("tracking_name", ""),
                "translation": (ta_arr * (1.0 - alpha) + tb_arr * alpha).tolist(),
                "size": (sa * (1.0 - alpha) + sb * alpha).tolist(),
                "rotation": [q.w, q.x, q.y, q.z],
            }
            if "tracking_score" in ba or "tracking_score" in bb:
                sa_sc = float(ba.get("tracking_score", bb.get("tracking_score", 0.0)))
                sb_sc = float(bb.get("tracking_score", ba.get("tracking_score", 0.0)))
                interp["tracking_score"] = sa_sc * (1.0 - alpha) + sb_sc * alpha
            if "velocity" in ba or "velocity" in bb:
                va = np.asarray(ba.get("velocity", bb.get("velocity", [0.0, 0.0])), dtype=np.float64)
                vb = np.asarray(bb.get("velocity", ba.get("velocity", [0.0, 0.0])), dtype=np.float64)
                interp["velocity"] = (va * (1.0 - alpha) + vb * alpha).tolist()
            out.append(interp)
        elif ba is not None:
            # prefer nearest side when only one exists
            out.append(dict(ba))
        else:
            out.append(dict(bb))
    return out


def build_keyframe_timeline(
    nusc: NuScenes,
    sample_tokens: Sequence[str],
    tracking_results: Dict[str, List[dict]],
) -> List[Dict[str, Any]]:
    """List of {ts, sample_token, boxes} for each keyframe, sorted by timestamp."""
    timeline = []
    for stok in sample_tokens:
        sample = nusc.get("sample", stok)
        timeline.append(
            {
                "ts": int(sample["timestamp"]),
                "sample_token": stok,
                "boxes": list(tracking_results.get(stok, [])),
            }
        )
    timeline.sort(key=lambda x: x["ts"])
    return timeline


def _boxes_at_timestamp(
    timeline: List[Dict[str, Any]],
    t: int,
) -> Tuple[List[dict], Optional[str], bool]:
    """Return (boxes, exact_keyframe_sample_token_or_None, is_exact_keyframe)."""
    if not timeline:
        return [], None, False
    # exact match
    for kf in timeline:
        if kf["ts"] == t:
            return kf["boxes"], kf["sample_token"], True
    # before first / after last
    if t <= timeline[0]["ts"]:
        return timeline[0]["boxes"], None, False
    if t >= timeline[-1]["ts"]:
        return timeline[-1]["boxes"], None, False
    # find bracketing keyframes
    for i in range(len(timeline) - 1):
        a, b = timeline[i], timeline[i + 1]
        if a["ts"] <= t <= b["ts"]:
            boxes = interpolate_tracking_boxes(
                a["boxes"], b["boxes"], float(t), float(a["ts"]), float(b["ts"])
            )
            return boxes, None, False
    return timeline[-1]["boxes"], None, False


def build_scene_map(
    nusc: NuScenes,
    scene_name: str,
    tracking_results: Dict[str, List[dict]],
    out_dir: Path,
    box_expand: float = 1.1,
    voxel_size: float = 0.1,
    self_range: Sequence[float] = (3.0, 3.0, 3.0),
    use_poisson: bool = False,
    poisson_depth: int = 9,
    write_pcd_also: bool = True,
    use_sweeps: bool = True,
) -> Dict[str, Any]:
    t0 = time.time()
    scene = None
    for s in nusc.scene:
        if s["name"] == scene_name:
            scene = s
            break
    if scene is None:
        raise ValueError(f"scene not found: {scene_name}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_tokens = list_sample_tokens(nusc, scene)
    if not sample_tokens:
        raise RuntimeError(f"no samples in {scene_name}")
    n_keyframes = len(sample_tokens)

    last_token = sample_tokens[-1]
    _, cs_last, pose_last, sd_last = load_lidar_points(nusc, last_token)
    cs_r_l, cs_t_l, ego_r_l, ego_t_l = pose_mats(cs_last, pose_last)

    static_chunks: List[np.ndarray] = []
    # track_id -> list of local points
    object_local: Dict[str, List[np.ndarray]] = defaultdict(list)
    # track_id -> list of (frame_idx, box_dict, is_keyframe)
    track_boxes: Dict[str, List[Tuple[int, dict, bool]]] = defaultdict(list)
    track_names: Dict[str, str] = {}

    n_static_raw = 0
    n_dynamic_raw = 0
    frames_meta = []

    if use_sweeps:
        sd_tokens = list_lidar_sample_data_tokens(nusc, scene)
        timeline = build_keyframe_timeline(nusc, sample_tokens, tracking_results)
        frame_iter = []
        for sd_tok in sd_tokens:
            sd = nusc.get("sample_data", sd_tok)
            frame_iter.append(
                {
                    "sd_token": sd_tok,
                    "timestamp": int(sd["timestamp"]),
                    "is_key_frame": bool(sd["is_key_frame"]),
                    "sample_token": sd["sample_token"],
                }
            )
    else:
        frame_iter = []
        for stok in sample_tokens:
            sample = nusc.get("sample", stok)
            sd_tok = sample["data"]["LIDAR_TOP"]
            sd = nusc.get("sample_data", sd_tok)
            frame_iter.append(
                {
                    "sd_token": sd_tok,
                    "timestamp": int(sd["timestamp"]),
                    "is_key_frame": True,
                    "sample_token": stok,
                }
            )
        timeline = None

    for fi, fr in enumerate(frame_iter):
        sd_tok = fr["sd_token"]
        pts_l, cs, pose, _ = load_lidar_points_from_sd(nusc, sd_tok)
        cs_r, cs_t, ego_r, ego_t = pose_mats(cs, pose)
        pts_g = lidar_to_global(pts_l, cs_r, cs_t, ego_r, ego_t)

        if use_sweeps:
            if fr["is_key_frame"]:
                # Keyframe: boxes keyed by this sample_token in tracking JSON
                stok_meta = fr["sample_token"]
                objs = tracking_results.get(stok_meta, [])
                is_kf = True
            else:
                objs, exact_stok, _ = _boxes_at_timestamp(timeline, fr["timestamp"])
                stok_meta = exact_stok if exact_stok else fr["sample_token"]
                is_kf = False
        else:
            stok_meta = fr["sample_token"]
            objs = tracking_results.get(stok_meta, [])
            is_kf = True

        centers = []
        sizes = []
        rots = []
        tids = []
        for o in objs:
            centers.append(o["translation"])
            sizes.append(o["size"])
            rots.append(o["rotation"])
            tid = str(o["tracking_id"])
            tids.append(tid)
            # Prefer recording keyframe appearances for placement; still append sweeps
            # but placement logic below filters to keyframes first.
            track_boxes[tid].append((fi, o, is_kf))
            track_names[tid] = o.get("tracking_name", track_names.get(tid, ""))

        centers_a = np.asarray(centers, dtype=np.float64).reshape(-1, 3) if centers else np.zeros((0, 3))
        sizes_a = np.asarray(sizes, dtype=np.float64).reshape(-1, 3) if sizes else np.zeros((0, 3))
        inbox = points_in_oriented_boxes(pts_g, centers_a, sizes_a, rots, expand=box_expand)

        if inbox.shape[1] > 0:
            any_dyn = inbox.any(axis=1)
        else:
            any_dyn = np.zeros((pts_g.shape[0],), dtype=bool)

        # per-track object points (global -> object local)
        for j, tid in enumerate(tids):
            sel = inbox[:, j]
            if not np.any(sel):
                continue
            local = world_to_object_local(pts_g[sel], centers[j], rots[j])
            object_local[tid].append(local)
            n_dynamic_raw += int(sel.sum())

        # static: not in any box; also drop near-ego in lidar frame
        static_mask = ~any_dyn
        keep_ego = self_range_mask(pts_l, self_range)
        static_mask = static_mask & keep_ego
        static_g = pts_g[static_mask]
        static_last = global_to_lidar(static_g, cs_r_l, cs_t_l, ego_r_l, ego_t_l)
        if static_last.shape[0] > 0:
            static_chunks.append(static_last)
            n_static_raw += int(static_last.shape[0])

        frames_meta.append(
            {
                "frame_idx": fi,
                "sample_token": stok_meta,
                "sample_data_token": sd_tok,
                "is_key_frame": bool(fr["is_key_frame"]),
                "timestamp": int(fr["timestamp"]),
                "n_lidar": int(pts_l.shape[0]),
                "n_boxes": len(objs),
                "n_static": int(static_mask.sum()),
                "n_dynamic_assigned": int(any_dyn.sum()),
            }
        )

    # concatenate + voxel static
    if static_chunks:
        static_all = np.concatenate(static_chunks, axis=0)
    else:
        static_all = np.zeros((0, 3), dtype=np.float64)
    static_vox = voxel_downsample(static_all, voxel_size)
    if use_poisson:
        static_export = optional_poisson(static_vox, depth=poisson_depth)
    else:
        static_export = static_vox

    # densify dynamics and place at last available KEYFRAME box (prefer last keyframe)
    last_fi = len(frame_iter) - 1
    # find last keyframe frame index among processed frames
    last_kf_fi = None
    for fr_i, fr in enumerate(frame_iter):
        if fr["is_key_frame"]:
            last_kf_fi = fr_i
    if last_kf_fi is None:
        last_kf_fi = last_fi

    dynamic_chunks: List[np.ndarray] = []
    boxes_last: Dict[str, Any] = {}
    tracks_placed = 0
    tracks_skipped_empty = 0

    for tid, locals_list in object_local.items():
        if not locals_list:
            tracks_skipped_empty += 1
            continue
        dens = np.concatenate(locals_list, axis=0)
        if dens.shape[0] == 0:
            tracks_skipped_empty += 1
            continue

        appearances = track_boxes.get(tid, [])
        if not appearances:
            tracks_skipped_empty += 1
            continue

        # Prefer last KEYFRAME appearance for place_box
        kf_apps = [(fi, b) for (fi, b, is_kf) in appearances if is_kf]
        if kf_apps:
            on_last = [b for (fi, b) in kf_apps if fi == last_kf_fi]
            if on_last:
                place_box = on_last[-1]
                place_source = "last_keyframe"
            else:
                place_box = kf_apps[-1][1]
                place_source = f"fallback_keyframe_{kf_apps[-1][0]}"
            n_frames_seen = len(kf_apps)
        else:
            # no keyframe appearance: fall back to last sweep appearance
            place_box = appearances[-1][1]
            place_source = f"fallback_sweep_{appearances[-1][0]}"
            n_frames_seen = len(appearances)

        placed_g = object_local_to_world(dens, place_box["translation"], place_box["rotation"])

        # optional crop to expanded last box
        crop_mask = points_in_oriented_boxes(
            placed_g,
            np.asarray([place_box["translation"]], dtype=np.float64),
            np.asarray([place_box["size"]], dtype=np.float64),
            [place_box["rotation"]],
            expand=box_expand,
        )[:, 0]
        if crop_mask.any():
            placed_g = placed_g[crop_mask]

        placed_l = global_to_lidar(placed_g, cs_r_l, cs_t_l, ego_r_l, ego_t_l)
        if placed_l.shape[0] > 0:
            dynamic_chunks.append(placed_l)
            tracks_placed += 1

        boxes_last[tid] = {
            "tracking_id": tid,
            "tracking_name": track_names.get(tid, place_box.get("tracking_name", "")),
            "translation": list(map(float, place_box["translation"])),
            "size": list(map(float, place_box["size"])),
            "rotation": list(map(float, place_box["rotation"])),
            "place_source": place_source,
            "n_local_points": int(dens.shape[0]),
            "n_placed_points": int(placed_l.shape[0]),
            "n_frames_seen": n_frames_seen,
        }

    if dynamic_chunks:
        dynamic_all = np.concatenate(dynamic_chunks, axis=0)
    else:
        dynamic_all = np.zeros((0, 3), dtype=np.float64)

    # light voxel on dynamic to control size (finer)
    dynamic_export = voxel_downsample(dynamic_all, max(voxel_size * 0.5, 0.05))

    if static_export.shape[0] and dynamic_export.shape[0]:
        combined = np.concatenate([static_export, dynamic_export], axis=0)
    elif static_export.shape[0]:
        combined = static_export
    else:
        combined = dynamic_export

    n_static = write_ply(out_dir / "static_map.ply", static_export)
    n_dyn = write_ply(out_dir / "dynamic_at_last.ply", dynamic_export)
    n_comb = write_ply(out_dir / "combined.ply", combined)
    if write_pcd_also:
        write_pcd(out_dir / "static_map.pcd", static_export)
        write_pcd(out_dir / "dynamic_at_last.pcd", dynamic_export)
        write_pcd(out_dir / "combined.pcd", combined)

    with open(out_dir / "boxes_last.json", "w") as f:
        json.dump(
            {
                "scene": scene_name,
                "last_sample_token": last_token,
                "last_lidar_sd_token": sd_last,
                "coordinate_frame": "last_keyframe_LIDAR_TOP",
                "boxes": boxes_last,
            },
            f,
            indent=2,
        )

    elapsed = time.time() - t0
    manifest = {
        "scene": scene_name,
        "n_frames": len(frame_iter),
        "n_keyframes": n_keyframes,
        "use_sweeps": use_sweeps,
        "first_sample_token": sample_tokens[0],
        "last_sample_token": last_token,
        "box_expand": box_expand,
        "voxel_size": voxel_size,
        "self_range": list(self_range),
        "use_poisson": use_poisson,
        "poisson_depth": poisson_depth if use_poisson else None,
        "n_static_raw_accumulated": n_static_raw,
        "n_static_after_voxel": n_static,
        "n_dynamic_raw_assigned": n_dynamic_raw,
        "n_dynamic_after_place_voxel": n_dyn,
        "n_combined": n_comb,
        "n_tracks_with_points": len(object_local),
        "n_tracks_placed": tracks_placed,
        "n_tracks_skipped_empty": tracks_skipped_empty,
        "elapsed_sec": round(elapsed, 3),
        "frames": frames_meta,
        "outputs": {
            "static_map.ply": n_static,
            "dynamic_at_last.ply": n_dyn,
            "combined.ply": n_comb,
        },
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest
