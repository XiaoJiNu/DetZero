#!/usr/bin/env python3
"""Pose-contract test for the nu-stack (bugfix 2026-09-03).

HEDNet info-pkl: car_from_global = global->ego, ref_from_car = ego->lidar.
The lidar->global transform used by T1/S1/N1 MUST be
    inv(ref_from_car @ car_from_global)
NOT the previously shipped inv(car_from_global @ ref_from_car).
Verified here against the devkit ego_pose/calibrated_sensor tables on every
frame of v1.0-mini (the 81 HEDNet val frames), which map onto the two
mini_val scenes.
"""
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.nustack import lidar_to_global  # noqa: E402

INFO = Path('/data/code/cv/AutoLabel/BEV-OD/HEDNet-qwen/data/nuscenes/v1.0-mini/nuscenes_infos_10sweeps_val.pkl')
DATAROOT = Path('/data/data/automomous/nuscenes/v1.0-mini')


def devkit_lidar_to_global(nusc, token):
    from pyquaternion import Quaternion
    sample = nusc.get('sample', token)
    sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ego = nusc.get('ego_pose', sd['ego_pose_token'])
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    e = np.eye(4); e[:3, :3] = Quaternion(ego['rotation']).rotation_matrix; e[:3, 3] = ego['translation']
    c = np.eye(4); c[:3, :3] = Quaternion(cs['rotation']).rotation_matrix; c[:3, 3] = cs['translation']
    return e @ c


def main():
    infos = pickle.load(INFO.open('rb'))
    assert len(infos) == 81, f'expected 81 val frames, got {len(infos)}'
    try:
        from nuscenes.nuscenes import NuScenes
    except ImportError:
        print('SKIP: nuscenes devkit not importable in this interpreter')
        return 0
    nusc = NuScenes('v1.0-mini', dataroot=str(DATAROOT), verbose=False)
    worst_good, worst_bad = 0.0, 0.0
    for m in infos:
        good = lidar_to_global(m)
        bad = np.linalg.inv(np.asarray(m['car_from_global'], dtype=np.float64)
                            @ np.asarray(m['ref_from_car'], dtype=np.float64))
        ref = devkit_lidar_to_global(nusc, m['token'])
        worst_good = max(worst_good, np.abs(good - ref).max())
        worst_bad = max(worst_bad, np.abs(bad - ref).max())
    assert worst_good < 1e-6, f'corrected composition off by {worst_good}'
    assert worst_bad > 100.0, f'old (wrong) composition unexpectedly close: {worst_bad}'
    print(f'PASS: corrected max-err {worst_good:.2e} (<1e-6), '
          f'old-order max-err {worst_bad:.2f} m (>100) on all {len(infos)} frames')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
