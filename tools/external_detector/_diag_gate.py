#!/usr/bin/env python3
"""Diagnostic 2: for detector obs of the SAME GT object across adjacent frames,
measure |next_center - (prev_center + v*dt)| where v = per-track GT velocity.
This is the association residual T1's gate must cover."""
import sys, pickle, json, collections
sys.path.insert(0, 'tools/external_detector')
import numpy as np
from scipy.optimize import linear_sum_assignment
import _diag_gt_tracks as D

OUT = 'output/nus-stage-a-20260831-165710-CST'
infos = pickle.load(open(D.INFO, 'rb'))
by_token = {m['token']: m for m in infos}

for scene in ['scene-0103', 'scene-0916']:
    toks = json.load(open(f'{OUT}/{scene}/data/gt_tokens.json'))['tokens']
    frames = [by_token[t] for t in toks]
    # GT tracks again (copy from diag 1)
    active, gt_traj, nid = {}, collections.defaultdict(list), 0
    for fi, m in enumerate(frames):
        c, nms = D.gt_global(m)
        for cls in D.CLASSES:
            di = [i for i, x in enumerate(nms) if x == cls]
            act = [k for k, v in active.items() if v['cls'] == cls and fi - v['f'] <= 2]
            taken = set()
            if di and act:
                P = np.array([active[k]['p'] for k in act])
                Dt = np.linalg.norm(c[di][:, None] - P[None], axis=2)
                r, cc = linear_sum_assignment(Dt)
                for x, y in zip(r, cc):
                    if Dt[x, y] < 5:
                        k = act[y]
                        active[k].update(p=c[di[x]], f=fi)
                        gt_traj[k].append((fi, di[x]))
                        taken.add(x)
            for x in di:
                if x not in taken:
                    active[nid] = {'cls': cls, 'p': c[x], 'f': fi}
                    gt_traj[nid].append((fi, x)); nid += 1

    # detector obs in global per frame, matched to GT ids
    det = pickle.load(open(f'{OUT}/{scene}/detector/hednet_frames_{scene}.pkl', 'rb'))
    det.sort(key=lambda f: int(f['frame_id']))
    det_g, det_n = [], []
    for f in det:
        b = np.array(f['boxes_lidar']); R = np.array(f['pose'])[:3, :3]; t = np.array(f['pose'])[:3, 3]
        det_g.append(b[:, :3] @ R.T + t)
        det_n.append(list(map(str, f['name'])))
    id_obs = collections.defaultdict(list)  # gt id -> [(frame, det_idx, center)]
    for gid, obs in gt_traj.items():
        for fi, gi in obs:
            gtid_of = (fi, gi)
    gtid_of = {}
    for gid, obs in gt_traj.items():
        for fi, gi in obs:
            gtid_of[(fi, gi)] = gid
    for fi in range(len(frames)):
        c, nms = D.gt_global(frames[fi])
        cg = det_g[fi]
        for i in range(len(cg)):
            if det_n[fi][i] not in D.CLASSES or len(c) == 0:
                continue
            same = np.array([x == det_n[fi][i] for x in nms])
            d = np.where(same, np.linalg.norm(c - cg[i, :2], axis=1), 1e9)
            j = int(np.argmin(d))
            if d[j] < 2 and (fi, j) in gtid_of:
                id_obs[gtid_of[(fi, j)]].append((fi, i, cg[i, :2]))

    # residual: predicted by GT-derived velocity vs naive
    res = collections.defaultdict(list)
    for gid, obs in id_obs.items():
        obs.sort()
        cls = det_n[obs[0][0]][obs[0][1]]
        for k in range(1, len(obs)):
            f0, _, p0 = obs[k - 1]; f1, _, p1 = obs[k]
            if f1 - f0 != 1 or k < 2:
                continue
            v = p0 - obs[k - 2][2]  # finite diff over 1 frame (EMA-ish naive velocity)
            res[cls].append(np.linalg.norm(p1 - (p0 + v)))
    print(scene, {c: (len(v), float(np.median(v)) if v else None,
                      float(np.percentile(v, 90)) if v else None) for c, v in res.items()})
    # also pure displacement (no velocity) for reference
    res2 = collections.defaultdict(list)
    for gid, obs in id_obs.items():
        obs.sort()
        cls = det_n[obs[0][0]][obs[0][1]]
        for k in range(1, len(obs)):
            if obs[k][0] - obs[k-1][0] != 1: continue
            res2[cls].append(np.linalg.norm(obs[k][2] - obs[k-1][2]))
    print('  raw jitter', {c: (round(float(np.median(v)),2), round(float(np.percentile(v,90)),2)) for c, v in res2.items()})
