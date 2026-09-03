#!/usr/bin/env python3
"""Throwaway diagnostic (not part of the pipeline): GT-track baseline,
detector->GT match rate, and T1 fragmentation/ID-switch counts."""
import pickle, json, collections
import numpy as np
from scipy.optimize import linear_sum_assignment

OUT = 'output/nus-stage-a-20260831-165710-CST'
INFO = '/data/code/cv/AutoLabel/BEV-OD/HEDNet-qwen/data/nuscenes/v1.0-mini/nuscenes_infos_10sweeps_val.pkl'
CLASSES = {'Vehicle': ['car', 'bus', 'truck', 'construction_vehicle', 'trailer'],
           'Pedestrian': ['pedestrian'], 'Cyclist': ['bicycle', 'motorcycle']}
MAP = {c: k for k, v in CLASSES.items() for c in v}


def gt_global(m):
    b = np.array(m['gt_boxes'])
    names = np.array(m['gt_names'])
    pose = np.linalg.inv(np.array(m['car_from_global']) @ np.array(m['ref_from_car']))  # lidar->global
    R, t = pose[:3, :3], pose[:3, 3]
    ok = np.array([x in MAP for x in names]) & (np.array(m['num_lidar_pts']) > -1)
    p = b[ok, :3] @ R.T + t
    return p[:, :2], [MAP[str(names[i])] for i in np.where(ok)[0]]


def main():
    infos = pickle.load(open(INFO, 'rb'))
    by_token = {m['token']: m for m in infos}
    for scene in ['scene-0103', 'scene-0916']:
        toks = json.load(open(f'{OUT}/{scene}/data/gt_tokens.json'))['tokens']
        frames = [by_token[t] for t in toks]

        # --- GT tracks (smooth, generous gate 5m, same max_age semantics) ---
        active, gt_traj, nid = {}, collections.defaultdict(list), 0
        motion = collections.defaultdict(list)
        for fi, m in enumerate(frames):
            c, nms = gt_global(m)
            for cls in CLASSES:
                di = [i for i, x in enumerate(nms) if x == cls]
                act = [k for k, v in active.items() if v['cls'] == cls and fi - v['f'] <= 2]
                taken = set()
                if di and act:
                    D = np.linalg.norm(c[di][:, None] - np.array([active[k]['p'] for k in act])[None], axis=2)
                    r, cc = linear_sum_assignment(D)
                    for x, y in zip(r, cc):
                        if D[x, y] < 5:
                            motion[cls].append(D[x, y])
                            k = act[y]
                            active[k].update(p=c[di[x]], f=fi)
                            gt_traj[k].append((fi, di[x]))
                            taken.add(x)
                for x in di:
                    if x in taken:
                        continue
                    active[nid] = {'cls': cls, 'p': c[x], 'f': fi}
                    gt_traj[nid].append((fi, x))
                    nid += 1
        lens = np.array([len(v) for v in gt_traj.values()])

        # --- detector obs -> GT id (nearest same-class GT center <2m) ---
        det = pickle.load(open(f'{OUT}/{scene}/detector/hednet_frames_{scene}.pkl', 'rb'))
        det.sort(key=lambda f: int(f['frame_id']))
        gtid_of = {}
        for gid, obs in gt_traj.items():
            for fi, gi in obs:
                gtid_of[(fi, gi)] = gid
        fp2gt = {}   # (frame, det_idx) -> gt id
        matched = total = 0
        for fi, f in enumerate(det):
            b = np.array(f['boxes_lidar'])
            nms = list(map(str, f['name']))
            R = np.array(f['pose'])[:3, :3]
            t = np.array(f['pose'])[:3, 3]
            cg = b[:, :3] @ R.T + t
            c, nms_g = gt_global(frames[fi])
            for i in range(len(b)):
                if nms[i] not in CLASSES:
                    continue
                total += 1
                if len(c) == 0:
                    continue
                same = np.array([g == nms[i] for g in nms_g])
                if not same.any():
                    continue
                d = np.linalg.norm(c - cg[i, :2], axis=1)
                d = np.where(same, d, 1e9)
                j = int(np.argmin(d))
                if d[j] < 2:
                    matched += 1
                    fp2gt[(fi, i)] = gtid_of.get((fi, j), -2)

        # --- T1 tracks: obs->trackid via tracked frames file (row order) ---
        t1 = pickle.load(open(f'{OUT}/{scene}/track/hednet_tracks_{scene}.pkl', 'rb'))[scene]
        # per-frame obs: detector frames kept by T1 have boxes == det boxes with dims swapped;
        # map back by matching center within 0.01m
        tf = pickle.load(open(f'{OUT}/{scene}/track/hednet_tracked_frames_{scene}.pkl', 'rb'))
        tf.sort(key=lambda f: int(f['frame_id']))
        fp2trk = {}
        for fi, (f, g) in enumerate(zip(det, tf)):
            db = np.array(f['boxes_lidar'])[:, :2]
            gb = np.array(g['boxes_lidar'])[:, :2]
            for row in range(len(gb)):
                dd = np.linalg.norm(db - gb[row], axis=1)
                j = int(np.argmin(dd))
                if dd[j] < 1e-3:
                    tid = list(sorted(set()))  # placeholder
            # need tid: recompute via tracks pkl sample_idx order == obs order used before
        # simpler: T1 track's obs (frame, det_idx) reconstructed from tracks pkl rows matching det centers
        for tid, tr in t1.items():
            sidx = np.array(tr['sample_idx'])
            boxes = np.array(tr['boxes_lidar'])
            for s, bx in zip(sidx, boxes):
                db = np.array(det[int(s)]['boxes_lidar'])
                dd = np.linalg.norm(db[:, :2] - bx[:2], axis=1) + np.linalg.norm(db[:, 6] - bx[6], axis=0)
                j = int(np.argmin(dd))
                if dd[j] < 0.05:
                    fp2trk[(int(s), j)] = tid

        frag = collections.defaultdict(set)   # gt id -> set(T1 ids)
        sw = collections.defaultdict(set)     # T1 id -> set(gt ids)
        for fp, gid in fp2gt.items():
            if gid < 0 or fp not in fp2trk:
                continue
            frag[gid].add(fp2trk[fp])
            sw[fp2trk[fp]].add(gid)
        pers = [g for g, obs in gt_traj.items() if len(obs) >= 10]
        frag_pers = sum(1 for g in pers if len(frag.get(g, {g})) > 1)
        idsw = sum(1 for t, s in sw.items() if len(s) > 1)
        print(f"{scene}: GT tracks={len(lens)} obs={lens.sum()} longest={lens.max()} >=10f={np.sum(lens >= 10)}"
              f" | det->GT matched {matched}/{total} ({matched / total:.0%})"
              f" | persistent-GT fragmented: {frag_pers}/{len(pers)}"
              f" | T1 tracks with >1 GT id (ID switch): {idsw}/{len(sw)}"
              f" | GT motion med(m): " + ','.join(f"{k}:{np.median(v):.2f}" for k, v in motion.items() if len(v)))


if __name__ == '__main__':
    main()
