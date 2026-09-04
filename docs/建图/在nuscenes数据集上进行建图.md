# 在 nuScenes 数据集上进行建图（SurroundOcc 风格静/动态分离）

| 字段 | 内容 |
| --- | --- |
| 作者 / 助手 | **grok**（算法复现） |
| 文档创建时间 | 2026-09-04 13:29:42 CST（2026-09-04T05:29:42Z） |
| 文档版本 | v1.0 |
| 工作区 | `/data/code/cv/AutoLabel/DetZero-nuscenes-simpletrack-trackgt` |
| Git 分支 | `nuscenes-mapping-surroundocc`（自 tip `acef8d7`） |
| 产品目标 | 离线场景点云建图：静态背景 + 动态物体致密（对齐末帧 LiDAR） |
| 上游依赖 | HEDNet + SimpleTrack 跟踪真值 `tracking_results.json` |

## 修订记录

| 时间 (CST) | 作者 | 版本 | 变更摘要 |
| --- | --- | --- | --- |
| 2026-09-04 13:29:42 CST | grok | v1.0 | 初版定稿：官方 nuScenes 位姿；SurroundOcc 静/动分离；HEDNet+SimpleTrack 框与 track_id；累积到末帧 LiDAR；物体局部致密后按末框位姿放置；可选 Open3D Poisson；不重跑 SLAM/KISS-ICP |

> 后续每次修正：在本表追加一行，并同步文首「文档版本 / 文档创建或修订时间」。文件名时间戳保留首次创建时刻。

---

## 1. 方案概述

### 1.1 一句话目标

在 **nuScenes v1.0-mini**（优先 `scene-0103` / `scene-0916`）上，用 **官方 ego_pose + calibrated_sensor** 把多帧 `LIDAR_TOP` 对齐到**场景末关键帧 LiDAR 坐标系**，借助 **HEDNet→SimpleTrack** 的 3D 框与稳定 `tracking_id` 做 **框内动态 / 框外静态** 分离，分别融合，导出点云地图与 BEV 可视化。

### 1.2 明确做什么 / 不做什么

| 做 | 不做 |
| --- | --- |
| 官方 nuScenes 位姿链：lidar→ego→global→末帧 lidar | 重跑 SLAM / KISS-ICP / 自研位姿优化 |
| SurroundOcc 式静/动分割（框内=动态，框外=静态） | 用速度阈值判定“是否在动” |
| 使用跟踪 JSON 的框 + `tracking_id`（非官方 GT instance） | 依赖 lidarseg 语义做地图标签（本阶段不做占用真值） |
| 动态：物体局部坐标致密 → 末次可见框位姿放置 → 末帧 LiDAR | 为每个关键帧单独出一套占用网格（SurroundOcc GT 路径） |
| 静态：多帧累积 + 体素下采样（默认 0.1 m） | 默认跑 Poisson（可选 `--poisson`，仅静态） |
| 导出 PLY / PCD / boxes_last.json / BEV PNG | 把大体积 `output/` 点云提交进 Git |

### 1.3 与 SurroundOcc 官方真值脚本的异同

| 项 | SurroundOcc `generate_occupancy_nuscenes.py` | 本方案 |
| --- | --- | --- |
| 位姿 | nuScenes 官方 | 同左 |
| 框来源 | 官方 `sample_annotation` + `instance_token` | HEDNet+SimpleTrack `tracking_id` |
| 参考系 | 先对齐到**首帧** LiDAR，再对每个关键帧变到该帧 | 直接累积到**末关键帧** LiDAR |
| 动态放置 | 每个关键帧用**该帧**框位姿放回 | 只在末帧地图中，用**末次可见（优先末帧）**框位姿放一次 |
| 中间 sweep | 沿 `sample_data.next` 含非关键帧 | **仅 keyframe sample**（与跟踪 JSON 对齐） |
| Poisson | 默认做，再体素化占用 | **默认关**；可选仅静态 |
| 语义 / 占用网格 | lidarseg + 0.5 m 占用 | 本阶段只出点云地图 |

### 1.4 端到端数据流

```
nuScenes v1.0-mini (LIDAR_TOP + ego_pose + calibrated_sensor)
        │
        ├──────────────────────────────────────────────┐
        │                                              │
tracking_results.json (global boxes + tracking_id)     │
        │                                              │
        ▼                                              ▼
  每关键帧：点云 → global；框 ×1.1；points_in_boxes
        │
        ├── 框外 → 静态点 → 变换到末帧 LiDAR → 累积 → voxel(0.1)
        │
        └── 框内 → 按 tracking_id → 减中心、去 yaw → 物体局部累积
                                                      │
                         末次框位姿（优先末帧）放置 → 末帧 LiDAR
        │
        ▼
  static_map.ply / dynamic_at_last.ply / combined.ply
  boxes_last.json / manifest.json / BEV PNGs
```

### 1.5 验收门

| 门 | 条件 |
| --- | --- |
| G1 链路 | 两 scene 跑通；产物非空；帧数与 scene keyframe 数一致 |
| G2 坐标 | 静态/动态均在**同一末帧 LiDAR**坐标系；BEV 与末帧 ego 附近对齐可读 |
| G3 静动 | 静态点大致不含车行人团块；动态点数 > 单帧框内点数（致密有效） |
| G4 边界 | 不改上游跟踪产物；不提交大 PLY；不重跑 SLAM |
| G5 可复现 | `tools/mapping/` 自包含；`mv2d` Python + open3d + nuscenes 可跑 |

---

## 2. 方案详述

### 2.1 输入资产

| 资产 | 路径 | 说明 |
| --- | --- | --- |
| nuScenes mini | `/data/data/automomous/nuscenes/v1.0-mini`，`version=v1.0-mini` | 官方位姿与点云 |
| 跟踪结果 | `output/track-gt-20260903-155401-CST/delivery/tracking_results.json` | 全局框；字段含 `translation/size/rotation/tracking_id/tracking_name` |
| Python | `/data/software/conda/anaconda3/envs/mv2d/bin/python` | open3d 0.18 + nuscenes；**避免 mmcv** |

### 2.2 位姿变换（与 SurroundOcc 一致）

对帧 \(i\) 的 LiDAR 点 \(p_L\)：

1. \(p_E = R_{L\rightarrow E}\, p_L + t_{L\rightarrow E}\)（`calibrated_sensor`）
2. \(p_G = R_{E\rightarrow G}\, p_E + t_{E\rightarrow G}\)（`ego_pose`）

变到末帧 LiDAR \(L^*\)：

1. \(p_{E^*} = R_{G\rightarrow E^*}\,(p_G - t_{E^*})\)
2. \(p_{L^*} = R_{E^*\rightarrow L^*}\,(p_{E^*} - t_{L\rightarrow E^*})\)

即：`lidar_to_world` 再用末帧的逆变换。**禁止**用 KISS-ICP 等替换官方位姿。

### 2.3 框与 points_in_boxes

- 跟踪框已在 **global**：`translation` 为中心，`size=[w,l,h]`，`rotation` 为四元数 \([w,x,y,z]\)。
- 扩框：`size *= 1.1`（与 SurroundOcc 一致）。
- 可选：将框底沿 z 略下移（`z -= 0.1` 或底心修正）以更包住轮地接触；实现默认做 **中心不变 + 尺寸×1.1**，并记录在 manifest。
- `points_in_boxes`：**numpy / Open3D 定向框**实现（点变到物体局部：去平移、乘 \(R^\top\)，再轴对齐判断），**不依赖 mmcv.ops**。
- 落在任一扩框内 → 记入该 `tracking_id` 的物体点列表；其余 → 静态。
- 自车近邻裁剪（可选，默认开启）：在**当前 LiDAR 系**剔除 \(|x|<3,|y|<3,|z|<3\)（或配置 `self_range`）内的静态点，避免车身/车盖伪影进地图。

### 2.4 静态融合

- 每帧静态点：先到 global，再到 **末关键帧 LiDAR**。
- 全场景 `concatenate` 后 **voxel_down_sample(0.1 m)**（Open3D）控制体积。
- 可选 `--poisson`：仅对静态估计法向 + Poisson，导出 mesh 顶点或网格；**默认关闭**以保证速度。

### 2.5 动态致密与放置

对每个 `tracking_id`：

1. 帧 \(i\)：框内点先到 **global**，再 \(p' = R_{\text{yaw}}^\top (p_G - c_i)\)（只用 yaw，与 SurroundOcc `Rotation.from_euler('z', -rots)` 同思路；实现可用完整四元数更稳）。
2. 跨帧 `concatenate` 得物体局部稠密云。
3. **放置位姿**：优先使用**末关键帧**上该 id 的框；若末帧缺失，用该 id 在本 scene **最后一次出现**的框。
4. \(p_G^{\text{place}} = R_{\text{last}}\, p' + c_{\text{last}}\)，再变到末帧 LiDAR。
5. 可选：放置后再按末框×1.1 裁一次，抑制拖影（SurroundOcc 有类似步骤）。

### 2.6 输出布局

```
output/mapping-<stamp>-CST/
  manifest.json                 # 全局跑参、耗时、scene 列表
  scene-0103/
    static_map.ply              # (+ 可选 .pcd)
    dynamic_at_last.ply
    combined.ply
    boxes_last.json             # 用于放置的末框（及缺失回退说明）
    manifest.json               # 帧数、点数、track 数、耗时
    bev_static.png              # 全尺寸 BEV，非 contact sheet
    bev_dynamic.png
    bev_combined.png
    index.html                  # 可选
  scene-0916/
    ...
```

### 2.7 实现位置与入口

| 路径 | 作用 |
| --- | --- |
| `tools/mapping/geometry.py` | 位姿、定向框内点、voxel、可选 Poisson |
| `tools/mapping/scene_map.py` | 单 scene 建图核心 |
| `tools/mapping/visualize_map_bev.py` | 末帧坐标系 BEV PNG |
| `tools/mapping/run_mapping.py` | CLI：多 scene 批跑 |

示例：

```bash
/data/software/conda/anaconda3/envs/mv2d/bin/python tools/mapping/run_mapping.py \
  --dataroot /data/data/automomous/nuscenes/v1.0-mini \
  --version v1.0-mini \
  --tracking-json output/track-gt-20260903-155401-CST/delivery/tracking_results.json \
  --scenes scene-0103,scene-0916 \
  --voxel-size 0.1 \
  --box-expand 1.1 \
  --out-dir output/mapping-<stamp>-CST
```

### 2.8 风险与诚实边界

- 跟踪漏检/错 ID → 动态点漏进静态，或同一物体碎成多团。
- 仅 keyframe（2 Hz）→ 致密弱于 SurroundOcc「含中间 sweep」。
- 扩框 1.1 仍可能切掉物体边缘或吞掉路边静态。
- 官方位姿在 mini 上通常足够；若出现重影，优先查跟踪框而非先上 SLAM。
- Poisson 慢且易填洞过度；默认关闭是有意的。

---

## 3. 参考

- SurroundOcc: `tools/generate_occupancy_nuscenes/generate_occupancy_nuscenes.py`
- 本仓解读（外部只读）：`/data/code/location/SurroundOcc/docs/代码解读/SurroundOcc真值生成与动态物体处理.md`
- 上游跟踪交付：`docs/推理复现/nuScenes-HEDNet-SimpleTrack跟踪真值完成报告-grok-20260903-155544-CST.md`
