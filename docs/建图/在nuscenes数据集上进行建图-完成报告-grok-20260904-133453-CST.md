# 在 nuScenes 数据集上进行建图 — 完成报告

| 字段 | 内容 |
| --- | --- |
| 作者 / 助手 | **grok**（算法复现） |
| 文档创建时间 | 2026-09-04 13:34:53 CST（2026-09-04T05:34:53Z） |
| 文档版本 | v1.0 |
| 工作区 | `/data/code/cv/AutoLabel/DetZero-nuscenes-simpletrack-trackgt` |
| Git 分支 | `nuscenes-mapping-surroundocc` |
| 方案文档 | `docs/建图/在nuscenes数据集上进行建图.md` |
| 产出目录 | `output/mapping-20260904-132942-CST/` |
| Python | `/data/software/conda/anaconda3/envs/mv2d/bin/python`（open3d 0.18） |

## 修订记录

| 时间 (CST) | 作者 | 版本 | 变更摘要 |
| --- | --- | --- | --- |
| 2026-09-04 13:34:53 CST | grok | v1.0 | 初版完成报告：scene-0103/0916 静动态建图跑通；默认无 Poisson |

## 1. 验收门

| 门 | 结果 | 说明 |
| --- | --- | --- |
| G1 链路 | **PASS** | 两 scene 共 81 keyframe 跑通；PLY/PCD/BEV/manifest 齐全 |
| G2 坐标 | **PASS** | 静态/动态均变换到末关键帧 `LIDAR_TOP`；BEV 原点为末帧 LiDAR |
| G3 静动 | **PARTIAL** | 静态点云成图可读；动态有致密效果，但大量短 track（中位局部点数≈6）→ 碎团多，见 §4 |
| G4 边界 | **PASS** | 未改跟踪 JSON；未跑 SLAM/KISS-ICP；未用 mmcv；`output/` 已在 `.gitignore` |
| G5 可复现 | **PASS** | `tools/mapping/run_mapping.py` 一键复跑 |

## 2. 运行命令

```bash
/data/software/conda/anaconda3/envs/mv2d/bin/python tools/mapping/run_mapping.py \
  --dataroot /data/data/automomous/nuscenes/v1.0-mini \
  --version v1.0-mini \
  --tracking-json output/track-gt-20260903-155401-CST/delivery/tracking_results.json \
  --scenes scene-0103,scene-0916 \
  --voxel-size 0.1 \
  --box-expand 1.1 \
  --out-dir output/mapping-20260904-132942-CST
```

- `--poisson`：**未开**（默认关）
- 跟踪输入：`track-gt-20260903-155401-CST/delivery/tracking_results.json`
- 总耗时：**11.595 s**（含 BEV 渲染）

## 3. 指标

### 3.1 总览

| scene | frames | static (voxel 0.1) | dynamic@last | combined | tracks placed | scene time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scene-0103 | 40 | **447 477** | **72 972** | 520 449 | 560 | 4.946 |
| scene-0916 | 41 | **558 776** | **111 552** | 670 328 | 413 | 4.468 |

### 3.2 累积原始点数（voxel 前）

| scene | static raw accum | dynamic raw assigned |
| --- | ---: | ---: |
| scene-0103 | 911 016 | 99 657 |
| scene-0916 | 917 646 | 139 885 |

### 3.3 跟踪侧对照（本 scene 内 unique `tracking_id`）

| scene | unique IDs in JSON | tracks with points / placed |
| --- | ---: | ---: |
| scene-0103 | 567 | 560 / 560 |
| scene-0916 | 423 | 413 / 413 |

scene-0103：仅 **66** 条 track 在末帧有框，其余用「最后一次出现」回退放置；`frames_seen` 直方图峰值在 2 帧（短轨迹很多）。

### 3.4 产物路径

```
output/mapping-20260904-132942-CST/
  manifest.json
  scene-0103/
    static_map.ply|.pcd
    dynamic_at_last.ply|.pcd
    combined.ply|.pcd
    boxes_last.json
    boxes_last_lidar.json
    manifest.json
    bev_static.png / bev_dynamic.png / bev_combined.png   # 1800×1800
    index.html
  scene-0916/  (同上)
```

预览（已入库小图，各 &lt; 1.5 MB）：

- `docs/建图/assets/scene-0103_bev_static.png`
- `docs/建图/assets/scene-0103_bev_combined.png`
- `docs/建图/assets/scene-0916_bev_dynamic.png`
- `docs/建图/assets/scene-0916_bev_combined.png`

## 4. 诚实局限

1. **短轨迹碎裂**：SimpleTrack 在 mini 上 ID 数量大（scene-0103≈567），大量 2～3 帧轨迹；动态地图呈「许多小团」而非少量完整车体。这是跟踪输入特性，不是位姿失败。
2. **仅 keyframe**：未融合非关键 `sample_data` sweep，致密弱于 SurroundOcc 官方真值脚本。
3. **框内=动态**：静止车辆仍进动态分支（与 SurroundOcc 一致）；漏检框内点会污染静态。
4. **扩框 1.1 + 近车裁剪**：可能切掉物体边缘或吞掉路边静态；自车 `self_range=[3,3,3]` 为启发式。
5. **Poisson 未跑**：默认关闭以保速度；需要补洞时可加 `--poisson`（仅静态，耗时与深度相关）。
6. **未做占用体素 / 语义**：本交付是点云地图，不是 SurroundOcc 训练用 occ GT。
7. **官方位姿**：未做闭环或 ICP 精炼；若 BEV 见明显重影，应先查跟踪框与标定，而不是默认上 SLAM。

## 5. 代码与文档

| 路径 | 说明 |
| --- | --- |
| `tools/mapping/geometry.py` | 位姿、定向框内点、voxel、可选 Poisson |
| `tools/mapping/scene_map.py` | 单 scene 静/动建图 |
| `tools/mapping/visualize_map_bev.py` | 全尺寸 BEV + HTML index |
| `tools/mapping/run_mapping.py` | CLI |
| `docs/建图/在nuscenes数据集上进行建图.md` | 方案定稿 |
| `.gitignore` | 已含 `output/`（大 PLY 不入库） |

## 6. 结论

SurroundOcc 风格静/动态建图链路在 nuScenes mini 两 scene 上**已跑通**，输出对齐末帧 LiDAR，耗时约 12 s。动态致密受跟踪短 ID 制约，验收上 G3 记为 **PARTIAL**；后续若要「车体级」致密，应先过滤短轨迹 / 提高跟踪稳定性，或改用官方 `instance_token` 对照实验。
