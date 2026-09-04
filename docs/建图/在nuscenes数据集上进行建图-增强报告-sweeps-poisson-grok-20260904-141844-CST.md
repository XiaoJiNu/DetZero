# 建图增强报告：全 sweep 融合 + Poisson 填洞

| 字段 | 内容 |
| --- | --- |
| 作者 / 助手 | **grok** |
| 报告时间 | 2026-09-04 14:18:44 CST（2026-09-04T06:18:44Z） |
| 工作区 | `/data/code/cv/AutoLabel/DetZero-nuscenes-simpletrack-trackgt` |
| Git 分支 | `nuscenes-mapping-surroundocc` |
| 基线产物 | `output/mapping-20260904-132942-CST`（keyframes only，无 Poisson） |
| 本次产物 | `output/mapping-20260904-141533-CST`（全 sweeps + Poisson depth=9） |

## 1. 变更摘要

1. **`tools/mapping/scene_map.py`**
   - `list_lidar_sample_data_tokens`：沿 `LIDAR_TOP` `prev/next` 收集全链（含非关键帧）。
   - `load_lidar_points_from_sd`：按 `sample_data` token 读点云/位姿。
   - `interpolate_tracking_boxes`：按 `tracking_id` 对齐；`translation`/`size` 线性插值；`rotation` 用 `Quaternion.slerp`；单侧存在则用该侧框。
   - `build_keyframe_timeline`：关键帧时间线 `{ts, sample_token, boxes}`。
   - `build_scene_map(..., use_sweeps=True)`：默认遍历全部 lidar SD；关键帧用跟踪 JSON 精确框，sweep 用插值框；动态放置仍优先**末关键帧**出现；manifest 增加 `n_keyframes` / `use_sweeps` / `use_poisson`；`frames_meta` 含 `sample_data_token` / `is_key_frame` / `timestamp`。
2. **`tools/mapping/run_mapping.py`**
   - `--use-sweeps`（默认开）与 `--keyframes-only`；根 manifest 写 `use_sweeps`；摘要打印 frames / sweeps / poisson。
3. **`tools/mapping/README.md`**：补充 sweeps + poisson 用法。

## 2. 复现命令

```bash
cd /data/code/cv/AutoLabel/DetZero-nuscenes-simpletrack-trackgt
/data/software/conda/anaconda3/envs/mv2d/bin/python tools/mapping/run_mapping.py \
  --dataroot /data/data/automomous/nuscenes/v1.0-mini \
  --version v1.0-mini \
  --tracking-json output/track-gt-20260903-155401-CST/delivery/tracking_results.json \
  --scenes scene-0103,scene-0916 \
  --poisson \
  --use-sweeps
```

- Python：`/data/software/conda/anaconda3/envs/mv2d/bin/python`
- Poisson 未 OOM；未降 `--poisson-depth`（保持默认 9）。

## 3. Before / After 指标

对比 `output/mapping-20260904-132942-CST`（KF only, poisson=off）→ `output/mapping-20260904-141533-CST`（sweeps+poisson）。

| scene | 项 | Before | After |
| --- | --- | ---: | ---: |
| scene-0103 | n_frames | 40 | **389** |
| | n_keyframes | 40 | 40 |
| | n_static_raw_accumulated | 911,016 | **8,878,050** (~9.7×) |
| | n_static 导出 | 447,477（voxel） | **98,929**（Poisson mesh 顶点，密度修剪后） |
| | n_dynamic_raw_assigned | 99,657 | **970,596** (~9.7×) |
| | n_dynamic 导出 | 72,972 | **347,356** (~4.8×) |
| | n_tracks_placed | 560 | 565 |
| | elapsed_sec | 4.9 | 92.4 |
| scene-0916 | n_frames | 41 | **399** |
| | n_keyframes | 41 | 41 |
| | n_static_raw_accumulated | 917,646 | **8,899,248** (~9.7×) |
| | n_static 导出 | 558,776（voxel） | **265,006**（Poisson mesh 顶点） |
| | n_dynamic_raw_assigned | 139,885 | **1,391,164** (~9.9×) |
| | n_dynamic 导出 | 111,552 | **525,509** (~4.7×) |
| | n_tracks_placed | 413 | 421 |
| | elapsed_sec | 4.5 | 88.5 |

总墙钟约 **183 s**。

说明：开启 Poisson 后 `n_static_after_voxel` 实际是 **Poisson 网格顶点**（低密度顶点已裁），点数可少于纯 voxel，但表面更连续、空洞被填。动态致密主要来自中间 sweep 点云 + 插值框。

## 4. 输出路径

```
output/mapping-20260904-141533-CST/
  manifest.json
  scene-0103/  # static/dynamic/combined .ply+.pcd, boxes_last*.json, BEV PNGs, manifest.json
  scene-0916/
```

大体积 `output/` 仍 gitignore，未入库。

## 5. 已知限制

- 插值框依赖跟踪在相邻关键帧的 `tracking_id` 稳定；漏检/换 ID 会在 sweep 上错分静/动。
- 仅单侧有框时直接复用该侧（不因 alpha 丢弃），快速运动物体边缘可能略偏。
- Poisson 慢、吃内存；本机 depth=9 通过。若 OOM 可试 `--poisson-depth 8`。
- Poisson 后静态点数不可与 voxel 点数直接横向对比“变少=更差”。
- 动态放置仍优先末关键帧框；sweep 外观不参与放置位姿选择（除非该 track 从未出现在关键帧）。

## 6. 验收对照

| 门 | 结果 |
| --- | --- |
| 两 scene 跑通、产物非空 | 通过 |
| n_frames ≈ 全 lidar 链（~389/399） | 通过 |
| 动态点数显著高于 keyframe-only | 通过 |
| 未提交 output/ 大文件 | 通过 |
