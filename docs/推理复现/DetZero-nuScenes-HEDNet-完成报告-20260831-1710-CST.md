# DetZero × nuScenes × HEDNet 真值生成 — 完成报告

> 时间：2026-08-31 CST ｜ 工作区：`DetZero-nuscenes-hednet-qwen3.8flash`（分支 `nuscenes-hednet-qwen3.8flash`）
> 方案：`docs/推理复现/DetZero-nuScenes-HEDNet-qwen3.8flash方案20260831.md` ｜ 状态：**链路 PASS / 效果 NOT_ESTABLISHED（tracking 环节被 GT 量化裁决为退化）**

## 0. 结论先行

| 维度 | 判定 | 依据 |
| --- | --- | --- |
| 可执行性 | **PASS** | S1→S6 两 scene（0103/40 帧、0916/41 帧）全部跑通，visuals 逐帧非空，全部产物带 sha256 manifest |
| 效果 | **NOT_ESTABLISHED** | R1 成立：Waymo tracking 在 nuScenes 2Hz 上把 mAP 从 0.74/0.83 打到 0.027，Pedestrian/Cyclist 轨迹全灭。R3 排除：pose 与坐标链实测零误差，退化是域差不是 bug |
| 发布就绪 | **不在范围** | 与方案 §0 一致：不接 coordinator、无 receipt/Release |

**一句话：DetZero 下游对 HEDNet-nuScenes 检测框的正确处理链已打通并证明坐标系无误，但"tracking+GRM/PRM 能提升 nuScenes 框质量"这一假设被 GT 量化否决——按方案 §2.1 R2 预案 fail-closed，正式真值输出应停在 detector 层。**

## 1. 交付物（全部在 worktree 内，未动 Waymo Stage-A 任何文件）

新增脚本（方案 §3.1 预告的 3 个）：
- `tools/external_detector/preprocess_nuscenes_scene.py` — S1
- `tools/external_detector/adapt_hednet_to_detzero.py` — S2
- `tools/external_detector/eval_nus_vs_gt.py` — S7

产物 generation root：`output/nus-stage-a-20260831-165710-CST/scene-{0103,0916}/`，每 scene 含
`data/`（Waymo 布局，无 GT）、`detector/`、`tracking/`、`refining/`、`final/`、`visuals/`（40/41 张 BEV png）、`eval_metric_report.json`；另有 `*-dt05` 变体（R1 对照实验，§4）。

## 2. 验收门逐条

| 门 | 结果 | 证据 |
| --- | --- | --- |
| G1 链路 | **PASS** | 两 scene S1→S6 exit 0；render_manifest 40/41 帧、逐帧 point_count>0、png 数=期望帧数 |
| G2 一致性 | **PASS** | R3 pose 对拍：`car_from_global`≡global→ego、`ref_from_car`≡ego→lidar，与 devkit 表误差 **0.0**；tracking 框经 `inv(pose)` 回 lidar 系后与 detector 源框最近邻距离中位 **0.00m** |
| G3 效果报告 | **PASS（按 fail-closed 如实记录）** | 三级 mAP 齐备（§3），R1=DEGRADED 如实判负 |
| G4 边界 | **PASS** | `git status`：0 个已跟踪文件改动；新增仅 tools/external_detector/ 3 脚本 + docs/ + output/ |

## 3. 核心数字（BEV 中心距 AP，0.5/1/2m 均值，≤50m，nuScenes 3 类映射）

| 阶段 | 0103 mAP | 0916 mAP | 0103 各类 | 0916 各类 |
| --- | --- | --- | --- | --- |
| detector (HEDNet) | **0.743** | **0.826** | V .757 / P .839 / C .633 | V .895 / P .916 / C .668 |
| +tracking | 0.027 | 0.027 | V .082 / P 0 / C 0 | V .081 / P 0 / C 0 |
| +GRM/PRM (no-CRM) | 0.026 | 0.028 | V .079 / P 0 / C 0 | V .084 / P 0 / C 0 |

GT 量级（scene-0103）：Vehicle 866 / Pedestrian 857 / Cyclist 47 框。detector 召回正常，说明评估器与类别映射工作正确。

## 4. 风险裁决与根因

- **R1 tracking 碎片化：成立（DEGRADED）。** detector 2769/2615 框 → tracking 仅剩 14/22 条轨迹，且全部是 Vehicle：Pedestrian/Cyclist 在 2Hz、0.4-0.5s 帧距下跨帧 IoUBEV 关联断链，随后被 `empty_track_delete LEAST_AGE: 5`（age≥5 才保命）整类删除——dropped.pkl 中可见 Ped 54+68、Cyc 16+43 框成批被丢。对照实验：按方案授权把 Kalman `DELTA_T` 0.1→0.5（`tracking-dt05` 全套重跑），mAP 0.026/0.054，仍判 DEGRADED；碎片化主因是 Waymo 域关联门限 + 最短轨迹年龄约束，非 dt 单一参数。
- **R2 GRM/PRM 域差：无法翻案也无需翻案。** final 相对 tracking 的 ΔmAP 在 ±0.02 内（0.027→0.026 / 0.027→0.028），NOT_DEGRADED——但这是在 tracking 已塌陷到 0.027 的地板上比较，GRM/PRM 对 Vehicle 轨迹有微幅正修正（0916: .081→.084），不构成可用性。
- **R3 pose 约定：排除。** 见 G2；S1 用 `inv(car_from_global @ ref_from_car)` 作 lidar→global，链路逆变换往返误差 0.00m。
- **R4 类别折叠：如设计执行。** manifest 记录丢弃 traffic_cone 140+37、barrier 7+7。

## 5. 对方案的两处偏离（均为方案错误，实测纠正）

1. **方案 §3.2-S1"补第 6 列 0"不可行**：`daemon/prepare_object_data.py:270-271` 按 `col5 == -1` 过滤 Waymo 首次回波，填 0 会把 nuScenes 全部点滤光、GRM/PRM 输入为空。实际填 **-1**（nuScenes 单回波=首回波），intensity 列同时按方案 ×255。
2. **方案 §3.3-S3 运行目录写错**：`run_track.py` 的 `_BASE_CONFIG_: cfgs/...` 相对路径要求 cwd=`tracking/tools` 且 cfg 用绝对路径——正式 `run_stage_a.py:450,478` 本来就是这么调度的（`execution_root/tracking/tools`），照抄即可，非新问题。

## 6. 真值标签产出建议（fail-closed）

本批 81 帧的可用真值 = **detector 层框**（`detector/hednet_frames_*.pkl`，0.74/0.83 mAP、含速度），而非 tracking/refined 输出。若要让 DetZero 下游在 nuScenes 域可用，需要的最小改动是（本方案未做，不擅自扩界）：tracking cfg 的关联门限与 `LEAST_AGE` 按 2Hz 重标、或放宽保命条件；根治则需要 nuScenes 域的运动模型参数集。GRM/PRM 权重不动权重合同的前提下无解（域差）。

## 7. 环境备忘

- 全部命令用 `mv2d` env python（与 Waymo Stage-A 同款）；CUDA 扩展 `.so` 不在 git 内，worktree 从主仓 `utils/` 复制（已 gitignore，不进提交）。
- 复现命令序列已固化在各 manifest 的 provenance 相对路径中；正式命令清单见 `output/nus-stage-a-20260831-165710-CST/`。

---

## 8. 增补（20260831-1735）：修订路线 §5 已实现 — 自研无 Waymo 跟踪出真值

新脚本 `tools/external_detector/track_hednet_boxes.py`（纯 numpy/scipy，零学习组件）：
匈牙利分配 + 全局系 EMA 有限差分速度外推，门限 Vehicle 6.0 / Cyc 3.0 / Ped 2.0m，max_age=3，尺寸=轨迹内 score 加权中值平滑。

**两处与方案的偏差由实测裁定**（§5.2 原设计据此修订）：
1. **不用 HEDNet vx/vy 做外推**：与 GT 位移对拍，其角度中位误差 31°、161°（lidar 系解释）——速度头不可信；改用轨迹自身差分速度，ID switch 从 89→56（0103）。vx/vy 仍原样写入真值框。
2. **门限按测得的帧间抖动定，不按物理尺寸**：2Hz 下 detector 同目标相邻帧中心漂移 2~5m，物理尺寸门限（2.5m）导致 86% 单帧轨迹；放宽后 obs/track 1.2→2.4（0103）/4.6（0916），0916 Vehicle 最长轨迹 40 帧=全片。

结果（scene-0103 / 0916）：

| 指标 | detector | 自研跟踪 | 判据 G3'（≥det−0.02） |
| --- | --- | --- | --- |
| mAP | 0.743 / 0.826 | **0.743 / 0.826** | **PASS**（不破坏中心） |
| 尺寸 L1 误差 vs GT | 0.160 / 0.151 | **0.144 / 0.148** | 平滑带来 −4%/−2% |
| R1 判定 | — | NOT_DEGRADED | Waymo 版为 DEGRADED |

产物：`scene-*/track/hednet_tracked_frames_*.pkl`（**真值输出**，schema 同 S2 帧列表，含逐帧框+轨迹号）+ `hednet_tracks_*.pkl`（按目标组织）+ `eval_metric_report_selftrack.json` + `visuals-selftrack/`（81 张）。GRM/PRM/CRM 未运行。

诚实边界：0103 行人碎片化仍偏多（persistent GT 71 个中 32 个被切成 ≥4 段、长轨迹 ID switch 56/103）——2Hz + detector 抖动下无外观特征的纯几何跟踪的固有上限；真值框质量不受影响（mAP/尺寸已证），受影响的仅是轨迹 ID 连续性。需要更稳 ID 时再上外观特征（方案 §5 范围外）。

---

## 9. 增补（20260903）：T1 跟踪升级 v2 — Kalman + 卡方门限，按实测碎片化根因修正

**触发**：用户查看 `output/nus-stage-a-20260831-165710-CST` 后反馈"跟踪后的结果很差"。

**先澄清看到的对象**：该目录同一 scene 下并存 5 套"跟踪输出"，质量天差地别：

| 目录 | 内容 | 每帧均框数(0103) | 状态 |
| --- | --- | --- | --- |
| `tracking/` `refining/` `final/`(含 `-dt05`) | 废弃的 Waymo tracking+GRM/PRM 链（§3，已被 R1 判负） | **3.8** | 仅存档，勿当真值 |
| `track/`（v1 自研 T1） | §5 自研轻量跟踪（EMA 差分速度+固定门限） | 69.2 | 被 v2 取代（见下） |
| `track-v2/`（本次新增） | T1 v2：Kalman CV + 卡方马氏门限 | 69.2 | **现行真值输出** |

用户看到的"很差"即 `final/`（Waymo 链输出）——该链在修订路线 §5 中已明确砍掉，其产物不是交付物。

**v1 的量化缺陷**（本次诊断 `_diag_gt_tracks.py`/`_diag_gate.py`）：
- v1 固定门限（Veh 6 / Ped 2 / Cyc 3 m）低于实测：同目标相邻帧原始位移中位数 Veh 3.0~3.3 / Ped 2.1~2.8 / Cyc 3.3 m，而 EMA 差分外推残差 p90 达 3.4~5.7 m → 高频断链。
- 结果：0103 Cyclist 98%、Pedestrian 78% 的轨迹只有 1 帧（v1 stats：Ped 883 条轨迹承载 1540 个观测，obs/track=1.7）。

**v2 改动**（`track_hednet_boxes.py`，纯 numpy + filterpy1.4.5(MIT，mv2d 已装)）：
- 关联改为每类 CV-Kalman（filterpy）在全局系的**马氏距离** + 卡方门限（df=2，χ²=9.21≈99%），门限随丢帧数自动放宽（P 阵增长），固定半径做不到；
- 量测噪声/过程噪声按上表实测抖动标定（R: V/C 2.5、P 2.0 m；Q: 加速度 V 1.0、P/C 1.5 m/s²）；匈牙利分配不变；
- 输出合同、尺寸中值平滑、vx/vy 直通均不变。
- 造轮子核查（用户要求优先找库）：唯一标准实现 AB3DMOT（3D KF+匈牙利，与本需求逐字对应）许可为 **CMU 非商用学术协议，不可引入**（nuScenes 场景亦受其非商用条款约束，与本项目商用真值定位冲突）；CenterPoint 的跟踪即"最近点贪心匹配"（≈v1 思路，已被实测否定）；最终复用已装的 filterpy（MIT）内核 + scipy 分配，新代码只有关联逻辑本身。

**结果（同口径对比，obs/track 与 1 帧轨迹占比）**：

| scene | 版本 | 轨迹数 | obs/track | 1帧轨迹 | ≥10帧轨迹 | mAP(G3') |
| --- | --- | --- | --- | --- | --- | --- |
| 0103 | v1 | 1168 | 2.37 | 73% | 69 | 0.743 |
| 0103 | **v2** | **251** | **11.03** | **5%** | **119** | 0.743 PASS |
| 0916 | v1 | 574 | 4.56 | 53% | 79 | 0.826 |
| 0916 | **v2** | **198** | **13.21** | **12%** | **104** | 0.826 PASS |

GT 对拍口径（`_diag_gt_tracks.py`）：detector→GT 关联率 66%/81% 为质量天花板（检测器本身的漏检+2Hz 抖动），v2 的长轨迹不再把同一目标切碎。逐帧框内容与 detector 完全一致（mAP、中心、yaw 直通），跟踪只改 ID 与尺寸平滑——**真值框质量不受跟踪环节损害，轨迹连续性提升 4~5 倍**。

**诚实边界**：
- 无外观特征的纯几何跟踪在遮挡/交叉处仍有 ID 混淆上限（v2 长轨迹与 GT-ID 纯度对拍 ~96% 存在"最近邻口径"污染，未逐例人审）；根治需 ReID 外观特征（超出 §5 范围）。
- 官方 nuScenes tracking 评估（AMOTA）本应加为旁证，但本机 devkit 与 motmetrics 1.4.0/pandas 2.2 不兼容（`pred_frequencies` 断链 + 空 MultiIndex），旁路尝试已放弃并删除脚本；如需正式 AMOTA，需钉 motmetrics==0.9.9 的隔离环境，属发布化工作。
- 参数网格（r_scale∈{0.8,1,1.4}×χ²∈{9.21,16}）显示 0103 行人 1 帧轨迹占比在所有设置下 ≤2%（v1 的 78% 是模型问题非参数噪声），取 rs=1.0/χ²=9.21 居中值。

**产物**：`scene-*/track-v2/hednet_tracked_frames_*.pkl`（真值）+ `hednet_tracks_*.pkl` + `eval_metric_report_trackv2.json`（三级 mAP）+ `visuals-trackv2/`（81 张 BEV，逐帧非空，40/41 帧）。自检查：`tests/test_track_hednet_v2.py`（合成 2Hz 双目标，断言 0 碎片/0 丢观测/0 串轨，PASS）。
