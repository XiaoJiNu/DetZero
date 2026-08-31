# DetZero × nuScenes × HEDNet 真值生成方案

> 时间：2026-08-31 CST ｜ 模型：qwen3.8-flash ｜ 性质：可行性分析与实施方案（含未执行命令，标注为"计划"）
>
> **20260831-1730 修订**：原 S3–S5（Waymo tracking + GRM/PRM）已被 GT 量化判负（见完成报告 20260831）。现行路线改走 **§5：detector → 自研轻量跟踪 → 真值**，不再使用任何 Waymo 训练权重/调参/后处理。S1/S2/S6/S7 与产物布局全部复用。

## 0. 结论先行

**总体判定：技术路线可行；nuScenes v1.0-mini 可行且反而是优选，不需要另下完整 clip。**

| 维度 | 判定 | 依据 |
| --- | --- | --- |
| 可执行性 | **PASS（预计 1~2 天工程量）** | HEDNet 输出已是 DetZero 框格式 `(N,9) float32 [x,y,z,l,w,h,yaw,vx,vy]`；Stage-A 各独立工具（adapter / run_track / run_stage_a_refining / combine_no_crm / render）均可命令行单独调用，插入一个 nuScenes 预处理脚本即可串成链 |
| mini 数据集 | **PASS** | `nuscenes_infos_10sweeps_val.pkl` 实测 81 帧 = scene-0103（40 帧）+ scene-0916（41 帧），两 scene 内部时间连续（0.4~0.5s 间隔），正好是 2 条 ~20s 的 clip；且 mini 自带 3D GT，可像 Waymo Stage-A 那样输出检测→跟踪→精修的逐环节对比 |
| 换完整 clip | **不需要**（作为增强可选项） | nuScenes 每个 scene 上限就 ~20-40s / 40 帧，trainval 完整 clip **不会更长**，只带来更多场景；Waymo 段 199 帧≈10Hz×20s 与 mini 单 scene 时长相当，差别只在帧率 |
| 效果（efficacy） | **NOT_EVALUATED，有明确风险点** | ① tracking/GRM/PRM 权重全部是 Waymo 训练的（域差）② nuScenes 2Hz vs Waymo 10Hz，跟踪运动模型的 dt 假设需实测 ③ intensity 量纲不同（Waymo [0,255] vs nuScenes [0,1]）。mini 有 GT，跑完即可量化裁决 |
| 发布就绪（release） | **不在本方案范围** | Waymo Stage-A 报告的结论就是发布链未闭合（comparator TOCTOU 等）；本方案目标是"跑通并产出真值标签 + 效果数字"，不重做 freeze/receipt/Release-C。若以后要正式化，先关闭 Waymo 报告的 14 节 blocker 再套同一框架 |

**GT 使用澄清：最终真值标签的生成链（S1→S6）完全不使用 nuScenes GT**——输入只有点云、ego 位姿和 HEDNet 推理框，tracking/GRM/PRM/final 全程不读 GT（Waymo Stage-A 即在零 GT 的 testing 段跑通全链，可证）。mini 自带 GT 仅用于两个旁路：S7 效果评估（裁决 R1/R2，不参与打标）与 R3 pose 约定一次性对拍。若连评估也不要，删 S7 即可，但效果状态只能记 `NOT_ESTABLISHED`。

一句话路线：**复用 HEDNet 已有 mini 推理产物（result.pkl，81 帧含速度）→ 新写一个 `preprocess_nuscenes_scene.py`（pcd.bin→npy、ego pose→Waymo 布局）→ 新写一个薄 adapter（10 类→Vehicle/Pedestrian/Cyclist，框格式直通）→ 依次手动调用 DetZero 现成的 tracking → refining(GRM/PRM, no-CRM) → final → render 五个工具，最后用 mini GT 算逐环节 mAP。**

---

## 1. 已核实的双侧现状

### 1.1 HEDNet 侧（`/data/code/cv/AutoLabel/BEV-OD/HEDNet-qwen`，报告见其 `docs/复现推理/完成报告.md`）

- `output/mini/eval/epoch_2/val/default/result.pkl`：81 帧，每帧 `metadata.token`、`name`、`score`、`boxes_lidar (N,9) float32`——**与 DetZero tracking 输入 schema 同形**（PCDet 约定 l,w,h + yaw + vx,vy），且比 Waymo Stage-A 多给了速度（Waymo adapter 只能 vx=vy=0）。
- 类别为 nuScenes 10 类（实测本帧出现 car/truck/pedestrian/bicycle/traffic_cone）。
- `data/nuscenes/v1.0-mini/nuscenes_infos_10sweeps_val.pkl`（同 `/data/data/automomous/nuscenes/v1.0-mini/`）：每帧含 `timestamp`、`token`、`lidar_path`（.pcd.bin 5 列）、`gt_boxes`/`gt_names`（**有 3D GT，效果可量化**）、`car_from_global`/`ref_from_car` 4×4 位姿。
- 环境 `detzero-open3dml-gpu`（torch 2.13+cu130）可跑 HEDNet 推理。

### 1.2 DetZero Stage-A 侧（Waymo，报告见 `DetZero-Waymo-Stage-A完成报告-20260827-133847-CST.md`）

链路：TFRecord 预处理 → 外部 detector → adapter（detzero_result.pkl）→ `tracking/tools/run_track.py` → `tools/external_detector/run_stage_a_refining.py`（prepare_object_data + GRM/PRM）→ `combine_grm_prm_no_crm.py`（no-CRM 直通 final score）→ `render_waymo_sequence.py` 可视化。

对 nuScenes 复用的硬约束（读码核实）：

| 约束点 | 位置 | 影响 |
| --- | --- | --- |
| 数据布局写死 Waymo：`root/waymo_processed_data/segment-<name>/%04d.npy` + `<name>.pkl`（info：`time_stamp` int、`pose` float64 4×4 v2w、`sample_idx`、`num_points_of_each_lidar` 5 元） | `pipeline.py:publish_preprocessed_records` / `_validated_points/_validated_pose` | 需新写 nuScenes 预处理脚本产出同布局；点数按 6 列 float32 校验 |
| 类别只有 3 类：`CLASS_NAME: [Vehicle, Pedestrian, Cyclist]`（tracking cfg、GRM/PRM 权重文件名、adapter class_map 三处一致） | `tracking/tools/cfgs/*`、`checkpoints/`、`adapt_open3dml_to_detzero.py` | nuScenes 10 类需映射（见 §3.2），barrier/traffic_cone 被丢弃 |
| 框 yaw 约定：Waymo Open3D-ML `[w,h,l]+yaw → [l,w,h]+heading(-yaw-π/2)` | `pipeline.py:open3dml_box_to_detzero` | **HEDNet 输出已是 PCDet/lidar 约定，nuScenes adapter 直通，不套这个转换** |
| 时间戳：int64、严格递增 | `pipeline.py:adapt_raw_predictions` | nuScenes `timestamp` 秒浮点 → ×1e6 转微秒整型 |
| refining 点特征：GRM/PRM 用 `pts[:, :3]` + `pts[:, [3]]`（强度）；权重 Waymo 域 | `waymo_geometry_dataset.py:103-106` | nuScenes intensity∈[0,1]，Waymo≈[0,255]：预处理时 ×255 拉齐量纲；效果仍存疑，用 GT 量化 |
| `run_stage_a.py` 正式 coordinator：199 帧合同、TFRecord sha、Open3D-ML 173-file 源码绑定、receipt/ledger | `run_stage_a.py`、`stage_a_source_policy.json` | **绕过它**，手动按序调 5 个工具。它本身也没过 Release-C |

无 CRM checkpoint → 与 Waymo Stage-A 相同的 `NOT_EXECUTED_NO_CRM_BY_DESIGN` 处理：`combine_grm_prm_no_crm.py` 已实现 GRM+PRM 几何/位置修正 + tracking score 直通，直接复用。

---

## 2. mini vs 完整 clip 的裁定（用户核心问题）

1. **"连续的 clip"在 nuScenes 里 = 一个 scene**，v1.0-mini 的两个 val scene（0103/0916）各自就是完整连续的 ~20s 片段，帧间无跳切（实测 gap 0.4-0.5s）。跟踪、GRM/PRM 轨迹窗口所需的时序连续性**已经满足**。
2. 换 trainval 完整 clip **不增加单场景帧数**（nuScenes 全场都 ~40 帧/scene），只增加场景数量与交通密度。收益是"演示更好看"，代价是 ~1TB 级下载 + 重跑 HEDNet info/推理。
3. 决定性优势：mini 有官方 3D GT。Waymo Stage-A 被卡在 `NOT_ESTABLISHED`（testing 段 laser_labels=0，无法算指标）；nuScenes-mini 路线反而能给出 tracking 前后 mAP/AMOTA 对比——**效果维度第一次可闭环**。
4. 结论：**先用 mini 双 scene 打通并量化；若数字显示需要更密场景（例如 916 场景动态目标过少），再挑 1 个 trainval scene（约 40 帧，只下载该 scene 的 samples/sweeps + v1.0-trainval meta）做加测**。HEDNet 复现脚本 `tools/process_tools/gen_mini_info.py` 改 split 参数即可为单 scene 生成 info，属低成本后备。

（备注：另有自采 bag 2103 帧连续点云可当"长 clip"，但域差严重（置信度<0.3、误检立柱为 bus），不适合做真值质量演示，仅作链路备选，不在本方案。）

## 2.1 主要风险与裁决手段

| # | 风险 | 严重度 | 裁决手段 |
| --- | --- | --- | --- |
| R1 | tracking 权重按 Waymo 10Hz 调的关联门限，2Hz 大 dt 下轨迹碎片化/误关联 | 高 | 跑完用 GT 算 det→track mAP 前后对比；碎片化则调 `waymo_detzero_track.yaml` 门限（允许改 yaml 不改权重） |
| R2 | GRM/PRM 是 Waymo 域精修头，对 nuScenes 框可能整体变差 | 高 | 输出 det/track/refined 三段指标对比，若 refined 变差则 final 停在 tracking 段并如实记录（fail-closed，不硬发布） |
| R3 | pose 方向/系约定搞反（car_from_global 的 v2w/w2v；lidar≈ego 差一个 lidar2ego 旋转） | 中 | 用 GT 框跨帧投影对拍校验一次（10 行脚本），确认后再进 tracking |
| R4 | 类别映射把 construction_vehicle/trailer 折进 Vehicle 造成尺寸离群 | 低 | 保留原类别 tag 进 manifest，便于事后剔除 |

---

## 3. 实施方案

### 3.1 目录与产物

```
DetZero/data/nuscenes-stage-a/            # nuScenes 版数据根（仿 Waymo 布局）
  waymo_processed_data/scene-0103/*.npy   # (N,6) float32: x,y,z,int*255,ring,t
  waymo_infos_test.pkl                    # 每 scene 的 info 列表（Waymo 字段名）
  ImageSets/test.txt
DetZero/output/nus-stage-a-<时间戳>/
  detector/hednet_frames_<scene>.pkl      # adapter 输出（detzero_result 同构）
  01-preprocess/ 02-detect/ 03-tracking/ 04-refining/ 05-final/ visuals/
  eval/metric_report.json                 # det/track/refined 对比 GT 的 mAP
```

新写代码只有 3 个小脚本（合计 ~350 行），全部放 `tools/external_detector/`，复用 `pipeline.py` 的 `_validated_pose/_sha256_file/rename_noreplace` 等既有工具函数：

### 3.2 步骤

**S1 preprocess_nuscenes_scene.py**（新，~120 行）
- 输入：`nuscenes_infos_10sweeps_val.pkl` + 场景名（scene-0103 / scene-0916）。
- 每帧：`pcd.bin` reshape(N,5) → 补第 6 列 0 → float32；timestamp×1e6→int64；pose 从 `car_from_global`/`ref_from_car` 组合成 lidar→world 4×4 float64（方向经 R3 对拍定标）；`num_points_of_each_lidar` 填 `[N,0,0,0,0]`。
- 按帧号排序写 `%04d.npy` + `waymo_infos_<split>.pkl`，字段名全部对齐 Waymo info（`time_stamp/sample_idx/sequence_name/pose/lidar_path/num_points_of_each_lidar`），使 tracking/refining/render 零改动读取。**info 产物不含 GT 框**；S7 评估直接读原始 nuScenes info pkl，与生成链物理解耦。
- 注：S3/S4 用 `--split test`——这是代码级 GT-free 保证：`tracking/detzero_track/datasets/dataset.py:26` 中 assign_mode 仅在 split≠test 时加载 GT，`run_stage_a_refining.py:147` 固定 split="test" 走 prepare_object_data 的无 GT 分支（Waymo Stage-A 零 GT 跑通即此路径）。

**S2 adapt_hednet_to_detzero.py**（新，~80 行；或给现有 adapter 加 `--source hednet` 分支，倾向独立小脚本不动 Waymo 合同）
- 输入：HEDNet `result.pkl` + S1 info（token 连接）。
- 类别映射：`{car,bus,truck,construction_vehicle,trailer}→Vehicle`，`{pedestrian}→Pedestrian`，`{bicycle,motorcycle}→Cyclist`，`{traffic_cone,barrier}→丢弃`（记入 manifest 丢弃计数）。
- 框直通 `(N,9)` float32，vx/vy 用 HEDNet 输出；输出与 `adapt_raw_predictions` 同构的 frame dict。score 阈值 0.1（mini_val 域内正常 >0.3）。

**S3~S6 复用现成工具，逐环节命令行**（计划，未执行）：

```bash
# 环境：mv2d（Waymo 复现同款，open3d/track 依赖在其中）
cd /data/code/cv/AutoLabel/DetZero
P=/data/software/conda/anaconda3/envs/mv2d/bin/python

# S3 tracking（每 scene 一次；run_track.py 的 dataset 类按 waymo_root 读 info，
#    若其 WaymoTrackDataset 强校验 gt_path，给 S1 同时产一份 GT info pkl 即可满足）
$P tracking/tools/run_track.py \
  --cfg_file tools/cfgs/tk_model_cfgs/waymo_detzero_track.yaml \
  --data_path output/nus-stage-a-<T>/02-detect/hednet_frames_scene-0103.pkl \
  --root_path data/nuscenes-stage-a --split test \
  --output_path output/nus-stage-a-<T>/03-tracking/scene-0103

# S4 refining: prepare_object_data + GRM/PRM（沿用 run_stage_a_refining.py，
#    它只吃 waymo-root + tracking 输出，S1 布局已满足）
$P tools/external_detector/run_stage_a_refining.py \
  --waymo-root data/nuscenes-stage-a \
  --tracking output/nus-stage-a-<T>/03-tracking/scene-0103 \
  --output-dir output/nus-stage-a-<T>/04-refining/scene-0103 --device cuda

# S5 final（no-CRM）+ S6 可视化
$P tools/external_detector/combine_grm_prm_no_crm.py \
  --tracking .../03-tracking/scene-0103 --detector-frames .../02-detect/...pkl \
  --refining-dir .../04-refining/scene-0103 \
  --output-dir .../05-final/scene-0103
$P tools/external_detector/render_waymo_sequence.py \
  --waymo-root data/nuscenes-stage-a --detector-frames ... --final-frames ... \
  --output-dir .../visuals/scene-0103 --expected-frames 40
```

（各工具实际参数名以 `--help` 为准，上面按读码结果给出，S3-S6 均为 Waymo 复现用过的同一入口。）

**S7 效果量化**（新，~100 行；**唯一读 GT 的环节，只出报告，不影响 S1-S6 任何输出框**）`eval_nus_vs_gt.py`：
- 以 S1 info 中的 `gt_boxes/gt_names` 为真值，按中心距离（0.5/1/2m）分别对 detector / tracking / final 三级框算各类 AP（nuScenes 风格 BEV，纯 numpy，不引新依赖）。
- 产出 `metric_report.json` + 一表：`detector → +tracking → +GRM/PRM` 的 mAP 增减，直接裁决 R1/R2。

### 3.3 明确不做

- 不接 `run_stage_a.py` 正式 coordinator、不生成 receipt/ledger/Release（其合同为 199 帧 Waymo/TFRecord/Open3D-ML 写死，改动成本 >> 收益；且 Waymo 侧 Release-C 本身未闭合）。
- 不动 `stage_a_source_policy.json`、不冻结源码身份。
- 不训练/微调任何模型；CRM 缺失维持 no-CRM 直通设计。
- 不下载 trainval（除非 S7 数字显示 mini 场景过稀，届时按 §2 第 4 条走加测）。

### 3.4 验收门

| 门 | 条件 |
| --- | --- |
| G1 链路 | 两 scene 各 40/41 帧完整走通 S1→S6，visuals 帧数=期望帧数，逐帧非空 |
| G2 一致性 | tracking 输出的 GT 关联轨迹肉眼抽查 3 帧与可视化一致；pose 对拍误差 <0.1m（R3） |
| G3 效果报告 | metric_report.json 三级 mAP 齐备；无论 refined 变好变差都如实记录状态（PASS / NOT_ESTABLISHED），禁止只报链路数字宣称效果 |
| G4 边界 | 不改 Waymo Stage-A 既有文件与 generation；新增文件全部在 tools/external_detector/ 与本 docs/ 下 |

预计工作量：脚本 3 个 ~350 行 + 串跑半天 + 指标半天 ≈ **1-2 个工作日**。

---

## 4. 与 Waymo Stage-A 报告的差异说明

本方案沿用其"外部 detector → adapter → tracking → GRM/PRM → no-CRM final → visuals"骨架与全部下游权重（Waymo 3 类），差别只在：detector 换成 HEDNet（自带速度、框格式直通）、数据源换成 nuScenes-mini（自带 GT）、以及**放弃其机械发布链**换取快速闭环——效果结论因此天然可建立（有 GT），这是 Waymo testing 段做不到的。

---

## 5. 修订路线（20260831-1730）：detector → 自研轻量跟踪 → 真值

### 5.1 为什么

原 S3-S5 复用 Waymo 训练的 tracking（10Hz 调参 + LEAST_AGE:5 保命门限）与 GRM/PRM 权重，GT 量化裁决 R1=DEGRADED（mAP 0.74→0.027，Ped/Cyc 整类被删），对照实验证明调 DELTA_T 无法翻案——是域差。因此：**跟踪要做，但必须用不依赖 Waymo 权重/参数/后处理的自研最小实现；GRM/PRM/CRM 全部砍掉。**

### 5.2 设计（纯 numpy/scipy，无任何学习组件）

新脚本 `tools/external_detector/track_hednet_boxes.py`（T1，~150 行），SORT 式恒速跟踪：

- **运动模型**：直接用 HEDNet 框自带的 vx/vy（lidar 系）做恒速外推 `x+vx·dt`，dt 取 S1 info 的真实时间戳差（2Hz≈0.5s），零拟合参数。
- **关联**：每类独立，代价 = BEV 中心距（外推后），匈牙利最优分配（scipy.linear_sum_assignment，已装），门限按物理尺寸放宽：Vehicle 2.5m / Cyclist 1.5m / Pedestrian 1.0m；超门槛视为新目标。
- **轨迹管理**：`max_age=2`（丢 2 帧断轨，容忍偶发漏检）、`min_hits=1`（不设 DetZero 式年龄保命门限——它正是 2Hz 下整类误删行人/骑车人的根因）。
- **真值平滑**（"跟踪优化"的全部替代）：每条轨迹的 (l,w,h) 取按 score 加权的**中值**（去抖），中心/朝向/速度保留逐帧原值；输出即最终真值框。
- **输出合同**：帧列表 schema 与 S2 detector 帧完全同构（可直接喂 S6 render 的 --final-frames 与 S7）；同时输出 tracking.pkl 同构 dict 供追溯。

### 5.3 步骤（复用既有件）

```
S1 data ──► S2 detector pkl ──► T1 track_hednet_boxes.py ──► track/hednet_tracked_frames_<scene>.pkl
                                                        └──► track/hednet_tracks_<scene>.pkl
S6 render(--detector-frames=S2, --final-frames=T1帧输出)   S7 eval(detector vs T1 帧输出)
```

验收门沿用 §3.4，改判据为 **G3'：T1 跟踪+中值平滑后的 mAP ≥ detector mAP − 0.02**（跟踪最多因删轨迹丢一点召回，不得引入位移/尺寸退化）；R2/R3 语义不变。

### 5.4 明确不做

GRM/PRM/CRM、Waymo tracking cfg/权重/DetZero post-process（motion_classify、static_drift_eliminate、track_merge）一概不碰；不改 S1/S2。
