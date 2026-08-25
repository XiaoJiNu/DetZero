# 单 Waymo 测试段与 CenterPoint 接入 DetZero 实现方案

> **For Hermes:** Use the subagent-driven-development skill to implement this plan task-by-task, completing TDD and review gates before advancing.

> **实施说明：** 本文是实施计划，不表示完整链路已经执行成功。实施时按阶段测试、失败关闭，并在每次正式运行前冻结输入、配置、代码和 checkpoint 哈希。

**目标：** 使用指定的 Waymo testing TFRecord 做真实点云推理，并把外部 CenterPoint 检测结果接入 DetZero tracking → GRM → PRM；不使用 CRM。

**推荐架构：** detector 始终在其原生代码和原生配置中运行，只在稳定的预测结果边界做一次显式适配。Waymo checkpoint 路线用于正式复现；nuScenes checkpoint 路线只用于跨域工程演示。

**技术栈：** Waymo Open Dataset SDK、OpenPCDet 或原始 CenterPoint、DetZero、PyTorch、CUDA、NumPy/Pickle 兼容输出和 JSON/NPZ 验收清单。

---

## 1. 整体简洁方案

### 1.1 结论

| 问题 | 结论 |
| --- | --- |
| 指定 TFRecord 能否作为推理数据 | **可以。** 它包含连续 199 帧、5 路 LiDAR、有效 pose 和严格递增时间戳，足以做单序列 detection → tracking → refining。 |
| 能否用它复现 Waymo AP/mAPH | **不可以。** 该 testing 文件没有本地 3D ground truth，只能做推理、结构验收和可视化。 |
| 用户提出的外部 detector → `result.pkl` → DetZero tracking 是否成立 | **成立。** DetZero tracking 接受 list 或嵌套 dict，只要字段、类别、坐标系、shape 和帧身份满足契约。 |
| 原始 CenterPoint + 获授权的官方 Waymo checkpoint | **推荐且可行。** 但 checkpoint 需要按 Waymo 条款取得，目前尚未提供。 |
| nuScenes-trained CenterPoint 用于该 Waymo 段 | **机械上可行，效果未证明。** 只能称为跨域 detector 工程演示，不能称为 DetZero/Waymo detector 复现。 |
| 把本机 nuScenes checkpoint 直接加载进 DetZero detector | **不可行。** 类别 head、输入特征、范围、voxel、velocity/IoU 分支均不一致。 |
| 不使用 CRM | **可以。** GRM 优化尺寸，PRM 优化中心和朝向，最终保留 tracking 输出的 detector-derived score；不能声称复现完整 CRM 置信度优化。 |

### 1.2 推荐路线

```text
推荐路线（正式复现）
指定 Waymo TFRecord
  → 流式解码为 Waymo 点云、pose、timestamp
  → 原始 CenterPoint/OpenPCDet 原生 Waymo 配置 + 获授权 Waymo checkpoint
  → 外部预测适配为 DetZero detection-result schema
  → DetZero tracking（split=test，无 GT）
  → object data preparation
  → Vehicle/Pedestrian/Cyclist 的 GRM + PRM
  → 不启用 CRM，保留 tracking 输出的 detector-derived score
  → 199 帧结构化结果和静态可视化

可立即尝试的候选路线（仅跨域演示）
指定 Waymo TFRecord
  → TOP LiDAR + 约 0.4 秒历史窗的 nuScenes-compatible 特征适配
  → 固定 OpenPCDet nuScenes checkpoint 原生前向
  → 显式类别/坐标/schema 转换
  → DetZero tracking → GRM → PRM
  → 输出标记 CROSS_DOMAIN_NUSCENES，禁止声明 Waymo 精度复现
```

### 1.3 对“方法 A”的修正

用户描述的“方法 A”包含两个不同前提，必须分开：

1. **原始 CenterPoint + 官方 Waymo checkpoint：** 路线正确，是首选。
2. **本机 nuScenes checkpoint：** 该文件是 OpenPCDet 格式，不是原始 CenterPoint 格式。它应在固定的 OpenPCDet 代码和 nuScenes 配置中运行；强行放进原始 CenterPoint 反而会重新引入 key/schema 不兼容。

因此，本机 nuScenes 权重的正确工程路线是：

```text
OpenPCDet 原生 nuScenes CenterPoint
  → Waymo-to-nuScenes-input adapter
  → 原生预测
  → DetZero-result adapter
```

而不是：

```text
原始 CenterPoint
  → 强行加载 OpenPCDet nuScenes checkpoint
```

### 1.4 结果应如何命名

若使用 nuScenes 权重，交付名称必须是：

> nuScenes 预训练 CenterPoint 在单个 Waymo testing segment 上的跨域检测，以及 DetZero tracking + GRM + PRM 工程链路演示。

不得称为：

- DetZero 官方 detector 复现；
- CenterPoint Waymo checkpoint 复现；
- DetZero 论文完整推理复现；
- Waymo AP/mAPH 复现。

---

## 2. 已完成的只读实测核验

### 2.1 指定 TFRecord

输入文件：

```text
/data/data/automomous/waymo/testing_0000/
segment-10084636266401282188_1120_000_1140_000_with_camera_labels.tfrecord
```

实测结果：

| 项目 | 实测值 |
| --- | --- |
| 文件类型 | regular file，非空 |
| 文件大小 | 1,053,628,548 bytes |
| SHA-256 | `84c960673cfbc55972392c606987e5eef9735321e65ba48adda9958e84140bc5` |
| TFRecord 记录数 | 199 |
| context name | `10084636266401282188_1120_000_1140_000` |
| 首帧 timestamp | `1558407840397346` µs |
| 末帧 timestamp | `1558407860197139` µs |
| 实际时间跨度 | 19,799,793 µs |
| 时间单调性 | 199 帧严格递增 |
| 每帧 LiDAR 数量 | 5 |
| 每帧相机数量 | 5 |
| pose | 199 个均为有限刚性 4×4；最大正交误差 `4.44e-16`，旋转行列式范围 `[0.9999999999999996, 1.0000000000000004]`，底行误差 0 |
| 199 帧 TOP LiDAR extrinsic | 全部数值一致，最大矩阵漂移 0；translation `[1.43, 0, 2.184]` m；raw sensor RPY 约 `[0.4466°, 0.0723°, 149.0304°]`，所以 raw TOP 轴不能直接当作 vehicle-forward model 轴 |
| `laser_labels` | 每帧 0 |
| `camera_labels` | 每帧 0 |
| TFRecord 容器闭合 | 精确 EOF 闭合；全部 record-length CRC32C 通过 |
| payload 抽检 | 第 0、99、198 条 payload CRC32C 通过 |

能力结论：

- **格式有效候选：是。**
- **可用于连续真实数据推理：是。**
- **可用于 tracking：是。**
- **可用于 GRM/PRM：是，前提是 detector 产生可跟踪目标。**
- **可用于本地 GT 评测：否。**
- **官方 checksum 认证：未证明。** 当前只有本地 SHA-256，没有取得 Waymo 针对该文件发布的权威 checksum manifest。

当前 `mv2d` 环境没有 TensorFlow 和 `waymo_open_dataset`。上述核验使用 TFRecord 与官方 protobuf 字段定义完成；实施时仍必须在隔离 Waymo SDK 环境中完成全部 199 帧 range-image 解压和点云有限性检查，才能把状态提升为“LiDAR payload 全量可解码”。

### 2.2 本机 nuScenes CenterPoint checkpoint

checkpoint：

```text
/data/models/pointfuse/centerpoint/
openpcdet-nuscenes-voxel0075-issue1704-reshare/
cbgs_voxel0075_centerpoint_nds_6648.pth
```

实测/已有绑定信息：

| 项目 | 值 |
| --- | --- |
| 文件大小 | 36,003,981 bytes |
| SHA-256 | `147486b6e078386ae884036757fc9469198e6c381a0e0013e9f3d0f72a1ca182` |
| checkpoint 顶层 | 仅 `model_state` |
| state entries | 558，全部 tensor |
| 原生代码候选 | OpenPCDet commit `8cacccec11db6f59bf6934600c9a175dae254806` |
| 原生配置 | `tools/cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint.yaml` |
| 配置结构覆盖 | 558/558 keys；537 原始 shape 匹配，21 个 spconv 布局适配 |
| 类别 | nuScenes 10 类、6 个 dense heads |
| 输入特征 | `x, y, z, intensity, timestamp` |
| sweep | 10 |
| point-cloud range | `[-54, -54, -5, 54, 54, 3]` |
| voxel size | `[0.075, 0.075, 0.2]` |
| GPU forward | 尚未验证 |
| 权重来源 | OpenPCDet issue #1704 社区参与者重分享 |
| 与已删除官方文件的字节同一性 | 未验证 |
| checkpoint 独立许可证/再分发权 | 未验证 |

source-frame 旁证（不是 checkpoint 权利或精度证明）：本机 nuScenes mini 元数据中 10 个 `LIDAR_TOP` calibration 的 sensor→ego yaw 范围为 `[-90.0311°, -89.8835°]`，roll 为 `[-2.64979°, -1.3884°]`，pitch 为 `[0.169199°, 0.338027°]`。这说明 OpenPCDet 源点帧近似 canonical RFU：`x=right, y=forward, z=up`，而不是 vehicle-forward。

重要结论：

- checkpoint 与固定 OpenPCDet nuScenes 配置在 key/shape 层面兼容。
- 这不等于它与 DetZero detector 配置兼容。
- 这也不证明 GPU 前向、nuScenes 官方回归或 Waymo 目标域效果。
- OpenPCDet 代码是 Apache-2.0，但代码许可证不能自动继承到 checkpoint。
- 本文件是社区重分享，发布或再分发前必须补齐 checkpoint 权利来源。

---

## 3. 可行性详细分析

### 3.1 为什么该 testing segment 可以用于推理

推理阶段需要的是：

- 连续 LiDAR 帧；
- 每帧 timestamp；
- 每帧 vehicle pose；
- 可确定的 sequence/frame identity。

该文件均具备，并且 199 帧接近完整 20 秒 Waymo segment。DetZero tracking 在 `split=test` 时关闭 GT target assignment；object preparation 也有独立的 test 分支。因此，缺少 GT 不会阻止机械推理。

缺少 GT 会阻止：

- detection AP/mAPH；
- tracking MOTA 等本地指标；
- GRM/PRM recall 提升的真实标签评估；
- 判断 nuScenes detector 在该段上的精度。

### 3.2 现有 DetZero 高层预处理不能直接使用

仓库当前存在两个 test-data 适配问题：

1. `detection/detzero_det/datasets/waymo/waymo_utils.py:188-208` 会把输入路径先去掉 `.tfrecord`，再追加 `_with_camera_labels.tfrecord`。若输入本身已经带该后缀，会得到：

   ```text
   ..._with_camera_labels_with_camera_labels.tfrecord
   ```

2. `detection/detzero_det/datasets/waymo/waymo_preprocess.py:101-115` 的 test 分支仍传 `has_label=True`，与底层函数对 test/new-sequence 应使用 `False` 的注释不一致。

所以不能直接把用户给出的绝对路径传给现有高层入口。实施时应增加一个受测的单段、test-only、流式预处理器，并调用已存在的单帧 range-image 解码逻辑。

### 3.3 DetZero tracking 的真实输入契约

用户给出的字段方向正确。推荐使用 frame-level list：

```python
[
    {
        "sequence_name": "10084636266401282188_1120_000_1140_000",
        "frame_id": 0,
        "pose": np.ndarray((4, 4), dtype=np.float64),
        "name": np.ndarray((N,), dtype=str),
        "score": np.ndarray((N,), dtype=np.float32),
        "boxes_lidar": np.ndarray((N, 9), dtype=np.float32),
    },
    # ... 共 199 帧
]
```

也可使用嵌套 dict，但必须是：

```text
{
  sequence_name: {
    frame_id_string: frame_record
  }
}
```

依据：

- `tracking/detzero_track/datasets/waymo_dataset.py:51-60` 接受 list 或 dict；
- `tracking/detzero_track/utils/data_utils.py:15-22` 将 list 按 `sequence_name` 和 `sample_idx/frame_id` 重组；
- `tracking/detzero_track/datasets/data_processor.py:85-95` 使用 pose 将 `boxes_lidar` 转到 global frame。

额外强制条件：

- `sequence_name` 必须对 199 帧完全一致；
- `frame_id` 必须唯一、连续且能转成整数；
- `pose` 必须是当前 Waymo vehicle frame 到 global frame 的 4×4 变换；
- `pose` 保持 float64，旋转块必须正交且行列式接近 `+1`；
- `boxes_lidar` 必须位于与 pose 对应的当前 vehicle frame；
- adapter 的 canonical box 为 `[x, y, z, dx, dy, dz, yaw, vx, vy]`；外部 detector 若只输出 Nx7，则显式补 `vx=vy=0`；
- `dx, dy, dz > 0`，全部数值有限；
- score 必须有限且位于 `[0, 1]`；
- name 只允许 `Vehicle/Pedestrian/Cyclist`；
- 空帧必须使用 `(0, 9)`、`(0,)`，不能使用 rank-1 空 box。

DetZero tracking 当前只消费 box 的前 7 维，并由 tracker 自己估计运动状态；`vx/vy` 是 detector-result canonical schema 的兼容字段，不会直接驱动该 tracker。

tracking 默认配置中的 overlap filter 会按类别查阈值；未映射的 nuScenes 类别会触发错误。因此类别转换必须在进入 tracking 前完成。

### 3.4 为什么 nuScenes checkpoint 不能直接加载到 DetZero detector

| 契约 | 本机 nuScenes checkpoint | DetZero Waymo CenterPoint 1-sweep |
| --- | --- | --- |
| 类别 | 10 类 | Vehicle/Pedestrian/Cyclist 3 类 |
| dense head | 6 个多类别 heads | 1 个 3 类 head |
| 回归 head | center/z/dim/rot/**vel** | center/z/dim/rot/**iou** |
| 输入第 5 维 | timestamp | elongation |
| sweep | 10 | 1 |
| point range | `[-54,54] × [-54,54] × [-5,3]` | `[-75.2,75.2] × [-75.2,75.2] × [-2,4]` |
| voxel | `[0.075,0.075,0.2]` | `[0.1,0.1,0.15]` |
| score threshold | 0.1 | 0.03 |

依据：

- nuScenes config：固定 OpenPCDet `cbgs_voxel0075_res3d_centerpoint.yaml`；
- DetZero config：`detection/tools/cfgs/det_model_cfgs/centerpoint_1sweep.yaml:1-85`；
- DetZero dataset：`detection/tools/cfgs/det_dataset_cfgs/waymo_1sweep.yaml:5-82`。

`strict=False` 或忽略 shape mismatch 只会留下随机/未加载的分类或回归 head，不能算真实 detector 推理。

### 3.5 原始 CenterPoint + Waymo checkpoint 是否可行

**架构上可行，也是更诚实的方案。** 原始 CenterPoint 仓库提供 Waymo 配置、测试集转换和预测输出。其 Waymo model zoo 明确要求注册确认和非商业用途后联系作者获取权重，因此该 checkpoint 不是匿名公共下载资产。

需注意：

- 原始仓库文档绑定旧 PyTorch/CUDA/TensorFlow 组合；当前 RTX 5090 环境不能假定原样可运行。
- 必须让 checkpoint、配置、代码 revision 三者闭合；不能只凭文件名选配置。
- 若原始运行时太旧，可在隔离容器/兼容 GPU 中运行 detector，再把预测结果转给当前 DetZero；适配边界仍然有效。

### 3.6 nuScenes checkpoint 路线是否可行

分四个独立 verdict：

| Verdict | 当前状态 |
| --- | --- |
| Schema-compatible | **有条件可行。** 可把输出完整表达成 DetZero tracking schema。 |
| Mechanically executable | **候选。** checkpoint/config key-shape 已闭合，但尚未完成真实 GPU forward。 |
| Target-domain efficacy | **未证明。** test split 无 GT，不能证明 Waymo 精度。 |
| Release/rights eligible | **阻塞。** 社区重分享 checkpoint 的许可证和原始字节身份未闭合。 |

主要域差异：

- nuScenes 与 Waymo 的 LiDAR 型号、安装、点密度和 intensity 统计不同；
- nuScenes 模型训练使用约 0.5 秒内的 10 sweeps，而 Waymo Perception frames 为 10 Hz；直接取 9 个 Waymo 历史帧会把时窗扩大到约 0.9 秒；
- 类别定义不同，尤其 `bicycle/motorcycle → Cyclist` 不是严格语义等价；
- score 未在 Waymo 上校准；
- nuScenes 只覆盖约 108 m × 108 m 范围，DetZero Waymo 原配置更大；
- voxel 和 z range 不同；
- testing split 无标签，不能用结果反向调阈值后声称性能提高。

### 3.7 不使用 CRM 的影响

无 CRM 的最终数据来源为：

```text
尺寸 dx/dy/dz       ← GRM
中心 x/y/z、yaw     ← PRM
score               ← tracking 输出的 detector-derived score，经 PRM 原样传递
```

代码依据：

- `refining/detzero_refine/datasets/waymo/waymo_position_dataset.py:231-253` 将输入 `pos_scores` 写回 position 输出；
- `tracking/detzero_track/models/tracking_modules/kalman_filter/kalman_filter.py:61-69,85-118` 显示匹配帧更新为当前 detector score；纯预测帧不改 score，因而沿用最近一次匹配分数；
- `daemon/combine_output.py:126-131` 组合 GRM 尺寸与 PRM 位置；只有启用 `combine_conf_res` 才替换 score；
- CLI 中 `--combine_conf_res` 默认关闭。

因此，无 CRM 不阻止 GRM/PRM 推理，但结果名称必须包含 `no_crm` 或 `score_passthrough`。

---

## 4. 详细实现方案

### 4.1 定义两个产品模式

实施一个入口，支持两个互斥模式：

| 模式 | detector | 允许的声明 |
| --- | --- | --- |
| `waymo-native` | 获授权 Waymo-trained CenterPoint | 单段 Waymo detector + DetZero GRM/PRM 推理复现；仍无 test GT 指标 |
| `nuscenes-cross-domain` | 当前 nuScenes-trained OpenPCDet checkpoint | 跨域工程链路演示；禁止 Waymo 精度和 DetZero detector 复现声明 |

`nuscenes-cross-domain` 必须要求显式开关，例如 `--ack-cross-domain`，避免误用默认模式。

### 4.2 使用独立、不可覆盖的 run root

不要把文件软链到已有 `data/waymo`，也不要覆盖 `output/inference_reproduction_v3`。建议：

```text
output/waymo_external_centerpoint_<UTC-or-CST-timestamp>/
├── source_manifest.json
├── data/
│   └── waymo/
│       ├── ImageSets/test.txt
│       ├── waymo_infos_test.pkl
│       ├── waymo_processed_data/
│       │   └── segment-10084636266401282188_1120_000_1140_000/
│       │       ├── 0000.npy
│       │       ├── ...
│       │       ├── 0198.npy
│       │       └── segment_info.pkl
│       └── waymo_detector_top_points/
│           └── segment-10084636266401282188_1120_000_1140_000/
│               └── 0000.npy ... 0198.npy
├── detection/
│   ├── raw_predictions.npz
│   ├── prediction_manifest.json
│   └── detzero_result.pkl
├── tracking/
│   ├── tracking_test.pkl
│   └── tracking_manifest.json
├── refining/
│   ├── Vehicle/
│   ├── Pedestrian/
│   ├── Cyclist/
│   └── result/
├── final/
│   ├── final_frame.pkl
│   ├── final_arrays.npz
│   └── final_manifest.json
└── visualization/
    ├── frames/0000.png ... 0198.png
    └── visualization_manifest.json
```

要求：目标目录必须不存在；使用随机 staging 目录、完整验收、无覆盖原子发布。

### 4.3 环境隔离

当前 `mv2d` 是 Python 3.10.20、PyTorch 2.7.1+cu128，没有 TensorFlow/Waymo SDK。仓库安装文档要求 Python 3.8 和 `waymo-open-dataset-tf-2-5-0`。

建议拆成两个环境：

1. **CPU 预处理环境** `detzero-waymo-tf25`：Python 3.8、TensorFlow 2.5、Waymo SDK；固定 `CUDA_VISIBLE_DEVICES=-1`。
2. **GPU detector/refiner 环境**：继续使用 `/data/software/conda/anaconda3/envs/mv2d`，但先补齐并验证固定 OpenPCDet runtime，不在预处理环境中运行 CUDA。

不得把旧 TensorFlow 直接安装进 `mv2d`，避免破坏已验证的 GRM/PRM 环境。

### 4.4 阶段 0：冻结来源和权利状态

正式运行前必须先在内存中构造 expected `source_manifest.json`，一次写入后立即重读校验；不能在运行结束后根据“实际用了什么”回填。它至少冻结：

- TFRecord path、size、SHA-256；
- 每个 checkpoint 的 canonical path、size、SHA-256、类别和阶段；
- OpenPCDet/CenterPoint revision；
- detector、tracking、refining 和 adapter config 的 canonical path 与 SHA-256；
- 递归 `_BASE_CONFIG_` 有向图的精确 path set、每文件 SHA-256 和 aggregate root；
- CLI argv、`--set` overrides，以及应用 base/override 后 canonical resolved config 的 SHA-256；
- Python、PyTorch、CUDA、spconv 版本；
- Waymo SDK 版本；
- 用户对 Waymo 和 nuScenes 非商业条款的确认；
- checkpoint 权利状态。

当前候选 tracking/refining config closure 的权威字节为：

| 角色 | 路径 | SHA-256 |
| --- | --- | --- |
| tracking entry | `tracking/tools/cfgs/tk_model_cfgs/waymo_detzero_track.yaml` | `835d0f4cee41ed64202aeb34e0eed4fb51ad09e0868f8a36f758b86912d53926` |
| tracking base | `tracking/tools/cfgs/tk_dataset_cfgs/waymo_dataset.yaml` | `c195de54aa4d25a8a07149d39e76b786ff5fd4f16a70f228a5b291b55cbabee8` |
| GRM dataset base | `refining/tools/cfgs/ref_dataset_cfgs/waymo_grm_dataset.yaml` | `f222f445071ae005f89a37a5f05ef859b889dddb23604c91537e19c5b3e2d87c` |
| PRM dataset base | `refining/tools/cfgs/ref_dataset_cfgs/waymo_prm_dataset.yaml` | `9fbf7e4562fae0556c734bcb2bc9dc329213d2a0c30fba9de46b74ffdbc44213` |

OpenPCDet entry config/checkpoint 的固定映射见 4.7；六个 refining class/stage 映射见 4.11。manifest builder 必须递归解析 base graph，拒绝环、root 逃逸、missing/extra node、重复角色或任一 hash 漂移，并在任何 dataset/model 构建前完成校验。每个阶段 receipt 必须绑定它使用的 closure root、effective-config hash 和 parent artifact hash；运行结束及发布前再次重算同一闭包，发现 TOCTOU 漂移时整次运行失败。

若使用当前社区重分享权重，机器清单必须保留：

```text
checkpoint_provenance_verified = false
release_eligible = false
```

不得因本地推理成功自动改成 true。

### 4.5 阶段 1：流式解码单个 TFRecord

拟新增：

```text
tools/external_centerpoint/preprocess_waymo_test_segment.py
```

实现要求：

1. 只接受一个 regular、non-symlink TFRecord；
2. 校验输入 basename 与每帧 `context.name` 一致；
3. 逐条迭代，不能把 199 个大 Frame 全部留在内存；
4. 每帧调用仓库现有 range-image → point-cloud 算法；
5. `has_label=False`，不得生成可被误认为 GT 的空 annotation；
6. 为后续 GRM/PRM 保存全部 5 路 LiDAR 的 `[x, y, z, intensity, elongation, NLZ]` float32；
7. 另外按 `LaserName.TOP` 显式保存 detector 用 TOP LiDAR 点，不能依赖“拼接后恰好排第一”的隐含顺序；
8. 保存 TOP LiDAR 到 vehicle 的 raw float64 extrinsic、后述 virtual model frame，以及 int64 `time_stamp`、`sample_idx`、`sequence_name`、float64 `pose`、`sequence_len`；
9. 每个 `.npy` 使用 `allow_pickle=False` 可读的纯数值数组；
10. 验证 all-LiDAR 和 TOP-LiDAR 点数非零、shape 为 `(N, 6)`、全部有限；
11. 验证 199 个 frame ID 与 `0..198` 精确相等；
12. 在发布前重读全部数组并生成 closed-world manifest。

不要调用会重复追加 `_with_camera_labels` 的现有高层路径逻辑。

### 4.6 阶段 2：构建 nuScenes-compatible TOP-centered、source-axis-aligned 历史输入

此阶段仅用于 `nuscenes-cross-domain`。

这不是“nuScenes-native 输入”：传感器、采样频率和目标域仍然是 Waymo。目标只是冻结一个比“5 路 LiDAR、vehicle 原点、vehicle-forward 轴、0.9 秒历史”更接近 nuScenes checkpoint 源契约的兼容 profile。

#### 模型坐标约定

定义：

```text
T_G<-V_i：第 i 帧 Waymo vehicle frame 到 global frame 的 pose
t_TOP_i：第 i 帧 raw TOP LiDAR extrinsic 的 vehicle-frame translation
M_i：以 t_TOP_i 为原点、轴采用 nuScenes source canonical RFU 的虚拟 model frame
R_V<-M = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
T_V<-M_i = [R_V<-M, t_TOP_i; 0, 1]
c：当前帧
i：历史帧
```

全 199 帧 Waymo wire-level 标定实测 raw TOP extrinsic 数值完全一致，yaw 约为 `149.0304°`；不能直接采用 raw Waymo sensor axes。另一方面，本机 nuScenes mini 的 10 个 `LIDAR_TOP` calibration sensor→ego yaw 均约为 `-90°`，说明 checkpoint 源点帧近似 RFU。Waymo 解码后的 TOP 点已经位于各自 vehicle frame，本 profile 因此只使用 Waymo raw extrinsic 的传感器中心 translation，轴则冻结为 source-derived canonical RFU。

将历史点变换到当前虚拟 model frame：

```text
T_Mc<-Vi = inverse(T_V<-M_c) @ inverse(T_G<-V_c) @ T_G<-V_i
p_Mc = T_Mc<-Vi @ p_Vi
```

detector 输出在当前虚拟 model frame；adapter 通过 `T_V<-M_c` 把 box center、heading direction 和 velocity 转回当前 vehicle frame。yaw 必须通过旋转后的 heading unit vector 重新计算，不能只硬编码加减 `π/2`。raw Waymo TOP rotation 只作为标定证据保存，不参与该虚拟 frame 的坐标变换。nuScenes 标定中不足 3° 的 roll/pitch 没有复制到 canonical RFU，这一残差必须保留为跨域限制。

实施时验证 199 帧 TOP calibration identity/刚性及 translation 稳定性，并用非单位、非对称 pose、非零 TOP-center translation 和 canonical RFU 旋转 golden test 验证两个方向；不能只测试 identity pose。

#### 历史窗规则

官方能力事实：Waymo Perception segments 为 10 Hz；nuScenes detection 允许约 0.5 秒、最多 10 个过去 LiDAR sweeps。固定本 profile 为：

- 当前帧 timestamp feature 为 0；
- 最多取前 4 个 Waymo frames，只使用 `0 < time_lag <= 0.45 s` 的历史，不使用未来帧；
- 不复制同一 Waymo frame 来伪造“10 个独立 sweeps”；OpenPCDet 接收的是拼接后的 `N×5` 点，不要求 checkpoint tensor 具有固定 sweep 维；
- 对每个历史 frame，先变换到其自身 virtual model frame，并按 OpenPCDet 原生规则删除 `|x| < 1 m 且 |y| < 1 m` 的 ego-near 点，再补偿到当前 frame；当前 frame 不应用这条历史-sweep 过滤；
- 时间特征采用 OpenPCDet nuScenes 原生定义：

  ```text
  time_lag_seconds = (current_timestamp - historical_timestamp) / 1e6
  ```

  对历史帧为非负值；
- 固定 newest-to-oldest 顺序，禁止随机选择历史帧；
- 前 4 个 warm-up frames 只使用实际可用历史；
- 记录每帧 unique frame count、最大 time lag、ego-near 删除数、拼接前后点数和 voxel cap 是否触发。

不能直接复用 DetZero `merge_sweeps` 的 timestamp：`detection/detzero_det/datasets/dataset.py:185-190` 使用 `target_time - current_time`，对历史帧是负数；OpenPCDet nuScenes 在 `nuscenes_utils.py:419-427` 使用 `reference_time - sweep_time`，符号相反。

#### 点特征

只使用 TOP LiDAR 点，输入必须重新构造为：

```text
[x, y, z, intensity, timestamp]
```

注意：Waymo 保存数组第 5 维是 `elongation`，不能误当 `timestamp`。

默认策略：

- 过滤 `NLZ != -1`；
- 使用 Waymo 原始 intensity，保留 source/target 统计报告；
- 不默认使用 `tanh(intensity)`，因为那会成为额外模型输入变更；
- 若后续选择 intensity 映射，必须作为单独、冻结、明确标记的实验 profile，不得根据无标签预测好坏临时选择。

全部 5 路 LiDAR 点仍保留给 DetZero object preparation 和 GRM/PRM；TOP-only 限制只作用于 nuScenes detector 输入。

range cropping、voxelization、voxel cap 使用固定 nuScenes config 完成。

### 4.7 阶段 3：在原生 OpenPCDet 中运行 detector

拟新增：

```text
tools/external_centerpoint/run_openpcdet_waymo_segment.py
```

本机 nuScenes 模式固定：

```text
OpenPCDet source root:
/data/code/location/pointFuse/output/vendor/OpenPCDet-8cacccec11db6f59bf6934600c9a175dae254806

OpenPCDet commit:
8cacccec11db6f59bf6934600c9a175dae254806

config:
tools/cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint.yaml

config SHA-256:
bb8d6575f9a96cc935c4b5a428557f9210434ab2f8f775fe85720d1e2069d7ab

raw _BASE_CONFIG_ value:
cfgs/dataset_configs/nuscenes_dataset.yaml

base config canonical source-relative path:
tools/cfgs/dataset_configs/nuscenes_dataset.yaml

base config SHA-256:
a9a561d3da8a05e0537da55c15817b19dd5ff60c968a562d9475bc71094e1036

recursive config closure file count:
2

recursive config closure root:
19986621790f8b7cfaeddad38cbbdc9e5227434a1c753c047b9f0c1e03a0e121

checkpoint SHA-256:
147486b6e078386ae884036757fc9469198e6c381a0e0013e9f3d0f72a1ca182
```

上述 closure root 的算法固定为：以 `<OpenPCDet source root>/tools` 为基准，将精确的两项 path set 按 POSIX 相对路径升序排列，把每项序列化成 UTF-8 `<relative_path>\t<lowercase_sha256>\n`，再对连接后的字节计算 SHA-256。当前 base 文件没有进一步的 `_BASE_CONFIG_`；expected path set 必须精确等于 `{cfgs/dataset_configs/nuscenes_dataset.yaml, cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint.yaml}`。manifest builder 必须先根据文档内置的 expected path/hash/root 校验这两个普通文件，拒绝 missing、extra、symlink、root 逃逸或 hash 不符，然后才允许 config loader 解析和合并；不得从当前磁盘递归发现结果反向生成 expected 值。

runner 必须从上述绝对 source root 导入 OpenPCDet，并把 config 的解析后路径限制在该 root 内。固定 vendor 的 `pcdet/config.py:51-78` 同样按进程 CWD 打开 `_BASE_CONFIG_`；因此 detector worker 必须使用 `<OpenPCDet source root>/tools` 作为 CWD，并以 `cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint.yaml` 载入配置。不得使用本机另一个同名工作区 `/data/code/cv/BEV/Lidar_AI_Solution/CUDA-PointPillars/OpenPCDet`：其同名 config 当前 SHA-256 为 `82da337d57b109babb3bae0e081620ed134037c579d3c5f0de6838aa22be4467`，与兼容性 receipt 固定的 `bb8d...` 不同。任何 source root、revision、worker CWD、entry/base config hash、closure path set/root 或 canonical resolved-config hash 漂移都必须在 dataset/model 构建前失败。

执行要求：

- checkpoint 只用 `torch.load(..., weights_only=True)` 读取；
- 经 OpenPCDet spconv layout adapter 后要求 558/558 keys 闭合；
- 不允许 missing/extra model key；
- `model.eval()` + `torch.inference_mode()`；
- batch size 1；
- 固定 Python/NumPy/PyTorch seed；
- 先执行 1 帧 GPU canary，再执行连续 10 帧，最后才执行 199 帧；
- 每帧保存原始 post-NMS `pred_boxes/pred_scores/pred_labels`；
- 原始结果写入无 pickle 的 NPZ + JSON manifest；
- 记录输入点数、range-filter 后点数、voxel 数、输出框数和各类别计数；
- 若出现 NaN、非法维度、CUDA error、checkpoint drift 或缺帧，整次运行失败。

checkpoint 当前只有 CPU 构建/key-shape 证据，没有真实 GPU forward。因此“1 帧 canary 通过”是继续 full segment 的硬门槛。

### 4.8 阶段 4：转换为 DetZero detection result

拟新增：

```text
tools/external_centerpoint/convert_openpcdet_to_detzero.py
```

#### 类别映射

采用固定映射，不读取 Waymo label/category 字段：

| nuScenes 输出 | DetZero 输出 | 说明 |
| --- | --- | --- |
| `car/truck/construction_vehicle/bus/trailer` | `Vehicle` | 跨域合并，非 Waymo GT 等价证明 |
| `pedestrian` | `Pedestrian` | 名称对应，但域仍不同 |
| `bicycle/motorcycle` | `Cyclist` | 语义较弱，Waymo Cyclist 往往是 rider-centric |
| `barrier/traffic_cone` | 丢弃 | DetZero tracking 无对应类别 |

该映射必须写入 manifest 并哈希绑定。

#### 坐标与框

detector 输出首先位于当前 TOP-centered、source-axis-aligned RFU 虚拟 model frame；adapter 必须先用同帧 `T_V<-M` 刚体变换到当前 vehicle frame，再生成 DetZero record。适配器必须验证：

- `boxes_lidar = [x,y,z,dx,dy,dz,yaw,vx,vy]`；
- 若原生输出只有 7 维，则 canonicalization 明确追加两个 float32 零值，而不是假造 detector velocity；
- center 使用完整 `R_V<-M @ center_M + t_TOP`；
- yaw 由 `R_V<-M @ [cos(yaw_M), sin(yaw_M), 0]` 后的 vehicle XY direction 重新计算；
- 非零 velocity 使用 `R_V<-M @ [vx_M, vy_M, 0]` 旋转；
- center-z 语义为 box center；
- yaw 单位为 rad，并归一到 `(-π, π]`；
- velocity 在当前 vehicle XY frame；
- pose 是同一帧 `T_G<-V`；
- 通过 box center/yaw/velocity 的非单位变换 golden test。

#### 分数

- 保留 OpenPCDet post-NMS score；
- 不使用 Waymo testing 输出调参；
- 默认保留原生 0.1 score threshold；
- DetZero tracking 的第二阶段同样有 0.1 threshold；两个 gate 必须分别记录，不能混成一个“最终阈值”。

#### 输出

生成：

```text
detection/detzero_result.pkl        # DetZero 兼容，仅由本地可信 producer 生成
detection/detzero_result_arrays.npz # 安全验收主数组
detection/adapter_manifest.json     # 身份、mapping、shape、hash、计数
```

`result.pkl` 必须包含 199 个 frame records，包括零检测帧。验证器以 NPZ/JSON 为安全依据，再逐字段对照本地生成的 pickle；不得把外部下载 pickle 直接交给 DetZero。

### 4.9 阶段 5：运行 DetZero tracking

拟修改：

```text
tracking/tools/run_track.py
```

增加显式：

- `--root_path`；
- `--output_path`；
- no-replace 输出；
- 输入 schema preflight。

`tracking/detzero_track/datasets.build_dataloader` 已支持 `root_path`，当前 CLI 只是没有传入。

`utils/detzero_utils/config_utils.py:59-69` 当前按进程 CWD 打开 `_BASE_CONFIG_`，而不是按外层 YAML 所在目录解析。所以下列命令必须把 tracking 的工作目录固定为 `tracking/tools`；wrapper 还必须先把 `RUN_ROOT` 规范化为绝对路径。

实施后的目标命令：

```bash
(
  cd /data/code/cv/AutoLabel/DetZero/tracking/tools
  /data/software/conda/anaconda3/envs/mv2d/bin/python \
    run_track.py \
    --cfg_file cfgs/tk_model_cfgs/waymo_detzero_track.yaml \
    --data_path "$RUN_ROOT/detection/detzero_result.pkl" \
    --root_path "$RUN_ROOT/data/waymo" \
    --output_path "$RUN_ROOT/tracking/tracking_test.pkl" \
    --split test \
    --batch_size 1 \
    --workers 1
)
```

关键点：

- 必须传 `--split test`；`split=val` 会尝试加载不存在的 GT；
- tracking 是 Kalman filter、关联和后处理，不加载学习 checkpoint；
- 当前 `points_in_box` processor 被注释，tracking 本身不要求读取点云；
- tracking 输出必须保留 sequence、object ID、frame ID、pose、name、score、global box 和 state；
- 输出 frame coverage 和 track frame IDs 必须是输入 frame IDs 的子集，不能出现外部帧。

### 4.10 阶段 6：生成 refining object data

拟修改：

```text
daemon/prepare_object_data.py
```

增加 `--root_path` 和 `--output_root`，移除对仓库固定 `data/waymo` 的依赖。

实施后的目标命令：

```bash
/data/software/conda/anaconda3/envs/mv2d/bin/python \
  daemon/prepare_object_data.py \
  --track_data_path "$RUN_ROOT/tracking/tracking_test.pkl" \
  --root_path "$RUN_ROOT/data/waymo" \
  --output_root "$RUN_ROOT/data/waymo/refining" \
  --split test \
  --workers 1
```

该阶段会：

- 按 Vehicle/Pedestrian/Cyclist 分开；
- 将 vehicle-frame 点通过 pose 变换到 global frame；
- 围绕 track box 裁剪 object points；
- 生成 GRM/PRM 所需 object-level `.pkl`。

必须验证裁剪后的每个 object/frame 点云有限、与 box frame 一致，并记录空轨迹/空类别。

### 4.11 阶段 7：运行六个 GRM/PRM checkpoint

拟修改：

```text
refining/tools/test.py
```

增加显式 `--root_path` 和 `--result_dir`；同时将 test split 的 recall 指标关闭，避免把零占位 GT 当成评测。

这里同样必须把进程 CWD 固定为 `refining/tools`，使六个模型 YAML 中的 `cfgs/ref_dataset_cfgs/...` base path 可解析；checkpoint、run root 和 result dir 均使用绝对路径。

六个模型必须由下面的 closed-world 映射选择，禁止只按 `${cls}_${stage}` 文件名拼接后直接信任磁盘内容：

| Class | Stage | model config SHA-256 | base config SHA-256 | checkpoint SHA-256 |
| --- | --- | --- | --- | --- |
| Vehicle | GRM | `1c5f43e2ddbd74b5db7fee2ee1656718f1da0c08ba130111db657b8222ca8904` | `f222f445071ae005f89a37a5f05ef859b889dddb23604c91537e19c5b3e2d87c` | `05f3b293d70f9b6a24486a9cfb93d9cb5a053f3987f65ffe11bc6ba44d2ba599` |
| Vehicle | PRM | `dd86fd0ef7d926bf248b05abe495f4b8fe5fc0190462c38a1b0699511eb426c8` | `9fbf7e4562fae0556c734bcb2bc9dc329213d2a0c30fba9de46b74ffdbc44213` | `32c757c1900bfe263a8b3786b91575092c35770dfc3b0b000a4b3636c0204fe5` |
| Pedestrian | GRM | `24c5732661cff342c330234fd663a282fb12cb64fe7fecef569c0c0bc3eb36b8` | `f222f445071ae005f89a37a5f05ef859b889dddb23604c91537e19c5b3e2d87c` | `f47cf8f5e9516a12a80fb77133317bf1c1ed8a570fb7025f8ba92ddb9789d385` |
| Pedestrian | PRM | `2f1efa1ab25ac22a47d56c8d1e26bd3affe84a0fa75e20388c9f9ce4b9de642f` | `9fbf7e4562fae0556c734bcb2bc9dc329213d2a0c30fba9de46b74ffdbc44213` | `31d2382f50cb02e786a3bd878a001ce984d48cbcbed829edbea427174da9a69f` |
| Cyclist | GRM | `17ea37b9bbe40411b848a037f74845e7b6e64e1c680843f2bf22d1df18add5a9` | `f222f445071ae005f89a37a5f05ef859b889dddb23604c91537e19c5b3e2d87c` | `92c101f98340d4f5bcf746f47cf5290bab734e6ec846f1404fcde47c86139990` |
| Cyclist | PRM | `1cc5f244e05678e4599719f973da887d4b2d41e6f736ec7a5e83f78d9110ef64` | `9fbf7e4562fae0556c734bcb2bc9dc329213d2a0c30fba9de46b74ffdbc44213` | `95f2bde035f1faddff62c24adc787d97ddf64f73251b05bba5da3cbc9b6995b0` |

`source_manifest.refining_models` 必须精确包含这六个 `(class, stage)` key 及各自 canonical config/base/checkpoint path 和上述 expected hash；missing、extra、角色互换或把某类 checkpoint 配给另一类都必须在构建模型前失败。`docs/INFERENCE_ASSET_MANIFEST.json` 可作为官方下载同一性的来源证据，但运行时不能信任其可变路径或布尔值，必须把 expected digest 复制进本次不可变 manifest 后直接重算普通文件。

实施后的目标命令模式：

```bash
(
  cd /data/code/cv/AutoLabel/DetZero/refining/tools
  for cls in vehicle pedestrian cyclist; do
    for stage in grm prm; do
      /data/software/conda/anaconda3/envs/mv2d/bin/python \
        test.py \
        --cfg_file "cfgs/ref_model_cfgs/${cls}_${stage}_model.yaml" \
        --ckpt "/data/code/cv/AutoLabel/DetZero/checkpoints/${cls}_${stage}_model.pth" \
        --root_path "$RUN_ROOT/data/waymo" \
        --result_dir "$RUN_ROOT/refining/result" \
        --save_to_file \
        --batch_size 1 \
        --workers 1 \
        --set \
          DATA_CONFIG.DATA_SPLIT.test test \
          MODEL.POST_PROCESSING.GENERATE_RECALL False
    done
  done
)
```

每个非空类别必须证明：

- checkpoint SHA-256 与官方公开副本一致；
- strict key/shape load；
- requested CUDA device；
- eval/inference mode；
- 至少一个真实 forward；
- 输出 frame/track identity 与输入闭合；
- 输出 box 全部有限且尺寸为正。

空类别处理必须失败关闭：

- 不生成假轨迹；
- 记录 `NOT_EXECUTED_NO_INPUT`；
- 生成显式空结果；
- 若验收目标要求“六个 checkpoint 都在真实 Waymo 目标上前向”，则该 run 不通过，需选择另一个有三类预测的 segment，而不能降低标准或伪造输入。

### 4.12 阶段 8：无 CRM 组合

不传 `--combine_conf_res`。建议新增一个名称明确、单次处理三类的薄封装：

```text
tools/external_centerpoint/combine_grm_prm_no_crm.py
```

原因：现有 `daemon/combine_output.py` 在类别循环外复用 `combine_dict`，并分别写以类别命名的累积文件；最终哪个文件包含三类不够直观。新封装仍复用相同组合语义，但输出一个明确文件：

```text
final/final_frame_grm_prm_score_passthrough.pkl
```

组合规则：

```text
final_box[:, 0:3]  = PRM center
final_box[:, 3:6]  = GRM size
final_box[:, 6]    = PRM yaw
final_score        = PRM carried tracking score
```

验证每个 final row 都能追溯到唯一 `(sequence_name, frame_id, object_id)`，并分别绑定 detector、tracking、GRM 和 PRM parent hash。

### 4.13 阶段 9：结构化结果与静态可视化

用户可直接查看的主交付应为 199 张静态图，而不是只给 HTML：

```text
visualization/frames/0000.png
...
visualization/frames/0198.png
```

每帧至少包含：

- Waymo LiDAR BEV；
- external detector boxes；
- DetZero tracking IDs；
- GRM/PRM final boxes；
- 类别、score 和 frame ID；
- 明显标记 `NO GT`、`NO CRM`；
- nuScenes 模式额外标记 `CROSS-DOMAIN / NOT WAYMO-VALIDATED`。

验收：

- 预期图片集合精确等于 `0000..0198`；
- 每张图可解码、尺寸固定、非空白；
- 图中 box 必须从 final NPZ 独立重渲染并做确定性字节比较；
- 发布 JSON manifest，记录 frame ID、图片路径、尺寸、hash、显示框数；
- 不用代表性帧替代全部 199 帧。

### 4.14 阶段 10：一键入口

拟新增：

```text
reproduce_waymo_external_centerpoint.sh
```

目标接口：

```bash
cd /data/code/cv/AutoLabel/DetZero

./reproduce_waymo_external_centerpoint.sh \
  --mode nuscenes-cross-domain \
  --ack-cross-domain \
  --openpcdet-root \
    /data/code/location/pointFuse/output/vendor/OpenPCDet-8cacccec11db6f59bf6934600c9a175dae254806 \
  --detector-config \
    tools/cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint.yaml \
  --input-tfrecord \
    /data/data/automomous/waymo/testing_0000/segment-10084636266401282188_1120_000_1140_000_with_camera_labels.tfrecord \
  --detector-checkpoint \
    /data/models/pointfuse/centerpoint/openpcdet-nuscenes-voxel0075-issue1704-reshare/cbgs_voxel0075_centerpoint_nds_6648.pth \
  --output-dir output/<必须不存在的新目录>
```

wrapper 应依次执行：

1. rights/source preflight；
2. TFRecord 全量点云解码；
3. 1-frame detector canary；
4. 10-frame canary；
5. 199-frame detector；
6. adapter；
7. tracking；
8. object preparation；
9. GRM/PRM；
10. no-CRM combination；
11. 199-frame visualization；
12. post-publication validator。

wrapper 必须先把 repo root、run root、checkpoint、TFRecord 和 OpenPCDet root 解析为受约束的绝对路径；启动 OpenPCDet/tracking/refining 时分别设置为 `<vendor>/tools`、`tracking/tools`、`refining/tools` 的显式 subprocess `cwd`，不得依赖调用者当前目录。

任一阶段失败，不得继续发布为 accepted result。

---

## 5. 拟新增或修改的文件

| 类型 | 路径 | 目的 |
| --- | --- | --- |
| Create | `tools/external_centerpoint/preprocess_waymo_test_segment.py` | 单 TFRecord、无标签、流式 Waymo 解码 |
| Create | `tools/external_centerpoint/run_openpcdet_waymo_segment.py` | 在 checkpoint 原生 OpenPCDet config 中执行 detector |
| Create | `tools/external_centerpoint/convert_openpcdet_to_detzero.py` | 类别、坐标、schema 转换 |
| Create | `tools/external_centerpoint/combine_grm_prm_no_crm.py` | 三类 GRM/PRM 明确组合，score passthrough |
| Create | `tools/external_centerpoint/validate_waymo_external_run.py` | 闭世界、跨阶段身份、数组、图像和 determinism 验收 |
| Create | `tools/external_centerpoint/render_waymo_sequence.py` | 199 帧静态 PNG |
| Create | `reproduce_waymo_external_centerpoint.sh` | 一键入口 |
| Modify | `tracking/tools/run_track.py` | 增加 `root_path/output_path` 和 schema preflight |
| Modify | `daemon/prepare_object_data.py` | 移除固定数据根，增加输出根 |
| Modify | `refining/tools/test.py` | 增加 `root_path/result_dir` 和真实 test 模式 |
| Test | `tests/test_waymo_external_preprocess.py` | TFRecord、命名、无标签、流式解码契约 |
| Test | `tests/test_openpcdet_detzero_adapter.py` | class/box/yaw/velocity/score/schema 转换 |
| Test | `tests/test_waymo_external_pipeline.py` | tracking → GRM/PRM → no-CRM 组合与失败关闭 |

只修改这些必要边界，不重构 DetZero 模型主体。

---

## 6. 测试与验收门槛

### 6.1 单元测试

- 已带 `_with_camera_labels` 的路径不得重复追加后缀；
- testing frame 必须使用 `has_label=False`；
- TFRecord 199 帧 identity 和时间严格递增；
- detector 历史窗只用过去、最大 0.45 秒且最多 4 个 Waymo 历史 frames；
- TOP LiDAR 选择、raw extrinsic 审计、canonical RFU 和 TOP-centered virtual frame ↔ vehicle box 变换有非 identity oracle；
- 不复制 Waymo frames 伪造 10 个独立 sweeps；
- 历史 frame 的 ego-near 过滤边界覆盖 `< 1 m`、恰好 `1 m` 和当前帧不应用三种情况；
- time lag 对历史帧为正；
- Waymo 第 5/6 点特征不得误当 timestamp；
- 非 identity pose 点转换 oracle；
- class mapping 闭集；
- box center/LWH/yaw/velocity golden vector；
- 空帧 shape；
- missing/extra/nonfinite/负尺寸/非法 score 必须拒绝；
- list 与嵌套 dict 均能被 tracking loader 接受；
- no-CRM 组合必须保留 PRM 携带的 tracking score；
- 空类别不得伪造 forward。
- 从任意调用者 CWD 启动 wrapper 时，OpenPCDet/tracking/refining 的 `_BASE_CONFIG_` 都必须解析到预期文件；错误 CWD、逃逸路径或错误同名 config 必须在模型构建前失败。
- 分别篡改 tracking entry/base、任一 GRM/PRM entry/base、effective override 或交换任意两个 class/stage checkpoint 时，pre-build closure gate 必须拒绝；运行中修改后，postflight TOCTOU gate 也必须拒绝。

### 6.2 集成测试

按成本递增：

1. TFRecord 首帧完整 range-image 解码；
2. 1 帧真实 OpenPCDet GPU forward；
3. 连续 10 帧 detector + adapter + tracking；
4. 199 帧 detector；
5. tracking → object data；
6. 每个非空类别 GRM/PRM；
7. no-CRM final；
8. 199 张静态图；
9. 第二个独立进程、独立输出根重复全流程。

### 6.3 双运行比较

- frame IDs、class IDs、box count 和 track IDs：精确一致；
- NPZ 中离散决策：精确一致；
- GPU 连续 box 值：使用预先声明的容差，同时要求阈值/NMS/关联决策不变；
- PNG：在最终数值相同的前提下字节级一致；
- volatile 字段如 elapsed time、PID、run ID 不参与 deterministic equality，但仍保留在 receipt。

### 6.4 失败关闭条件

以下任一发生，最终状态必须是 `passed=false`：

- TFRecord、source、checkpoint、任一递归 config closure 或 canonical resolved config hash 漂移；
- tracking/refining 的 class-stage-config-base-checkpoint closed-world 映射 missing、extra、角色互换或 parent binding 不一致；
- Waymo SDK 不能解码任一帧；
- 少于 199 个输入或 detector frame records；
- checkpoint key/shape 不闭合；
- GPU forward 未执行；
- class mapping 出现未知值；
- 坐标/pose/yaw oracle 失败；
- 非有限值、非正尺寸、错误 shape；
- tracking 引入不存在的 frame ID；
- GRM/PRM 结果无法绑定到 track；
- validator 接受 extra/missing 文件；
- 第二次运行改变 NMS、class、track 或 final box identity；
- nuScenes 模式的报告声称 Waymo 指标或 DetZero detector 复现；
- rights gate 未闭合却标记 release eligible。

---

## 7. 当前不可避免的风险和阻塞项

1. **Waymo 使用授权尚未在本任务中明确确认。** 本地存在文件不等于已证明用途符合条款。
2. **当前 `mv2d` 没有 Waymo SDK。** 必须新建隔离 Python 3.8 预处理环境，不能污染已验证的 GRM/PRM 环境。
3. **本机 nuScenes checkpoint 的权利来源不闭合。** 它来自社区 issue 重分享，原始官方文件已失效，字节同一性未证明。
4. **真实 GPU forward 未验证。** 目前只验证了 checkpoint 与固定 OpenPCDet config 的 CPU key/shape 兼容。
5. **目标域质量不可评测。** testing split 无 GT；只能检查几何合理性、点支持和时间稳定性，不能得到 precision/recall/AP。
6. **跨域可能导致类别为空。** 若某一类没有轨迹，对应 GRM/PRM 无真实输入，必须记为 `NOT_EXECUTED_NO_INPUT`。
7. **Cyclist 语义映射较弱。** `bicycle/motorcycle` 不严格等价于 Waymo rider-centric Cyclist。
8. **score 未校准。** 无 CRM 且 detector 为 nuScenes 域时，score 的 Waymo 排序意义未知。
9. **现有脚本有固定路径和 test 分支缺陷。** 必须先完成薄适配和测试，再运行正式链路。
10. **Pickle 是本地兼容格式，不是安全交换格式。** 外部预测先进入 NPZ/JSON 验收，再由本地受信 producer 生成 pickle。

---

## 8. 实施决策建议

### 如果目标是“尽快跑通真实数据链路”

选择：

```text
nuscenes-cross-domain
```

但必须保留所有 `CROSS_DOMAIN / NO GT / NO CRM / RELEASE NO-GO` 标记，并接受可能出现空类别或低质量预测。

### 如果目标是“可信的 DetZero 推理复现”

优先取得：

```text
获授权的 Waymo-trained CenterPoint checkpoint
```

然后使用相同的外部预测适配边界进入 DetZero。这样既避免 DetZero detector fork 的 IoU/head 差异，又避免 nuScenes→Waymo 的主要数据域和类别域问题。

### 最终推荐

先实现通用、受测的两个边界：

1. `Waymo TFRecord → 标准逐帧点云/pose/info`；
2. `外部 detector native prediction → DetZero result.pkl`。

随后：

- 用当前 nuScenes checkpoint 做明确标记的 1 帧/10 帧 canary；
- canary 通过后再决定是否跑 199 帧跨域演示；
- 正式复现等待获授权 Waymo checkpoint，复用同一 adapter，不重写 tracking/refining。

---

## 9. 证据索引

仓库证据：

- `detection/detzero_det/datasets/waymo/waymo_utils.py:175-223`：单序列保存和后缀处理；
- `detection/detzero_det/datasets/waymo/waymo_utils.py:226-302`：TFRecord、pose 和点云解码；
- `detection/detzero_det/datasets/waymo/waymo_preprocess.py:101-119`：test 分支当前仍传 `has_label=True`；
- `detection/detzero_det/datasets/dataset.py:167-195`：DetZero sweep 坐标和时间特征；
- `tracking/detzero_track/datasets/waymo_dataset.py:51-85`：list/dict 输入和 test/GT 行为；
- `tracking/detzero_track/utils/data_utils.py:15-22`：frame list 转 sequence dict；
- `tracking/detzero_track/datasets/data_processor.py:42-95`：yaw 和 global transform；
- `tracking/detzero_track/datasets/dataset.py:25-27`：test split 禁用 assign mode；
- `daemon/prepare_object_data.py:193-313`：test object data 和点云裁剪；
- `refining/detzero_refine/datasets/waymo/waymo_position_dataset.py:231-253`：PRM 保留输入 score；
- `daemon/combine_output.py:102-167`：GRM/PRM/CRM 组合；
- `detection/tools/cfgs/det_model_cfgs/centerpoint_1sweep.yaml`：DetZero Waymo detector head；
- `detection/tools/cfgs/det_dataset_cfgs/waymo_1sweep.yaml`：DetZero Waymo 输入范围和特征；
- `utils/detzero_utils/config_utils.py:59-69`：`_BASE_CONFIG_` 的当前 CWD 相对解析行为；
- 固定 vendor `pcdet/config.py:51-78`：OpenPCDet `_BASE_CONFIG_` 的 CWD 相对解析行为；
- `docs/INFERENCE_ASSET_MANIFEST.json`：六个公开 GRM/PRM checkpoint 的官方下载枚举和本地字节同一性证据；
- `docs/INSTALL.md:3-44`：仓库要求的 Python/Waymo SDK；
- `LICENSE`：DetZero Apache-2.0。
- nuScenes source-axis 元数据：`/data/data/automomous/nuscenes/v1.0-mini/v1.0-mini/{sensor,calibrated_sensor}.json`。

外部权威/来源证据：

- Waymo Dataset License Agreement：<https://waymo.com/open/terms/>；
- Waymo Perception 10 Hz 与传感器说明：<https://waymo.com/open/about/>；
- Waymo `dataset.proto`：<https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/dataset.proto>；
- 原始 CenterPoint Waymo 指南：<https://github.com/tianweiy/CenterPoint/blob/master/docs/WAYMO.md>；
- 原始 CenterPoint Waymo model zoo：<https://github.com/tianweiy/CenterPoint/blob/master/configs/waymo/README.md>；
- OpenPCDet：<https://github.com/open-mmlab/OpenPCDet>；
- OpenPCDet checkpoint 重分享讨论：<https://github.com/open-mmlab/OpenPCDet/issues/1704>；
- nuScenes detection 的约 0.5 秒/最多 10 LiDAR sweeps 规则：<https://www.nuscenes.org/object-detection>；
- nuScenes 非商业条款：<https://www.nuscenes.org/terms-of-use>。

文档生成时间（CST，UTC+08:00）：2026-08-25T15:17:53+08:00
