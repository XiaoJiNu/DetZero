# 单 Waymo 测试段与开源 Detector 接入 DetZero 最终方案（GPT-5.6 Pro 评审版）

> **评审对象：** [`单Waymo测试段与CenterPoint接入DetZero方案.md`](./单Waymo测试段与CenterPoint接入DetZero方案.md)  
> **目标分支：** `develop-yr`  
> **评审日期：** 2026-08-25  
> **目标：** 使用一个真实、连续但无 3D GT 的 Waymo testing segment，完成纯 LiDAR 3D detector → DetZero tracking → GRM → PRM → no-CRM final output，并为以后替换高精度 detector 保留稳定接口。

---

## 0. 最终结论

原方案的**总体架构可行**，而且“让外部 detector 在原生代码与原生配置中运行，只在预测结果边界适配为 DetZero schema”是正确方向。但是，原方案把本机 nuScenes CenterPoint 跨域推理作为“可立即尝试路线”，这不应成为当前默认方案。

经过对 GitHub 上公开代码、公开 checkpoint、数据域、类别、运行环境和 DetZero 接口的进一步核查，当前最合理的结论如下。

| 问题 | 最终结论 |
| --- | --- |
| 单个 Waymo testing TFRecord 能否完成真实推理链路 | **可以。** 199 帧连续点云、时间戳和 vehicle pose 足以完成 detection、tracking 和 refining；但没有 GT，不能计算 AP、MOTA 或证明精度提升。 |
| 外部 detector 能否接入 DetZero | **可以。** 只要输出逐帧 `sequence_name/frame_id/pose/name/score/boxes_lidar`，无需把外部 checkpoint 强行加载进 DetZero detector。 |
| 当前可直接获得、Waymo 原生、三类别、纯 LiDAR 的公开权重 | **Open3D-ML PointPillars Waymo checkpoint。** 它不是当前 SOTA，但来源、数据域和类别契约比 nuScenes 跨域权重可靠。 |
| 当前是否应继续优先做 nuScenes CenterPoint → Waymo 跨域输入适配 | **不建议作为主线。** 只保留为显式诊断模式；它增加 TOP-only、虚拟坐标系、历史窗、类别合并和 score 域偏移等大量风险。 |
| 最终高精度 detector 应选什么 | **获授权的 Waymo-trained CenterPoint**；若后续允许自训，则可再评估 SAFDNet-4f、FSHNet 等更高精度模型。 |
| 是否必须有 CRM | **不必须。** 当前目标可运行 `DET → TRK → GRM → PRM`，最终 score 透传；但不能称为完整 DetZero 同权重复现。 |

因此，本项目建议采用**两阶段方案**：

```text
阶段 A：立即跑通、建立可信基线
Waymo testing TFRecord
  → Waymo SDK 解码全部 5 路 LiDAR
  → Open3D-ML Waymo PointPillars 公共 checkpoint
  → 显式 box/schema adapter
  → DetZero tracking（split=test）
  → GRM + PRM
  → no CRM / score passthrough
  → 199 帧结果与可视化

阶段 B：最终高精度替换
保持 TFRecord 预处理、DetZero adapter、tracking、GRM/PRM 全部不变
  → 只把 detector 替换为获授权的 Waymo CenterPoint
  → 重新执行 1 帧、10 帧和 199 帧验收
```

**当前默认 detector 推荐：Open3D-ML PointPillars Waymo。**  
**最终 detector 推荐：获授权的原始 CenterPoint Waymo 模型。**

---

## 1. 对原方案的可行性评审

## 1.1 原方案中正确且应保留的部分

### 1. detector 与 DetZero 解耦

正确的数据边界是：

```text
外部 detector 原生推理
  → 原生预测结果
  → 单次显式转换
  → DetZero result.pkl
```

而不是：

```text
把来源不同、head 不同、特征不同的 checkpoint
强行加载到 DetZero CenterPoint 网络
```

这样可以避免以下问题：

- checkpoint key 和模块命名不一致；
- 类别 head 数量不一致；
- velocity、IoU 等回归分支不一致；
- voxel、point range、点特征和 sweep 不一致；
- 部分权重加载成功但检测 head 保持随机初始化。

### 2. 单 Waymo testing segment 可用于机械推理

原文核验的指定 TFRecord 有连续 199 帧、5 路 LiDAR、有效 pose 和严格递增时间戳。它足以验证：

- TFRecord 解码；
- detector 真实前向；
- 逐帧 schema 转换；
- tracking；
- object crop；
- GRM/PRM；
- no-CRM 组合；
- 全序列可视化。

因为 testing 文件没有本地 3D GT，它不能用于：

- Waymo AP/mAPH；
- tracking MOTA；
- 判断 GRM/PRM 是否真正提升精度；
- 根据结果反复调阈值后宣称性能改善。

### 3. `split=test`、无 CRM 和 score passthrough 的判断正确

DetZero tracking 在 `split=test` 时不依赖 GT assignment。当前公开的 6 个 GRM/PRM checkpoint 已完成本地严格核验，因此后半段具备真实执行基础。

不运行 CRM 时，组合语义应固定为：

```text
中心 x/y/z、yaw  ← PRM
尺寸 dx/dy/dz     ← GRM
score             ← tracking/PRM 透传的 detector score
```

结果名称必须包含：

```text
NO_GT
NO_CRM
SCORE_PASSTHROUGH
```

### 4. 失败关闭、哈希冻结和不可覆盖输出目录值得保留

原方案中的以下工程原则正确：

- 输入、checkpoint、配置和代码 revision 固定；
- 1 帧 → 10 帧 → 199 帧逐级 canary；
- 输出目录不得覆盖已有结果；
- class、box、pose、frame identity 必须闭合；
- 非有限数、负尺寸、缺帧和未知类别立即失败；
- pickle 只作为本地兼容产物，同时输出 NPZ/JSON 供安全验收。

这些措施不需要因为更换 detector 而删除。

---

## 1.2 原方案需要修改的关键点

### 1. 不应把 nuScenes 跨域 detector 作为默认“立即跑通”方案

本机 nuScenes CenterPoint checkpoint 虽然与固定 OpenPCDet nuScenes 配置在 key/shape 层面兼容，但它与 Waymo 目标域存在多重差异：

- nuScenes 10 类，DetZero 只接受 Vehicle/Pedestrian/Cyclist；
- 10 sweeps 与 Waymo 单帧/10 Hz 序列契约不同；
- point range、voxel 和输入特征不同；
- bicycle/motorcycle 到 Waymo Cyclist 的语义不严格等价；
- score 未在 Waymo 上校准；
- checkpoint 是社区重分享，独立权利来源尚未闭合；
- testing segment 没有 GT，无法判断跨域检测是否有效。

机械上能前向，不等于检测结果有用。

### 2. TOP-only + 虚拟 RFU + 历史窗会引入不必要的前置复杂度

原方案为兼容 nuScenes checkpoint，设计了：

- 只取 TOP LiDAR；
- TOP-centered 虚拟 model frame；
- Waymo vehicle frame 与 source-derived RFU 变换；
- 最多 4 个历史帧、0.45 秒窗口；
- intensity/time-lag 重构；
- 10 类到 3 类合并。

这些步骤本身可以实现，但每一步都会增加坐标、时间和类别错误的机会。对于“先跑通真实 Waymo detector 链路”而言，没有必要先承担这些风险。

使用 Waymo 原生 checkpoint 后，可以直接：

```text
全部 5 路 Waymo LiDAR、vehicle frame、单帧、Waymo 三类别
```

无需虚拟传感器坐标系，也无需跨域历史窗。

### 3. 原方案的实现规模大于验证问题所需规模

原文提出了完整 manifest、递归配置闭包、双运行字节级比较、全 199 帧重渲染等严格要求。这些适合最终可审计交付，但第一阶段应优先完成最小闭环：

```text
真实点云 → 真实 detector → adapter → tracking → GRM/PRM
```

建议分层：

- **P0：** 1 帧真实 detector 与 box 几何正确；
- **P1：** 10 帧 detection + tracking；
- **P2：** 199 帧 + GRM/PRM + no-CRM；
- **P3：** 完整不可变 manifest、双运行和发布级验收。

避免在 detector 能否正常产生有效框尚未验证前，先实现全部发布级设施。

---

## 1.3 可行性评分

| 部分 | 评分 | 结论 |
| --- | ---: | --- |
| 外部 detector → DetZero adapter 架构 | 9/10 | 正确，应保持 |
| 单 Waymo testing segment 机械推理 | 9/10 | 可行，无 GT 指标 |
| tracking → GRM → PRM → no CRM | 8/10 | 可行，但存在 detector 分布差异 |
| nuScenes checkpoint 作为跨域实验 | 5/10 | 仅适合诊断和研究演示 |
| nuScenes checkpoint 作为默认正式路线 | 2/10 | 不推荐 |
| Open3D-ML Waymo PointPillars 作为首个真实 detector | 8/10 | 当前最稳妥的公开基线 |
| 获授权 Waymo CenterPoint 作为最终 detector | 9/10 | 最符合 DetZero，当前受权重获取阻塞 |

---

## 2. 当前可使用的开源 detector 核查

这里区分三种状态：

1. **代码公开且 Waymo checkpoint 公开，可立即下载推理；**
2. **代码公开、Waymo checkpoint 需要授权申请；**
3. **只公开代码或其他数据集 checkpoint，不能直接用于 Waymo。**

## 2.1 候选对比

| 候选 | Waymo-trained 权重 | 类别 | 框架 | 主要问题 | 结论 |
| --- | --- | --- | --- | --- | --- |
| **Open3D-ML PointPillars Waymo** | **公开 Model Zoo 直链** | Vehicle/Pedestrian/Cyclist | PyTorch/Open3D-ML | 精度较旧、非 SOTA；新 GPU 可能需源码编译 Open3D | **当前默认推荐** |
| **原始 CenterPoint Waymo** | 需提交 Waymo 注册证明和非商业用途后申请 | 3 类 | det3d/CenterPoint | 不能匿名下载；老运行时 | **最终高精度推荐** |
| **Hale423/CenterPoint TensorRT** | 仓库内有 Waymo ONNX | 原配置为 3 类 | C++/TensorRT | 社区权重来源说明不足；TensorRT 8/CUDA 11.3 过旧；指标非标准 | 实验备选，不做默认 |
| `sean-wade/simtrack_waymo` | 仓库内有 `.pth` | Vehicle/Pedestrian | 旧 det3d | 只有 2 类；环境老；仓库记录有坐标疑点 | 不作为完整方案 |
| `BotRunner64/RefineMoE` | 提供部分 Waymo 权重 | Car only | FSHNet/OpenPCDet 风格 | 只有车辆；验证子集不同 | 仅车辆实验 |
| SAFDNet-4f / FSHNet / LION | Waymo checkpoint 未公开 | 3 类 | OpenPCDet 风格 | 需要自训或另行申请 | 高精度训练候选，不是立即可用权重 |
| 本机 nuScenes CenterPoint | 本地已有 | nuScenes 10 类 | OpenPCDet | 跨域、类别和输入不匹配、权利来源未闭合 | 仅诊断模式 |

---

## 2.2 当前首选：Open3D-ML PointPillars Waymo

GitHub：[`isl-org/Open3D-ML`](https://github.com/isl-org/Open3D-ML)

官方仓库 README 的 Waymo object detection 表格公开了可下载 checkpoint：

```text
pointpillars_waymo_202211200158utc_seed2_gpu16.pth
```

官方配置：

```text
ml3d/configs/pointpillars_waymo.yml
```

其关键契约为：

```yaml
classes: [VEHICLE, PEDESTRIAN, CYCLIST]
point_cloud_range: [-74.88, -74.88, -2, 74.88, 74.88, 4]
voxel_size: [0.32, 0.32, 6]
input: x, y, z, intensity
score_thr: 0.1
```

官方 Waymo 预处理会解码全部 5 路 LiDAR、两个 return，并保存点云和 pose。PointPillars 模型实际取前四维：

```text
[x, y, z, intensity]
```

### 为什么优先选它

- checkpoint 明确在 Waymo 上训练；
- Vehicle/Pedestrian/Cyclist 三类与 DetZero 完全对齐；
- 点云直接位于 Waymo vehicle frame；
- 不需要 TOP-only、虚拟 RFU 或 nuScenes 时间特征；
- PyTorch 输出可直接取得 3D box、类别和 score；
- 官方仓库提供 config、预处理代码、模型实现和模型下载链接；
- 代码采用 MIT 许可证，来源比社区重分享 checkpoint 更清楚。

### 必须认识的限制

Open3D-ML README 报告的是其自定义表格中的 Waymo BEV/3D mAP@0.50，不是 Waymo 官方 L2 mAPH，因此不能与 DetZero、SAFDNet、FSHNet 的 L2 mAPH 数值直接比较。它也不是现代高精度 SOTA detector。

它适合的角色是：

> **可信、公开、Waymo 原生的真实 detector 基线，用于先完成 DetZero 全链路。**

它不适合的声明是：

> **当前精度最好的 Waymo detector。**

### 环境风险

Open3D-ML 官方预编译组合主要面向较旧的 PyTorch/CUDA。当前 RTX 5090、PyTorch 2.7、CUDA 12.8 环境不能假定其 CUDA ops 直接兼容。

建议顺序：

1. 在隔离环境完成 CPU 单帧 canary，验证 checkpoint、配置、预处理和 adapter；
2. 再尝试为当前 PyTorch/CUDA 从源码编译 Open3D 的 PyTorch/CUDA ops；
3. GPU canary 通过后再跑 199 帧；
4. 不把旧 Open3D 依赖直接安装进已验证的 DetZero refining 环境。

实施时必须下载 checkpoint 后计算并冻结 SHA-256。本文不填写尚未在本机重新下载核验的哈希。

---

## 2.3 最终首选：获授权的原始 CenterPoint Waymo 模型

GitHub：[`tianweiy/CenterPoint`](https://github.com/tianweiy/CenterPoint)

原始 CenterPoint Waymo model zoo 要求提供：

- 姓名和机构；
- Waymo 注册确认；
- 预期用途；
- 非商业使用说明。

取得权重后，不应把它直接塞进 DetZero detector，而应在其原生 CenterPoint 代码、原生 Waymo 配置中推理，再转换输出。

优点：

- 与 DetZero 原 detector 技术路线最接近；
- Waymo 三类别；
- 精度显著高于旧 PointPillars 基线；
- 后续 tracking、GRM、PRM 可以复用同一个 adapter 接口。

因此它是**最终正式方案**，但不是当前“匿名下载即运行”的方案。

---

## 2.4 实验备选：Hale423/CenterPoint

GitHub：[`Hale423/CenterPoint`](https://github.com/Hale423/CenterPoint)

仓库内有：

```text
models/pfe_baseline32000.onnx
models/rpn_baseline.onnx
```

并提供 Waymo TensorRT 推理代码。它可以作为第二个 detector 对照，但不应排在 Open3D-ML 前面，原因是：

- 权重由社区仓库提供，缺少独立 checkpoint 训练与许可说明；
- README 指标采用其自身 2D IoU 设置，不能作为官方 Waymo 精度；
- 运行环境绑定 TensorRT 8.0.1.6 和 CUDA 11.3；
- 当前 RTX 5090 环境大概率需要重写或升级 TensorRT 接口；
- ONNX 被拆为 PFE/RPN，前后处理和输出还需 C++ 适配。

建议只在以下条件满足后启用：

```text
checkpoint/ONNX 来源审计通过
+ TensorRT 10/CUDA 12 端口完成
+ 与 PyTorch/Waymo 坐标 oracle 对齐
```

---

## 3. 最终推荐架构

## 3.1 定义三个模式

| 模式 | detector | 默认状态 | 用途 |
| --- | --- | --- | --- |
| `open3dml-waymo-pointpillars` | Open3D-ML Waymo PointPillars | **默认启用** | 当前真实 Waymo 全链路基线 |
| `centerpoint-waymo-authorized` | 获授权的原始 CenterPoint Waymo checkpoint | 权重取得后启用 | 最终正式、高精度方案 |
| `nuscenes-cross-domain` | 本机 nuScenes OpenPCDet CenterPoint | 默认关闭，必须显式确认 | 只做跨域诊断研究 |

默认入口不得自动选择 `nuscenes-cross-domain`。

---

## 3.2 数据流

```text
指定 Waymo testing TFRecord（199 帧）
  │
  ├─ Waymo SDK 流式解码
  │    ├─ all-LiDAR: [x,y,z,intensity,elongation,NLZ]
  │    ├─ vehicle pose: T_G<-V
  │    ├─ timestamp
  │    └─ frame/sequence identity
  │
  ├─ Open3D-ML detector 输入
  │    └─ all-LiDAR[:, 0:4] = [x,y,z,intensity]
  │
  ├─ Open3D-ML PointPillars 原生前向
  │    └─ BEVBox3D / raw bboxes + score + class
  │
  ├─ Open3D-ML → DetZero adapter
  │    └─ frame-level detzero_result.pkl
  │
  ├─ DetZero tracking（split=test）
  │
  ├─ DetZero object data preparation
  │    └─ 使用全部 5 路 LiDAR 点
  │
  ├─ Vehicle/Pedestrian/Cyclist GRM + PRM
  │
  ├─ no-CRM combination
  │    └─ score passthrough
  │
  └─ 199 帧结果、NPZ/JSON manifest 和静态可视化
```

---

## 4. 详细实施方案

## 4.1 阶段 1：单 TFRecord 无标签流式预处理

保留原方案拟新增的：

```text
tools/external_detector/preprocess_waymo_test_segment.py
```

或兼容原命名：

```text
tools/external_centerpoint/preprocess_waymo_test_segment.py
```

要求：

1. 只处理指定的一个 TFRecord；
2. `has_label=False`；
3. 不重复追加 `_with_camera_labels`；
4. 逐帧解码，不把 199 个 Frame 全部留在内存；
5. 保存全部 5 路 LiDAR、两个 return；
6. 点云保持 Waymo vehicle frame；
7. 保存：

   ```text
   [x, y, z, intensity, elongation, NLZ]
   ```

8. `pose` 保持 float64，含义固定为当前 vehicle frame 到 global frame；
9. timestamp 保持 int64；
10. 生成 `0000.npy ... 0198.npy` 和逐帧 info；
11. 读回全部文件验证 shape、有限性、点数和 frame coverage；
12. 不生成伪 GT annotation。

### 与原方案相比的简化

在默认 `open3dml-waymo-pointpillars` 模式中：

- 不生成 TOP-only detector 数据；
- 不构造虚拟 RFU；
- 不融合历史帧；
- 不构造 timestamp feature；
- 不做 nuScenes 类别映射。

---

## 4.2 阶段 2：Open3D-ML 原生推理

拟新增：

```text
tools/external_detector/run_open3dml_waymo_pointpillars.py
```

固定：

```text
repository: isl-org/Open3D-ML
config: ml3d/configs/pointpillars_waymo.yml
checkpoint: pointpillars_waymo_202211200158utc_seed2_gpu16.pth
input: all-LiDAR [x,y,z,intensity]
batch size: 1
```

执行门槛：

1. 固定 Open3D-ML commit；
2. 固定 config SHA-256；
3. 下载 checkpoint 后固定 size 和 SHA-256；
4. checkpoint 严格加载；
5. `model.eval()`；
6. `torch.inference_mode()`；
7. 首先运行 1 帧；
8. 通过后运行连续 10 帧；
9. 最后运行 199 帧；
10. 每帧保存 raw box、score、class 和输入统计；
11. 任何 NaN、缺帧、空 rank、非法标签或模型加载差异立即停止。

Open3D-ML PointPillars 是单帧模型，因此不需要历史 pose 补偿。后续时间一致性由 DetZero tracking 处理。

---

## 4.3 阶段 3：Open3D-ML box 转 DetZero box

拟新增：

```text
tools/external_detector/convert_open3dml_to_detzero.py
```

### 4.3.1 类别映射

直接一一映射：

| Open3D-ML | DetZero |
| --- | --- |
| `VEHICLE` | `Vehicle` |
| `PEDESTRIAN` | `Pedestrian` |
| `CYCLIST` | `Cyclist` |

不允许出现第四类；未知类别立即失败。

### 4.3.2 中心和尺寸

Open3D-ML `BEVBox3D` 内部：

```text
center = 3D 几何中心
size   = [width, height, depth/length]
```

DetZero Waymo box 使用：

```text
[x, y, z, length, width, height, heading]
```

因此转换为：

```python
x, y, z = box.center
length = box.size[2]
width  = box.size[0]
height = box.size[1]
```

不能直接把 `BEVBox3D.to_xyzwhlr()` 原样送入 DetZero，因为该函数返回的是**底面中心**，并保留 Open3D 原生 yaw 表达。

### 4.3.3 yaw 转换

Open3D-ML Waymo 预处理将 Waymo heading 写成：

```text
yaw_open3d = -heading_waymo - π/2
```

因此 adapter 的逆变换固定为：

```text
heading_waymo = wrap_to_pi(-yaw_open3d - π/2)
```

不能根据可视化“看起来差不多”临时加减角度。必须使用带非零 heading 的 golden test 验证：

- `heading_waymo = 0`；
- `heading_waymo = π/2`；
- `heading_waymo = -π/4`；
- 非对称长宽车辆框。

### 4.3.4 canonical box

最终每个框为：

```text
[x, y, z, length, width, height, heading, vx, vy]
```

PointPillars 不输出可靠 velocity，因此明确设置：

```text
vx = 0
vy = 0
```

DetZero 当前 tracker 主要消费前 7 维并自行估计运动状态，零 velocity 只是 schema 兼容字段。

### 4.3.5 frame record

每帧输出：

```python
{
    "sequence_name": str,
    "frame_id": int,
    "pose": np.ndarray((4, 4), dtype=np.float64),
    "name": np.ndarray((N,), dtype=str),
    "score": np.ndarray((N,), dtype=np.float32),
    "boxes_lidar": np.ndarray((N, 9), dtype=np.float32),
}
```

强制：

- 199 帧全部存在，包括零检测帧；
- 空 box shape 为 `(0, 9)`；
- score 在 `[0,1]`；
- 尺寸严格为正；
- box 和 pose 都是同一帧 vehicle frame 契约；
- `sequence_name` 对 199 帧完全一致；
- `frame_id` 精确为 `0..198`。

---

## 4.4 阶段 4：DetZero tracking

沿用原方案，但把入口从 `external_centerpoint` 泛化为 `external_detector`。

建议修改：

```text
tracking/tools/run_track.py
```

增加：

```text
--root_path
--output_path
--schema_preflight
```

目标调用：

```bash
cd tracking/tools
python run_track.py \
  --cfg_file cfgs/tk_model_cfgs/waymo_detzero_track.yaml \
  --data_path "$RUN_ROOT/detection/detzero_result.pkl" \
  --root_path "$RUN_ROOT/data/waymo" \
  --output_path "$RUN_ROOT/tracking/tracking_test.pkl" \
  --split test \
  --batch_size 1 \
  --workers 1
```

注意：

- 必须使用 `split=test`；
- tracking 不加载学习 checkpoint；
- 必须保留 detector score；
- 输出 frame ID 只能是输入 frame ID 的子集；
- 记录 track 长度、空帧、类别计数和纯预测帧比例。

---

## 4.5 阶段 5：object data 与 GRM/PRM

保留原方案对以下脚本的薄修改：

```text
daemon/prepare_object_data.py
refining/tools/test.py
```

增加显式：

```text
--root_path
--output_root / --result_dir
```

object crop 使用全部 5 路 Waymo LiDAR 点，而不是只使用 detector 输入的某个子集。

### 重要分布差异

DetZero GRM/PRM 是在 DetZero/Waymo detector 和 tracking 分布上训练的；Open3D-ML PointPillars 的框误差和 score 分布可能不同。因此：

- GRM/PRM 可以机械前向；
- 不能因为模型成功运行就断言框一定更准确；
- 必须同时保留 `detector box`、`tracking box` 和 `GRM/PRM final box`；
- 对 refined box 执行有限值、正尺寸、中心跳变和尺寸变化的结构检查；
- testing 无 GT 时，不根据结果主观选择“更好”的版本后宣称精度提升。

若某一类别没有真实轨迹：

```text
NOT_EXECUTED_NO_INPUT
```

不得制造假目标来满足“六个模型都执行”的形式要求。

---

## 4.6 阶段 6：no-CRM 组合

新增明确入口：

```text
tools/external_detector/combine_grm_prm_no_crm.py
```

输出：

```text
final/final_frame_grm_prm_score_passthrough.pkl
```

固定组合：

```python
final_box[:, 0:3] = prm_box[:, 0:3]
final_box[:, 3:6] = grm_box[:, 3:6]
final_box[:, 6]   = prm_box[:, 6]
final_score       = prm_or_tracking_score
```

不传：

```text
--combine_conf_res
```

最终结果不称为 CRM confidence refined result。

---

## 4.7 阶段 7：输出和可视化

建议输出目录：

```text
output/waymo_open3dml_detzero_<timestamp>/
├── source_manifest.json
├── data/waymo/
├── detection/
│   ├── raw_open3dml_predictions.npz
│   ├── detzero_result.pkl
│   └── adapter_manifest.json
├── tracking/
│   └── tracking_test.pkl
├── refining/
│   ├── Vehicle/
│   ├── Pedestrian/
│   ├── Cyclist/
│   └── result/
├── final/
│   ├── final_frame_grm_prm_score_passthrough.pkl
│   ├── final_arrays.npz
│   └── final_manifest.json
└── visualization/
    └── frames/0000.png ... 0198.png
```

每张图至少显示：

- 当前帧 Waymo LiDAR BEV；
- detector box；
- tracking ID；
- GRM/PRM final box；
- class、score 和 frame ID；
- `NO GT`；
- `NO CRM`；
- detector 名称 `Open3D-ML Waymo PointPillars`。

---

## 5. 建议新增或修改的文件

| 类型 | 路径 | 目的 |
| --- | --- | --- |
| Create | `tools/external_detector/preprocess_waymo_test_segment.py` | 单 TFRecord、无标签、全部 Waymo LiDAR 流式解码 |
| Create | `tools/external_detector/run_open3dml_waymo_pointpillars.py` | 原生 Open3D-ML Waymo 推理 |
| Create | `tools/external_detector/convert_open3dml_to_detzero.py` | 类别、z 中心、尺寸顺序、yaw 和 schema 转换 |
| Create | `tools/external_detector/combine_grm_prm_no_crm.py` | 明确的三类 no-CRM 合并 |
| Create | `tools/external_detector/render_waymo_sequence.py` | 199 帧静态结果图 |
| Create | `tools/external_detector/validate_waymo_external_run.py` | 跨阶段结构和身份验收 |
| Create | `reproduce_waymo_external_detector.sh` | 支持多 detector profile 的统一入口 |
| Modify | `tracking/tools/run_track.py` | 显式 root/output/schema preflight |
| Modify | `daemon/prepare_object_data.py` | 显式 root/output，不写死仓库数据目录 |
| Modify | `refining/tools/test.py` | test split、root/result 和无 GT 指标关闭 |
| Test | `tests/test_waymo_test_preprocess.py` | 199 帧、无标签、点云/pose 契约 |
| Test | `tests/test_open3dml_detzero_adapter.py` | z、尺寸顺序、yaw、class、空帧 golden tests |
| Test | `tests/test_external_detector_pipeline.py` | detector → tracking → GRM/PRM → no CRM |

建议将原文的 `tools/external_centerpoint/` 泛化为 `tools/external_detector/`。这样后续替换 CenterPoint、SAFDNet 或 FSHNet 时无需复制整套流程。

---

## 6. 分阶段执行与验收门槛

## 6.1 P0：资产和预处理

验收：

- TFRecord hash 与原文记录一致；
- 199 帧全部可由 Waymo SDK 解码；
- 每帧 all-LiDAR 点数大于 0；
- 点云 shape `(N,6)`，全部有限；
- pose 为有限刚性 4×4；
- frame ID、timestamp 严格递增；
- Open3D-ML commit/config/checkpoint hash 冻结。

## 6.2 P1：1 帧 detector canary

验收：

- checkpoint 严格加载；
- 真实 CPU 或 GPU forward；
- 输出类别闭合；
- box 有限、尺寸为正；
- Open3D → Waymo heading golden test 通过；
- box 中心和尺寸重排 golden test 通过；
- 可视化方向与点云一致。

## 6.3 P2：10 帧 detector + tracking

验收：

- 10 个 frame record 全部存在；
- tracking 不加载 GT；
- track frame IDs 不越界；
- 车辆/行人/骑行者计数记录；
- 至少一个非空 track 才继续 object crop。

## 6.4 P3：199 帧 detector

验收：

- 输入和 detector 输出 frame coverage 精确等于 `0..198`；
- 零检测帧显式存在；
- 无 NaN/Inf；
- 每帧 box 数量、类别和 score 统计完整；
- 不使用 testing 结果回调 score threshold。

## 6.5 P4：tracking → GRM/PRM → no CRM

验收：

- object crop 可读取同一 run root 下的全部 5 路 LiDAR；
- 每个非空类别的 GRM/PRM checkpoint 严格加载并真实前向；
- final box 可追溯到唯一 track；
- final score 与 tracking/PRM 输入一致；
- 不读取 CRM 文件；
- 空类别明确标记而不伪造输入。

## 6.6 P5：发布级验收

验收：

- 199 张图可解码且非空白；
- NPZ 使用 `allow_pickle=False` 可读；
- JSON 拒绝重复 key；
- 输入、配置、checkpoint、代码和产物 hash 完整；
- 第二个独立输出目录重复运行，离散结果一致；
- 结果说明不包含未被证据支持的精度声明。

---

## 7. 风险与缓解措施

| 风险 | 影响 | 缓解措施 |
| --- | --- | --- |
| Open3D-ML PointPillars 精度较旧 | 漏检远距离目标，影响轨迹完整性 | 只作为当前可信基线；后续替换授权 CenterPoint |
| 当前 RTX 5090 与 Open3D CUDA ops 兼容性 | GPU 无法直接运行 | 隔离环境、CPU canary、源码编译 Open3D CUDA/PyTorch ops |
| GRM/PRM 输入分布与 PointPillars 不同 | refined box 不一定更准 | 保存前后结果、结构门槛、无 GT 时不宣称提升 |
| testing segment 无 GT | 无法计算 precision/recall/AP | 只做机械推理与几何/时序验收；另下载有标签 validation segment 做精度评测 |
| no CRM | score 未轨迹级重标定 | 明确 score passthrough；不使用过高阈值；保留原始 score |
| 某类无预测 | 对应 refiner 无输入 | 明确 `NOT_EXECUTED_NO_INPUT`，必要时换有三类目标的 segment |
| Waymo 权重和数据许可 | 不能随意商用或再分发 | 使用前确认 Waymo 条款和用途；manifest 记录许可状态 |
| 社区 CenterPoint ONNX 来源不充分 | 结果难以审计 | 不作为默认；只有来源和运行时验证闭合后再启用 |

---

## 8. 最终决策

### 当前立即实施

```text
Open3D-ML Waymo PointPillars
  → DetZero tracking
  → GRM + PRM
  → no CRM
```

理由不是它精度最高，而是它是当前核查到的、同时满足以下条件的最稳妥公开资产：

- 纯 LiDAR；
- Waymo 原生训练；
- Vehicle/Pedestrian/Cyclist 三类；
- checkpoint 由正式开源项目 Model Zoo 公开；
- 原生预处理、模型代码和推理输出齐全；
- 不需要跨域类别、坐标和时间特征重构。

### 最终正式版本

```text
获授权的原始 CenterPoint Waymo checkpoint
  → 原生 CenterPoint 推理
  → 复用同一 DetZero adapter
  → tracking → GRM → PRM → no CRM
```

若以后取得 CRM 或自行训练 CRM，再新增独立 profile，不修改当前 no-CRM 产物。

### 不推荐作为主线

```text
nuScenes CenterPoint
  → TOP-only Waymo
  → 虚拟 RFU
  → 历史帧适配
  → 类别合并
```

该路线可以保留为 `nuscenes-cross-domain` 研究模式，但不应消耗当前主线资源，也不能作为 Waymo detector 或 DetZero 官方复现结果。

---

## 9. 结果命名与允许声明

当前阶段允许的结果名称：

> Open3D-ML Waymo PointPillars 在单个无标签 Waymo testing segment 上的检测，以及 DetZero tracking、GRM、PRM、no-CRM 工程链路复现。

不得称为：

- DetZero 官方 detector 同权重复现；
- DetZero 论文完整推理复现；
- Waymo AP/mAPH 复现；
- 当前最精确的 Waymo 3D detector；
- CRM confidence refined result；
- GRM/PRM 已被证明提升真实精度。

---

## 10. 证据与参考

### 本仓库

- [原始实施方案](./单Waymo测试段与CenterPoint接入DetZero方案.md)
- [`FULL_INFERENCE_ASSET_AUDIT.md`](../FULL_INFERENCE_ASSET_AUDIT.md)
- [`INFERENCE_ASSET_MANIFEST.json`](../INFERENCE_ASSET_MANIFEST.json)
- [`INFERENCE_REPRODUCTION.md`](../INFERENCE_REPRODUCTION.md)

### Detector

- [Open3D-ML](https://github.com/isl-org/Open3D-ML)
- [Open3D-ML Waymo PointPillars config](https://github.com/isl-org/Open3D-ML/blob/main/ml3d/configs/pointpillars_waymo.yml)
- [Open3D-ML Waymo preprocessing](https://github.com/isl-org/Open3D-ML/blob/main/scripts/preprocess_waymo.py)
- [Open3D-ML PointPillars implementation](https://github.com/isl-org/Open3D-ML/blob/main/ml3d/torch/models/point_pillars.py)
- [Open3D-ML Waymo checkpoint](https://storage.googleapis.com/open3d-releases/model-zoo/pointpillars_waymo_202211200158utc_seed2_gpu16.pth)
- [Original CenterPoint Waymo Model Zoo](https://github.com/tianweiy/CenterPoint/blob/master/configs/waymo/README.md)
- [Hale423 CenterPoint TensorRT](https://github.com/Hale423/CenterPoint)
- [SAFDNet/HEDNet](https://github.com/zhanggang001/HEDNet)
- [FSHNet](https://github.com/Say2L/FSHNet)
- [LION](https://github.com/happinesslz/LION)

### DetZero 缺失权重

- [DetZero issue #49：Detection checkpoint](https://github.com/PJLab-ADG/DetZero/issues/49)
- [DetZero issue #66：CRM checkpoint](https://github.com/PJLab-ADG/DetZero/issues/66)

### 数据许可

- [Waymo Open Dataset License Agreement](https://waymo.com/open/terms/)

---

## 11. 一句话执行建议

> **先用 Open3D-ML 的 Waymo 三类 PointPillars 公共权重跑通 199 帧真实全链路；adapter 稳定后，再把 detector 无缝替换为获授权的 Waymo CenterPoint。不要把 nuScenes 跨域 detector 作为当前默认方案。**
