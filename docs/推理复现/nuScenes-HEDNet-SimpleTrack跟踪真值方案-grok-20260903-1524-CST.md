# nuScenes × HEDNet × SimpleTrack 跟踪真值方案

| 字段 | 内容 |
| --- | --- |
| 作者 / 助手 | **grok**（算法复现） |
| 文档创建时间 | 2026-09-03 15:24:04 CST（2026-09-03T07:24:04Z） |
| 文档版本 | v1.0 |
| 工作区 | `/data/code/cv/AutoLabel/DetZero-nuscenes-simpletrack-trackgt` |
| Git 分支 | `nuscenes-simpletrack-trackgt`（自 `develop-yr` @ `776d975`） |
| 产品目标 | **B：出跟踪真值**（稳定 `track_id`，可评官方 AMOTA） |
| 对照旧仓 | `DetZero-nuscenes-hednet-qwen3.8flash`（只读，不改） |

## 修订记录

| 时间 (CST) | 作者 | 版本 | 变更摘要 |
| --- | --- | --- | --- |
| 2026-09-03 15:24:04 CST | grok | v1.0 | 初版：按目标 B 定稿实现方案（概述 + 详述）；跟踪优先 SimpleTrack(MIT)；不做 DetZero GRM/PRM 精修 |

> 后续每次修正：在本表追加一行，并在文首「文档版本 / 文档创建或修订时间」同步更新；文件名中的时间戳保留**首次创建**时刻，修订以表内记录为准（若重大改版可另存 `...-grok-<新时间戳>.md` 并在旧文顶部加指向）。

---

## 1. 方案概述

### 1.1 一句话目标

在 **nuScenes v1.0-mini**（scene-0103 / scene-0916）上，用已有 **HEDNet** 检测结果，经开源 **SimpleTrack（MIT）** 做 3D 多目标跟踪，产出带稳定 `track_id` 的跟踪真值（伪标签），并可用 **nuScenes 官方 tracking 评估**（AMOTA / AMOTP / IDS / FRAG 等）验收。

### 1.2 明确做什么 / 不做什么

| 做 | 不做 |
| --- | --- |
| HEDNet `result.pkl` → nuScenes / SimpleTrack 检测输入适配 | 重训或微调 HEDNet |
| SimpleTrack 离线跟踪（优先官方 nuScenes 配置与流程） | 自研 Kalman/匈牙利跟踪（旧 `track_hednet_boxes.py` 仅对照） |
| 导出官方 tracking `results.json` + 帧级带 `track_id` 的内部 pkl | 复活 Waymo 域 DetZero tracking / GRM / PRM / CRM 作为主链 |
| 隔离环境跑官方 AMOTA | 用「跟踪前后 mAP 不变」当作跟踪好坏的主指标 |
| 新 worktree 开发，旧 qwen3.8flash 只读 | 在旧 worktree 上继续堆交付物 |

### 1.3 为什么不是完整 DetZero 链

旧方案在同一数据上已用 GT 证明：Waymo 训练的 DetZero tracking + GRM/PRM 在 nuScenes 2Hz 上 **R1=DEGRADED**（mAP 0.74→0.027）。  
目标 B 要的是 **跨帧 ID 与官方跟踪指标**，不是「把框修到比检测更准」。SimpleTrack 是 TBD：以检测框为主，滤波主要稳中心/速度，**基本不修尺寸**；朝向弱处理——这与 B 一致。

### 1.4 端到端数据流（逻辑）

```
HEDNet result.pkl (mini val, 81 帧)
        │
        ▼
S1  adapt_hednet_to_nusc_det.py     # 10 类检测 → tracking 用类别集合 + score/box/token
        │
        ▼
S2  SimpleTrack (mot_3d)            # 官方 nuScenes 推理入口 / 等价配置
        │
        ├──────────────────────────►  delivery/tracking_results.json   # 官方格式
        └──────────────────────────►  delivery/frames_with_track_id.pkl
        │
        ▼
S3  nuScenes tracking evaluate      # AMOTA 等（隔离 env，钉 motmetrics 兼容版本）
        │
        ▼
S4  render / 抽查                   # BEV 可视化 + 与旧自研 track-v2 对照（可选）
```

### 1.5 验收门（概述）

| 门 | 条件 |
| --- | --- |
| G1 链路 | 两 scene（或整份 mini val）跑通 S1→S2，产物非空、帧数对齐 |
| G2 格式 | `results.json` 能被 nuScenes tracking 评估器加载；每框有 `tracking_id` / 官方字段齐全 |
| G3 指标 | 产出 `metrics_summary.json`（AMOTA 等）；如实记录，不与「检测 mAP」混报 |
| G4 边界 | 旧 worktree 零改动；本仓新增限于适配脚本、vendor/submodule、docs、output |
| G5 许可 | 引入代码 MIT/Apache；**不引入 AB3DMOT 非商用协议** |

### 1.6 工作量与执行位置（概述）

- **位置**：本机拯救者（已连 local-exec）；云端只做读码/写文档/轻量检查。  
- **GPU**：本阶段复用已有 HEDNet 推理结果，**默认不新占 5090**；若需重跑 HEDNet 再申请。  
- **预计**：适配 + SimpleTrack 接入 + mini 跑通 + 官方评估环境 ≈ 1～2 天工程（不含冲榜调参）。

---

## 2. 方案详述

### 2.1 需求与真值定义（目标 B）

**跟踪真值（本项目交付含义）** = 每个 keyframe（或评估所需帧）上的一组 3D 框，且：

1. 几何来自 HEDNet（经跟踪器关联；中心可能经运动滤波轻微偏离检测）；  
2. 同类目标跨帧共享稳定 **`tracking_id` / `track_id`**；  
3. 类别属于 **nuScenes tracking 评测集合**（见 §2.3）；  
4. 可序列化为官方 tracking 提交/评估 JSON；  
5. 可用官方指标量化，而不是仅靠自研 BEV 中心距 mAP。

**非目标**：尺寸/姿态的点云级精修；用伪标签反超 HEDNet 检测 mAP；Release/receipt 发布链。

### 2.2 输入资产（已核实路径，实现时再 sha 固化）

| 资产 | 预期路径 | 说明 |
| --- | --- | --- |
| HEDNet 检测 | `/data/code/cv/AutoLabel/BEV-OD/HEDNet-qwen/output/mini/eval/epoch_2/val/default/result.pkl` | 81 帧；`boxes_lidar (N,9)`、`name`、`score`、`metadata.token` |
| nuScenes mini | `/data/data/automomous/nuscenes/v1.0-mini/`（及 HEDNet 侧 info pkl） | 含 GT，供 AMOTA 与抽查 |
| 旧对照产物 | `DetZero-nuscenes-hednet-qwen3.8flash/output/nus-stage-a-20260831-165710-CST/` | 只读；含废 Waymo 链与自研 track-v2 |

实现第一步：写 `assets_manifest.json`（路径 + sha256 + 帧数），避免 TOCTOU。

### 2.3 类别策略

nuScenes **detection** 10 类与 **tracking** 集合不同。本方案：

- **主交付 / 官方评估**：使用 tracking 挑战常用类别（以 nuScenes 官方 tracking 配置为准，通常为  
  `bicycle, bus, car, motorcycle, pedestrian, trailer, truck` 等；实现时以 `nuscenes.eval.tracking` 的 `TRACKING_NAMES` 钉死）。  
- **检测 10 类中的** `barrier` / `traffic_cone` / `construction_vehicle` 等：  
  - 默认：**不进入 tracking 主交付**（计入 `dropped_class_counts`）；  
  - `construction_vehicle` 是否映射进某 tracking 类：实现前对照官方定义，**禁止静默乱折**。  
- **不再**为迁就 DetZero 而强制折成 Vehicle/Pedestrian/Cyclist（旧链需要时可另开旁路导出，非主交付）。

### 2.4 跟踪器选型与滤波语义

| 候选 | 许可 | 结论 |
| --- | --- | --- |
| **SimpleTrack** (`tusen-ai/SimpleTrack`) | MIT | **主选**：文档含 nuScenes 推理说明，`mot_3d` 库可本地 `pip install -e` |
| Poly-MOT | MIT | **备选**：冲 AMOTA 再换；接口预留 |
| AB3DMOT | 非商用学术条款 | **禁止** |
| 自研 `track_hednet_boxes.py` | — | 仅对照基线，不作为交付引擎 |

**滤波会不会改框（写进合同，避免预期偏差）**：

| 量 | SimpleTrack 典型行为 | 本方案承诺 |
| --- | --- | --- |
| 中心位置 | 运动模型 + 滤波，可与检测中心有差 | 允许；在 manifest 记录「geometry_source=detector+motion_filter」 |
| 速度 | 滤波状态常见 | 写入官方字段（若格式需要） |
| 朝向 yaw | 多跟检测或弱处理 | 不承诺系统性修正 |
| 尺寸 l,w,h | 基本跟检测 | **不承诺修正** |

### 2.5 目录与产物布局（本 worktree）

```
DetZero-nuscenes-simpletrack-trackgt/
  docs/推理复现/
    本方案文档
  third_party/                         # 或 git submodule
    SimpleTrack/                       # upstream MIT，记录 commit
  tools/track_gt/                      # 本方案新增薄封装（名称实现时可微调）
    adapt_hednet_to_simpletrack.py     # S1
    run_simpletrack_nuscenes.py        # S2 调用封装
    export_nusc_tracking_json.py       # 若 SimpleTrack 输出需再转官方 JSON
    eval_nusc_tracking.sh              # S3
    render_track_bev.py                # S4 可选
  output/track-gt-<时间戳>-CST/
    assets_manifest.json
    s1_detections/                     # 适配后的检测中间件
    s2_simpletrack/                    # tracker 原始输出
    delivery/
      tracking_results.json            # ★ 主交付（官方）
      frames_with_track_id.pkl         # ★ 主交付（内部）
      class_drop_manifest.json
    eval/
      metrics_summary.json
      ...
    visuals/                           # 可选
    run_manifest.json                  # 命令、env、commit、耗时
```

**唯一交付目录**：`delivery/`。禁止把实验废产物与交付混放在同一层无说明目录（吸取旧仓 `tracking/` vs `track-v2/` 误导教训）。

### 2.6 步骤详述

#### S0 环境与依赖

- Python：新建或复用隔离 conda env，例如 `nusc-track-gt`（**不要**把 motmetrics 钉死进会破坏其他项目的全局 env）。  
- 关键点：`nuscenes-devkit`、SimpleTrack `requirements`、以及与官方 tracking eval 兼容的 **`motmetrics` 版本**（旧仓曾因 motmetrics 1.4.0 × pandas 2.2 失败；本方案在 S0 用一次性 smoke：对空/最小结果跑通 `evaluate.py` 再开全量）。  
- SimpleTrack：`git clone` 固定 commit → `pip install -e ./`；commit 写入 `run_manifest.json`。  
- 本 worktree 可 symlink：`checkpoints`（若仍需要）、数据根；**不**复制大数据集。

#### S1 检测适配 `adapt_hednet_to_simpletrack.py`

输入：HEDNet `result.pkl` + nuScenes dataroot / version。  

处理：

1. 按 `sample_token` 对齐 keyframe；校验 token ∈ mini。  
2. 框约定：HEDNet 已是 PCDet/lidar `(x,y,z,l,w,h,yaw[,vx,vy])`，**不做** Waymo Open3D-ML 那套 yaw 翻转。  
3. 坐标：按 SimpleTrack nuScenes 文档要求转到其期望系（通常 global 或 lidar——**以 SimpleTrack `docs/nuScenes.md` 为准**，实现时读码钉死，并在 manifest 写 `box_frame=`）。  
4. 类别过滤/映射 → tracking names；丢弃类写计数。  
5. score 阈值：默认沿用旧经验 `0.1` 起，可配置；写入 manifest。  
6. 输出：SimpleTrack 可读的 detection 目录或 pkl/json（格式跟 upstream）。

验收：帧数=81（或所选 scene 子集）；每帧 token 唯一；无空文件。

#### S2 跑 SimpleTrack

- 严格按 upstream nuScenes 推理文档的 config 路径与命令；本仓只留薄 wrapper，**不 fork 改核心算法**（除非 bugfix 并回馈记录）。  
- 输出保留 upstream 原始结构到 `s2_simpletrack/`；再导出到 `delivery/`。  
- 随机性：固定配置；若有并行 worker，记录 worker 数。

#### S3 官方评估

```text
python <nuscenes-devkit>/nuscenes/eval/tracking/evaluate.py \
  delivery/tracking_results.json \
  --output_dir output/.../eval \
  --eval_set mini_val   # 或以 mini 实际 split 名为准，实现时核对 \
  --dataroot <NUSCENES_MINI>
```

- 成功标志：生成官方 summary，无 traceback。  
- 报告中同时附：轨迹条数、平均 track 长度、1 帧轨迹占比（自研统计，便于和旧 track-v2 对照）。  
- **禁止**把检测 BEV mAP 写进「跟踪主结论」标题；若附检测旁证，单独一节标注 `旁证-非G3`。

#### S4 可视化与对照（可选但建议）

- BEV：点云 + 检测框 + 跟踪框（按 id 着色）。  
- 与旧 `track-v2`：同 scene 对比 IDS/碎片率（若旧格式可对齐）；不对齐则只做定性抽查。

### 2.7 与旧 qwen3.8flash 方案的关系

| 旧项 | 本方案 |
| --- | --- |
| Waymo 布局伪装 `waymo_processed_data` | **取消**（跟踪主链不需要） |
| DetZero `run_track` + GRM/PRM | **不跑** |
| 自研 T1/T1-v2 | 对照基线，不进 `delivery/` |
| 假 Waymo 三类 | 改为 nuScenes tracking 类别 |
| G3'：mAP≥det−0.02 | 改为官方 AMOTA 等（G3） |

旧方案文档与完成报告留在旧 worktree，本仓概述中引用路径即可，不复制整份废链命令。

### 2.8 风险与预案

| ID | 风险 | 预案 |
| --- | --- | --- |
| T1 | SimpleTrack 输入坐标/yaw 约定搞反 | S1 后用 GT 中心投影对拍（10～20 框）；错则只改适配层 |
| T2 | motmetrics / pandas 不兼容 | 独立 env 钉版本；S0 smoke 不过不进 S2 全量 |
| T3 | mini 场景短、AMOTA 方差大 | 主结论标注「mini 仅链路+相对对照」；加测单个 trainval scene 需另批数据下载 |
| T4 | HEDNet 漏检导致碎轨 | 属检测上限；报告分开写 detection recall vs tracker frag |
| T5 | 许可污染 | 依赖锁定 + 禁止 AB3DMOT；第三方 commit 进 manifest |

### 2.9 明确不做（详述）

1. 不训练/微调 SimpleTrack 或任何学习式 ReID。  
2. 不引入 AB3DMOT。  
3. 不把 DetZero GRM/PRM/CRM 接回主交付（若未来要「框更准」，另开方案文档，不在本文件范围偷偷加）。  
4. 不改旧 worktree 已跟踪文件；不在本机 `git push` 除非用户明确说可以。  
5. 不 sudo 乱装系统包；长时间占满 GPU 前先停问。  
6. 不对外发布评测数字，除非用户在本线程明确授权。

### 2.10 建议实施顺序（落地 checklist）

1. [ ] S0：建 `nusc-track-gt` env + 装 SimpleTrack（钉 commit）+ tracking eval smoke  
2. [ ] 写 `assets_manifest.json`（HEDNet pkl / dataroot sha）  
3. [ ] 实现 S1 适配 + 单 scene 干跑  
4. [ ] S2 全 mini val  
5. [ ] S3 官方评估 + 写 `eval/README` 解读  
6. [ ] S4 可视化抽查  
7. [ ] 完成报告：`docs/推理复现/nuScenes-SimpleTrack跟踪真值完成报告-grok-<时间戳>.md`  

### 2.11 开放决策（已拍板 / 仍可调）

| 项 | 状态 |
| --- | --- |
| 产品目标 B | **已拍板** |
| 跟踪器 SimpleTrack 优先 | **已拍板**（冲榜改 Poly-MOT） |
| 主指标官方 AMOTA | **已拍板** |
| score 阈值、是否输出预测框（漏检续命） | 实现时按 SimpleTrack 默认 config，变更记入修订记录 |
| 是否 submodule vs 拷贝 third_party | 实现时选一种，写入 run_manifest |

---

## 3. 附录

### 3.1 关键参考

- DetZero 论文/仓库：离线检测+跟踪+精修（本方案仅借用仓作为 worktree 宿主，**主算法不用其精修**）  
- SimpleTrack：https://github.com/tusen-ai/SimpleTrack （MIT）  
- Poly-MOT：https://github.com/lixiaoyu2000/Poly-MOT （MIT，备选）  
- 旧完成报告：`DetZero-nuscenes-hednet-qwen3.8flash/docs/推理复现/DetZero-nuScenes-HEDNet-完成报告-20260831-1710-CST.md`

### 3.2 文档维护约定（给后续 grok / 人工）

1. 每次改方案：更新文首版本号 + 修订记录表（**作者填 grok 或人名 + CST 时间**）。  
2. 命令、路径、阈值变更必须进修订记录，避免「正文与真实命令漂移」。  
3. 完成报告与方案分文件；完成报告同样带 `grok-<时间戳>`。
