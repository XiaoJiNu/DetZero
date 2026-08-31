# DetZero Waymo 阶段 A 会话交接

Updated: 2026-08-26 10:11:14 CST (+0800)
Snapshot ID: 20260826-101114-CST
Supersedes: none
Workspace: `/data/code/cv/AutoLabel/DetZero`
Branch: `develop-yr`
Final release complete: **否**。P0–P4 已形成真实 199 帧候选产物，P5 的 199 张静态图、正式跨阶段验证、第二次独立生成和最终只读审查尚未完成。

## 1. 目标与不可变约束

本任务只完成阶段 A：

```text
单个 Waymo testing segment（199 帧）
→ Open3D-ML Waymo PointPillars
→ DetZero tracking
→ GRM + PRM
→ no CRM
→ 结果、状态记录和每帧静态诊断图
```

必须继续遵守：

- 仅阶段 A；不要提前实现依赖外部 Waymo CenterPoint 权重的阶段 B。
- 用途为非商业研究；用户已确认接受 Waymo 条款。
- testing 段没有 GT，只能证明管线执行、schema、有限值和几何自洽，禁止宣称检测、跟踪或 refining 精度正确。
- detector 已执行但零框、无输入未执行、模型或环境失败必须使用不同状态。
- 无输入类别必须在模型启动前短路为 `NOT_EXECUTED_NO_INPUT`，不得伪造输入。
- CRM 明确不执行；最终分数必须为 tracking score passthrough。
- 正式输出必须使用 fresh/no-replace 路径，禁止覆盖既有候选产物。
- 静态图只作诊断，不是几何正确性的唯一 oracle。
- 不得提交、推送或清理用户工作树，除非用户另行明确要求。

## 2. 权威输入、权重与运行环境

### 2.1 输入段

- TFRecord：`/data/data/automomous/waymo/testing_0000/segment-10084636266401282188_1120_000_1140_000_with_camera_labels.tfrecord`
- bytes：`1,053,628,548`
- SHA-256：`84c960673cfbc55972392c606987e5eef9735321e65ba48adda9958e84140bc5`
- sequence：`10084636266401282188_1120_000_1140_000`
- 帧范围：`0000`–`0198`，共 199 帧

### 2.2 PointPillars

- checkpoint：`/home/yr/Downloads/pointpillars_waymo_202211200158utc_seed2_gpu16.pth`
- checkpoint SHA-256：`080cee14452c376295a2e69e3f3746509a65e3739c714e5e20a145e355c785cd`
- config：`/data/software/venvs/detzero-open3dml/lib/python3.10/site-packages/open3d/_ml3d/configs/pointpillars_waymo.yml`
- config SHA-256：`a9360f7994e577beed4858a19f522d0685bd03d70a80a079f38b126373055666`
- 权重许可状态：上游 model-zoo 来源已记录，但尚未找到 checkpoint 独立权重许可证；这仍是 release blocker。

### 2.3 环境

- 通用测试、tracking、refining：`/data/software/conda/anaconda3/envs/mv2d`
- Open3D-ML detector：`/data/software/venvs/detzero-open3dml`
- Waymo/TensorFlow 预处理：`/data/software/venvs/detzero-waymo-tf213`
- Hermes 配置：`/home/yr/.hermes/config.yaml`
- `agent.max_turns` 已设置并读回为 `500`；需要在新 Hermes 进程中使用。

## 3. 当前工作树

交接前实时状态：

- 分支：`develop-yr`
- 工作树：dirty，未提交。
- tracked diff：11 个文件，约 `100 insertions / 55 deletions`。
- untracked：`output/waymo-stage-a-20260825-181832-CST/`、`tests/test_waymo_external_detector.py`、`tools/`，以及本交接文档。

已修改的 tracked 文件：

- `daemon/prepare_object_data.py`
- `requirements.txt`
- `tracking/detzero_track/datasets/__init__.py`
- `tracking/detzero_track/datasets/data_processor.py`
- `tracking/detzero_track/datasets/waymo_dataset.py`
- `tracking/detzero_track/models/__init__.py`
- `tracking/detzero_track/models/tracking_modules/data_association/data_association.py`
- `tracking/detzero_track/models/tracking_modules/data_association/distance.py`
- `tracking/detzero_track/models/tracking_modules/kalman_filter/kalman_filter.py`
- `tracking/detzero_track/models/tracking_modules/track_manager.py`
- `tracking/tools/run_track.py`

新增 external-detector 文件：

- `tools/external_detector/__init__.py`
- `tools/external_detector/pipeline.py`
- `tools/external_detector/preprocess_waymo_test_segment.py`
- `tools/external_detector/run_open3dml_waymo_pointpillars.py`
- `tools/external_detector/adapt_open3dml_to_detzero.py`
- `tools/external_detector/run_stage_a_refining.py`
- `tools/external_detector/combine_grm_prm_no_crm.py`
- `tools/external_detector/render_waymo_sequence.py`

重要源码闭包缺口：

- `tracking/detzero_track/version.py` 当前存在，30 bytes，但被 `.gitignore:20:*version.py` 忽略。
- source-tree import 依赖该文件；正式 release 前必须用测试先复现 fresh checkout/import 行为，再决定最小修复，不能让正式 source bundle 漏掉它。

## 4. 已完成并真实执行的阶段

### 4.1 P0：199 帧预处理

状态：**完成候选实现与只读检查**。

产物：

- `/data/code/cv/AutoLabel/DetZero/output/waymo-stage-a-20260825-181832-CST/data/waymo`
- manifest：`data/waymo/preprocess_manifest.json`

已确认：

- 199 帧连续覆盖；frame id 为 0–198。
- 首末时间戳为 `1558407840397346` / `1558407860197139`。
- 合并所有 LiDAR 与 returns，并应用 `NLZ != 1`。
- 点云、pose、dtype、有限值和发布路径经过只读检查。

### 4.2 P1：真实 1 帧 detector + adapter

状态：**完成 canary**。

产物：

- `detector-canary-0001/`
- `adapter-canary-0001/`

已确认 Open3D box 到 DetZero 的明确转换，包括：

- Open3D 底面中心到 DetZero 几何中心的 z 修正。
- size 顺序转换为 `[length, width, height]`。
- heading 转换：`wrap(-open3d_yaw - pi/2)`。
- 速度不可用时明确记录 `vx=vy=0`。

### 4.3 P2：连续 10 帧 detector + tracking

状态：**完成 canary**。

产物：

- `detector-canary-0010/`
- `adapter-canary-0010/`
- `tracking-canary-0010/`

先前独立检查结果：

- 10 帧完整。
- 20 条轨迹，184 个轨迹观测。
- 10 个 dropped-detection frame 均保留。
- 实际出现 Vehicle 和 Pedestrian。

已修复 tracking 的前导空检测帧、9 维输入 box、scalar `sample_idx` 和 fresh/no-replace 输出路径问题。

### 4.4 P3：完整 199 帧 detector + adapter

状态：**完成真实候选执行与只读检查**。

产物：

- `detector-full-0199/`
- `adapter-full-0199/`

观测结果：

- detector：199 帧，6,736 个框。
- Vehicle：5,888。
- Pedestrian：671。
- Cyclist：177。
- 每帧最少 18 个框，最多 60 个框，零检测帧数为 0。
- detector CPU 耗时：`472.54365925700404` 秒。
- raw NPZ SHA-256：`1301d5e8ef0e224ed0869ead770476e23bc53d08c3c3a49565f917b4c76c083b`。
- adapter pickle SHA-256：`e574bbedb449fe2e301b12e4486a7da28e8946d763314f8622a51a5d59b2fc7b`。

### 4.5 P4：tracking、object crop、GRM/PRM、no-CRM combine

状态：**完成真实候选执行与只读检查**。

主要产物：

- `tracking-full-0199/tracking.pkl`
- `tracking-full-0199/dropped.pkl`
- `refining-full-0199/`
- `final-full-0199/`（旧版候选，无最终 NPZ）
- `final-full-0199-v2/`（当前诊断候选，含安全 NPZ）

实际 tracking/refining 结果：

| 类别 | 轨迹数 | 轨迹观测数 | GRM forward | PRM forward |
| --- | ---: | ---: | ---: | ---: |
| Vehicle | 302 | 16,306 | 302 | 302 |
| Pedestrian | 22 | 981 | 22 | 22 |
| Cyclist | 7 | 351 | 7 | 7 |
| 合计 | 331 | 17,638 | 331 | 331 |

其他已确认事项：

- GRM/PRM 总 forward 数：662。
- 每个 checkpoint 均 strict load；GRM 为 104/104 tensors，PRM 为 112/112 tensors。
- CRM 状态：`NOT_EXECUTED_NO_CRM_BY_DESIGN`。
- score policy：`TRACKING_SCORE_PASSTHROUGH`。
- final：199 帧，331 条轨迹。
- `final_arrays.npz` 可通过 `np.load(..., allow_pickle=False)` 读取。
- `final_arrays.npz` SHA-256：`fcdc66fdd286ec7023383ac5aff5a47dfc8bd39590260a1aa8a55f736a64fe7b`。
- final frame pickle SHA-256：`c0863e28d8b66a2c17abc87a666e738f76022e50802ca39ff763cac13c13cb52`。
- final track pickle SHA-256：`e297c521f7988faa64d7c3e1f4cf633d5ccdc67b0a0b910a1bfb9db11984a793`。
- 先前临时只读检查发现 2,820 个 object-frame crop 为空点集；模型确实完成 forward，但该现象必须作为质量限制保留，不能宣称有效性已证明。

## 5. 测试证据与当前测试货币性

最新交接前执行：

```bash
PYTHONDONTWRITEBYTECODE=1 /data/software/conda/anaconda3/envs/mv2d/bin/python \
  -m pytest \
  tests/test_waymo_external_detector.py::test_static_renderer_cli_pins_inputs_output_and_frame_count \
  -q -p no:cacheprovider
```

结果：`1 passed in 1.07s`。

其他此前 focused GREEN：

- renderer 数据绑定：不同点云输入会改变像素输出，尺寸为 1280×640。
- no-CRM combine：PRM 提供位置/yaw，GRM 提供尺寸，tracking score 原样透传。
- final safe NPZ：显式 frame offsets，允许空帧，不包含 object array。
- 真实非空类别 GRM/PRM canary。
- tracking 前导空帧与 scalar `sample_idx` 回归。

仍未形成当前源码快照的测试结论：

- `tests/test_waymo_external_detector.py` 全文件尚未在最后两次 renderer/NPZ 修改后重跑。
- 整个仓库 canonical test suite 尚未运行。
- linter/static checks 尚未形成最终证据。
- 因源码仍会变化，已有生成不能作为当前 release 的 source-bound 正式证据。

## 6. 生成账本

| Generation | 状态 | 角色/限制 |
| --- | --- | --- |
| `detector-canary-0001` + `adapter-canary-0001` | 完成 | 真实 1 帧 contract canary |
| `detector-canary-0010` + `adapter-canary-0010` + `tracking-canary-0010` | 完成 | 真实连续 10 帧 tracking canary |
| `detector-full-0199` + `adapter-full-0199` | 完成候选 | 199 帧 detector/adapter；无 generation-local source bundle |
| `tracking-full-0199` | 完成候选 | 199 帧 tracking；当前没有独立 tracking manifest |
| `refining-full-0199` | 完成候选 | 三类别均真实执行 GRM/PRM；CRM 未执行 |
| `final-full-0199` | 历史候选 | 不含 `final_arrays.npz`，不要选为当前候选 |
| `final-full-0199-v2` | 当前诊断候选 | 含 final PKL/NPZ/JSON；尚无静态图、正式 auditor、source bundle 或双跑证明 |
| 199 帧静态图目录 | 不存在 | 交接时递归 PNG 数为 0 |
| 正式 release A/B | 不存在 | 必须在源码冻结后 fresh 生成，不可复制旧目录充当第二次运行 |

交接时无运行中的 bounded producer；此前 full detector 后台进程已退出，exit code 为 0。

## 7. 当前第一个未完成依赖：P5 renderer

`tools/external_detector/render_waymo_sequence.py` 当前状态：

- `render_frame(...)` 已实现并有数据绑定测试。
- CLI 参数已经存在并通过 `--help` 测试。
- `main()` 当前只执行 `parser.parse_args()`；尚未读取 199 帧点云、detector/final pickle，尚未保存 PNG 或 render manifest。
- 交接时没有任何正式 PNG。

下一会话必须继续 TDD，不要直接写完整 producer：

1. 先增加一个最小 2 帧 batch-render 测试，要求 fresh output 下精确生成 `0000.png`、`0001.png` 和 manifest；先观察 RED。
2. 最小实现 batch renderer 和 staged/no-replace publish，使该测试 GREEN。
3. 再运行 `tests/test_waymo_external_detector.py` 全文件。
4. 使用真实 199 帧输入生成新目录，禁止写入或覆盖旧 candidate。
5. 独立验证 199 个精确 frame ID、PNG 解码、1280×640、非空/方差、manifest hash/count、无 HTML，并做至少一个输入敏感性检查。

建议真实输出名仅供未来命令使用，尚不存在：

```text
output/waymo-stage-a-20260825-181832-CST/visuals-full-0199
```

## 8. 后续任务（依赖顺序）

### P5-A：完成 199 帧静态图

通过标准：

- 精确 199 张 standalone PNG，文件名与 frame id 一一对应。
- 每张图同时绑定原始点云、detector boxes 与最终 tracking+GRM+PRM boxes。
- 输出 manifest 记录 frame id、相对路径、bytes、尺寸、SHA-256、点数和两类 box 数。
- 目录内 HTML 文件数必须为 0。
- staged/no-replace 发布；失败不得留下可被误认成完成的 canonical 目录。

### P5-B：实现独立、只读、fail-closed validator

validator 不得导入 producer helper 来重算关键结论，至少独立检查：

- P0–P5 cross-stage sequence/frame identity 和 199 帧 coverage。
- JSON duplicate-key、字段集合与有限数值。
- NPZ 使用 `allow_pickle=False`，shape/dtype/offset/label domain/有限值。
- pose 刚体性、global↔lidar 变换和 heading/velocity 约定。
- detector→adapter box 数与类别映射。
- tracking/refining/final key 对齐。
- PRM 位置/yaw、GRM 尺寸、tracking score passthrough。
- 空类别状态与真实输入计数一致；CRM 确实没有执行。
- 199 张 PNG 完整、可解码且数据层非空。
- 所有 manifest hashes 与真实文件一致。

### P5-C：冻结一键命令和正式源码闭包

在任何正式长跑前完成并测试：

- 一键 launcher/orchestrator。
- renderer、validator、tests、配置和 contract 文档。
- `tracking/detzero_track/version.py` fresh-checkout/source-bundle 闭包问题。
- generation-local immutable source bundle 与 per-file/aggregate manifest。
- source start/end equality gate。
- 环境/依赖身份记录。

不得把当前 mutable workspace 的后验 hash 附到旧产物并称为生成时源码证明。

### Release-A：当前源码的 fresh 正式全链生成

通过标准：

- fresh 不存在的输出路径。
- P0→detector→adapter→tracking→object crop→GRM/PRM→no CRM→NPZ/JSON→199 PNG 全部由冻结源码生成。
- post-publish 独立 validator PASS。
- testing 段无 GT 的限制和 checkpoint license blocker仍明确记录。

### Release-B：第二次独立生成与比较

通过标准：

- 第二个 fresh 不存在路径和独立 run receipt；禁止复制 A。
- 比较所有 branch-sensitive arrays、状态、frame IDs、classes、轨迹键和 PNG。
- 对连续浮点量使用预先声明的容差并报告差异；决策/计数不得变化。
- A、B 各自先满足 schema/source/provenance gate，再进行比较。

### Release-C：最终检查与只读审查

- 在正确 runtime 下运行 focused、全文件和 canonical suite。
- 执行静态检查。
- 冻结最终 dirty-worktree allowlist/hash。
- 进行独立只读代码与产物审查；异步 reviewer 未返回不算通过。
- 任何 reviewer 驱动源码修复均必须 RED→GREEN，并重新生成受影响的正式产物。
- 不提交、不推送，除非用户明确要求。

## 9. 下一会话恢复步骤

先读取，不要先修改：

```bash
cd /data/code/cv/AutoLabel/DetZero
hermes config get agent.max_turns
git branch --show-current
git status --short
```

然后读取：

```text
docs/handoff/detzero-waymo-stage-a-handoff-20260826-101114-CST.md
docs/推理复现/单Waymo测试段与CenterPoint接入DetZero方案-5.6pro中.md
tools/external_detector/render_waymo_sequence.py
tests/test_waymo_external_detector.py
output/waymo-stage-a-20260825-181832-CST/final-full-0199-v2/final_manifest.json
```

状态建立测试：

```bash
PYTHONDONTWRITEBYTECODE=1 /data/software/conda/anaconda3/envs/mv2d/bin/python \
  -m pytest \
  tests/test_waymo_external_detector.py::test_static_renderer_cli_pins_inputs_output_and_frame_count \
  -q -p no:cacheprovider
```

下一条开发动作：为 2 帧 batch renderer 写一个会因尚未生成 PNG/manifest 而失败的测试，然后做最小实现使其通过。

建议新会话首条指令：

```text
请读取 /data/code/cv/AutoLabel/DetZero/docs/handoff/detzero-waymo-stage-a-handoff-20260826-101114-CST.md，核对真实工作树和产物后，从 P5 renderer 的第一个 RED 测试继续；不要重做已完成 canary，不要只总结，不要提交代码。
```

## 10. 限制与禁止声明

- 无 testing GT：不得宣称 detector/tracking/refining 精度正确或优于基线。
- 2,820 个 object-frame crop 为空点集，仍需作为质量风险呈现。
- PointPillars checkpoint 独立权重许可未闭合，release readiness 不能通过。
- `final-full-0199-v2` 是诊断候选，不是 source-bound、双跑、独立审查通过的正式 release。
- renderer 尚未生成 199 张图。
- 当前没有完整测试集、当前源码的 fresh E2E、第二次独立执行或最终 reviewer verdict。

## 11. Release 判断

**阶段 A 的真实计算链 P0–P4 已跑通并生成 199 帧候选结果，但用户要求的完整阶段 A release 尚不存在；P5 静态图、正式独立验证、源码闭包、双跑比较和最终只读审查全部完成之前，状态必须保持 `NOT_RELEASED`。**
