# DetZero Waymo Stage-A 续作交接（2026-08-26 12:39:57 CST）

## 状态

- 工作区：`/data/code/cv/AutoLabel/DetZero`
- 分支：`develop-yr`（跟踪 `origin/develop-yr`）
- 原始交接：`docs/handoff/detzero-waymo-stage-a-handoff-20260826-101114-CST.md`
- 本文档：因长任务在上下文压缩后仍未完成而提前保留的、不可覆盖的中间交接；不是 Release PASS。
- 当前没有已知后台 producer 进程。
- 用户已有 dirty worktree，禁止 reset/stash/覆盖；不得 commit/push，除非用户另行要求。

## 当前任务结论

Release 尚未完成。历史目录 `output/waymo-stage-a-20260825-181832-CST` 只能作为诊断输入；它不是当前源码绑定的 fresh/no-replace Release-A。

### 已完成并有当前证据的工作

1. 已读取原始交接、方案文档、producer/manifest 代码和真实 199 帧产物结构。
2. `tests/test_waymo_external_detector.py` 的既有 batch renderer 测试此前通过，完整外部检测器文件此前为 `33 passed`；这些结果早于本轮 validator 变更，不能作为最终 Release-C 证据。
3. 历史 199 帧可视化已实际生成到：
   - `output/waymo-stage-a-20260825-181832-CST/visuals-full-0199`
   - 文件名表面覆盖 `0000.png`–`0198.png`，且有 `render_manifest.json`。
   - 仍需用当前独立 validator 完成逐张解码、哈希、点层数据绑定和 closed-world 验证后，P5-A 才能完成。
4. 新增 `tools/external_detector/validate_stage_a.py`，目前已实现并分别通过的 TDD 纵向切片：
   - 严格 JSON：拒绝重复键和非有限数；focused：`1 passed`。
   - detector NPZ → adapter PKL 独立几何/类别/分数重放；focused：相关两项 `2 passed`。
   - tracking → GRM/PRM → no-CRM final 独立键对齐、代数、global→lidar、final NPZ/manifest 重放；focused：`1 passed`。
5. 修复 `tools/external_detector/pipeline.py` 的真实空帧 schema 根因：adapter 的空 `name` 数组现在固定为 `<U10`，不再退化为 `float64`；对应测试先 RED 后 GREEN。
6. `validate_visuals(...)` 的测试已先观察到缺失符号 RED，随后已实现；截至本文档写入时尚未执行 GREEN，因此当前证据状态是“implemented, untested”。

## 本轮任务自有改动

- `tools/external_detector/validate_stage_a.py`（新增，仍在开发）
- `tools/external_detector/pipeline.py`（空帧 name dtype 修复）
- `tests/test_waymo_external_detector.py`（validator TDD 与空帧回归）

仓库中其余 modified/untracked 文件来自更早阶段或用户工作，不能据此推断任务所有权。当前 `git status` 仍显示 11 个 tracked modified 文件，以及 `docs/handoff/`、`output/...`、`tests/test_waymo_external_detector.py`、`tools/` 等 untracked 路径。

## 当前测试证据（按最新源码有效性）

- PASS：`test_stage_a_validator_rejects_duplicate_json_keys`
- PASS：adapter schema + detector/adapter boundary 两项 focused
- PASS：`test_no_crm_combiner_publishes_track_aligned_full_frame_output`
- RED→已实现但未复测：`test_static_renderer_batch_publishes_two_frames_and_manifest` 中的 `validate_visuals(...)`
- 未运行：当前源码下完整 `tests/test_waymo_external_detector.py`
- 未运行：canonical suite、静态检查、真实 199 帧当前 validator

## 尚未闭合

1. P5-A：运行 `validate_visuals` 对真实 199 PNG 做 closed-world、逐张 PNG 解码/尺寸/方差、点文件哈希、点层像素绑定、输入敏感性验证。
2. P5-B：补齐顶层 CLI/事务性失败报告、preprocess/detector/adapter/refining/render manifest 的 exact-field/hash 链、dropped-frame 语义、source/run receipt gate，以及语义篡改 adversarial 测试；再对真实产物运行。
3. P5-C：修复 fresh checkout 缺失 `tracking/detzero_track/version.py`；实现最小一键 launcher、完整 transitive source closure、generation-local source bundle、环境身份、start/end source gate 和 no-replace 发布。
4. Release-A：冻结源码后在全新目录运行完整 199 帧链，并 post-publish 验证。
5. Release-B：第二个全新目录独立运行，保存不可伪造为复制的独立 receipt，并比较全部 release-relevant 产物。
6. Release-C：当前冻结源码下 focused、全文件、canonical suite、静态检查和最终独立只读审查。
7. Release eligibility 必须保持 fail-closed：Waymo 无 GT，且 Open3D-ML PointPillars 权重许可/训练来源仍未解决，所以即使结构审计通过也不得声称 production release eligible。

## 关键风险

- `validate_stage_a.py` 目前仍读取兼容性 PKL；对外部不可信输入这不是安全反序列化。正式 release 必须由 source/hash/closed-world 边界把这些 PKL 限定为本次受控 producer 输出，并以安全 NPZ 为主要消费格式；不能把 PKL 结构有效等同于安全。
- 任何后续 producer/validator/test/launcher/doc/source-cone 修改都会使此前 real-run 证据对“当前 release”失效；必须先冻结再执行 Release-A。
- 不得在历史目录补写/reseal 以伪装 source-bound release；Release-A/B 必须是新的不存在路径。
- 完整 199 帧 detector/refining 运行耗时较长；启动前必须先闭合 validator/source bundle/launcher，预留 postflight 与交接容量。

## 下一步命令（从仓库根目录执行）

```bash
PYTHONDONTWRITEBYTECODE=1 /data/software/conda/anaconda3/envs/mv2d/bin/python -m pytest tests/test_waymo_external_detector.py::test_static_renderer_batch_publishes_two_frames_and_manifest -q -p no:cacheprovider
```

若 GREEN，立即运行当前文件级回归：

```bash
PYTHONDONTWRITEBYTECODE=1 /data/software/conda/anaconda3/envs/mv2d/bin/python -m pytest tests/test_waymo_external_detector.py -q -p no:cacheprovider
```

然后用 `validate_visuals(...)` 对以下真实路径执行只读验证：

- Waymo root：`output/waymo-stage-a-20260825-181832-CST/data/waymo`
- detector frames：`output/waymo-stage-a-20260825-181832-CST/adapter-full-0199/detzero_detector_frames.pkl`
- final frames：`output/waymo-stage-a-20260825-181832-CST/final-full-0199-v2/final_frame_grm_prm_score_passthrough.pkl`
- visuals：`output/waymo-stage-a-20260825-181832-CST/visuals-full-0199`
- expected frames：`199`

## 完成判据

只有以下全部成立才能结束：P5-A/B/C 完整、fresh Release-A postflight PASS、独立 fresh Release-B 有可信 receipt 且 release-relevant 比较通过、当前源码完整测试/静态检查通过、冻结 allowlist/hash 后的独立只读审查返回当前 verdict；并明确把技术完整性与许可/GT release eligibility 分开报告。
