# DetZero Waymo Stage-A 续作交接（2026-08-26 13:08:13 CST）

## 边界与身份

- 工作区：`/data/code/cv/AutoLabel/DetZero`
- 分支：`develop-yr`，跟踪 `origin/develop-yr`
- HEAD：`8db6f7dc1e62656b204314db69d424f5a5542681`
- 本文档为 `docs/handoff/detzero-waymo-stage-a-handoff-20260826-123957-CST.md` 的后继；不得覆盖任一既有交接。
- 创建时没有后台进程。
- 发布链尚未冻结；Release-A/Release-B 尚未启动，因此当前仍可修改源码/测试。

## 当前 verdict

- P5-A：完成。
- P5-B：实现完成并有 fresh focused/full-file 证据；尚需在正式 fresh generation 上获得正向 post-publish PASS。
- P5-C：进行中，source bundle/环境身份/start-end gate/一键 launcher 尚未实现。
- Release-A/B/C：未完成。
- 当前没有可发布 generation。历史 `20260825` 目录仅用于诊断，不能代表当前源码。
- 效果门仍为 `NOT_EVALUATED_NO_GROUND_TRUTH`；PointPillars 权重许可证仍为 `UNRESOLVED`，均必须继续阻断 release eligibility，不能被技术完整性 PASS 覆盖。

## 已完成且已验证

### P5-A renderer

- `tools/external_detector/render_waymo_sequence.py` 已支持真实 batch renderer。
- `output/waymo-stage-a-20260825-181832-CST/visuals-full-0199` 已独立验证：
  - `frame_count=199`
  - `unique_image_count=199`
  - 连续文件 `0000.png`–`0198.png`
  - 每张 PNG 解码、尺寸、哈希、帧身份、点层与 detector/final 框绑定均通过。
- 注意：正确 adapter 文件是 `adapter-full-0199/detzero_result.pkl`，不是前一交接里误写的 `detzero_detector_frames.pkl`。

### P5-B validator

新增 `tools/external_detector/validate_stage_a.py`，不导入 producer helper，现覆盖：

- duplicate-key-rejecting JSON；
- preprocess 文件闭包、199 帧 roster、点云 dtype/shape/finiteness/NLZ、info/manifest/input 绑定；
- detector NPZ 安全解析、adapter 精确 replay、真实 producer manifest schema；
- dropped detections 对 adapter 的逐帧 exact multiset-subset replay；
- tracking/refining/final 的 track/frame/PRM/GRM/no-CRM/score passthrough/NPZ/manifest/hash 闭包；
- 199 PNG exact coverage、hash、decode、尺寸、非平凡像素和点层绑定；
- transactional no-replace JSON audit report。

真实历史链审计结果：

- 报告：`output/waymo-stage-a-audits/historical-20260826-130430-CST-validation.json`
- `passed=false`
- `preprocess_detector_adapter` 与 `dropped_frames` 通过；
- 阻断点：`ValueError: refining output is not closed-world`。
- 原因：旧 refining producer 保留了 6 个未在 manifest outputs 中声明的 `Vehicle/Pedestrian/Cyclist` 中间 pickle。当前 `run_stage_a_refining.py` 已改为每类消费完成后删除中间目录；必须 fresh run，不得修补历史目录。

### 当前 fresh 测试证据

最后一次完整外部流水线测试文件：

```bash
PYTHONDONTWRITEBYTECODE=1 /data/software/conda/anaconda3/envs/mv2d/bin/python -m pytest tests/test_waymo_external_detector.py -q -p no:cacheprovider
```

结果：`37 passed in 5.81s`。

最后一次语法检查：

```bash
PYTHONDONTWRITEBYTECODE=1 /data/software/conda/anaconda3/envs/mv2d/bin/python -B -m py_compile tools/external_detector/validate_stage_a.py tools/external_detector/run_stage_a_refining.py tools/external_detector/pipeline.py
```

结果：exit 0。

## 本轮实际修改/新增

本续作至少涉及：

- `tests/test_waymo_external_detector.py`
- `tools/external_detector/validate_stage_a.py`（新增）
- `tools/external_detector/pipeline.py`（空帧 `name` 固定为 `<U10`，避免 adapter dtype 不确定）
- `tools/external_detector/run_stage_a_refining.py`（消费后删除未声明中间目录）
- `docs/handoff/detzero-waymo-stage-a-handoff-20260826-123957-CST.md`
- 本交接文档

工作树还包含此前任务的 11 个 tracked 修改文件以及整个 untracked `tools/`、测试、历史 output；不得 reset/stash/删除用户工作。Git 的简短状态见本文件创建前 live 查询，HEAD 如上。

## P5-C 尚需完成

1. 查清并修复 `tracking/detzero_track/version.py` 的 fresh-checkout/source-bundle 缺失闭包；不得依赖生成后或外部未声明文件。
2. 用严格 TDD 实现最小 source capture/verify：
   - 显式 allowlist/闭包；
   - 拒绝 symlink/special/path escape；
   - 复制 generation-local immutable bytes；
   - canonical per-file SHA-256 + aggregate tree SHA-256；
   - start/end 对同一冻结集合重算；
   - embedded bundle 与 live workspace currency 分离；
   - bundle-only `--help` replay，不从 live repo 借 first-party imports。
3. 冻结环境身份：至少 interpreter 路径/hash/version、关键 distributions/versions、CUDA/torch/Open3D-ML、detector/refiner checkpoints/config hashes；不要把凭据写入任何 receipt。
4. 实现一个真实的一键 Stage-A launcher，固定顺序：capture source → preprocess → detector → adapter → tracking → refining → final combine → 199 PNG → end gate → immutable receipt/ledger → post-publish validator。
5. 输出必须 fresh/no-replace；A、B 必须是两个不同目标且有独立外部 receipts，不能通过复制目录伪造第二次执行。
6. 在 source bundle 发布到 repo 下之前，验证 canonical pytest 不会递归收集 `output/.../source_bundle/.../tests`；优先 `norecursedirs = output`，但先核对现有 runner 配置和 canonical test locations。

## 下一步字面命令

先重新读取当前源码/调用链，不依赖旧摘要：

```bash
cd /data/code/cv/AutoLabel/DetZero
git status --short --branch
```

然后搜索 version/import/runner 配置和一键入口：

```bash
# 使用 Hermes search_files/read_file，不要用 grep/find/cat。
```

首个 P5-C RED 应验证 source bundle 独立回放时 `tracking/detzero_track/version.py` 或其生成依赖缺失会失败；再做最小 root-cause 修复。每个行为严格 RED→GREEN，随后重跑：

```bash
PYTHONDONTWRITEBYTECODE=1 /data/software/conda/anaconda3/envs/mv2d/bin/python -m pytest tests/test_waymo_external_detector.py -q -p no:cacheprovider
```

在 launcher/source/validator/test/contract 文档全部完成并 GREEN 前，不得启动 Release-A。启动后必须冻结源码锥，禁止边跑边改；如发现缺陷，保留失败 generation，修复后改用全新目标重跑。

## 剩余发布门

- P5-C source/runtime/launcher/start-end gate。
- Release-A fresh full 199 + canonical-path post-publish validator。
- Release-B 第二次独立 fresh full 199 + 外部 run receipt + 全 release-relevant comparator。
- Release-C focused/full/canonical/static。
- 冻结 dirty-worktree task allowlist/hash，完成独立只读 code+artifact review；任何后续源码/测试/launcher/validator 变更都会使 verdict stale。

Final artifact exists: no
