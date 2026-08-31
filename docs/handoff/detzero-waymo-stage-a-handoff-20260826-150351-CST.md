# DetZero Waymo Stage-A 交接（2026-08-26 15:03:51 CST）

## 结论先行

当前仍是 **P5-C 源码闭包/真实 start-end tracer 阶段**；Release-A、Release-B、Release-C 均未完成。没有后台 producer。两次名为 release-a 的路径都是明确失败、保留 `.unaccepted` 的诊断尝试，禁止晋升或复用。

最新代码状态是一个有意义的 RED：refining manifest 仍写绝对内部路径，导致 A/B 根目录不同即 manifest/hash 不同。测试已先行要求 `tracking.path == ../tracking.pkl`、`waymo_root == ../waymo`，生产代码尚未修。

## 仓库与进程快照

- 仓库：`/data/code/cv/AutoLabel/DetZero`
- 分支：`develop-yr`（跟踪 `origin/develop-yr`）
- HEAD：`8db6f7dc1e62656b204314db69d424f5a5542681`
- 工作树：dirty；含用户既有修改与本任务新增/修改，禁止 reset、stash、覆盖或清理。
- 当前后台进程：无。
- 已退出进程：
  - `proc_d275e22e94e6`，exit `-15`，主动停止，因为 frozen bundle 中 refining 必然找不到外部 checkpoint。
  - `proc_eb977f06c820`，exit `1`，tracking 默认 `SPLIT=val` 读取不存在的 `waymo_infos_val.pkl`。

## 当前真实进展

### 已通过

1. `validate_stage_a.py` 顶层已接通：
   - `--run-root`
   - source bundle/current workspace 校验
   - 三 runtime bundle replay（preprocess / detector / pipeline）
   - run ledger
   - transactional failure report
2. source bundle 闭包已加入：
   - git-ignored DetZero CUDA `.so`
   - `.gitignore` 与 `LICENSE`
   - virtualenv entrypoint 保留（不再 `resolve()` 成 base interpreter）
   - 独立 preprocess Python：`/data/software/venvs/detzero-waymo-tf213/bin/python`
3. refining 已显式接收并使用外部、preflight 已哈希的 checkpoint root：`/data/code/cv/AutoLabel/DetZero/checkpoints`。
4. tracking 根因已修：Stage-A launcher 显式传 `--split test` 与 `--root_path`；通用 tracking CLI 的默认 `val` 未被破坏。
5. 当前代码对失败 run 已完成最小真实下游 tracer：
   - tracking：`/tmp/detzero-tracking-probe-20260826-1449`
   - refining：`/tmp/detzero-refining-probe-20260826-1450`
   - final：`/tmp/detzero-final-probe-20260826-1451`
   - 199 PNG：`/tmp/detzero-visuals-probe-20260826-1451`
6. 独立 stage validators 对该拼接 tracer 全部通过：
   - 199 frames
   - 37,378,745 points
   - 6,736 detector boxes
   - 331 tracks / 17,638 observations
   - 181 dropped boxes
   - 199/199 unique PNG
7. 新增 `tools/external_detector/compare_stage_a_runs.py`：
   - 两侧先独立验证 closed-world run ledger
   - 每个 ledger path 必须归入 exact、semantic JSON、stable metadata 或显式 volatile execution evidence
   - source tree 必须相同
   - A/B role 必须分别为 A/B
   - 重封 ledger 后修改 payload 的 negative 测试会失败
   - comparator 已加入 producer 与 validator 的 bundle replay entrypoints。

### 最新测试证据

- 在 comparator / tracking / refining 相对路径 RED 之前，完整 focused 文件曾为：
  - `54 passed in 13.71s`
- tracking 修复 focused：
  - `2 passed, 52 deselected`
- comparator focused：
  - `1 passed`
- comparator bundle replay static focused：
  - `1 passed`
- 当前权威状态为 RED（旧 broad GREEN 已失效）：
  - `tests/test_waymo_external_detector.py::test_stage_a_refining_runs_real_models_only_for_nonempty_classes`
  - 预期：`../tracking.pkl`
  - 实际：绝对 `/tmp/.../tracking.pkl`

## 两次失败诊断尝试（只读保留）

### 尝试 1

- 命令冻结：`output/waymo-stage-a-formal-commands/release-a-20260826-142946-CST.sh`
- 目录：`output/waymo-stage-a-release-a-20260826-142946-CST`
- 状态：主动停止，`.unaccepted` 保留；无成功 receipt；不得复用。

### 尝试 2

- 命令冻结：`output/waymo-stage-a-formal-commands/release-a-20260826-143643-CST.sh`
- 目录：`output/waymo-stage-a-release-a-20260826-143643-CST`
- 失败 receipt：`output/waymo-stage-a-receipts/release-a-20260826-143643-CST.json`
- 状态：exit 1，tracking 误走 val/GT；`.unaccepted` 保留；不得复用。
- 已成功且可作为诊断输入的阶段：preprocess、PointPillars detector、adapter。

## 当前正在实现的 RED→GREEN

目标：让 `refining/refining_manifest.json` 对同一 sibling run tree 可搬迁且 A/B 字节稳定。

最小修改：

1. `tools/external_detector/run_stage_a_refining.py`
   - 用 stdlib `os.path.relpath(..., start=output_dir)` 写入：
     - `tracking.path`
     - `waymo_root`
2. `tools/external_detector/validate_stage_a.py`
   - 绝对路径保持兼容；相对路径必须以 `refining_dir` 为基准解析并 containment/regular-file 检查。
3. 将 `test_no_crm_combiner_publishes_track_aligned_full_frame_output` 的 hand-authored refining manifest 改为相对路径，证明 validator 端真实读取该格式。

随后依次运行：

```bash
PYTHONDONTWRITEBYTECODE=1 /data/software/conda/anaconda3/envs/mv2d/bin/python -m pytest \
  tests/test_waymo_external_detector.py::test_stage_a_refining_runs_real_models_only_for_nonempty_classes \
  tests/test_waymo_external_detector.py::test_no_crm_combiner_publishes_track_aligned_full_frame_output \
  -q -p no:cacheprovider

PYTHONDONTWRITEBYTECODE=1 /data/software/conda/anaconda3/envs/mv2d/bin/python -m pytest \
  tests/test_waymo_external_detector.py -q -p no:cacheprovider
```

## 剩余发布顺序（不得跳步）

1. 完成上述相对 provenance GREEN。
2. 检查 comparator 对真实 manifest 的 canonical/exact 分类，不以宽泛忽略项掩盖差异。
3. 跑最新 focused 全文件与 `py_compile`。
4. 重新生成全新、不可存在的正式 Release-A 命令/目录/receipt；先实际 semantic preflight，再 full 199 全链。
5. Release-A 必须 exit 0、无 `.unaccepted`、pre/post validator 报告均 PASS、receipt PASS。
6. 源码冻结不变，生成独立 fresh Release-B；同样完整运行与 receipt。
7. 用 bundled comparator 比较全部 release-relevant ledger paths；不能只比 final pickle。
8. 正确 runtime 下跑 focused、全文件、canonical no-argument pytest、`py_compile`/静态检查。
9. 冻结 dirty-worktree task allowlist + 每文件 SHA-256，做独立只读 Release-C 审查；审查 callback 未返回前不得宣称完成。
10. 机械完整性与 release eligibility 分开；预计保留：
    - `NOT_EVALUATED_NO_GROUND_TRUTH`
    - `POINTPILLARS_WEIGHT_LICENSE_UNRESOLVED`

## 关键约束

- 历史 `output/waymo-stage-a-20260825-181832-CST` 永远只读，不能作为 Release-A。
- 所有正式/失败 generation 路径均不可覆盖、不可修补、不可重封、不可删除。
- producer 运行后冻结完整 source cone；运行中禁止编辑源码/测试/validator/comparator/docs。
- source bundle replay 禁止借用 live repo 第一方代码。
- 主 Python：`/data/software/conda/anaconda3/envs/mv2d/bin/python`
- detector Python：`/data/software/venvs/detzero-open3dml/bin/python`
- preprocess Python：`/data/software/venvs/detzero-waymo-tf213/bin/python`
- 使用 `PYTHONDONTWRITEBYTECODE=1`；pytest 使用 `-p no:cacheprovider`。
- 凭据不得进入报告/交接；如发现必须写为 `[REDACTED]`。
- Ponytail full：复用现有 validator/stdlib；不新增抽象、依赖或 shell wrapper，除非真实合同缺口证明需要。
