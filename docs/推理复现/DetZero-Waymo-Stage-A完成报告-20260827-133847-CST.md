# DetZero Waymo Stage-A 推理复现完成报告

> **报告时间：** 2026-08-27 13:38:47 CST（UTC+08:00）
>
> **工作目录：** `/data/code/cv/AutoLabel/DetZero`
>
> **分支：** `develop-yr`
>
> **基线 HEAD：** `8db6f7dc1e62656b204314db69d424f5a5542681`
>
> **最终被审查源码快照：** 129 files，SHA-256 `6f79e980dac746ce60fdb7d9fea4fdc77065430948fe674a7592f15ca887b007`
>
> **报告性质：** Release-C 冻结后的完成情况说明；本文件不是冻结内的发布凭据，也不改变 fail-closed 裁定。

---

## 0. 结论先行

**推理复现已经完成，发布验收尚未完成。**

当前已经在真实 Waymo testing segment 上完整执行两次独立 Stage-A 推理，生成了 fresh Release-A 和 fresh Release-B。两次运行都完成了：

```text
Waymo TFRecord（199 帧）
  → 全 5 路 LiDAR 预处理
  → Open3D-ML Waymo PointPillars
  → DetZero schema adapter
  → DetZero tracking
  → GRM + PRM refining
  → no-CRM final output
  → 199 帧可视化
  → 单次独立验证
  → A/B 完整路径比较
```

但是，最终 Release-C 独立 trust-boundary 审查复现了 comparator 在最后一次重验后仍存在的 TOCTOU false-success 窗口。因此，当前结果可作为**真实可运行、内容完整、可重复的候选产物**，但不能作为**满足严格不可变发布合同的正式 release**。

| 判定维度 | 当前状态 | 结论 |
| --- | --- | --- |
| 推理链路实现 | `COMPLETED` | 真实 199 帧端到端链路已完成 |
| Fresh Release-A | `PASS` | 独立生成、独立验证完成 |
| Fresh Release-B | `PASS` | 独立生成、独立验证完成 |
| A/B 内容可重复性 | `PASS` | 559-file profile、558-file ledger 完整闭合 |
| 当前候选字节的数据合同 | `PASS_CURRENT_BYTES` | A/B 当前字节均通过 validator |
| 技术可执行性 | `PASS` | 两次 fresh 完整执行证明运行环境可用 |
| 机械发布完整性 | `BLOCKED` | comparator post-revalidation TOCTOU 未关闭 |
| 目标域效果 | `NOT_ESTABLISHED` | 无 3D GT、权威指标和预冻结阈值 |
| 权重发布授权 | `NOT_AUTHORIZED` | detector/refining 权重权利证据未闭合 |
| 最终 `release_eligible` | `false` | deny-overrides 生效 |

**推荐下一步：** 先修复 comparator 的不可变读取/最终提交边界和 Release-C 只读审查问题，再重新冻结源码、重跑 regression、fresh A/B 和 Release-C。未经该轮重跑，不得把当前候选提升为正式 release。

---

## 1. 本次工作的目标与边界

### 1.1 已实现目标

1. 使用一个真实、连续的 Waymo testing TFRecord 完成 199 帧推理。
2. 以 Open3D-ML Waymo PointPillars 作为外部 detector。
3. 将 detector 输出适配为 DetZero tracking 输入。
4. 执行 DetZero tracking、GRM、PRM，并生成 no-CRM final output。
5. 为每帧生成可浏览的静态可视化。
6. 将源码、输入、配置、checkpoint、运行时和生成物绑定到可验证身份。
7. 生成两个事先不存在的独立 A/B generation，并比较完整路径集合。
8. 用 fail-closed 独立审查区分机械完整性、效果和授权状态。

### 1.2 明确不属于本次已完成范围的事项

- 没有 CRM checkpoint，因此 final score 采用 tracking score passthrough。
- 没有目标 segment 的 3D ground truth，不能计算 AP、MOTA 或证明精度。
- 没有覆盖全部 detector/refining checkpoint 的肯定 redistribution/deployment 授权凭据。
- 当前并非完整 DetZero 同权重、同 detector 的官方指标复现。
- 当前 Release-C 没有通过严格机械发布验收。

---

## 2. 已完成的工程实现

### 2.1 正式 Stage-A coordinator

已形成可执行的单命令 Stage-A coordinator，负责：

- preflight 输入、运行时、源码和 Open3D-ML 物理源码身份；
- 创建事先不存在的 generation root 和外部 receipt；
- 依次执行 preprocess、detector、adapter、tracking、refining、final 和 visuals；
- 生成 source bundle、run metadata、run ledger、pre/post audit 和 receipt；
- 失败 generation 保留 `.unaccepted`，不得覆盖或重新密封；
- receipt no-replace 发布并 strict readback 后才移除 `.unaccepted`。

主要入口：

- `tools/external_detector/run_stage_a.py`
- `tools/external_detector/validate_stage_a.py`
- `tools/external_detector/compare_stage_a_runs.py`

### 2.2 外部 detector 接入

已完成：

- Waymo TFRecord 解析；
- 5 路 LiDAR 点云合并；
- Open3D-ML Waymo PointPillars 推理；
- detector config/checkpoint SHA-256 绑定；
- Open3D-ML 173-file 源码树与 commit/tag 绑定；
- detector 原始 NPZ 到 DetZero pickle schema 的显式 adapter。

Open3D-ML 身份：

| 字段 | 值 |
| --- | --- |
| commit | `fcf97c07bf7a113a47d0fcf63760b245c2a2784e` |
| upstream tag | `v0.18.0` |
| source files | 173 |
| source tree SHA-256 | `cc52a3eb417e0fd754d732856fa7eef219985e138e32b3b22a3536b18c8598ca` |
| source license | MIT |

源码 license 只覆盖 Open3D-ML 源码，不自动授予外部分发 checkpoint 的权利。

### 2.3 DetZero tracking/refining/final

已完成：

- tracking 数据集和入口支持当前 Waymo testing generation；
- tracking 输出及 dropped boxes 写入固定 schema；
- Vehicle、Pedestrian、Cyclist 的 GRM/PRM checkpoint 实际加载和 forward；
- 无 CRM 时采用显式 `NOT_EXECUTED_NO_CRM_BY_DESIGN`；
- final output 记录 score passthrough 策略；
- 199 帧最终可视化完整生成。

### 2.4 信任边界加固

已实现并有测试覆盖的主要措施：

- formal profile 不由 generation 自报，绑定 `stage_a_source_policy.json` 和 199-frame 合同；
- closed-world 同时约束文件和目录，拒绝未声明空目录、symlink 和特殊节点；
- formal comparator 默认强制双边 receipt；
- receipt 绑定 role、source tree、ledger、pre/post audit 路径、内容和 hash；
- 外部输入在使用前 snapshot 到 generation 内；
- Open3D-ML 物理源码 tree snapshot 支持两个合法零字节源码文件；
- runtime module origin 必须位于声明的 Open3D-ML root；
- Python bytecode cache 定向 generation 内临时目录并在发布前清理；
- formal pickle 读取改用 bounded、`O_NOFOLLOW`、restricted unpickler；
- NPZ/NPY 使用 `allow_pickle=False`；
- generation 内 provenance 使用相对路径并验证 containment；
- source bundle 在无 `.git` 环境下可执行 preflight 和重新 capture/verify。

---

## 3. 回归与静态门

最终源码快照 `6f79e980...b007` 上获得以下结果：

| 门 | 命令范围 | 结果 |
| --- | --- | --- |
| Stage-A 目标测试 | `tests/test_waymo_external_detector.py` | 71 passed，1 warning |
| 无参数全仓库 pytest | 项目默认 pytest | 120 passed，13 warnings |
| Python 编译检查 | 当前项目 Python 文件 | 829 files passed |
| `git diff --check` | 当前工作树 diff | PASS |
| formal source policy | 129-file roster | PASS |
| formal pickle 静态边界 | 列明的 Stage-A 调用图 | PASS |
| 真实 Open3D-ML snapshot | 173 files，含 2 个零字节源码文件 | PASS |
| bundle preflight | 无 `.git` source bundle | PASS，未生成 run/receipt |

测试使用：

```text
/data/software/conda/anaconda3/envs/mv2d/bin/python
PYTHONDONTWRITEBYTECODE=1
pytest -p no:cacheprovider
--basetemp 位于 /tmp
```

注意：Release-C trust reviewer 执行目标测试时，测试本身在项目 `output/` 下创建并删除了 `.pytest-source-bundle-probe`。内容虽被清理，且正式源码字节未变化，但这仍违反“审查过程不得写项目树”的严格合同，详见第 8 节。

---

## 4. Fresh Release-A

### 4.1 产物

Generation：

`output/waymo-stage-a-release-a-20260827-121010-CST`

Receipt：

`output/waymo-stage-a-receipts/release-a-20260827-121010-CST.json`

独立验证：

`output/waymo-stage-a-audits/release-a-20260827-121010-CST-independent-20260827-122726-CST.json`

正式 launcher：

`output/waymo-stage-a-formal-commands/release-a-20260827-121010-CST.sh`

### 4.2 运行证据

| 字段 | 值 |
| --- | --- |
| role | A |
| destination absent before launch | `true` |
| start | `2026-08-27T12:11:33.310446+08:00` |
| end | `2026-08-27T12:26:37.554131+08:00` |
| elapsed | 904.24 s |
| profile files | 559 |
| ledger payload files | 558 |
| profile tree SHA-256 | `c8d834cbece37bec7e3df7af028d143edae35333ebc78a290c8b13c696f824d0` |
| receipt SHA-256 | `a37f00d42b19dec615f4d527586b345525f9874f2abba3ca4c0da968c9d48b8a` |
| canonical marker | absent |
| symlink/special nodes | 0 |
| undeclared empty directories | 0 |
| producer status | PASS |
| independent artifact audit | PASS |
| release eligible | `false` |

---

## 5. Fresh Release-B

### 5.1 产物

Generation：

`output/waymo-stage-a-release-b-20260827-122900-CST`

Receipt：

`output/waymo-stage-a-receipts/release-b-20260827-122900-CST.json`

独立验证：

`output/waymo-stage-a-audits/release-b-20260827-122900-CST-independent-20260827-125505-CST.json`

正式 launcher：

`output/waymo-stage-a-formal-commands/release-b-20260827-122900-CST.sh`

### 5.2 运行证据

| 字段 | 值 |
| --- | --- |
| role | B |
| destination absent before launch | `true` |
| start | `2026-08-27T12:30:22.463980+08:00` |
| end | `2026-08-27T12:53:08.045864+08:00` |
| elapsed | 1365.58 s |
| profile files | 559 |
| ledger payload files | 558 |
| profile tree SHA-256 | `3b5b9090ec62411c4fab9e924937980d0da5f639de90e9190c9428ab644e358b` |
| receipt SHA-256 | `aa19c0e6b99224812daded3c3056a46dc5e11a34d69a67aea568dc81cd3a8505` |
| canonical marker | absent |
| symlink/special nodes | 0 |
| undeclared empty directories | 0 |
| producer status | PASS |
| independent artifact audit | PASS |
| release eligible | `false` |

A、B 的起止时间、role、generation root、receipt、ledger hash 和多个 volatile stage log hash 均不同，能够证明 B 不是 A 的复制或重新密封。

---

## 6. A/B 完整路径比较

比较报告：

`output/waymo-stage-a-audits/release-a-121010-vs-release-b-122900-comparison-20260827-125505-CST.json`

比较结果：

| 类别 | 数量 |
| --- | ---: |
| expected profile paths | 559 |
| classified ledger paths | 558 |
| exact bytes | 541 |
| semantic JSON | 8 |
| stable metadata | 1 |
| volatile execution | 8 |
| 未分类路径 | 0 |

两边均满足：

```text
expected profile - {run_ledger.json} == ledger path set
```

两边 ledger path set 完全相等；541 + 8 + 1 + 8 = 558，当前比较报告为 `PASS`。

这证明当前 A/B 内容按现有 comparator 规则可重复，但不消除第 8.1 节所述 comparator 返回前 TOCTOU 窗口。

---

## 7. 真实推理结果摘要

A、B 的稳定语义结果一致：

| 指标 | 数值 |
| --- | ---: |
| frames | 199 |
| source points | 37,378,745 |
| detector boxes | 6,736 |
| Vehicle boxes | 5,888 |
| Pedestrian boxes | 671 |
| Cyclist boxes | 177 |
| dropped boxes | 181 |
| tracks | 331 |
| Vehicle tracks | 302 |
| Pedestrian tracks | 22 |
| Cyclist tracks | 7 |
| track-aligned observations | 17,638 |
| visual frames | 199 |
| unique visual images | 199 |

所有 199 帧均有对应静态可视化：

```text
visuals/0000.png
...
visuals/0198.png
```

这些数据证明流水线实际执行并产出了完整候选结果，但它们不是效果指标。

---

## 8. Release-C 独立审查结果

### 8.1 当前确认的机械 blocker

#### `COMPARATOR_POST_REVALIDATION_TOCTOU`

位置：

- `tools/external_detector/compare_stage_a_runs.py:326-347`
- `tools/external_detector/validate_stage_a.py:611-636`
- `tools/external_detector/validate_stage_a.py:1534-1543`

问题：comparator 完成最后一次 child-file validation 后，仍会继续执行 ledger hashing、receipt validation 和 report construction。攻击者可在最后一次 B validation 返回后修改 child file，或重新加入 `.unaccepted`，从而让 comparator 对已经失效的 run 返回 `PASS`。

该问题已有 `/tmp` disposable repro，属于真实 false-success，而不是理论建议。严格发布合同要求 comparator 使用不可变 snapshot、descriptor-bound closure 或等价的事务锁/提交边界，保证判定使用的字节在返回 PASS 时仍是同一组字节。

### 8.2 Release-C 证据闭合问题

#### `FINAL_FREEZE_EVIDENCE_NOT_FULLY_CLOSED`

最终 checker 正确验证了 freeze SHA、129 个 source descriptors、双边 559-file profile、ledger 和 receipt binding，但 external-evidence 比较时错误地把精简实测 descriptor 与 freeze 中包含额外字段的 descriptor 做 whole-object equality，导致 17 个 external entries 出现系统性 mismatch。由于审查被硬停止，没有完成字段归一化后的最终复核。

这不证明 17 个文件当前损坏，但意味着不能声称 Release-C 已独立闭合全部 external descriptor。

#### `COMPARATOR_CLASSIFICATION_NOT_INDEPENDENTLY_RECONCILED`

formal comparator 报告：

```text
541 exact + 8 semantic JSON + 1 stable metadata + 8 volatile = 558
```

独立 checker 错误地主要按 A/B byte-difference 分类，得到 `548/1/1/8`。虽然 558 个路径被覆盖一次，但尚未用 comparator 的正式 policy roster 独立重建同一分类。因此 Release-C 分类复核不完整。

### 8.3 Release-C 只读合同问题

#### `READ_ONLY_AUDIT_SCOPE_VIOLATED_BY_TARGET_TEST`

目标测试会在项目树创建并删除：

`output/.pytest-source-bundle-probe`

最终内容已被清理，源码字节也保持一致，但严格 Release-C 要求审查期间不得在项目树生成或删除任何文件。因此该审查记录为：

```text
snapshot_verified = false
project_writes_made = true
tmp_cleaned = true
```

测试 fixture 必须改为只使用经过验证的 `/tmp` private root，之后重新执行完整只读 Release-C。

### 8.4 仍需当前快照因果复核的安全项

较早、非最终快照审查曾提出：

1. checkpoint hash validation 与底层普通 `torch.load` 之间可能存在 ABA 替换窗口；
2. bundle replay verifier 的可执行字节是否有外部可信固定摘要。

这些较早报告不能直接作为最终 `6f79...b007` 快照的当前 blocker，但最终 Release-C 也没有提供足够因果证据将其关闭。下次发布前必须在当前继任快照上分别做 RED/GREEN probe，不能仅靠源码阅读或成功 A/B 运行推断安全。

---

## 9. Release-C freeze

Freeze：

`output/waymo-stage-a-audits/release-c-20260827-130251-CST-review-freeze.json`

| 字段 | 值 |
| --- | --- |
| freeze SHA-256 | `7d8ba71c79a2c8dd8df26bf775dbdb153edceeb5ecdedc61e0ef1fb0fdf64be0` |
| source files | 129 |
| source tree SHA-256 | `6f79e980dac746ce60fdb7d9fea4fdc77065430948fe674a7592f15ca887b007` |
| explicit frozen descriptors | 150 |
| A profile root | `c8d834cbece37bec7e3df7af028d143edae35333ebc78a290c8b13c696f824d0` |
| B profile root | `3b5b9090ec62411c4fab9e924937980d0da5f639de90e9190c9428ab644e358b` |

Artifact/runtime reviewer 在严格只读模式下重新验证：

- freeze 和 150 个 descriptors 在开始、结束时一致；
- A/B 各 559 个普通文件；
- A/B 各 558 个 ledger payload；
- 无 symlink 或特殊节点；
- A/B in-memory validator 均通过；
- comparator in-memory 返回当前内容 `PASS`；
- 私有 `/tmp` scratch 已清理。

Trust-boundary reviewer 的 adversarial probe 仍复现第 8.1 节 blocker，因此采用 deny-overrides，不能用 artifact reviewer 的 `NO_BLOCKER` 覆盖 trust reviewer 的 blocking finding。

---

## 10. 为什么不能声称效果已经建立

本次使用的 Waymo testing TFRecord 实测：

```text
frames = 199
laser_labels = 0
camera_labels = 0
```

当前 frozen profiles 中不存在：

- 目标域 3D ground truth；
- 权威目标指标；
- 预先冻结的效果阈值；
- 可据此做 PASS/FAIL 的 efficacy report。

因此以下事实不能替代效果证明：

- detector 生成了 6,736 个 boxes；
- tracking 生成了 331 条 tracks；
- refining model 实际 forward；
- 199 帧均有非空可视化；
- A/B 结果可重复。

目标域效果状态必须保持：

```text
NOT_ESTABLISHED_NO_LABELS_METRICS_OR_FROZEN_THRESHOLDS
```

---

## 11. 为什么权重发布授权未建立

当前 detector manifest 明确记录：

```text
upstream-model-zoo-source-recorded; no separate weight license found
```

当前 freeze 没有为下列精确权重 hash 提供肯定的 redistribution/deployment grant：

- PointPillars Waymo detector checkpoint；
- Vehicle GRM/PRM；
- Pedestrian GRM/PRM；
- Cyclist GRM/PRM。

Open3D-ML 源码为 MIT，不代表单独下载或分发的模型权重自动适用 MIT。Waymo 数据仅按本次声明的 non-commercial research 条件使用，也不能替代 checkpoint 权利证明。

因此授权状态为：

```text
NOT_AUTHORIZED_WEIGHT_RIGHTS_EVIDENCE_UNRESOLVED
```

---

## 12. 当前可以如何使用这些结果

### 12.1 可以做

- 作为本机真实 Waymo Stage-A 推理复现结果查看；
- 检查 199 帧静态可视化；
- 研究 detector → tracking → GRM/PRM 的数据合同；
- 复现实验环境中的技术执行；
- 作为后续 detector 替换和安全加固的候选基线；
- 对 A/B 当前字节做只读分析。

### 12.2 不可以宣称

- 不可宣称达到某个 3D detection/tracking 精度；
- 不可宣称优于原始 detector 或官方 DetZero；
- 不可宣称满足商业部署要求；
- 不可宣称拥有 checkpoint 分发或部署授权；
- 不可把当前 A/B 提升为严格机械完整的 canonical release；
- 不可把 comparator 当前 `PASS` 解释为 TOCTOU 已关闭。

---

## 13. 完成发布还需要什么

必须按以下顺序执行：

1. 为 comparator post-revalidation TOCTOU 写最小 RED test。
2. 让 comparator 在不可变 snapshot、descriptor-bound closure 或等价事务边界上完成全部判定。
3. 将 canonical marker 纳入最终 closed-world authority，避免最后重验后的重新注入。
4. 修正 Release-C external descriptor 的归一化比较。
5. 用 comparator 正式 policy roster 独立重建 558-path 分类。
6. 把测试中的项目树 probe 移到 verified private `/tmp`。
7. 在继任快照上复核 checkpoint load ABA 和 bundle verifier 固定摘要。
8. 运行 focused、目标文件、无参数 pytest、py_compile 和静态门。
9. 冻结新的 source identity。
10. 创建全新、事先不存在的 Release-A 和 Release-B；不得修补现有 generation。
11. 分别独立验证 A/B，再比较完整路径集。
12. 创建新的 no-replace Release-C freeze。
13. 完成严格只读 Release-C，并核验审查前后 snapshot。
14. 即使机械门全部通过，若效果和授权仍未建立，保持 `release_eligible=false`。

任何源码、测试、validator、launcher 或合同文档的修改都会产生新的 source identity。现有 A/B 和 Release-C freeze 只能保留为 `6f79...b007` 的历史候选证据，不能重新密封为继任 release。

---

## 14. 最终完成情况

| 工作项 | 状态 |
| --- | --- |
| Waymo 199 帧预处理 | 完成 |
| Open3D-ML PointPillars 推理 | 完成 |
| DetZero adapter | 完成 |
| Tracking | 完成 |
| GRM + PRM | 完成 |
| no-CRM final output | 完成 |
| 199 帧静态可视化 | 完成 |
| 信任边界基础加固 | 完成，但仍有 comparator TOCTOU blocker |
| 回归门 | 完成并通过 |
| Fresh Release-A | 完成并通过单次 artifact audit |
| Fresh Release-B | 完成并通过单次 artifact audit |
| A/B 完整路径比较 | 当前内容通过 |
| Release-C artifact/runtime audit | 当前字节无 blocker |
| Release-C trust-boundary audit | 阻断 |
| 目标域 efficacy | 未建立 |
| checkpoint release authorization | 未建立/阻断 |
| 正式 release eligibility | `false` |

**最终结论：工程推理复现完成；严格发布完成度未达标。当前最重要的未完成事项不是再次运行模型，而是关闭 comparator 的最终 TOCTOU、修复 Release-C 只读审查和证据闭合，然后从新源码身份重新执行完整发布链。**
