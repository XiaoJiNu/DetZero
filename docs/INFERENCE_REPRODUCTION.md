# DetZero 推理复现说明

## 结论

已在本机 `mv2d` 环境中完成 DetZero refining 阶段的真实权重推理，并生成可直接查看的静态可视化。

当前 `checkpoints/` 中只有 3 个类别的 GRM（Geometry Refine Model）和 PRM（Position Refine Model）权重，没有 detection 或 tracking 权重。因此，本次复现覆盖 3 类 × 2 个 refining 模型，共 6 次 checkpoint 前向；不声称覆盖原始点云到检测、跟踪的完整 DetZero 链路。

输入采用确定性生成的最小目标轨迹：每类 5 帧、每帧 320 个点。它用于验证配置、官方特征提取、权重加载、CUDA 前向、世界坐标恢复、结果组合和可视化链路，不是 Waymo 精度评测数据，输出中的合成参考误差不能作为模型精度结论。

## 一键运行

默认输出目录保留了首次运行；完成失败关闭验收器加固后，当前验收候选结果位于 `inference_reproduction_v3`。再次运行时应指定一个不存在的新目录：

```bash
./reproduce_inference.sh \
  --output-dir output/inference_reproduction_v3
```

脚本会在目标目录已存在时拒绝覆盖。可用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--output-dir` | `output/inference_reproduction` | 新的输出目录；必须不存在 |
| `--device` | `cuda` | PyTorch 推理设备 |
| `-h`, `--help` | - | 显示帮助 |

## 本次产物

当前验收候选产物目录：

```text
/data/code/cv/AutoLabel/DetZero/output/inference_reproduction_v3
```

| 文件 | 用途 | SHA-256 |
| --- | --- | --- |
| `results.json` | 运行环境、6 个权重的加载记录、预测框和验证信息 | `0c730b1366721007b298a07495e2f229b04aab85d0dc346f265037f179d4fed5` |
| `results.npz` | 3 类的点云、输入框、GRM/PRM/组合输出和合成参考框，共 18 个数组 | `ef7692eeae5bad5a52687defa7dc52db2893ada49e8db56d55d67534c41ddd38` |
| `visualization.png` | 3 类首帧与全轨迹 BEV 对比图 | `f1f761e03de4ee713b1332bacc879e8a24db6b6f3d566e9ad8e8184f6207b81b` |

直接查看：

```text
/data/code/cv/AutoLabel/DetZero/output/inference_reproduction_v3/visualization.png
```

可视化已独立解码并检查：尺寸为 2240 × 1680，RGB 三通道最小标准差为 39.57，非白色像素比例为 10.61%，不是空白图。

## 权重执行情况

| 类别 | 模型 | checkpoint | 已加载张量 / 模型张量 |
| --- | --- | --- | --- |
| Vehicle | GRM | `checkpoints/vehicle_grm_model.pth` | 104 / 104 |
| Vehicle | PRM | `checkpoints/vehicle_prm_model.pth` | 112 / 112 |
| Pedestrian | GRM | `checkpoints/pedestrian_grm_model.pth` | 104 / 104 |
| Pedestrian | PRM | `checkpoints/pedestrian_prm_model.pth` | 112 / 112 |
| Cyclist | GRM | `checkpoints/cyclist_grm_model.pth` | 104 / 104 |
| Cyclist | PRM | `checkpoints/cyclist_prm_model.pth` | 112 / 112 |

每个模型都要求 checkpoint 键集合与张量形状严格匹配模型；任何缺失键、额外键或形状不一致都会停止运行。旧 checkpoint 中的 NumPy 标量元数据通过 PyTorch `weights_only=True` 白名单方式读取，没有退回任意对象反序列化。

## 环境

本次真实运行环境：

- Python 3.10.20
- PyTorch 2.7.1+cu128
- CUDA 12.8
- NVIDIA GeForce RTX 5090 Laptop GPU，compute capability 12.0
- 解释器：`/data/software/conda/anaconda3/envs/mv2d/bin/python`

已在 `mv2d` 中补齐 `easydict`、`tensorboardX`、`pyyaml`、`tabulate`、`kornia` 和 `addict`，并编译、导入和执行 refining 路径需要的 `iou3d_nms`、`roiaware_pool3d`、`roipoint_pool3d` CUDA 扩展。

## 实现入口

- 一键脚本：`reproduce_inference.sh`
- 推理实现：`refining/tools/reproduce_inference.py`
- checkpoint 兼容修复：`utils/detzero_utils/model_utils.py`
- 集成测试：`tests/test_reproduce_inference.py`
- checkpoint 加载回归测试：`tests/test_model_utils.py`

实现读取仓库原有的 6 份模型 YAML，并调用原有 Waymo geometry/position dataset 的 `extract_track_feature` 与 `collate_batch`，没有自造模型输入张量接口。PRM 结果负责中心和航向，GRM 结果负责尺寸，组合后输出世界坐标 7 自由度框。

## 验证

本次 v3 一键命令退出码为 0；发布后重新读取 canonical 路径，完成以下检查：

- 输出只包含声明的 3 个文件，且不存在 `.UNACCEPTED` 标记；
- 6 个 checkpoint 的类别/模型组合完整且无重复；
- 每个 checkpoint 加载张量数等于模型张量数；
- `results.npz` 使用 `allow_pickle=False` 读取，18 个数组的精确键集合、`float32` 类型、形状、有限值和正尺寸检查通过；
- `results.json` 使用拒绝重复键的解析器复核；
- JSON 中的五组框逐数组绑定 NPZ；GRM 保留输入位姿，组合框使用 PRM 位姿和 GRM 尺寸；
- PNG 可解码、尺寸和非空像素检查通过，并在输出树外从 NPZ 重新渲染后逐字节比对；
- 产物记录的推理脚本、checkpoint loader 和 6 个 checkpoint 哈希均与当前普通文件一致；
- 输出使用 Linux `renameat2(RENAME_NOREPLACE)` 原子无覆盖发布；canonical 路径验证失败时保留 `.UNACCEPTED` 标记。
- 使用两个新的输出目录、两个独立进程，在清除 `PYTHONPATH` 的环境中重复执行；两个 `results.npz` 和两个 `visualization.png` 分别逐字节一致，JSON 除推理耗时外语义一致。

当前 v3 已完成上述本地严格验证和双进程重放。独立终审必须绑定当前代码、测试、文档与 v3 产物哈希；终审返回前不把候选状态表述为终审通过。

测试命令：

```bash
export PYTHONPATH="$PWD/utils:$PWD/refining:$PWD/detection"
/data/software/conda/anaconda3/envs/mv2d/bin/python -m pytest -q
```

当前源码快照的结果为 `49 passed, 12 warnings`。警告来自仓库中一个带构造函数的 `TestTimeAugmentor` 类无法被 pytest 收集，以及当前 Matplotlib/pyparsing 组合的弃用提示；无测试失败。
