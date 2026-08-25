# DetZero 完整推理资产核查

## 结论

当前仓库已经完整且无损地包含 DetZero 官方公开发布的全部模型权重，但这些公开资产不足以执行论文意义上的完整 `detection → tracking → GRM → PRM → CRM` 推理链路。

- 官方 Google Drive 实际仅发布 3 个类别的 GRM/PRM，共 6 个 checkpoint。
- 已把官方目录重新下载到隔离临时目录，并对官方副本与本地 `checkpoints/` 逐文件比较字节数和 SHA-256；结果为 `6/6` 完全一致。
- DetZero 官方没有发布 detection checkpoint；公开请求 [issue #49](https://github.com/PJLab-ADG/DetZero/issues/49) 仍为 open。
- DetZero 官方没有发布 CRM checkpoint；公开请求 [issue #66](https://github.com/PJLab-ADG/DetZero/issues/66) 仍为 open。
- tracking 阶段是 Kalman filter、数据关联和后处理流程，不需要学习型 checkpoint。
- 本机没有 Waymo TFRecord、DetZero 处理后的 Waymo 数据或可用的 Waymo detection 结果。
- 本机另有一个 nuScenes CenterPoint checkpoint，但其数据域、类别/head 契约与 DetZero Waymo detector 不兼容，因此不能作为完整复现权重。

因此，当前不能诚实地把已有 refining v3 结果描述成完整 DetZero 复现，也不能用随机初始化、nuScenes 权重或未经授权的第三方文件补齐缺口。

## 官方公开权重核验

机器可读清单位于 [INFERENCE_ASSET_MANIFEST.json](INFERENCE_ASSET_MANIFEST.json)。

| checkpoint | 字节数 | SHA-256 | 本地状态 |
| --- | ---: | --- | --- |
| `cyclist_grm_model.pth` | 16,381,633 | `92c101f98340d4f5bcf746f47cf5290bab734e6ec846f1404fcde47c86139990` | 与官方副本一致 |
| `cyclist_prm_model.pth` | 14,711,261 | `95f2bde035f1faddff62c24adc787d97ddf64f73251b05bba5da3cbc9b6995b0` | 与官方副本一致 |
| `pedestrian_grm_model.pth` | 16,381,633 | `f47cf8f5e9516a12a80fb77133317bf1c1ed8a570fb7025f8ba92ddb9789d385` | 与官方副本一致 |
| `pedestrian_prm_model.pth` | 14,702,045 | `31d2382f50cb02e786a3bd878a001ce984d48cbcbed829edbea427174da9a69f` | 与官方副本一致 |
| `vehicle_grm_model.pth` | 16,381,633 | `05f3b293d70f9b6a24486a9cfb93d9cb5a053f3987f65ffe11bc6ba44d2ba599` | 与官方副本一致 |
| `vehicle_prm_model.pth` | 14,702,045 | `32c757c1900bfe263a8b3786b91575092c35770dfc3b0b000a4b3636c0204fe5` | 与官方副本一致 |

官方来源：

- DetZero 模型目录：<https://drive.google.com/drive/folders/1SUzMox9oNte_DYkeceERDSNMWvKKUeGn>
- DetZero 仓库：<https://github.com/PJLab-ADG/DetZero>

官方 README 声明发布模型遵循 Waymo Open Dataset 的非商业使用条款。

## 完整推理还需要的模型资产

| 阶段 | 所需资产 | 当前状态 | 说明 |
| --- | --- | --- | --- |
| Detection | 与 `detection/tools/cfgs/det_model_cfgs/centerpoint_*.yaml` 严格兼容的 Waymo CenterPoint checkpoint | 缺失、DetZero 未公开 | 完整链路的首个必要学习模型 |
| Tracking | 无学习权重 | 源码可用 | `tracking/tools/cfgs/tk_model_cfgs/waymo_detzero_track.yaml` 配置 Kalman filter、两阶段关联、反向跟踪与后处理 |
| GRM | Vehicle/Pedestrian/Cyclist 三个 checkpoint | 已有且与官方一致 | 已在 refining v3 中真实前向 |
| PRM | Vehicle/Pedestrian/Cyclist 三个 checkpoint | 已有且与官方一致 | 已在 refining v3 中真实前向 |
| CRM | Vehicle/Pedestrian/Cyclist 三个 checkpoint | 缺失、DetZero 未公开 | 完整 confidence refinement 和最终 score 合并需要 |

原始 CenterPoint 的 Waymo 模型也不是匿名公共下载：其官方 Waymo model-zoo 要求申请者提供 Waymo 注册确认、所属机构和用途，并明确限制为非商业用途。OpenPCDet model zoo 同样说明不能公开分发 Waymo 预训练权重。

## 最小合法输入契约

完整链路至少需要一个连续 Waymo 序列，而不是互不相关的单帧点云，因为 tracking 和 trajectory refining 依赖时序轨迹。

### Detection

`detection/detzero_det/datasets/waymo/waymo_dataset.py` 和数据配置要求：

```text
data/waymo/
├── ImageSets/<split>.txt
└── waymo_processed_data/
    └── <sequence-name>/
        ├── <sequence-name>.pkl
        ├── 0000.npy
        ├── 0001.npy
        └── ...
```

每个 info 中的 `lidar_path` 必须能解析到对应帧的 NumPy 点云。合法来源应是用户已接受条款后取得的 Waymo TFRecord，再通过仓库预处理脚本生成。

### Tracking

`tracking/tools/run_track.py` 接收 detection 输出 pickle：

```bash
cd tracking/tools
python run_track.py \
  --cfg_file cfgs/tk_model_cfgs/waymo_detzero_track.yaml \
  --data_path <DETECTION_RESULT.pkl> \
  --split <val-or-test>
```

tracking 会在 `data/waymo/tracking/` 下生成 track 和 drop pickle。若使用需要 GT assignment/evaluation 的 split，还需要对应的 `data/waymo/waymo_infos_<split>.pkl`；纯 test inference 不应依赖 GT。

### Refining 数据准备

`daemon/prepare_object_data.py` 读取 tracking pickle，并再次读取同一连续序列的帧点云：

```text
data/waymo/waymo_processed_data/segment-<sequence>/<frame>.npy
```

随后为三个类别生成：

```text
data/waymo/refining/Vehicle/<sequence>.pkl
data/waymo/refining/Pedestrian/<sequence>.pkl
data/waymo/refining/Cyclist/<sequence>.pkl
```

GRM、PRM、CRM 分别由对应配置和 checkpoint 推理。`daemon/combine_output.py --combine_conf_res` 只有在每类 geometry、position、confidence 结果都存在时，才能生成包含 CRM score 的完整 final 输出。

## 当前已完成与未完成

已完成：

1. 官方发布目录枚举。
2. 官方六个 checkpoint 的隔离重下载。
3. 本地与官方副本的 `6/6` 字节数和 SHA-256 一致性验证。
4. 现有六个 GRM/PRM 的真实 CUDA 前向、确定性重放和严格产物验收。

尚未完成且当前被外部资产阻塞：

1. Waymo-trained DetZero-compatible detection checkpoint。
2. Vehicle/Pedestrian/Cyclist 三个 CRM checkpoint。
3. 至少一个合法、连续的 Waymo 序列及其处理后数据。
4. 在上述资产上执行真实 detection → tracking → GRM → PRM → CRM。

## 继续执行所需信息

继续完整官方复现前，需要用户明确：

1. 本次用途为非商业研究，且已注册并接受 Waymo Open Dataset 条款。
2. 一个合法 Waymo TFRecord 或 DetZero processed-data 本机路径。
3. 一个获授权的、与 DetZero detection 配置严格兼容的 Waymo CenterPoint checkpoint 路径。
4. 三个 DetZero CRM checkpoint 路径；若确实无法取得，则必须明确选择“不运行 CRM”或“自行训练”，且相应结果不能称为官方同权重完整复现。

拿到这些路径后，应先做 checkpoint 键/shape 全量严格加载和输入 schema 预检，再启动新的不可覆盖全链路输出目录；不得覆盖现有 `output/inference_reproduction_v3`。
