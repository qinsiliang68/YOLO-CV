# 冻结 Ultralytics 接入说明

## 原则

SCTSR v4 使用窄 overlay，不修改 `YOLOv11/ultralytics`。冻结上游 learner 的 base loss、模型、optimizer 构建、scheduler、AMP 和 EMA 语义；只替换训练循环的 replay 注入和证据采集。

## 实际调用链

1. `stage1_sctsr_v4/training_system.py::bind_upstream` 校验 `integrations/ultralytics/UPSTREAM_FILES_MANIFEST.json` 和六个实际引用的上游 blob。
2. `stage1_sctsr_v4/formal_cli.py::build_prepared_trainer` 校验 lock、weights、assets、identity manifest 和 trainer overrides。
3. 它临时把 `YOLOv11`、仓库根和 `integrations/ultralytics` 加入 import path，实例化 `SctsrClassificationTrainer`。
4. 只调用冻结 upstream `_setup_train()`；随后强制 `accumulate=1` 并审计 train/val loader 角色。
5. `integrations/ultralytics/sctsr_classification_trainer.py` 给 base dataset 附加身份但不改变长度，并暴露按 sample ID 获取 replay batch 的 provider。
6. `stage1_sctsr_v4/formal_training.py` 编排 common parent/branch；`ultralytics_overlay.py` 执行每个 fixed base step。

正式路径永不调用 upstream `_do_train()`，因为其自动 batch/accumulation 行为不符合固定合同；也永不调用 `final_eval()`，因此 `best.pt` 不能进入选择或 endpoint。

## 冻结的上游文件

`UPSTREAM_FILES_MANIFEST.json` 绑定：

- `ultralytics/engine/trainer.py`；
- `ultralytics/models/yolo/classify/train.py`；
- `ultralytics/nn/tasks.py`；
- `ultralytics/nn/modules/head.py`；
- `ultralytics/utils/loss.py`；
- `ultralytics/utils/torch_utils.py`。

source-tree manifest 同时绑定 overlay 本身。上游或 overlay 任一 bytes/SHA 改变都需要新的 source freeze 和审查，不能静默接受。

## Loader 角色检查

prepared train loader 必须精确包含 120,000 个 canonical identity，长度为 938 batch；prepared validation loader 只能对应登记的 `val_model/study` 两个 component。`val_op` 不进入 trainer 的过程验证，它只在 E200 endpoint publisher 中按登记 split bundle读取。`test` 和 blind holdout 没有合法入口。

## Replay step 不变量

- base loss 直接使用 `trainer.model(batch)` 返回的 upstream loss；
- replay logits 经同一 model 得到，CE reduction 固定为 sum/128；
- replay forward 前保存全局 RNG 和所有 BN running buffer；
- replay backward 后恢复 RNG/BN，但不恢复 parameter gradient；
- 每个 base batch 只有一次 unscale/clip/step/update/zero-grad/EMA；
- scheduler、warmup 和 global step 不因 replay forward 增加；
- OOM 立即上抛，由 epoch transaction quarantine，不做降配。

## Endpoint 模型

`runtime_policy_v1.json` 把正式 endpoint 冻结为 E200、`val_op`、`EMA`、batch 128。publisher 会确认所用模型 state 与 E200 checkpoint 的 EMA state 字节级 digest 一致，并验证 logits-softmax 一致性。

## 审查命令

```powershell
git diff a70ba60485dd32c2f8b4268b8f28ea2d3549f42f -- YOLOv11/ultralytics stage1_gapvalue240 stage1_dynamic_replay_v3
uv run pytest tests/stage1_sctsr_v4/test_real_yolo_integration.py tests/stage1_sctsr_v4/test_training_system.py tests/stage1_sctsr_v4/test_ultralytics_overlay.py -q
```

第一条必须没有 v4 提交造成的旧代码或 upstream diff；第二条只证明 integration 机制，不是正式训练结果。
