# Stage1 GapValue 240-Run 来龙去脉与交接状态

## 1. 这套实验为什么存在

旧 120-run 只能说明：按单次最终置信度挑选高置信误报 normal 的 hard-negative 规则没有稳定优于随机回流。它不能证明不存在高价值样本，也不能作为当前 GapValue 方法有效的证据。因此旧 120-run 已降级为 historical/development background，不再继承旧 best-run 结论。

新的研究入口来自 120,000 个训练样本的完整 OOF 动态：10 folds、200 epochs、2,000 个 fold-epoch 文件、24,000,000 行逐样本预测。研究问题由“最终哪个样本最难”改成：训练状态从坏 gap 变成好 gap 时，哪些样本沿业务目标的正确方向移动，以及回流这些样本是否真的改善模型。

专家提供的 PDF 与 DOCX 内容相同，原题为《Stage-1 OOF Gap-Value 训练动态样本价值实验计划与专家交接书》，日期为 2026-07-10，共 52 页。它是一份实验前协议，不是结果报告。其核心要求是：观察性训练动态只能提出 hypothesis，必须通过公平回流、随机对照、负对照、时间/fold/group/provenance 敏感性和独立确认，才能形成样本价值结论。

专家随后提供 `Stage1_GapValue_240Run_Complete_20260710.zip` 作为代码底稿：35,819,634 bytes、387 entries、240 个独立 run 入口，ZIP SHA-256 为 `9EB4F5C84A87C5A752A996B284C6B351E847C68F6622ABD6B66AEBF85078E7D3`。当前仓库是在该底稿上做兼容审查和稳定性修复，不重写 OOF 动态链，也不修改历史训练入口。

## 2. 科学假设的浓缩版本

normal 的主分数为：

```text
GapCritical(x) = mean_p_defect(bad-gap epochs) - mean_p_defect(good-gap epochs)
```

高分表示该 normal 在坏 gap 状态容易被误判为 defect，但在好 gap 状态被正确压回 normal。defect guard 使用相反方向：

```text
GapGuard(x) = mean_p_defect(good-gap epochs) - mean_p_defect(bad-gap epochs)
```

当前数据只证明这两个分数具有可解释的观察性结构，尚未证明因果训练价值。正式主命题是：`GapCritical-Strict` 在相同 replay budget、训练 seed、初始化、增强和 optimizer steps 下，能否同时胜过两类随机对照，并在 FN 安全约束下改善 `TN_at_FN95`。

必须保留的机制对照包括：Confidence、Boundary、Persistent、LearnableBucket-Random、Early-Late、Anti-Gap，以及 fold/group、LOFO、TailGap、epoch178 排除等敏感性。Gradient 方法仍然 blocked，因为当前五个 gradient 字段为空。

## 3. 从专家资源档到最终 240-run

专家 PDF 给出的是分层资源建议：最低机制版约 45-60 runs、推荐发现版约 87-90 runs、发表准备版 120+，Phase B 再按资源追加。项目随后依据 12 台机器和 1:2 随机对照要求冻结为一套确定矩阵：

```text
19 个 Phase A 条件 + 6 个 Phase B 条件 = 25 个发现条件
25 × 3 discovery seeds = 75 个发现期 Treatment
Phase C 为主配置增加 5 个新 seeds = 5 个 Treatment
Treatment 总数 = 80
每个 Treatment 配 R1 和 R2 两个随机对照
80 × (T + R1 + R2) = 240 个实际训练 run
```

最终结构固定为 80 T、80 R1、80 R2，即一组实验配两组随机对照。`R1` 是 Global-Random-Clean；`R2` 是 Method-Matched-Random，匹配 fold 和预先冻结的 hardness 特征。B600/B3000 是主要机制预算，B6000 主要解释为扩量、饱和和稀释实验。

10 台主机各绑定 24 runs（8 个完整 triads）；machine 11、12 的 shard 为空，只作备用。12 个 shard、240 行矩阵、selection index 和每份 selection CSV 都由 runtime contract v1.2 的 SHA-256 固定。训练机只消费已冻结 manifest，不现场重抽样。

## 4. Replay 与 Phase B 的公平性

Phase A 使用 additive replay：120,000 基础样本保持一次正常曝光，再追加 B 个已冻结 replay 样本。相同 budget 的 T/R1/R2 使用相同训练参数和 optimizer-step 目标。

Phase B 只在 replay 槽位内部 replacement，不删除基础训练样本。例如 B3000、guard 10% 为 2,700 normal replay + 300 defect replay，总 replay 数仍为 3,000。不同 guard 方法在同 seed/ratio 下必须复用同一 normal 集合，只改变 defect 来源。

## 5. 数据与测试集职责

- Train OOF dynamics：只能用于 ranking、bucket 和 epoch proxy。
- `val_model`：训练期模型验证。
- `val_cal`：拟合 Platt calibration。
- `val_op`：发现阶段 operational 指标比较。
- 历史 120k test：已多次使用，逻辑角色降级为 `development_benchmark_120k`。
- blind/external test：方法、预算、seed 和统计脚本冻结后一次性确认。

240 个发现 run 不能反复查看 blind test 后再修改方法，否则 blind test 会被污染成 development set。

## 6. 为什么没有继续使用旧训练入口

`scripts/train_stage1_cls_sweep.py` 和历史 evaluator 是实验档案，保持不动。正式 GapValue 路径新增适配器，原因不是更换模型或科学配置，而是旧入口不能稳定满足以下合同：精确 checkpoint 路径、`patience=0`、严格 200 epochs、同卷 hardlink-only staging、稳定 `training_state/last.pt`、隔离 GPU 子进程、严格 postflight 和 runtime v1.2 身份校验。

模型仍固定为 YOLO11l，正式参数仍由冻结科学合同给出：epochs=200、batch=128、imgsz=224、deterministic=true、cache=false。机器管理员只能修改路径、GPU、workers 和磁盘阈值，不能修改方法、seed、budget、epoch 或 selection。

## 7. 本轮稳定性修复解决了什么

- runtime v1.2 同时固定 science contract、matrix、selection index、240 selections、12 个 machine shards、checkpoint 和发布 tag。
- 每台机器先生成一次 384,000 图片资产报告；正式 run 复用报告，只复核 manifest/checkpoint SHA，不每次重扫图片。
- staging 与 dataset 必须同卷，只允许 hardlink；建立一次 120k train + 24k val_model 基础缓存，每个 run 只临时增加 replay links，结束时清理 replay 和 YOLO cache。
- 训练和 val_cal/val_op prediction 都在独立子进程中执行；长驻 shard controller 不导入 PyTorch/YOLO。
- 原生 resume 记录为 `native_approximate`，保存 resume count、checkpoint SHA、起止 epoch 和 segment；所有 Phase 可纳入统计，但汇总保留 resume_count 供敏感性分析。
- `status.json` 是权威状态；run、GPU、staging 和 shard controller 均有互斥锁。已验证 run 自动跳过，训练完成后评估失败只重跑评估，评估完成后验证失败只重跑验证。
- VALIDATED 前严格检查 200 epoch、steps、batch、resolved args、optimizer/augmentation 记录、checkpoint reload、预测 ID/标签/行数、NaN/Inf 和 operational metric 重算。
- 聚合只读取正式目录中当前 release、合同、matrix、selection、input snapshot 全部匹配且 artifact manifest 校验通过的 VALIDATED run；dry-run 永不进入统计。

## 8. 与 AIOps 的责任边界

本仓库不再建设新的调度或监控平台。代码只提供容易运维的确定接口：

- exit 0：完成、已经完成或没有任务；
- exit 20：可重试的训练/预测/锁/子进程故障；
- exit 30：输入、版本、合同或科学配置错误，禁止盲目重试；
- 原子 `status.json`：包含 run_slot、phase、PID、last_epoch、resume_count、retryable 和 error_code；
- 受控失败时终止完整子进程树并保留 attempt 证据。

AIOps 负责发现卡死、告警、按退出码重启、控制最大重试次数、磁盘不足时清理可再生 staging/失败 attempt、以及人工批准后的备用机接管。代码不自动修改科学矩阵，不自动把两台备用机转成主机。

## 9. 当前已验证事实

- Stage1 GapValue 测试：70 passed。
- 240 份 selection 与冻结 index：0 mismatch。
- selection index SHA-256：`C58C2BED62FF49859F05C852D53BE2B6935EF438B23FB29E59AC9B8CB64C58F8`。
- selection 集合摘要：`013475D7A585DCE59F67C1C1CAA1B01A20192463847AA78C90C84D5C22941E23`。
- base checkpoint SHA-256：`6B56513A5D8BDAE6B8F0A36DACAF01B26D5A522BA1B34197C3BAC9FA6463366C`。
- 本地连续两个 YOLO11l resource smoke 均通过：batch128、workers8、48 train/16 val、3 epochs、每 epoch 1 step。
- 两个 smoke 耗时分别约 19.46s 和 19.23s；结束后与训练前 WDDM 显存基线相比为 -17MiB、-21MiB；未发现残留 `local_resource_smoke` Python 进程。
- 每个 smoke 的 `results.csv` 为 3×9，`args.yaml` 含 110 个 resolved fields，audit 字段完整；测试 checkpoint 仅作哈希验证后删除，未混入科学结果。

## 10. 尚未完成、不得写成已通过的门槛

- 真实 120.6k、B600 的 T/R1/R2 完整 3-epoch canary。
- 对其中一臂人工中断并验证真实 native resume。
- 至少一台代表性训练机的 200-epoch canary。
- 12 台机器分别完成资产、显存和吞吐基准。
- 10 台主机在 25% 故障缓冲下仍能在 15 天内完成的实测估算。
- blind/external test 及任何样本价值科学结论。

因此当前状态是“代码候选已通过本地测试”，不是“240-run 已批准放量”。固定 tag `stage1-gapvalue240-runtime-v1.2.0` 暂不创建；只有上述外部门槛完成并复核当前 commit 后才创建。

## 11. 正式启动顺序

1. 在每台机器修改自己的 machine YAML，仅改路径、GPU、workers 和资源字段。
2. 执行 runtime links、240 selections 和 machine asset report 校验。
3. 在代表性训练机完成全量 triad 3-epoch、中断恢复和 200-epoch canary。
4. 12 台分别记录吞吐；确认 15 天容量后，由负责人批准 release commit。
5. 在该 commit 创建且只创建一次 `stage1-gapvalue240-runtime-v1.2.0` tag。
6. AIOps 启动 machine 01-10 的冻结 shards；machine 11-12 保持 reserve。
7. 汇总只读取 VALIDATED；最终 blind test 在方法冻结后一次性执行。

## 12. 不可破坏的交接约束

- 不修改 240 份 selection CSV，不重新抽样。
- 不修改旧训练入口和历史 OOF 动态链。
- 不使用最终测试反馈选样或调整排序。
- 不把 dry-run、resource smoke、单个最好 seed 或旧 120-run 当科学结论。
- 不在没有新版本合同和新 release 的情况下移动正式 tag。
- 不因磁盘或机器故障偷偷更换 seed、budget、arm 或 replay exposure。

