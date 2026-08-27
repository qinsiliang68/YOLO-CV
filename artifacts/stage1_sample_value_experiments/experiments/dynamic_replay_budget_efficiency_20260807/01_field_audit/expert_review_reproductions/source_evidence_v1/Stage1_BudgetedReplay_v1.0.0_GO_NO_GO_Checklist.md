# Stage1 Budgeted Replay v1.0.0：GO / NO-GO 验收清单

当前状态：**NO-GO**

## A. 数据与 OOF（全部必须通过）

- [ ] trajectory axis 不再包含 fold；fold 为 per-sample assignment metadata。
- [ ] 真实 K-fold fixture：每个样本仅由 held-out fold 模型预测，strict conversion 通过。
- [ ] 每个 sample×epoch×seed/repetition 均有且仅有一个 OOF prediction。
- [ ] 验证模型训练集合不包含该 sample；输出可审计 fold mapping。
- [ ] OOF dynamics 不存在结构性 NaN；任何实际缺失均 hard-fail。
- [ ] video_id / pipe_id / source_session 分组切分无跨角色泄漏。
- [ ] file SHA + pHash/embedding 近重复跨角色审计通过。

## B. 数据角色与评估（全部必须通过）

- [ ] 新增独立 val_target/val_study；Critical-CLD/gradient 不能使用 val_op。
- [ ] val_cal 仅校准；val_op 仅选 operating thresholds；test 最终一次性使用。
- [ ] `canonical_source=test_oracle_curve` 从默认配置和开发汇总删除。
- [ ] 分别冻结 `threshold_fn95` 与 `threshold_tn68253`。
- [ ] test 分别输出两阈值 confusion matrix。
- [ ] checkpoint selection 使用固定 last epoch，或在 val_model 上按冻结业务目标选择。
- [ ] selected epoch、criterion 和 checkpoint SHA 全部写入 ledger。

## C. 预算与训练（全部必须通过）

- [ ] 明确 B600/B3000/B6000 是 per-epoch slots 还是 total-run exposures。
- [ ] 报告 unique samples、repeat histogram、per-epoch slots 和 cumulative exposures。
- [ ] 单卡实际 sample fetch ledger 与 schedule 完全一致。
- [ ] DDP 每 rank actual exposure ledger 汇总后与目标完全一致。
- [ ] DistributedSampler 不会通过 padding 额外重复样本，或额外重复被精确记录并对齐三臂。
- [ ] resume 前后 epoch/sample schedule 连续且无重复/遗漏。
- [ ] physical replay 每次访问具有可审计 augmentation seed/hash。
- [ ] 显式冻结 optimizer/lr/momentum；禁止 `optimizer:auto`。
- [ ] 精确冻结 Python/PyTorch/CUDA/cuDNN/Ultralytics/driver/container digest。

## D. 方法实现（全部必须通过）

- [ ] USEFUL 文献标题修正；方法标为 Budgeted-USEFUL adaptation。
- [ ] current-state reducibility 缺 current predictions 时 hard-fail。
- [ ] historical proxy 作为独立 arm，不与 current reducibility 同名。
- [ ] Critical-CLD 在 matched seed/model trajectory 内计算，再聚合。
- [ ] Critical-CLD 使用独立 val_target，并保留 anti-CLD control。
- [ ] alignment provenance 校验 checkpoint SHA/layer/preprocess/class mapping。
- [ ] cluster quota 在 caps 后仍满足最终 quota，或明确报告偏离并失败。
- [ ] one canonical implementation per feature/metric/calibrator/backend/statistic；legacy 路径禁用。

## E. 矩阵与统计（全部必须通过）

- [ ] 唯一 frozen_240run.yaml 恰好生成 80 triads / 240 arms。
- [ ] matrix manifest、config 和 source code 均有 SHA-256。
- [ ] primary endpoint、safety endpoint 和 success rule 预注册。
- [ ] T/R1/R2 overlap policy 预注册；报告 Jaccard；完成 disjoint sensitivity。
- [ ] 3 seed 仅用于筛选，不写成确认性“稳压”。
- [ ] 短名单使用更多独立 seed 做确认实验。
- [ ] paired randomization/permutation 以 seed/triad 为单位。
- [ ] 多方法/预算/终点使用 Holm 或预注册的同时推断控制。
- [ ] 报告 worst seed、leave-one-seed-out、双对照安全胜率。

## F. 真实环境 smoke（全部必须通过）

- [ ] 真实 YOLO11m-cls 单卡 1 epoch T/R1/R2 parity smoke。
- [ ] 真实多 GPU/DDP 1 epoch parity smoke。
- [ ] best/last/checkpoint selection 与冻结规则一致。
- [ ] prediction/calibration/threshold/test 全链路重算一致。
- [ ] 中断恢复 smoke；原子状态和 stale-lock 恢复通过。
- [ ] 只有所有上述项通过，才允许启动正式大矩阵。
