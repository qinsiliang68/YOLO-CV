# SCTSR v4 训练与证据系统架构

## 1. 信任与身份平面

正式入口先验证 HMAC-SHA256 release、trust key、nonce/expiry 和七个签名绑定：baseline commit、taskbook blob、source tree、contract、assets、runtime、seed registry。当前 trust registry 为空，所以 formal 必须在 trainer 构建和数据访问前拒绝。

矩阵 release 不能直接启动训练。每次 START/RESUME 还必须使用一个 owner 签名、只允许消费一次的 execution token。十台机器必须访问同一个共享 claim registry；`CLAIM_REGISTRY.json.registry_root_digest` 绑定该共享根的规范化绝对路径。把 descriptor 和空 `claims/` 复制到另一台机器的本地目录不会产生第二个合法 registry，同一 token 在克隆目录中必须失败。正式部署应使用所有机器解析一致的 UNC 共享路径，并在签发 token 前验证 exclusive-create 语义。

source tree 不只记录文件列表，还覆盖 v4 source/scripts/configs/tests/docs、Ultralytics overlay、依赖锁、任务书和实际导入的六个 upstream 文件。正式验证重新扫描 include roots、重算 bytes/SHA、核对当前 Git HEAD 和 tracked-clean 状态；include root 内新增未登记文件会失败。

## 2. 数据角色平面

- `train`：canonical base，固定 120,000 optimizer-visible occurrence；
- `OOF/reference`：只提供预终端 dynamics 和 reference signal；
- `val_model/study`：upstream trainer 的过程观察，不得选最终方法；
- `val_cal`：登记但不参与第一阶段 selector；
- `val_op`：E200 冻结 endpoint 的安全前沿评价；
- `val_target`：当前不存在，A 必须 blocked；
- `test/blind_holdout`：未登记到任何 runner，保持密封。

asset registry 对每个 component 保存 role、path、bytes、SHA、row count 和 identity digest；split bundle 把多个 component 合成一个不可变 sample-label identity。

## 3. 选择与调度平面

T/R1/R2 是离线生成并哈希绑定的固定 identity pool。R2 的白名单投影发生在 matcher
之前；先按 `(label, historical_dynamic_bucket, oof_fold, oof_group_id)` 耗尽 exact
capacity，再仅在同 `(label, historical_dynamic_bucket, oof_fold)` cell 内执行已批准的
在原 378 个 quota deficit 上再排除 1 个 T content alias，形成 379 个最小 group
displacement。`R2_U`、`R2_F` 与 fallback 共享一个 pool identity/content digest。
五个 identity group 由 ID 稳定散列产生，不读取原 rank。

Schedule 是完整 materialized E1–E200 occurrence plan，不把 seed 当成计划。每个 epoch 记录 rate、sample IDs、slot skeleton、identity policy、fallback state 和累计守恒摘要。跨臂 validator 联合检查八臂、U/F parity、共同前缀、stop/fallback 和逐 ID multiplicity。

## 4. Parent/branch 执行平面

每个 seed 的 E1–E120 只运行一次 no-replay parent。E120 checkpoint 是所有八个 child 的共同父状态。BranchLineage 绑定 parent ID/path/SHA、seed、arm、source、contract 和创建身份；裸 checkpoint 不能启动 branch。

LogicalArtifactIndex 不复制 parent 产物：logical E1–E120 指向 parent physical root，E121–E200 指向 child physical root。它逐项绑定 checkpoint bytes，避免把复制文件伪装成 child 原生产物。

## 5. Fixed-step execution 平面

`integrations/ultralytics/sctsr_classification_trainer.py` 只负责准备冻结 ClassificationTrainer、长度不变的 identity dataset 和 replay-by-ID provider。`formal_training.py` 不调用 upstream `_do_train` 或 `final_eval`；真正的 epoch 循环由 `ultralytics_overlay.py` 驱动。

每个 base step 先计算 upstream base loss，再在隔离 RNG/BN 域内计算 replay loss。两次 backward 共享同一 parameter gradient，之后只有一次 optimizer-visible update。base dataset、base order、base augmentation、938 steps、scheduler/warmup/EMA clock 都与 NR 对齐。

## 6. Evidence plane

`EpochEvidenceRecorder` 接收运行时逐 occurrence 和逐 step 事件，并写四类 partition：

- occurrence：身份、角色、增强、logits/probability、CE、margin、correctness、OOF/RHO 非 utility 信号、累计 replay 与 lag；
- optimizer-step：base/replay loss、gradient、clip、LR、AMP、EMA、BN/RNG digest 和耗时；
- exposure：计划/实际分子分母、unique/repeat、累计 occurrence、steps 和各 partition SHA；
- telemetry：1 秒 cadence 下的进程、系统、GPU、CUDA、磁盘和 provider reason。

selection ledger 保存候选全集、排除原因、quota、选择原因和终端字段未读取证明，不只保存 selected IDs。

## 7. Transaction/recovery plane

每个 epoch 是独立 generation。只有 schema、row count、Parquet codec、曝光守恒、step count、checkpoint identity、RNG 闭环和 SHA 全部通过，`.inprogress` 才原子发布为 `.complete`。append-only receipt 和 rolling pointer 都绑定前一 generation。

恢复器扫描完整 generation 和 quarantine，重建 replay history，验证最后 checkpoint，并拒绝 generation 跳跃、错误 parent/source/assets/RNG 或已 terminal 的 run。恢复不会覆盖旧 generation。

## 8. Prediction/evaluation plane

E200 endpoint publisher 从登记的 `val_op` split bundle 解析真实图片，经冻结 transform 得到 raw logits 和 probability，写 Zstd Parquet，再计算 96 点 tie-safe frontier。一个 probability tie group 要么整体进入阈值一侧，要么整体不进入，不能为满足 FN budget 拆 tie。

`TN_at_FN95` 和 `FN_at_TN68253` 各有独立 threshold 和 reachability；不得拼成同一 confusion matrix。统计层按 seed 配对，报告全 delta、win rate、worst seed、dual-end degradation、exact sign-flip 和 Holm。

## 9. Validation/closeout plane

`validate_run_tree` 先验证 exhaustive artifact index，再按 synthetic/formal 语义检查全部交易、ledger、checkpoint、lineage、pool、parent、prediction 和 frontier。正式 closeout 还重新读取运行前的外部输入；训练结束后替换外部文件不会被快照摘要掩盖。

仓库级 self-audit 直接解析 taskbook 的 206 个 SA ID。每个 ID 都有独立状态与证据字段。任何一个实施项失败都会保留 `SELF_AUDIT_FAIL`，不能被长报告、总体测试数或 blocker 文案覆盖。
