# SCTSR v4 独立审查清单

这份清单供未来独立审查者使用。本轮 206 项报告必须标为 self-review，不能伪称独立审稿人。

## 1. 身份与范围

- 核对 branch 的共同祖先是 `a70ba60485dd32c2f8b4268b8f28ea2d3549f42f`。
- 核对 taskbook blob 为 `b201d021712e9c6614e119d35f0e14bdf405c6be`。
- 对 `stage1_gapvalue240`、`stage1_dynamic_replay_v3`、`YOLOv11/ultralytics` 做 base-to-head diff；v4 提交不得修改它们。
- 核对只提交登记 v4 路径，未 stage 无关 literature、training evidence 或用户工作树。
- 从 clean checkout 重建 source-tree manifest，确认 `integrations/ultralytics` 和未跟踪 importable file 检测在覆盖范围内。

## 2. 合同、资产和数据角色

- 运行 contract、schema、asset validator，并保留独立 stdout/stderr、exit、bytes、SHA。
- 检查 ReplayRateSpec 拒绝 float、absolute count、缺失 denominator 和不可整除值。
- 检查八臂集合与顺序，CURRENT_LOSS_U 必须 HELD。
- 检查 val_model/val_cal/val_op bundle 只由登记 component 构造；test/blind 不应出现在 choices。
- 检查 `val_op` 不进入 pool、schedule、trainer config 或 method selection。

## 3. T/R1/R2

- 从原始 T bytes 重算 path/bytes/SHA、3,000 IDs 和冻结 digest。
- 确认 T role 是 stress set，不是 validated selector。
- 确认 R1 universe 是完整 eligible base，自然 T overlap 被报告。
- 逐行检查 terminal whitelist projection 在 R2 matcher 前发生。
- 主动注入 terminal-field access，必须抛错。
- 在真实资产复现旧四字段 R2 joint-stratum infeasibility，再独立物化 owner-approved
  addendum 与 content repair：3,000 unique、T sample/content overlap=0、三字段 exact、
  379 displacement、group TV=`0.12633333333333333`。
- 确认 `R2_U`、`R2_F` 与 fallback 共用同一 R2 digest，并主动注入第二字段 relaxation、
  错 seed、错 digest、replacement 和 T overlap，全部必须 fail closed。
- 检查 selection ledger 是候选全集而非 selected-only。

## 4. Schedule 与 common parent

- 重建八臂完整计划，逐 ID 比较 U/F multiplicity vector。
- 检查 E1–E120 全臂 no-replay；T stop/fallback 的共同前缀逐 epoch 相同。
- 检查 U/F 的累计 occurrence、identity digest 相同，唯一差异是 epoch distribution。
- 检查每 seed 只有一个 E120 parent，八 child parent SHA 字节相同。
- 破坏 parent seed/source/assets/epoch/SHA，branch 必须拒绝。
- 检查 logical timeline 的 E1–120/121–200 physical owner 分界。

## 5. Fixed-step 逐行复核

必须给出精确文件和行号，至少覆盖：

- optimizer/scaler/clip/zero-grad/EMA 调用边界；
- replay CE `sum / 128` 和尾 batch cap；
- upstream base loss 未替换；
- BN 全部 running buffers 保存/恢复；
- Python/NumPy/Torch CPU/CUDA RNG 保存/恢复；
- replay gradient 保留；
- OOM/accumulation/world-size 失败路径不会改变合同。

用 mock call counter 和真实 tiny YOLO integration 两类测试交叉验证；只看源代码或只看 mock 都不够。

## 6. Evidence、telemetry、事务和恢复

- 对 occurrence/step/exposure/selection/telemetry 的每个 taskbook 字段做 schema 对照。
- 检查正式模式只写 Zstd Parquet；portable fallback 只能 synthetic 显式启用。
- 检查每 epoch 的 row count、938 step、120,000 base、planned/actual replay 和所有 partition SHA。
- 检查 telemetry 1 秒 cadence、进程/系统/GPU/CUDA/磁盘字段和 null reason。
- 分别注入 kill、OOM、disk-full、半写 Parquet、半写 JSON、坏 receipt、错 generation/RNG/source/parent/assets。
- 确认 quarantine 不覆盖旧证据，pointer 不越过失败 epoch，resume 只从最后完整 generation。

## 7. Prediction、frontier 和统计

- 从登记 E200 checkpoint 和 val_op component 重建 split bundle。
- 检查模型 state 与 checkpoint EMA state digest 相同。
- 检查每行 raw logits 与 softmax probability 一致，sample/label 无缺失、重复、多余或错位。
- 手算 tie fixture，确认 FN=0..95 恰 96 行且 tie group 不拆。
- 分别复核 normalized AUC、TN_at_FN95、FN_at_TN68253、两个 threshold 和 unreachable 语义。
- 对小 n 穷举 exact paired sign-flip；复核 Holm 排序、阈值和 decision。

## 8. Formal input 与 closeout

- 替换训练前外部 contract/asset/source/seed/identity/pool/parent 任一字节，closeout 必须失败。
- 检查 input snapshot 同时绑定原路径和 run 内复制字节。
- 检查正式 endpoint 缺失时 branch closeout 失败；不得接受外部随意 JSON prediction。
- 检查 summary completion 和 206 项 detailed self-audit 都是 closeout 必需输入。
- 检查 detailed audit 的日志 hash、任务书行号、ID 顺序、状态计数和 source commit。
- 确认 closeout 只输出 implementation acceptance，不创建 release/assignment/gate/pilot。

## 9. 必跑命令

```powershell
uv lock --check
uv run pytest tests/stage1_sctsr_v4 -q
uv run pytest tests/stage1_dynamic_replay_v3 -q
uv run python -m compileall -q stage1_sctsr_v4 scripts/stage1_sctsr_v4 integrations/ultralytics tests/stage1_sctsr_v4
git diff --check
```

必须报告完整结果，包括 skip/xfail、最初失败原因、日志原始 bytes/SHA 和实际 source commit。当前 v3 结果低于任务书 231 时不得输出整体 PASS。

## 10. 最终否定性证明

逐字段证明均为 JSON boolean false：formal training、engineering gate、assignments、pilot release、blind holdout、selector training、method effectiveness claim、synthetic scientific registration。旧历史 gate/release/assignment 必须单独登记为 legacy detected，不能被写成 active v4，也不能删除历史来制造 false。
