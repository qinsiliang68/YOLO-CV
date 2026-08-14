# SCTSR v4 训练部署逐项检查表

本表每项只能为 `PASS`、`FAIL` 或 `BLOCKED`，不能用勾选符替代证据。每个 PASS
需附 command、exit code、artifact path、bytes 和 SHA。

## A. Owner 科学决定

- [ ] A01 R2 addendum 有 owner 明确接受文本、日期和 SHA。
- [ ] A02 addendum 明确 estimand、匹配字段、不可避免 imbalance 和分析方式。
- [ ] A03 新 R2 matcher 有 failure-first/green tests。
- [ ] A04 R2 生成 3,000 unique、T overlap=0，quota audit符合 addendum。
- [ ] A05 CURRENT_LOSS 仍 HELD；Q/R/A/D weighted score仍禁止。
- [ ] A06 val_target缺失只阻断 A，不被其他 split冒充。
- [ ] A07 discovery/confirmation/stop/multiplicity/Holm规则冻结。
- [ ] A08 blind/test仍密封。

## B. Source 和 protected history

- [ ] B01 exact clean commit已指定，`git status --porcelain`为空。
- [ ] B02 source-tree manifest覆盖所有 v4、CLI、config、overlay、tests、lock和实际
      imported upstream files。
- [ ] B03 source manifest重新验证 PASS。
- [ ] B04 `stage1_gapvalue240`、`stage1_dynamic_replay_v3`、`YOLOv11/ultralytics`
      相对 `a70ba604` 无实现变化。
- [ ] B05 历史已训练产物、queue/release/assignment未删除、改写或冒充 v4。
- [ ] B06 taskbook blob、review fix commits 和 runbook manifest登记。

## C. 代码质量

- [ ] C01 Python 3.11 v4全套 PASS，无核心 skip。
- [ ] C02 Python 3.12 v4全套 PASS，无核心 skip。
- [ ] C03 v3 regression fresh result和skip原因登记。
- [ ] C04 compileall 3.11/3.12 PASS。
- [ ] C05 `uv lock --check` PASS。
- [ ] C06 `git diff --check` PASS。
- [ ] C07 synthetic canary同 seed两次 deterministic。
- [ ] C08 本地真实图+真实 yolo11l engineering canary PASS。
- [ ] C09 无 open P0/P1；P2/P3有明确接受/缓解。

## D. 资产

- [ ] D01 training lock 4,623 bytes/SHA匹配。
- [ ] D02 yolo11l checkpoint 28,553,700 bytes/SHA匹配。
- [ ] D03 base manifests 60k+60k、identity digest匹配。
- [ ] D04 OOF 120k、group semantic正确。
- [ ] D05 T canonical path/bytes/SHA/digest匹配。
- [ ] D06 sample value/dynamics表 bytes/SHA匹配且 terminal guard生效。
- [ ] D07 val_model/val_cal/val_op components row/SHA/互斥PASS。
- [ ] D08 dataset content ledger 384,000/Zstd/SHA/content digest PASS。
- [ ] D09 本机逐图物理验证 384,000/82,637,967,451 bytes PASS。
- [ ] D10 test/blind未进入 registry/ledger/runner。

## E. 公平性

- [ ] E01 每 seed唯一 E1-E120 common parent。
- [ ] E02 八 child同一 E120 checkpoint SHA。
- [ ] E03 base rows 120,000、steps 938。
- [ ] E04 NR/replay arm base order和augmentation digest匹配。
- [ ] E05 replay不改变 DataLoader length。
- [ ] E06 每 base step只有一次 optimizer/scaler/EMA update。
- [ ] E07 replay CE=sum/128，microbatch≤actual base batch/4。
- [ ] E08 replay RNG/BN恢复且 parameter gradients保留。
- [ ] E09 U/F pool、dose 48,000、per-ID multiplicity 16一致。
- [ ] E10 stop/fallback planned/actual E160切换一致。
- [ ] E11 R2 terminal fields不可访问。
- [ ] E12 planned/actual occurrence和cumulative history守恒。

## F. Release、token 和 intent

- [ ] F01 release trust key已由 authority登记。
- [ ] F02 discovery seeds在签名 release前冻结。
- [ ] F03 release绑定 source/contract/assets/runtime/seeds和expiry。
- [ ] F04 每 logical process有唯一 token和action START/RESUME。
- [ ] F05 execution claim registry为全机器同一 canonical shared root。
- [ ] F06 copied registry descriptor不能重复 claim。
- [ ] F07 run-intent acknowledgement字段完整、digest和runbook SHA有效。
- [ ] F08 acknowledgement bind exact role/run/arm/seed/output/parent/schedule/pool/
      release/token。
- [ ] F09 trainer构造前完成 release/token/ack/asset/data验证。
- [ ] F10 无 assignment/gate/pilot side effect由 validator暗中生成。
- [ ] F11 acknowledgement在token claim和trainer构造前完成独立context重派生验证。
- [ ] F12 START/RESUME intent attempt均进入append-only snapshot chain并由终态receipt绑定。

## G. 机器

- [ ] G01 GPU为预期 RTX 3090，UUID/driver/CUDA/Torch登记。
- [ ] G02 world size=1，numeric device恰好一个。
- [ ] G03 该 GPU无第二个 formal process。
- [ ] G04 系统时间同步，UTC可用。
- [ ] G05 每台至少 500 GB 独占可用 NVMe（正式 benchmark后可修订）。
- [ ] G06 central evidence store容量和SHA复制流程就绪。
- [ ] G07 CPU/RAM/workers符合 runtime。
- [ ] G08 telemetry可读 process/system/GPU/CUDA/disk真实值。
- [ ] G09 Windows长路径策略和目标output路径通过canary。
- [ ] G10 receipt root在run root外部且可原子写。

## H. Common parent

- [ ] H01 START output root不存在。
- [ ] H02 parent identity manifest恰好120k，无pool annotation。
- [ ] H03 token/ack角色为 COMMON_PARENT/COMMON_PARENT_NR。
- [ ] H04 E1-E120连续complete。
- [ ] H05 每epoch checkpoint+4 Parquet+summary+manifest完整。
- [ ] H06 receipt chain/pointer/index末端一致。
- [ ] H07 parent terminal receipt与E120 checkpoint SHA一致。
- [ ] H08 validate_run PASS。
- [ ] H09 parent只发布给同seed branch。

## I. Branch

- [ ] I01 branch parent/parent index/lineage全量重验。
- [ ] I02 schedule arm/seed/pools绑定正确。
- [ ] I03 identity manifest和pool membership一致。
- [ ] I04 E121-E200连续complete。
- [ ] I05 E160前后schedule实际切换正确。
- [ ] I06 E200 fixed endpoint存在，EMA/val_op/batch128。
- [ ] I07 best.pt未使用。
- [ ] I08 prediction sample/label/logit/probability/checkpoint identity完整。
- [ ] I09 frontier恰好96点且tie-safe。
- [ ] I10 validate_run和closeout PASS。

## J. Failure/resume

- [ ] J01 kill/OOM/disk/half-write/corrupt receipt故障注入均fail closed。
- [ ] J02 partial/orphan进入append-only quarantine。
- [ ] J03 root cause不被cleanup failure遮蔽。
- [ ] J04 RESUME只从last contiguous receipt commit point。
- [ ] J05 resume前remaining disk preflight PASS。
- [ ] J06 resume使用新one-use token/action=RESUME。
- [ ] J07 resume setup root canonical且隔离。
- [ ] J08 source/assets/data/parent/RNG/history与原run一致。

## K. 统计和发布

- [ ] K01 只使用canonical complete同seed pairs。
- [ ] K02 discovery和confirmation seeds隔离。
- [ ] K03 primary raw frontier nAUC、两个锚点独立threshold。
- [ ] K04 seed win rate、worst seed、dual-end degradation报告。
- [ ] K05 exact sign-flip与Holm family预注册执行。
- [ ] K06 未看test/blind。
- [ ] K07 结果类型区分paper/expert/code/synthetic/engineering/Stage1 observation。
- [ ] K08 在干预结果前没有method effectiveness claim。

## L. 最终布尔值

部署 release 前机器审计必须明确：

```json
{
  "formal_training_started": false,
  "assignments_generated": false,
  "engineering_gate_generated": false,
  "pilot_release_generated": false,
  "blind_holdout_opened": false,
  "test_accessed": false,
  "method_effectiveness_claimed": false
}
```

这些值只描述 release 前状态；正式 training release 签发后，只有
`formal_training_started` 会在对应 run 真正进入 epoch 后变为 true，其他禁止项仍为
false。不得提前把未来训练状态写成当前事实。
