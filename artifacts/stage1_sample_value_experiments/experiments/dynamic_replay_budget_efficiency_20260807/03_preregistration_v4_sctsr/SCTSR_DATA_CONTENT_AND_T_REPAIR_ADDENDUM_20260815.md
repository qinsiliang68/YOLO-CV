# SCTSR v4 数据内容隔离与 T 去重修订

状态：`IMPLEMENTED_NOT_FORMALLY_RUN`。本修订只修复数据身份、公平性和运行时绑定，
不构成正式训练授权，也不声称 T、R2 或 SCTSR 有效。

## 1. 触发证据

对冻结的 384,000 行非 test 图像内容账本按 `image_sha256` 复核得到：

- 383,887 个唯一图像 SHA；
- 113 个重复内容组；
- 76 个跨科学角色内容组；
- 历史 canonical base 内有 11 个重复内容组；
- 原 T 有 3,000 个 sample ID，但只有 2,999 个唯一图像 SHA。

路径身份互斥不能替代图像字节互斥。正式 validator 因此必须同时验证 sample ID 和
图像 bytes/SHA。

## 2. 冻结的评价集修复规则

历史 base 和历史训练产物不可改写。有效评价集使用唯一确定的优先级：

`base > val_model > val_cal > val_op`

同一图像 SHA 若出现在 base 和评价角色，保留全部历史 base occurrence，排除全部评价
occurrence；若只出现在评价角色，按上述角色优先级和 canonical sample ID 保留一个，
排除其余 occurrence。不得人工挑选，不得读取 test/blind，不得用标签、loss、confidence、
RHO、gradient 或 endpoint 结果决定保留项。

冻结 overlay：

- `assets/DATASET_CONTENT_EXCLUSIONS_v1.csv`
- 102 行；
- SHA-256 `CED3DE0E070F82DD5AE4B692D478FF6D3FA0015DB62186F72110E16DFCD0C417`；
- 有效 `val_model=23,996`、`val_cal=119,962`、`val_op=119,940`；
- 应用后跨角色内容冲突为 0，评价集内部内容重复为 0。

## 3. T 的最小内容去重

历史 RUN_010 文件原地保留。其重复内容组涉及：

- 保留 rank 495：`Det/images/normal_train/00175370.png`；
- 移出 rank 1001：`Det/images/normal_train/00859781.png`；
- 以 `Det/images/normal_train/00859064.png` 填回 rank 1001。

替换项与移出项均为 label 0、`learnable_hard`、OOF fold 8、
`filename_bucket_1000:859`，因此 T 的四字段 quota 完全不变。候选只在该精确 cell 内按
历史 GapCritical score 降序、sample ID 破同分选择；图像 SHA 只用于可靠性去重门。

派生 T：

- `assets/T_STRESS_CONTENT_UNIQUE_v1.csv`；
- 3,000 IDs / 3,000 unique image SHA；
- SHA-256 `62F7CE1E1DD4E34470E7C085D1DA290F4EAFC618DCB9C5AFE94DF46D6E219BD6`；
- identity digest `D9702F54DA3D9C7C4E27B657B7EC7A5FD235DEC72E2257AE5029E0C62D7482C7`。

T 仍只是历史符号反转压力集合，不是已验证 selector。

## 4. R2 联动冻结

T 身份变化后必须重建共享 R2，禁止继续引用旧 digest。新 R2：

- 3,000 unique；
- 与派生 T overlap=0；
- label、historical dynamic bucket、OOF fold 精确匹配；
- 仅 `oof_group_id` 发生容量下界 378 次位移；
- group TV=0.126；
- selection seed=20260812；
- identity digest `957346D5178CA9397181D0DB47250533E9D659A74A3E7AAFF171FBEE5A0D194B`；
- `R2_U`、`R2_F`、`T_TO_R2_AT_160` fallback 必须复用同一 pool。

## 5. 运行时强制条件

训练 staging 的实际物理文件必须逐项与内容账本 bytes/SHA 一致，不能只按 basename+label
对齐。val_model 必须先应用有效身份 overlay 再建 batch。E200 endpoint 必须使用 trainer
已经验证并写入 binding 的同一个绝对 dataset root；禁止退回仓库硬编码默认路径。

本机真实 Sewer-ML 全量复核读取 384,000 张图、82,637,967,451 字节并 PASS。该结果是
工程身份验证，不是科学结果。新 source/contract/release/seed/token 尚未冻结前，
`formal_training_started` 必须保持 false。
