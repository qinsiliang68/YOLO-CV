# SCTSR v4 条件公平与因果可解释合同

## 1. 公平的定义

本合同中的“公平”不是所有 arm 完全相同。它表示每个正式 contrast 中，除预注册
treatment difference 外，所有会改变训练轨迹或评价身份的条件都必须相同并有
逐字节证据。若无法证明，run 即使 exit code 为 0 也作废。

## 2. 所有 arm 共同冻结的条件

每个 training seed 必须共同冻结：

- model family：`yolo11l`；
- initial checkpoint SHA：
  `6B56513A5D8BDAE6B8F0A36DACAF01B26D5A522BA1B34197C3BAC9FA6463366C`；
- canonical training lock SHA：
  `7AFD96784F4994903892BD5BA8477396AAF5EFFBFF158A1C57225428BF01F74E`；
- 200 epochs、batch 128、workers 4、imgsz 224、AMP true；
- canonical base denominator 120,000；
- 每 epoch 938 个 base batches/base optimizer steps；
- base sample identity、base order 和逐样本 augmentation counter-domain；
- optimizer、scheduler、warmup、AMP scaler、EMA 和 global-step 时钟；
- E1-E120 common parent 全部 checkpoint 状态与 SHA；
- source tree、runtime policy、asset registry、contract 和 seed identity；
- E200/EMA/val_op endpoint 与评价代码。

八个 child 必须引用同一 E120 parent checkpoint SHA。不能分别重跑 E1-E120 后
声称“配置相同”就是 common parent；必须是同一物理 checkpoint 字节。

## 3. Base process 锁

基础 DataLoader 只能包含 canonical base：60,000 defect + 60,000 normal。replay
不得拼入 dataset、不得改变 loader length、不得产生第 939 个 step。

每个 base step 只允许一次：

1. base forward；
2. base backward；
3. 可选 replay forward/backward；
4. unscale；
5. gradient clip；
6. optimizer/scaler update；
7. zero-grad；
8. EMA update。

禁止 replay 自己调用 optimizer step、scheduler step、EMA update 或 global-step
increment。禁止隐式 accumulation、auto batch reduction、OOM 后减 batch 继续、
多 GPU/DDP 或拆分一个 base step。

尾 batch 为 64 时，replay microbatch 上限随实际 base batch 变为 16；普通 batch
128 的上限为 32。replay loss 必须为：

`sum(per_sample_cross_entropy) / 128`

分母固定为 canonical base batch size 128，不得改成 replay microbatch size，
也不得用 mean CE 抵消 replay dose。

## 4. RNG、augmentation 和 BatchNorm

- base order seed 域只由 `training_seed + epoch` 推导；
- base augmentation seed 域只由
  `training_seed + epoch + canonical sample_id` 推导；
- replay augmentation 使用独立域和 occurrence token；
- replay 前后 Python、NumPy、Torch CPU、全部 CUDA RNG 必须字节一致；
- replay forward 后恢复所有 BatchNorm running mean/variance/
  `num_batches_tracked`；
- replay parameter gradients必须保留，不能连同 BN/RNG 一起回滚。

任何 arm 如果改变后续 base augmentation、base order 或 BN base trajectory，身份/
timing contrast 即失去解释性。

## 5. Identity pool 公平

### 5.1 T

T 固定为 3,000 个历史 sign-reversal stress IDs。不得用 RUN_013、RUN_016 或
“相同 digest 的另一个方便文件”替代 canonical path；path、bytes、SHA 和 digest
均需一致。

### 5.2 R1

R1 是 canonical eligible base 上的全局随机。它不匹配 T 条件，作用是估计普通
随机额外曝光。随机过程必须由登记 selection seed 决定，pool 生成后冻结身份与
SHA；不得依据本 seed 结果重抽。

### 5.3 R2

R2 是方法匹配随机，目的不是做另一个 selector，而是在 treatment 前条件相同的
范围内剥离 T 的身份效应。禁止给 matcher 访问 loss、confidence、RHO、gradient、
AUM、correct rate、未来 checkpoint 或 endpoint 字段。

2026-08-15 owner addendum 后，R2 要求为：

- 3,000 unique；
- 与 T identity overlap 为 0；
- exact label + historical dynamic bucket + OOF fold；
- `oof_group_id` 语义是
  `FILENAME_BUCKET_SURROGATE_NOT_TRUE_VIDEO_ID`；
- 只允许对 `oof_group_id` 做容量下界为 378 的最小 displacement；
- 三字段 quota 不可满足或试图再放宽第二字段时 fail closed。

真实资产审计得到 172 个 shortage strata、378 个缺口。批准算法先耗尽全部可用
四字段 exact capacity，再在相同三字段 cell 内 counter-hash 填充 378 个缺口；
group TV 必须恰为 0.126。R2 固定为 3,000 unique、T overlap=0、digest
`075FC31...19B6BECC`。`R2_U`、`R2_F` 和 fallback 共用该 pool；不得各自重抽。
任何额外 relaxation、nearest outside coarse cell、replacement、回用 T、改标签或
少于 3,000 unique 都是实验无效，不是“临时工程修复”。

## 6. Schedule、dose、unique 和 repeat

| arm family | active epochs | slots/epoch | total replay | pool unique | per-ID multiplicity |
| --- | ---: | ---: | ---: | ---: | ---: |
| U | 80 | 600 | 48,000 | 3,000 | 16 |
| F | 40 | 1,200 | 48,000 | 3,000 | 16 |
| T→R2 | 40 T + 40 R2 | 600 | 48,000 | 3,000 + 3,000 | 8 + 8 |
| T→NR | 40 T | 600 | 24,000 | 3,000 | 8 |
| NR | 0 | 0 | 0 | 0 | 0 |

每个 epoch receipt 必须分别报告 planned/actual numerator、denominator、rate、slots、
unique、within-epoch repeat、cumulative replay occurrences、每 ID replay count 和
epochs since last replay。不能只报告“回流了 3,000 张”。

U/F 的身份、total occurrence、per-ID multiplicity 必须一致，才能解释为 timing。
T→R2 与 T→NR 的累计 dose 故意不同，分析时必须明确，不能伪称 dose matched。

## 7. 数据内容和 split 公平

正式 asset registry 除 manifest identity 外，还绑定 384,000 张非 test 图像的
byte-level ledger：

- 120,000 train；
- 24,000 val_model；
- 120,000 val_cal；
- 120,000 val_op；
- 82,637,967,451 个物理图像字节；
- ledger SHA：
  `B2B61509AB4451C881FE7E9D0AAFB3F9D3CC0981A78AB9337C54C320E3E96D2C`。

相同文件名/标签但不同图像字节必须失败。不得自动搜索另一个数据目录、相似文件、
软链接目标或 latest dataset。test/blind 不在 ledger 中，也不得在 formal runner
参数中出现。

## 8. Endpoint 和统计公平

- 所有 arm 只在固定 E200/EMA 产生正式 prediction；
- val_op 只做 endpoint evaluation；
- prediction 每行绑定 sample ID、label、logits、raw probability、split manifest
  SHA 和 checkpoint SHA；
- FN=0..95 必须恰有 96 个 tie-safe 点；
- `TN_at_FN95` 与 `FN_at_TN68253` 各自保存 threshold/tie/reachability；
- paired analysis 只能比较同 training seed；
- 缺 arm、缺 seed、错 parent 或错 endpoint 不得改用 unpaired mean；
- Holm family 必须预注册，不得看结果后删比较。

## 9. 直接作废条件

以下任一项发生，run/contrast 直接 invalid，不允许仅打 warning：

- parent SHA、seed、source、contract、asset、runtime、schedule 或 pool SHA 错；
- trainer 读错数据根、漏图、同名错字节或 split role 错；
- replay 改变 base loader length/order/augmentation 或 optimizer-step count；
- replay 未恢复 RNG/BN，或误恢复 parameter gradient；
- OOM 后改 batch/accumulation/steps 继续；
- E200 以外 checkpoint、MODEL 替代 EMA、`best.pt` 或 val_op 选点；
- resume 跳过 receipt chain、跨 generation/seed/arm/parent；
- R2 overlap、超出 owner addendum 的 quota relaxation 或 terminal leakage；
- Parquet fallback、半写文件、缺 SHA/row count/schema；
- test/blind access；
- 只有 exit code 0，没有 canonical completion audit。

## 10. 公平性证明所需产物

每个 run 必须至少有：formal identity、authorization snapshot、execution claim、
run-intent acknowledgement、prepared-trainer binding、dataset-content binding、schedule、
pool binding、parent/lineage binding、每 epoch generation manifest、occurrence/step/
exposure/telemetry Parquet、完整 checkpoint、receipt chain、prediction、frontier、
artifact index、validation report 和 closeout。缺一项不能用 Markdown 声明补齐。
