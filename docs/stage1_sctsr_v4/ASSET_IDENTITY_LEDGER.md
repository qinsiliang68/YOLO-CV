# SCTSR v4 资产身份账本

## 1. 使用规则

本文件解释正式资产身份；机器权威来源始终是
`configs/stage1_sctsr_v4/asset_registry_v1.json`。训练机不得凭本文件中的文件名
猜路径，也不得自动寻找“差不多的文件”。每次 release 必须重算 registry file
SHA、registry semantic digest 和其中每项 bytes/SHA。

当前 registry：

- file bytes：12,110；
- file SHA-256：
  `ABF938C143537C823DC6FE0513130513190BCCFC27B4AD9ED08C60B118D37124`；
- semantic digest：
  `6DB17B9627013A8CBD1327A6B9D3F4F705AA403D15C07A5F52C81512F505F0D7`；
- canonical base denominator：120,000；
- base sample-label digest：
  `7884B823E8A55D4B1BF1A3285CF551FB03E63A2A35F2A95E868F5301A87AA686`；
- `val_target_available=false`。

任一后续合法修订会改变上述 file SHA 和 semantic digest，必须有独立 addendum、
source manifest 和签名 release；不得手改本表后继续使用旧 release。

## 2. 规范、代码和 runtime

| role | canonical path | bytes | SHA-256 / digest |
| --- | --- | ---: | --- |
| taskbook | `artifacts/.../03_preregistration_v4_sctsr/SCTSR_EXPERT_IMPLEMENTATION_TASKBOOK.md` | 87,333 | file SHA `732FF49F...98A66`; Git blob `b201d021...05c6be` |
| contract | `configs/stage1_sctsr_v4/contract_v1.json` | 2,885 | file SHA `DF30244A...E0271` |
| R2 addendum policy | `configs/stage1_sctsr_v4/r2_matching_policy_v1.json` | 1,122 | file SHA `E58E9D70...5EA3C`; digest `33B4681F...E7D4F` |
| arms | `configs/stage1_sctsr_v4/arms_phase1_v1.json` | 6,126 | file SHA `DD7A3FF7...07277` |
| contract+arms semantic | derived by validator | N/A | `8AA9CC9F...E9E0C` |
| runtime policy | `configs/stage1_sctsr_v4/runtime_policy_v1.json` | 1,138 | file SHA `5628D5BE...49626`; digest `12A07710...8838` |
| canonical training lock | `configs/stage1_gapvalue240/CANONICAL_TRAINING_LOCK_v1.json` | 4,623 | `7AFD9678...F74E` |
| initial checkpoint | `yolo11l-cls.pt` | 28,553,700 | `6B56513A...3366C` |

省略号只用于本说明的视觉缩短；机器配置、acknowledgement、release 和 receipt 必须
使用完整 64 位 SHA。

## 3. Canonical train

| asset ID | path | rows | bytes | SHA-256 | label/split |
| --- | --- | ---: | ---: | --- | --- |
| `canonical_base_defect_manifest` | `data/final_sewerml_dataset/manifests/train_manifest.csv` | 60,000 | 16,703,999 | `964193654D91483BA5C28411F56716E4057B6E1D5C9A0DA1EBBD24D30018C07C` | 1 / `train` |
| `canonical_base_normal_manifest` | `data/final_sewerml_dataset/manifests/normal_train_manifest.csv` | 60,000 | 24,189,732 | `196474FC6EEF08D982FBECEE94DD544A4A5AC7220587DD58C9B2C186EAD7652C` | 0 / `normal_train` |

两个 manifest 必须 identity-disjoint，合并后恰好 120,000。分母从这两个已哈希
manifest 推导；schedule 中出现的 120,000 只是派生校验值，不是另一个可修改输入。

## 4. 注册 validation roles

| role/component | rows | bytes | SHA-256 |
| --- | ---: | ---: | --- |
| val_model defect | 12,000 | 3,468,523 | `9606AE90933008E49D7EE354616BA998351CCFB578AEFF25F07C9D8B0058F2D9` |
| val_model normal | 12,000 | 5,624,569 | `73B53C0498E74A76BAF43F926DE318CF3AB2179FFE2958F78A5333FE900D7AB4` |
| val_cal defect | 20,000 | 5,513,354 | `0AD8A680031C8193A6818C418E1C7CDF1EFA9AD573E17156F94B44E9F90EA2C4` |
| val_cal normal | 100,000 | 40,931,163 | `4284D6788F53898919719A68AD1E6359F30807578C0AF719DAC444A7772A8276` |
| val_op defect | 20,000 | 5,453,825 | `7869591D76A8E55C0B14B34C7AA06CFDBEBA30EC531C294C6331EFE4839DD6ED` |
| val_op normal | 100,000 | 40,631,374 | `C255A261DF8EEFAFD573A1EBBD2D88C375D6D9244BB3077B319E648FFFC1E33E` |

完整 path 由 registry 给出。train/val_model/val_cal/val_op 跨角色 identity-disjoint。
test/blind manifest 不在 v4 registry 中。

## 5. 图像字节账本

`DATASET_CONTENT_LEDGER_v1.parquet` 是正式数据内容身份，不是普通日志：

- path：
  `artifacts/stage1_sample_value_experiments/experiments/dynamic_replay_budget_efficiency_20260807/03_preregistration_v4_sctsr/assets/DATASET_CONTENT_LEDGER_v1.parquet`；
- rows：384,000；
- bytes：19,350,859；
- SHA-256：
  `B2B61509AB4451C881FE7E9D0AAFB3F9D3CC0981A78AB9337C54C320E3E96D2C`；
- content digest：
  `EDA93977CE43E946D4C795A8FBA30BF39B6AF510034276E739BC51D88DB1DD6E`；
- physical image bytes represented：82,637,967,451；
- format：canonical Zstd Parquet；
- test/blind rows：0。

每行包含 canonical relative path、split、label、source manifest asset/SHA、图像
bytes/SHA、width、height、mode 和 format。正式训练前及 closeout 都逐图重算；
filename、label 或 hardlink 名称相同不能替代内容 SHA。

## 6. OOF 与预终端参考

| asset | rows | bytes | SHA-256 |
| --- | ---: | ---: | --- |
| OOF metadata | declared 120,000 | 1,049 | `B4AE826649C8924388B118B0738A341A36013ACEE0B0418B2814E2F3A6C8D4F0` |
| OOF assignments | 120,000 | 51,199,918 | `EE82D19D8B8BC2875842B1DF433CC1B6098D7F88A300B330CA5E45B3642AE0C6` |
| sample value table | 120,000 | 31,819,678 | `376674FCFD5C8378051FC5D1A588ED415CA8DEFA06A19AAD72505AB49B20B980` |
| dynamics summary | 120,000 | 30,737,786 | `84EE044693E9D47295C5A4C6DF05470379FA5CE79B838EDE0CEE3EB1E0DE959F` |

任务书曾记录 OOF metadata 的工作树换行版本 1,076 bytes/SHA `759B...`；clean
checkout 实际 tracked bytes 为上表 1,049/`B4AE...`，已由失败优先修订绑定。不能
恢复旧工作树字节来迎合任务书文字。

`oof_group_id` 全部是 filename numeric bucket surrogate，不是真实 video/source
identity。R2 builder 只能从参考表先投影白名单字段，再交给 matcher。

## 7. T 与 R2 审计资产

- T manifest：590,747 bytes，SHA
  `DFDADC5D75B39A78E0C8995BD46F063C5D969BEA1024E4A332A051D00A172689`；
- T identity digest：
  `85D462C1D95F30FB8B519162BBAD762CC4E9506A185C07D719145F07FE003B4B`；
- R2 old strict status：`R2_QUOTA_INFEASIBLE`；
- shortage：172 joint strata / 378 occurrences；
- owner-approved active policy pool digest：
  `075FC31FE487D3646E89BA1043E5124D9FE49CE9FCC61C1A8041A9CB8196BECC`。

该 digest 已通过 TDD 和真实 120,000-row 物化验证，但仍必须由新的 clean source
commit、contract、pool artifact、schedule 和签名 release共同冻结，不能沿用旧 review
分支的 release identity。

## 8. 训练机最低核验

每台机器必须分别获得 PASS receipt：

1. `validate_contract.py`；
2. `validate_assets.py`；
3. `validate_dataset_content.py`（不得使用 `--ledger-only` 代替）；
4. `build_source_tree_manifest.py` + clean validation；
5. Python/Torch/CUDA/GPU runtime identity；
6. exact trainer override/data root/model path；
7. disk capacity；
8. per-job release/token/acknowledgement。

任何本地镜像都要以 SHA 证明与本表一致；复制成功、文件可读或相同文件大小都不够。
