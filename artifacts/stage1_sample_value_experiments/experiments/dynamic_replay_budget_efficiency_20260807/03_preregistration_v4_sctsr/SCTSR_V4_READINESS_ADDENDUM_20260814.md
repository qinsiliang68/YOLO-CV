# SCTSR v4 训练就绪补充合同（2026-08-14）

## 1. 身份与适用范围

本补充合同绑定：

- 历史基线：`a70ba60485dd32c2f8b4268b8f28ea2d3549f42f`；
- 实现冻结提交：`e9b6df61b0eb02e1d32c29175644f1c2af545afc`；
- 本轮审查输入提交：`f285754108c7b8e37afd7f5f0fa58fe8fb23d38a`；
- 原任务书 blob：`b201d021712e9c6614e119d35f0e14bdf405c6be`。

原任务书保持不可修改。本文件仅订正训练前复核发现的、不会改变科学问题与训练方法的实施身份字段，并登记后续明确批准的规格变更。未被本文件逐项覆盖的原任务书条款继续有效。

本文件不是正式训练 release，不生成 seed、assignment、engineering gate 或 pilot release，也不授权读取 `val_op`、`blind_holdout` 或 `test`。

## 2. OOF metadata 字节身份订正

### 2.1 发现

原任务书和 `asset_registry_v1.json` 登记的是某个历史 Windows 工作树中的 CRLF 表示：

- bytes：`1076`；
- SHA-256：`759B7D7E01506694FA508C6F2B040B510458E91056E9192A9F1D0F9101A6F97C`。

但是仓库 `.gitattributes` 对 JSON 明确执行 `text eol=lf`。因此任意 clean checkout 实际得到：

- path：`artifacts/stage1_oof_folds_10fold_20260617/metadata.json`；
- bytes：`1049`；
- SHA-256：`B4AE826649C8924388B118B0738A341A36013ACEE0B0418B2814E2F3A6C8D4F0`。

旧 CRLF 文件和 Git 跟踪的 LF 文件经 JSON 解析后字段和值完全相同。变化仅为 27 个行尾从 CRLF 规范化为 LF；`seed=20260606`、`n_folds=10`、`total_rows=120000`、`group_sources.numeric_filename_bucket=1156` 以及 OOF 分组语义均未变化。

### 2.2 规范性决定

从本补充合同起，所有 clean checkout、source manifest、asset registry、formal authorization 和训练机预检必须使用 Git 实际检出的 LF 字节身份：

```text
bytes=1049
sha256=B4AE826649C8924388B118B0738A341A36013ACEE0B0418B2814E2F3A6C8D4F0
```

任何训练机若观察到旧 CRLF 身份、其他字节数或其他 SHA，必须停止；不得现场改换行后继续，也不得用“JSON 能解析”绕过字节绑定。正确动作是从冻结提交重新 clean checkout，并重新运行资产验证。

该订正不改变数据行、样本身份、标签、fold、group、T/R1/R2 构造、训练预算、训练 seed 或评价规则，因此不能被解释为方法调整或结果选择。

## 3. 尚未授权事项

以下内容仍保持阻断，直到本文件后续章节和对应机器验证给出明确结论：

- 当前严格 R2 quota 不可行；
- 正式作业级一次性执行令牌尚未定案；
- 独立 `val_target` 不存在，因此 A 保持 `BLOCKED_BY_VAL_TARGET`；
- 未签发正式 release、正式 seeds 或训练作业；
- SCTSR 方法效果未知。
