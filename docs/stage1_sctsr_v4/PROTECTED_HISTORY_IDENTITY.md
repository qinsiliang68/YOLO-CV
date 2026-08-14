# SCTSR v4 历史保护文件身份审计说明

## 结论

历史训练源码、已训练产物、旧 queue、release 和 assignment 的内容身份不得因 SCTSR v4 实现或审查而改变。仓库审计以 Git blob 身份和逐文件 SHA-256 作为内容不变性的权威证据；文件系统 `mtime` 只如实记录，不作为内容身份或训练有效性证据。

## 为什么不能把 `mtime` 当成内容身份

Git 不保存工作树文件的 `mtime`。clone、checkout、linked worktree、解压或从同一 commit 重建 clean checkout 时，即使文件字节完全相同，操作系统也会生成新的 `mtime`。因此，在审查专用 clean worktree 中声称“历史文件的原始 mtime 没有变化”是不可验证的，也会把正常 checkout 误报成历史内容被篡改。

审计器仍保存实际 `mtime_utc`、实现开始时间及二者的先后关系，但明确标记：

```text
mtime_verification_status=INFORMATIONAL_NOT_CONTENT_IDENTITY
```

该标记不是豁免内容变化，也不是把未知项写成 PASS。

## 必须同时满足的内容不变条件

每个受保护历史文件必须同时满足：

1. `baseline_git_blob_oid == implementation_source_git_blob_oid`；
2. baseline blob 与 implementation-source blob 的 SHA-256 相同；
3. 当前 clean worktree 经 Git 属性规范化后的 `git hash-object --path` 与 source blob OID 相同；
4. 当前 tracked worktree clean；
5. 相对冻结 baseline 的 protected-path diff 为空；
6. 文件存在，且未被删除、替换、移动或冒充 SCTSR v4 产物。

任一内容条件失败，审计必须 fail closed。仅 `mtime` 晚于实现开始时间、但以上六项均通过时，不得虚假声称 mtime 未变；只能报告它不是 Git 可重建的内容证据。

## 边界

本说明只修正 clean-checkout 审计中的文件系统时间戳语义，不改变任何训练方法、数据角色、arm、seed、预算、schedule、评价指标或科学 estimand，也不构成正式训练授权。
