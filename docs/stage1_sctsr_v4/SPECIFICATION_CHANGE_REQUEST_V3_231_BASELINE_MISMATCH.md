# SCTSR v4 v3 回归基线规格变更请求

状态：`OPEN_TASKBOOK_ACCEPTANCE_BLOCKER`

本请求不降低 v4 测试要求，也不授权训练。它登记 taskbook SA-266 与当前冻结仓库可复现事实之间的矛盾。

## 冻结要求

Taskbook 附录 C 声称当前 v3 基线为 `231 passed`；SA-266 要求：

```powershell
uv run pytest tests/stage1_dynamic_replay_v3 -q
```

至少得到 231 passed。

## 当前可复现事实

在完整 `C:\GitHub\YOLO-CV` checkout、Python 3.11 `uv` 环境中，tracked `tests/stage1_dynamic_replay_v3` 当前包含 34 个 Python test files。上述命令得到：

`181 passed, 3 skipped`

3 项 skip 都是本地 evidence integration：1 项需要登记 Desktop mirror，2 项需要 21 份 literature anchor source files。这些依赖不应伪造，也不能通过复制任意目录强行消除。

Taskbook、现有 tracked tree 和本机 clean-checkout evidence 仍不能复现 `231 passed`。因此，历史 `231 passed` 文本不能替代当前命令证据。

## 对验收的影响

- v4 suite 可以独立全绿；
- v3 源码没有被 v4 提交修改；
- 但 SA-266 在当前 taskbook 下必须是 `FAIL`；
- detailed self-audit 和 strict closeout 不得输出 PASS；
- 不允许新建 48 个无语义测试来凑数量。

## Owner 必须选择其一

1. 恢复产生历史 231 结果的缺失 test files、fixtures 和原始 receipt，并使当前 clean checkout 可复现；或
2. 通过新的 preregistration/taskbook blob，把 SA-266 改成当前 tracked test inventory 的明确基线，同时解释 231 与 183 的差异；或
3. 提供可哈希验证的历史 source tree/bundle，在隔离环境复跑 231，然后明确它是历史兼容基线而非当前 tree 的测试数量。

在 owner 决定前，本实现只报告实际结果，不篡改 taskbook，不伪造 PASS。
