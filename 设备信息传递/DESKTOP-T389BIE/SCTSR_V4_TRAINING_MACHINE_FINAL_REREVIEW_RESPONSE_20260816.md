# SCTSR v4 训练机最终复核回复（2026-08-16）

## 最终结论

```text
PASS
READY_FOR_FORMAL_AUTHORIZATION_REVIEW
NOT_AUTHORIZED_FOR_FORMAL_TRAINING
```

训练机已对开发分支精确 HEAD `6d1ab662fa4baa8d030dd2e3629ea9829e3497c0` 完成独立复核。PR #3 提出的两个阻断缺口均已在代码、负向测试和真实 RTX 3090 工程 canary 中通过；本轮未发现新的代码阻断项。

该结论只表示可以进入正式授权审查，不表示已经签发 assignment、engineering gate、pilot release 或 execution token，也不允许直接启动正式训练。

## 复核对象与训练机

- 远端分支：`codex/sctsr-v4-training-output-fixes`
- 复核 HEAD：`6d1ab662fa4baa8d030dd2e3629ea9829e3497c0`
- 复核结束时远端分支仍指向同一 HEAD。
- 训练机：`DESKTOP-DGQTJ9O` / `192.168.100.9`
- GPU：NVIDIA GeForce RTX 3090，UUID `GPU-396ff5c3-1c32-1614-80fe-f2822833da4b`
- Python：3.11.15
- Torch：2.11.0+cu128
- CUDA build：12.8
- NVIDIA driver：531.29
- 锁文件 SHA-256：`7CB1067D3C7098B6A10EC74230523A7A75B6A7E5C94D2A78CEC61977DEE61237`
- 训练机源码：`C:\Users\ASUS\Desktop\ssh\AI\sctsr_rereview_6d1ab66`
- 已验收私有环境：`C:\Users\ASUS\Desktop\ssh\AI\sctsr_rereview_6d1ab66\.venv`

训练机源码已绑定真实 Git metadata，最终核验为 HEAD 一致、`git status` clean。共享旧环境中残留的 `ultralytics-opencv-headless` 顶层 `tests` 包已从本项目私有环境剔除；随后执行 locked/offline sync，最终为 57 个锁定包且 `uv pip check` 通过。该私有环境可在锁文件不变时复用，不需要每次重装。

## 两个缺口的独立复核

### 1. claim 后异常终止与立即 RESUME

- `run_common_parent.py` 和 `run_branch.py` 在 claim 后的 resume preparation、trainer setup、训练调用及 branch finalization context 均位于 `execute_claimed_phase` 保护边界内。
- claim 后异常会立即写入 `FAILED`，原异常保持向上抛出。
- 合法 RESUME 可立即 generation+1 接管。
- 训练机负向测试：`5 passed, 21 deselected`。
- 完整 v4 套件进一步覆盖该路径，无残留 `RUNNING` 阻断。

开发回复中的“五阶段”测试本身复用了同一个 wrapper 故障注入入口，集成覆盖描述偏强；训练机因此额外检查了两个 runner 的实际 claim 后代码边界。结构检查与动态终止/RESUME 行为一致，此项不再构成阻断。

### 2. loader 精确 role-tree 与 junction/reparse 拒绝

- loader 只接受注册的 `train/val` role roots 和精确 class/file ancestor closure。
- sibling class、sibling role、额外文件以及祖先 symlink/junction/reparse 均被拒绝。
- 训练机定向测试：`17 passed`。
- 控制端另用真实 Windows junction（非 monkeypatch）复现路径逃逸，返回 `DATASET_CONTENT_MISMATCH`。

## 训练机测试结果

```text
uv lock --check                                      PASS
claim 负向测试                                       5 passed
dataset_adapter 负向测试                             17 passed
tests/stage1_sctsr_v4                                468 passed in 301.67s
tests/stage1_dynamic_replay_v3                       181 passed, 3 skipped
compileall（v4 + v3 代码与脚本）                      PASS
protected trees（stage1_gapvalue240/v3/YOLOv11）     与已审 HEAD 一致
```

完整 v4 最终复跑在 `GIT_NO_LAZY_FETCH=1` 下通过，所需 Git blobs、冻结资产和 evidence 均已本地化，不依赖运行时访问 GitHub。

## RTX 3090 工程 canary

```text
status:                         PASS
source_tree_git_head:           6d1ab662fa4baa8d030dd2e3629ea9829e3497c0
source_tree_git_dirty:          false
source_tree_digest:             5AD51BE111606DEB8661689BD762825A05CCF10733F16DDB4A2959035F7CAD52
device:                         NVIDIA GeForce RTX 3090 / cuda:0
real_image_count:               4
real_image_bytes_verified:      true
real_yolo_weight_verified:      true
forward_passed:                 true
base_backward_passed:           true
replay_backward_passed:         true
optimizer_step_count:           1
ema_update_count:               1
checkpoint save/reload:         PASS
corrupt checkpoint rejection:   PASS
partial generation quarantine:  PASS
total_seconds:                  11.328092099633068
```

- D 盘产物根：`D:\ssh\AI\artifacts\sctsr_rereview_6d1ab66_canary_20260816`
- CLI receipt SHA-256：`6DD6303053243C74AAE797A2FE2DFA7C4AEF6A968AD2AA25A45F3D923A3D6407`
- canary artifact manifest：15/15 文件 bytes/SHA 全部独立复算匹配。
- artifact manifest digest：`00666CF5EF41E93C72DC285DEC95E3211C8114EF4A81AA2B36F7A848FCB2349B`
- canary 结束后：无 `python/uv/yolo` 残留进程；GPU 回落到 399 MiB、0% utilization、35°C。

GitHub 同步了两份轻量 JSON 证据：

- `artifacts/.../sctsr_v4_training_machine_final_rereview_20260816/ENGINEERING_CANARY_CLI_RECEIPT.json`
- `artifacts/.../sctsr_v4_training_machine_final_rereview_20260816/ENGINEERING_CANARY_ARTIFACT_MANIFEST.json`

训练机原始 receipt 与 Git 副本均为 9,601 bytes / `6DD630...D6407`；原始 artifact manifest 与 Git 副本均为 2,980 bytes / `1228D70C5946DCCA7EB6FDCF5C6BB6AD901896740FF2D2BE0C0868E35BA6ED8F`。两份 Git 证据与训练机原始文件 bytes/SHA 完全一致。

## 正式训练边界

本轮明确核验以下值均为 false：

```text
assignments_generated
engineering_gate_generated
pilot_release_generated
formal_training_started
blind_holdout_opened
test_accessed
method_effectiveness_claimed
```

因此当前不能直接执行正式 runner。下一步只能对该精确 HEAD 和上述 source-tree/lock identity 进行正式授权审查，并签发相互绑定的 assignment、engineering gate、pilot release 与 execution token。任何一个缺失或 identity 不一致都必须 fail closed。

工程 canary 的 checkpoint、frontier、prediction 和 synthetic 指标不得转作正式结果，也不得作为 SCTSR 有效性证据。
