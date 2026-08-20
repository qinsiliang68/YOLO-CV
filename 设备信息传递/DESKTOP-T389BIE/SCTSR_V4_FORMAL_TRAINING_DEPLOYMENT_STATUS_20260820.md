# SCTSR V4 正式训练部署状态（2026-08-20）

## 当前结论

`FORMAL_TRAINING_ACTIVE_EPOCH_1_IN_PROGRESS`

正式训练已在训练机启动，并已确认进入真实 GPU 批次；这表示部署成功，不表示整轮训练已经完成。

## 运行定位

- 训练机：`DESKTOP-DGQTJ9O`（`192.168.100.9`，RTX 3090）
- 活跃源码提交：`70e54940d8d91114c066ded12da514e085b4c905`
- 活跃任务：`SCTSR_V4_FORMAL_PILOT_PARENT_2019192314_V6`
- 运行 ID：`PARENT_2019192314`
- 控制根：`C:\Users\ASUS\Desktop\ssh\AI\sctsr_v4_formal_control_20260820_v6_exact_view`
- 数据视图：`C:\Users\ASUS\Desktop\ssh\AI\datasets\sctsr_v4_classification_view_70e5494_exact`
- claim 根：`C:\Users\ASUS\Desktop\ssh\AI\sctsr_v4_claim_registry_release_20260820_70e5494_v6`
- 产物根：`D:\ssh\AI\artifacts\sctsr_v4_formal_discovery_pilot_parent_20260820_v6`

## 已确认事实

- 任务状态为 `Running`，只有一个 V6 正式任务，无重复训练进程。
- train 为 120,000 张（两类各 60,000）；val_model 为 23,996 张（11,997 + 11,999）；总计 143,996 张。
- 数据视图仅使用同卷硬链接；train/val_model 的 materialized binding 均已成功写出。
- `PREPARED_TRAINER_BINDING.json`、`FORMAL_AUTHORIZATION_BINDING.json` 和输入快照已写入产物根。
- `2026-08-20 16:21:19 +08:00`：GPU 显存 8,561 MiB，利用率 35%，epoch 事务开始写入。
- `2026-08-20 16:22:22 +08:00`：GPU 显存 8,561 MiB，利用率 47%，训练持续。
- `2026-08-20 16:24:25 +08:00`：GPU 显存 8,561 MiB，利用率 54%，`epoch_0001.generation_1.inprogress` 持续存在。
- blind/test 未打开。

## 前序失败与处置

- V4 在 epoch 前停止：Ultralytics 自动生成的 `train.cache` 被精确 role-tree 校验拒绝。代码已限定并隔离 train/val cache，提交为 `70e54940d8d91114c066ded12da514e085b4c905`。
- V5 在 epoch 前停止：旧 val 视图包含 4 张已冻结排除的图片。未修改旧视图，重新生成上述精确硬链接视图后启动 V6。
- V4/V5 均未产出 epoch 或 checkpoint；失败 claim 和错误证据保留，未被覆盖。

## 后续动作

训练机继续运行 V6。只需监控任务存活、GPU/温度/磁盘和 epoch/checkpoint 产物；若失败，直接依据该任务的 terminal receipt 和 claim 恢复，不重新安装环境、不创建重复任务、不打开 blind/test。

本状态文档是在 V6 启动后提交；活动训练继续绑定 `70e54940d8d91114c066ded12da514e085b4c905`，不会热更新到本状态文档的提交。
