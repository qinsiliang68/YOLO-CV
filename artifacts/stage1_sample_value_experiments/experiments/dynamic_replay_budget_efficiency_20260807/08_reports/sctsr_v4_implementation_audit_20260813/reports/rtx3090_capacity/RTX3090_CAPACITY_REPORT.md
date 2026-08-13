# SCTSR v4：10×RTX 3090 训练容量审计

状态：`PASS_WITH_EXPLICIT_RUNTIME_UNCERTAINTY`。这是工程排期证据，不是训练授权或方法有效性证据。

## 可直接采用的排期结论

- 只计算历史同卡基础训练速度时：约 125.5–152.0 小时，即 5.23–6.33 天。
- 建议实际预留：`8 个连续自然日`（182.4 小时容量），不含 discovery 决策停顿。
- 若 discovery 门失败并按合同停止：p90 加 20% 容量约 70.5 小时。

## 同型号历史证据

- 40/40 个 run 为 RTX 3090、yolo11l、200 epochs、batch 128、imgsz 224 且 status=ok。
- 历史全体中位数 15.507 小时；最近似 120,600-image 两次均值 14.263 小时；nearest-rank p90 17.271 小时。

## 依赖与工作量

- discovery：8 个 parent + 64 个 child，在 10 卡上的依赖 makespan 为 680 epoch-time units。
- confirmation：14 个 parent + 112 个 child，最优容量粒度下为 1080 epoch-time units。
- 全部通过时：198 个物理训练 job、16,720 个 base epochs、15,683,360 个 base optimizer steps。
- 另有 6,402,880 次独立 replay microbatch 调用、6,864,000 次 replay occurrence。
- occurrence 证据规模为 2,006,400,000 条 base rows，必须单独计入 CPU、磁盘和 Parquet 开销。

## 为什么不是一个精确小时数

历史 3090 run 将附加样本并入 Dataset，而 v4 在固定 base step 内执行独立 replay forward/backward，并记录全量 occurrence/telemetry/事务证据；历史记录没有直接测量这些 v4 开销。因此 20% 是容量缓冲，不是测得的加速/减速系数，也不是置信区间。正式排队前仍需在一台目标 3090 上跑 NR 与 replay 的一 epoch 工程校准。

Taskbook SHA-256：`732FF49FC9DF90707B2C64F962ACC17256514612AED89B8B74B4E3E643498A66`。
