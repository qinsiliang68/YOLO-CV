# Stage1 动态回流实验训练机操作员说明 v2

## 0. 文档身份

| 项目 | 内容 |
| --- | --- |
| 实验 ID | `dynamic_replay_budget_efficiency_20260807` |
| 当前状态 | `CODE_READY_FOR_OWNER_CANARY` |
| 当前日期 | 2026-08-08 |
| 十机使用截止日 | 2026-09-10 |
| 正式训练入口 | `scripts/stage1_gapvalue240/dynamic_campaign_train_worker.py` |
| 活动预注册 | `03_preregistration_v2/` |
| 活动任务队列 | `04_run_queue_v2/` |
| 当前可进入工程门禁的周期 | `CYCLE_1` |
| 当前仍被科学门禁锁定的周期 | `CYCLE_2`、`CYCLE_3`、`CYCLE_4` |
| blind holdout | `UNBOUND`，最终方案冻结前禁止打开 |

这是一份给训练机操作员、值班人员和接班工程师使用的背景说明。它解释这轮实验为什么存在、
实际训练什么、296 个物理 job 如何对应科学实验、训练期间应监控什么、什么情况必须停止并上报。

具体命令、release、assignment、coordination root、lease 和 fencing 规则以
[DYNAMIC_REPLAY_CAMPAIGN_OPERATIONS_V2.md](DYNAMIC_REPLAY_CAMPAIGN_OPERATIONS_V2.md) 为准。
本文件不替代冻结预注册，也不授权操作员修改科学参数。

## 1. 一句话说明这轮实验

本轮使用完全相同的 YOLO11l、12 万张基础训练样本和冻结超参数，只改变：

```text
回流比例
回流发生的训练阶段
累计回流曝光量
是否使用弱缺陷保护
```

目标是验证：同一批静态 OOF 候选样本能否通过训练阶段控制，跨训练 seed 更稳定地改善
`FN=0..95` 的安全前沿，而不是继续发明新的静态“照片价值分数”。

形式上研究的是：

```text
Value = V(selection, model_state, training_stage,
          replay_schedule, cumulative_exposure, seed, context)
```

而不是假设每张照片都有固定不变的 `V(x)`。

## 2. 为什么要做这轮实验

### 2.1 旧 120-run 给出的结论

旧 120-run 说明，按一次置信度或一次 hard-negative 分数挑选 normal，并不能稳定优于随机回流。
它只能否定“单次高置信误报就是高价值样本”这个简单代理，不能证明不存在有价值样本。

### 2.2 240-run 和完整训练时序给出的新事实

后续审计使用了：

```text
240 个有效 canonical run
80 个 Treatment / R1 / R2 三联
48,000 条逐 epoch 训练记录
120,000 个 OOF 训练样本
10 folds x 200 epochs
```

核心事实是：完全相同的 Treatment 样本集合，在不同 training seed 下可能一个有益、一个有害。
静态样本字段不变，但最终安全前沿发生反转。因此样本作用不能只解释为照片自身的固定属性。

旧结果中，只有极少数局部实例在 `FN=0..95` 原始安全前沿上同时支配两个随机对照，且没有方法
在完整 FN 范围内稳定支配。相对可信的机制线索是：

```text
有效 run 不只是压低困难 normal；
它还必须避免把最弱 defect 的分数一起压低。
```

训练后期约 150 至 160 epoch 附近出现过额外拟合风险，但旧数据不足以把它写成规则。因此本轮先做
严格的时间表和累计剂量对照，而不是直接部署一个未经验证的智能停止器。

### 2.3 为什么仍使用 GapCritical-Strict 候选集合

前两轮冻结 `TREATMENT_GAPCRITICAL_NESTED` 作为 normal replay 候选来源。这里的含义是：

- 它是已知会跨 seed 反转的固定扰动源，适合研究训练阶段控制；
- 三个回流比例可以从同一排名中嵌套截取；
- 它不是已被证明正确的“高价值样本公式”；
- 本轮成功也不能自动证明 GapCritical 是普遍最优静态排序。

## 3. 操作员必须理解的术语

### 3.1 基础训练集

每个正式 run 都包含相同的 120,000 张基础图片：

| 角色 | 数量 |
| --- | ---: |
| defect train | 60,000 |
| normal train | 60,000 |
| 合计 | 120,000 |

这些基础样本在每个 epoch 正常曝光一次。回流不是替换基础训练集，也不是引入新标签。

### 3.2 回流槽位

回流槽位表示在一个 epoch 内额外重复曝光多少张已冻结样本。定义为：

```text
rho = 每个 epoch 的额外回流槽位数 / 120,000
```

| 注册比例 | 常规峰值槽位 | 当 epoch 配置样本数 |
| --- | ---: | ---: |
| 0.5% | 600 | 120,600 |
| 1.0% | 1,200 | 121,200 |
| 2.5% | 3,000 | 123,000 |
| no-replay | 0 | 120,000 |

正式科学名称只使用百分比。绝对张数是程序根据 120,000 自动换算的执行结果。

### 3.3 三种回流时间表

#### 持续回流 `CONTINUOUS`

```text
epoch 1..200: 始终使用注册比例 rho
```

#### 同峰值衰减 `SAME_PEAK_TAPER`

```text
epoch 1..140: rho
epoch 141..160: 从 rho 线性衰减到 0
epoch 161..200: 0
```

它与持续回流具有相同峰值，但累计回流曝光量是持续回流的 75%。

#### 总剂量匹配衰减 `DOSE_MATCHED_TAPER`

```text
epoch 1..140: 4/3 * rho
epoch 141..160: 从 4/3 * rho 线性衰减到 0
epoch 161..200: 0
```

它与持续回流具有相同累计曝光量，但把曝光集中到前中期，用于区分“时间效应”和“总量效应”。

| 比例 | 持续累计曝光 | 同峰值衰减累计曝光 | 总剂量匹配累计曝光 |
| --- | ---: | ---: | ---: |
| 0.5% | 120,000 | 90,000 | 120,000 |
| 1.0% | 240,000 | 180,000 | 240,000 |
| 2.5% | 600,000 | 450,000 | 600,000 |

### 3.4 弱缺陷保护

第三周期会在总回流槽位不变的条件下，把一部分 normal replay 槽位替换为 defect guard：

```text
total replay slots = normal replay slots + defect guard slots
```

guard 不是额外增加训练量。它用于验证：保护 OOF 动态中可学习但容易漏检的弱 defect，是否能防止
normal replay 把 defect low-tail 一起压低。

## 4. 冻结模型与训练参数

每个正式 arm 使用同一个 canonical learner：

| 参数 | 冻结值 |
| --- | --- |
| 模型 | `yolo11l-cls.pt` |
| 模型家族 | `yolo11l` |
| 任务 | binary classification，`target_defect` 对 `no_target` |
| epochs | 200 |
| batch | 128 |
| imgsz | 224 |
| workers | 4 |
| AMP | true |
| deterministic | true |
| patience | 0 |
| cache | false |
| optimizer | `auto`，保持 canonical 行为 |
| 初始权重 SHA-256 | `6B56513A5D8BDAE6B8F0A36DACAF01B26D5A522BA1B34197C3BAC9FA6463366C` |
| canonical lock SHA-256 | `7AFD96784F4994903892BD5BA8477396AAF5EFFBFF158A1C57225428BF01F74E` |

OOM、速度慢或机器型号不同，都不能授权操作员修改这些值。OOM 必须作为失败 attempt 保留。

## 5. 四个七天周期

### 5.1 Cycle 1：高压力下的后期暴露实验

研究问题：2.5% 高压力回流时，减少后期回流是否降低弱缺陷伤害？

8 个配对 discovery seed，每个 seed 三个 arm：

| arm | 回流策略 | 科学作用 |
| --- | --- | --- |
| `C1_T_RHO_2P5_CONTINUOUS` | 2.5% 持续到 200 epoch | 高压力参考 |
| `C1_T_RHO_2P5_SAME_PEAK_TAPER` | 2.5% 同峰值衰减 | 检验减少后期暴露 |
| `C1_NR_NO_REPLAY` | 无回流 | 判断回流相对基础 learner 的作用 |

规模：

```text
8 seeds x 3 arms = 24 个逻辑 run
编译为 88 个物理 segment job
当前状态 = ENGINEERING_GATE
```

Cycle 1 没有给每个条件重复配置 R1/R2，因为本周期的主问题不是重新筛选静态排名，而是同一 selection
下的时间表因果对照。最终确认周期仍保留全局随机和不重合难度匹配随机。

### 5.2 Cycle 2：比例、时间和累计剂量拆分

Cycle 2 使用与 Cycle 1 相同的 8 个 discovery seed：

| 比例 | 持续 | 同峰值衰减 | 总剂量匹配衰减 |
| --- | --- | --- | --- |
| 0.5% | 有 | 有 | 有 |
| 1.0% | 有 | 有 | 有 |
| 2.5% | Cycle 1 已有 | Cycle 1 已有 | Cycle 2 补齐 |

规模：

```text
8 seeds x 7 arms = 56 个逻辑 run
编译为 208 个物理 segment job
当前状态 = HELD
依赖 = Cycle 1 的注册决策
```

解释规则：

| 结果 | 主要解释 |
| --- | --- |
| 两种衰减都优于持续回流 | 后期暴露本身可能有害 |
| 只有同峰值衰减更好 | 主要问题可能是累计回流量过大 |
| 总剂量匹配衰减也更好 | 回流发生时间比累计总量更关键 |
| 两种衰减都无改善 | 当前后期暴露假设被削弱或否定 |

### 5.3 Cycle 3：弱缺陷 guard

Cycle 3 当前只有模板，不能执行。Cycle 2 完成后必须先冻结一个比例和一个时间表，再生成新的
版本化 queue/release/assignment。预注册模板包括：

```text
无 guard
历史 OOF GapGuard-Raw，guard 占 10%
可学习弱缺陷 guard，guard 占 10%
可学习弱缺陷 guard，guard 占 20%
匹配随机 defect guard，guard 占 10%
匹配随机 defect guard，guard 占 20%
```

操作员不得根据中间成绩自行挑 guard 或修改比例。

### 5.4 Cycle 4：14 个全新 seed 正式确认

Cycle 4 当前只有模板。Cycle 3 冻结最终方案后，才允许在 14 个完全未见 seed 上生成六臂确认：

```text
最终完整策略
同一 selection 的持续回流
同一动态策略但无 guard
全局随机回流 R1
与 Treatment 不重合的难度匹配随机回流 R2
no-replay
```

规模计划为：

```text
14 unseen seeds x 6 arms = 84 个逻辑 run
```

blind holdout 在最终策略、对比和分析规则冻结前禁止打开。

## 6. 为什么是 80 个逻辑 run 和 296 个物理 job

当前 `03_preregistration_v2/EXPERIMENT_MATRIX.csv` 冻结了 Cycle 1/2 的 80 个完整 200-epoch
逻辑 run。执行时每条训练轨迹被拆为四个边界：

```text
segment 1: epoch 1..140
segment 2: epoch 141..150
segment 3: epoch 151..160
segment 4: epoch 161..200
```

持续回流和同峰值衰减在前 140 epoch 完全相同，因此程序只训练一次共享前缀，然后从 epoch 140
的同一 checkpoint 分支。这能减少重复计算，并让后期策略比较拥有完全相同的前史。

最终计数为：

| 周期 | 逻辑 run | 物理 job | 状态 |
| --- | ---: | ---: | --- |
| Cycle 1 | 24 | 88 | `ENGINEERING_GATE` |
| Cycle 2 | 56 | 208 | `HELD` |
| 合计 | 80 | 296 | 仅 Cycle 1 可进入下一工程门禁 |

因此：

```text
296 个 job != 296 个独立 200-epoch 模型
296 个 job = 80 条逻辑训练轨迹的分段与共享前缀
```

## 7. 从 queue 到单任务训练的执行链

```text
preregistration_v2
  -> run_queue_v2
  -> engineering_gate_v2
  -> release_v2
  -> assignment_v2
  -> ACTIVE_ASSIGNMENT.json
  -> shared coordination root
  -> single-job worker
  -> checkpoint / telemetry / predictions / result
```

关键规则：

1. 每个训练进程必须恰好接收一个 `--job-id`。
2. 禁止一个 worker 连续隐式领取多项任务。
3. controller 只是可选调度层，退出不会终止已领取的 worker。
4. 同一 cycle/seed block 必须由同一台机器执行。
5. assignment 只改变任务放置，不能改变 seed、selection、schedule 或依赖图。
6. 训练前必须同时校验 release、assignment、canonical lock、machine config 和输入 manifest 的身份。
7. 未在 release 内的 job、仍为 `HELD` 的 job、旧 v1 queue 中的 job都会被拒绝。

## 8. 每个逻辑 run 产生什么

正式输出根：

```text
artifacts/stage1_sample_value_experiments/experiments/
  dynamic_replay_budget_efficiency_20260807/
    05_training_runs/<logical_run_id>/
```

典型目录如下：

```text
<logical_run_id>/
  inputs/
    replay_identity_manifest.csv
    monitor manifest 副本
    machine_assets_validation.json
    code_provenance.json
    input_identity.json

  trainer/
    args.yaml
    results.csv
    weights/best.pt
    weights/last.pt

  training_state/
    last.pt
    checkpoint_epoch_0120.pt
    checkpoint_epoch_0140.pt
    checkpoint_epoch_0150.pt
    checkpoint_epoch_0160.pt
    checkpoint_epoch_0180.pt
    checkpoint_epoch_0200.pt
    对应 checkpoint sidecar

  process_telemetry/
    epoch_0001_process_telemetry.parquet
    epoch_0001_process_telemetry.json
    epoch_0001_role_loss_summary.json
    ...
    epoch_0200_process_telemetry.parquet
    epoch_0200_process_telemetry.json
    epoch_0200_role_loss_summary.json

  key_checkpoint_predictions/
    probe_manifests/
    epoch_0120/
      val_op_predictions.csv
      val_op_predictions.manifest.json
      causal_train_probe_predictions.csv
      causal_train_probe_predictions.manifest.json
      checkpoint_prediction_manifest.json
    epoch_0140/
    epoch_0150/
    epoch_0160/
    epoch_0180/
    epoch_0200/

  resource_logs/
    resource.csv

  dynamic_training_audit.json
  job_results/<job_id>.json
```

稳定文件使用原子发布。临时文件、半写 CSV、缺失 sidecar 或校验和不一致都不能被标记为 `COMPLETE`。

## 9. “全 epoch 训练动态”具体保存什么

这里的“全 epoch”表示 epoch 1 至 200 每一轮都采集，而不是每 10 轮采集一次。

### 9.1 覆盖全部训练曝光的聚合记录

每个 epoch 记录：

- 实际样本曝光总数；
- base normal、base defect、normal replay、defect guard 的曝光数；
- 各角色 loss 和 `p_defect` 的均值、标准差、最小值、最大值与分位数；
- minibatch 顺序摘要、batch-size 序列摘要和数据增强实现摘要；
- 实际 optimizer steps、学习率、scaler、RNG 状态摘要；
- epoch 训练、评估、checkpoint、写入和 DataLoader 等待耗时；
- GPU 显存、CUDA peak、CPU、RSS 和 child process 数量。

### 9.2 逐样本详细记录

逐样本明细保存以下集合：

```text
冻结的 1,200 个 OOF-only causal monitor 样本
所有实际 replay normal
所有实际 defect guard
```

每个逐样本记录包含 sample ID、标签、角色、曝光次数、loss 分布、`p_defect` 分布和增强摘要。

其余 12 万基础样本仍全部参与训练和角色级聚合，但不会在新实验中每轮逐张永久展开。旧 OOF 证据已经
保存了 `200 x 120,000` 的逐样本预测动态；新实验把逐样本存储集中在因果探针和实际回流集合上。

## 10. 六个关键 checkpoint 的预测

冻结关键 epoch：

```text
120, 140, 150, 160, 180, 200
```

每个关键 checkpoint 对两个 split 进行 raw-score 预测：

1. `val_op`：用于重建业务工作点和原始安全前沿；
2. `causal_train_probe`：冻结的困难 normal、弱 defect 和机制监控样本。

预测 CSV 至少包含：

```text
sample_id
y_true
score
score_raw
p_defect
p_normal
probability_margin
log_odds_defect
entropy_nats
predicted_y
```

训练机只负责生成并校验 raw prediction。阈值扫描、图表和 HTML 由中央只读汇总重建，不能由操作员
在单机上根据中间结果改变下一项训练。

## 11. 最终分析看什么

主分析不使用任意固定加权总分。预注册端点是：

| 层级 | 指标 |
| --- | --- |
| 主安全曲线 | `raw_safe_frontier_fn_0_95` |
| 安全端点 | `FN_at_TN68253` |
| 效率端点 | `TN_at_FN95` |

附加机制分析包括：

- 最弱 defect 的分数和 loss 是否被保护；
- 困难 normal 是否下降；
- seed 反转从哪个 epoch 开始；
- 同峰值衰减和总剂量匹配衰减的差异；
- replay 暴露、optimizer steps、学习率积分和结果的关系；
- no-replay 是否表明回流本身值得继续；
- 最终方案是否在 unseen seeds 上稳定，而不是只出现一个最好 run。

## 12. 梯度诊断的边界

仓库已经提供独立的最后分类层梯度探针，但它不是全部正式 job 的默认训练步骤。它属于 P1 机制 pilot，
需要单独的冻结候选 manifest 和放行命令。

计划输出：

```text
gradient_probe_scalars.parquet
gradient_feature_payload.npz
gradient_probe_manifest.json
```

主要字段包括：

- 单样本最后一层 CE 梯度范数；
- 与困难 normal 目标梯度的点积和余弦；
- 与弱 defect 目标梯度的点积和余弦；
- 跨 checkpoint 的方向一致性；
- 好 seed 与坏 seed 的梯度方向差异。

梯度诊断不自动替代终点复现，也不得未经 pilot 就扩展为 12 万样本、200 epoch、全网络梯度。

## 13. 操作员启动前检查

正式 Cycle 1 pilot 放行前必须全部满足：

1. 本机 checkout 对应 release 绑定的精确 git commit；
2. canonical lock SHA 与本文件一致；
3. 初始 checkpoint SHA 与本文件一致；
4. 本机 machine YAML 仅修改允许的路径和设备字段；
5. 数据集、各 split manifest 和 hardlink staging 在同一文件系统卷；
6. GPU、CPU、RAM、磁盘和文件权限 preflight 通过；
7. 共享 coordination root 十机 canary 为 `10/10 PASS`；
8. 十机 one-job real-data canary 为 `10/10 PASS`；
9. engineering gate v2 已读取并校验底层 evidence，而不是手写顶层 PASS；
10. release v2、assignment v2 和 `ACTIVE_ASSIGNMENT.json` 身份一致；
11. 当前 job 明确出现在 release 和 assignment 中；
12. 当前周期不是 `HELD`；
13. blind holdout 仍为 `UNBOUND`。

当前仓库只完成了本机 RTX 4060 真实图片 canary 和故障恢复演练。十机共享目录 canary、十机真实数据
one-job canary、各机 preflight、正式 engineering gate、release 和 assignment 尚未执行。因此现在不能
直接把 296 个 job 全部启动。

## 14. 训练期间监控什么

### 14.1 任务身份

每个进程应满足：

```text
一个进程
一个 job_id
一个 release
一个 assignment
一个 lease token
一个 fencing generation
```

如果日志显示重复 job、隐式批量领取、身份 hash 不一致或旧 assignment holder 仍在写入，立即停止并上报。

### 14.2 训练进度

操作员应观察：

- 当前 segment 的起止 epoch 是否与 queue 一致；
- resume epoch 是否恰好等于 segment 起点减 1；
- `trainer/results.csv` 是否持续增加且 epoch 唯一；
- 实际 optimizer steps 是否符合每轮配置样本数和 batch=128；
- 140、150、160、200 边界 checkpoint 是否按时形成；
- 120、140、150、160、180、200 的保留 checkpoint 和预测是否完成；
- 每个已完成 epoch 是否同时存在 Parquet、sidecar 和角色汇总；
- heartbeat 是否持续更新，lease 是否仍属于本进程。

### 14.3 GPU 与数据供给

目标是让 GPU 在训练批次期间保持高利用率。瞬时下降可能来自验证、checkpoint、分段切换或磁盘写入，
不能只看某一秒。需要结合以下字段判断：

```text
gpu_util_pct
memory_used_mb
process_cpu_pct
system_ram_available_bytes
disk_read/write deltas
dataloader_wait_fraction
step_time_mean_seconds
eval_seconds
checkpoint_seconds
```

若训练阶段长期低 GPU 利用率，同时 `dataloader_wait_fraction` 持续升高，应检查磁盘、hardlink 数据集、
杀毒软件、CPU 和 worker 子进程。但操作员不能自行把 workers 从 4 改为其他值，也不能改 batch。

### 14.4 磁盘与产物

监控重点：

- staging 与 dataset 必须保持同卷；
- 不允许从 hardlink 静默退化为复制图片；
- 磁盘余量低于 machine config 的门槛时必须在新 job 前 fail closed；
- `.tmp`、attempt 目录和失败日志不是完成态；
- 不能删除 failed attempt、checkpoint sidecar、lease 或 provenance；
- 不应把大 checkpoint、图片或本地 canary 目录提交到 Git。

## 15. 正常状态与异常状态

任务状态机：

```text
UNCLAIMED -> CLAIMED -> RUNNING -> COMPLETE
                    \-> FAILED

CLAIMED/RUNNING --TTL expired--> STALE -> REAPED -> new CLAIMED
旧 generation holder --assignment switch--> FENCED
```

### 15.1 正常完成

只有同时满足以下条件才是完成：

- worker 退出码为 0；
- stdout 返回 `status=COMPLETE`；
- segment 结束 epoch 与 queue 一致；
- `job_results/<job_id>.json` 存在并通过身份校验；
- machine job state 为 `COMPLETE`；
- segment 边界 checkpoint 存在且 SHA 可验证；
- 本 segment 的逐 epoch telemetry 完整；
- 应保留 checkpoint 的 raw predictions 完整；
- lease token 和 active assignment 身份一致。

单独看到 `last.pt`、`results.csv` 或日志最后出现 200/200 都不足以判定完成。

### 15.2 OOM

OOM 是失败 attempt，不允许自动降低 batch、imgsz、AMP、workers 或模型大小。

- 如果尚未完成任何 epoch，保留失败 workspace 后从固定初始 checkpoint 重启；
- 如果已有合法、完整、可恢复的本机 checkpoint，按注册边界 resume；
- 如果状态包不完整或身份不一致，拒绝拼接结果并升级处理。

### 15.3 进程被杀或机器重启

- 保留中断日志、job state、checkpoint 和 telemetry；
- 只从最后一个完整 checkpoint 恢复；
- 下一个 epoch 已出现半写产物时，不得把它算作完成；
- 跨机器默认不拼接孤立 arm；机器失效时按 `full-block restart` 在热备机重跑整个 cycle/seed block；
- 旧 attempt 标记为 `SUPERSEDED` 或 `FENCED`，不得进入正式配对统计。

### 15.4 telemetry 半写或 sidecar 损坏

只出现 Parquet、只出现 sidecar、行数错误、sample ID 重复、SHA 不一致都属于失败。程序会拒绝 resume
和 `COMPLETE`，操作员不得手工补一个 JSON 或改状态字段。

### 15.5 lease、heartbeat 或 fencing 异常

- 共享盘争抢只有一个 winner；loser 应正常退出；
- ACL 或目录权限错误不能伪装成普通 lock contention 无限重试；
- assignment 切换后，旧 holder 不得继续 heartbeat 或发布完成态；
- stale job 必须按 TTL、reap 和新 generation 流程接管，不能手工删除 claim 文件抢任务。

## 16. 操作员绝对不能做的事

```text
不能修改 batch=128
不能修改 workers=4
不能修改 imgsz=224
不能修改 epochs=200
不能修改 optimizer、LR、增强或 AMP
不能替换 yolo11l 权重
不能改变 training seed
不能改 replay 比例或衰减曲线
不能换 selection IDs
不能把 HELD job 提前放行
不能在训练机本地重新随机选样
不能打开 blind holdout
不能删除失败 attempt 来伪装完整率
不能手工写 COMPLETE、PASS 或修改 SHA
不能把孤立 arm 拼进另一台机器的 seed block
```

如果资源不足，正确动作是停止、保留证据、上报和重新 assignment，不是修改科学实验。

## 17. 七天滚动出结果的方式

每个周期独立完成以下闭环：

```text
小规模 release
-> 训练与逐 epoch 监控
-> 完整性验证
-> paired 中央汇总
-> cycle closeout
-> 注册科学决策
-> 生成下一周期的新版本化 queue/release/assignment
```

不要求等待一个月后一次性看全部结果。也不允许因为某个早期 seed 看起来很好，就绕过当前周期完整性
和预注册决策提前改变后续矩阵。

每日只读状态报告应至少汇总：

- 每个 machine 的当前 job 和 segment；
- `COMPLETE/RUNNING/FAILED/STALE/FENCED` 数量；
- 每个 cycle/seed block 的完整性；
- 最近 heartbeat；
- GPU、显存、RSS、磁盘余量和吞吐异常；
- 失败 attempt 和重跑归属；
- 缺 checkpoint、缺 telemetry、缺 prediction 的 job；
- 本周期距离 closeout 还缺哪些注册产物。

## 18. 操作员交接最小模板

换班或机器移交时，至少记录：

```text
machine_id:
machine_config_sha256:
active_assignment_id:
active_assignment_sha256:
release_id:
canonical_lock_sha256:
current_job_id:
cycle_id / seed_id / arm_id:
segment_start / segment_end:
last_complete_epoch:
last_checkpoint_sha256:
lease_state / heartbeat_time:
gpu / memory / disk status:
telemetry_complete_through_epoch:
prediction_complete_epochs:
failed_attempt_ids:
operator_action_taken:
next_required_action:
```

不要只写“训练正常”或“跑到 150 轮”。必须留下可核验身份和完整边界。

## 19. 当前已验证和尚未验证的内容

### 已验证

- Stage1 本轮相关测试：`463 passed, 1 skipped`；
- 本机 RTX 4060 真实 SewerML 图片训练；
- `workers=4` 的 Windows DataLoader spawn；
- telemetry 与无 telemetry 基线数值一致；
- OOM 零 epoch 恢复；
- 进程强杀后 checkpoint resume；
- telemetry 写入中断不发布半成品；
- 损坏 sidecar 被拒绝；
- 热备完整重跑；
- v2 预注册和 296-job queue 内容校验。

唯一 skip 是本机未挂载 canonical 240-run source ledger，不代表正式十机训练已经完成。

### 尚未执行

- 真实共享 coordination root 的十机 canary；
- 十机一机一 job 的真实数据 canary；
- 每台机器正式资源 preflight；
- engineering gate v2 正式证据绑定；
- pilot release v2；
- assignment v2 激活；
- Cycle 1 正式 200-epoch 训练；
- Cycle 2/3/4 科学放行；
- blind holdout。

因此当前状态准确表述为：

```text
代码已经达到 owner canary 前的准备状态；
科学实验尚未正式放行。
```

## 20. 权威文件顺序

发生冲突时按以下顺序判断：

1. `03_preregistration_v2/CANONICAL_TRAINING_LOCK_BINDING.json` 和 canonical lock；
2. `03_preregistration_v2/` 中的冻结科学矩阵、schedule 和 job graph；
3. `04_run_queue_v2/JOB_EXECUTION_REGISTRY.csv`；
4. 当前正式 release v2；
5. 当前活动 assignment v2；
6. [DYNAMIC_REPLAY_CAMPAIGN_OPERATIONS_V2.md](DYNAMIC_REPLAY_CAMPAIGN_OPERATIONS_V2.md)；
7. 本操作员背景说明；
8. 临时聊天记录或口头描述。

旧的无版本号 `03_preregistration/` 和 `04_run_queue/` 只作为历史证据，不能用于新 release。

## 21. 相关入口

- 科学预注册：`artifacts/stage1_sample_value_experiments/experiments/dynamic_replay_budget_efficiency_20260807/03_preregistration_v2/PREREGISTRATION.md`
- 活动 queue：`artifacts/stage1_sample_value_experiments/experiments/dynamic_replay_budget_efficiency_20260807/04_run_queue_v2/JOB_EXECUTION_REGISTRY.csv`
- 操作手册：[DYNAMIC_REPLAY_CAMPAIGN_OPERATIONS_V2.md](DYNAMIC_REPLAY_CAMPAIGN_OPERATIONS_V2.md)
- queue 合同：[RUN_QUEUE_V2_README.md](RUN_QUEUE_V2_README.md)
- 字段与容量：[DYNAMIC_REPLAY_CAMPAIGN_FIELD_AUDIT.md](DYNAMIC_REPLAY_CAMPAIGN_FIELD_AUDIT.md)
- 机器配置说明：`configs/stage1_gapvalue240/README_MACHINE_CONFIG.md`
- 文献综合：`artifacts/stage1_sample_value_experiments/experiments/dynamic_replay_budget_efficiency_20260807/02_literature/RESEARCH_SYNTHESIS.md`
- 正式单 job 入口：`scripts/stage1_gapvalue240/dynamic_campaign_train_worker.py`
