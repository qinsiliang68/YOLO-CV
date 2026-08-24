# P35 / S007 / R1_U 部署备注

记录日期：2026-08-25（Asia/Shanghai）

状态：`DEPLOYED_AND_FORMAL_TRAINING_ACTIVE`

- 节点：`P35` / `DESKTOP-Q3M30PJ` / NVIDIA GeForce RTX 3090
- 任务：`SCTSR_P35_DISCOVERY_S007_R1_U_20260825_V2_PIN_MEMORY_RESUME`
- Run / arm / seed：`SCTSR_DISCOVERY_S007_R1_U` / `R1_U` / `51447201`
- 正式输出：`D:\ssh\AI\artifacts\sctsr_v4_formal_discovery_s007_r1_u_20260825_20a9558`
- 冻结源码提交：`20a9558e36b8782857f54670ae8cf79d3fb2554d`
- P35 source-tree digest：`51D8F0309A6CAEA1FCCBC0BDE0B7ACDC17A3A646888C1912F73570A74ACEEA35`
- P35 runtime-environment digest：`E1A2E5071056FBB2CE1F2574CBAA149258800B985BDFCD5FC113E1858DCC26AC`
- E120 父 checkpoint SHA-256：`BD9764A40D7775F1B9B3767D45ADC47C41759864693B7797D2E183BC9B7B4206`
- 冻结 schedule digest：`079262CAE6DDDBDB56332EAEA980E71761E317E76C795AB30AAC3D3E198C6CB6`
- R1 pool digest：`DDF5655A5D74B514CFF2C124A0196FFB8CCA40BE1198DFF9064326403D9F1F7C`

## 数据与恢复

没有复制 28 GiB 数据，也没有经本机中转数据集。P35 直接使用 C 盘已有的 canonical 数据和同卷 classification hardlink view；本次 loader 扫描 `train=120000`、`val=23996`，两者均为 `0 corrupt`。已有完整物理内容验证绑定覆盖 384,000 个文件、82,637,967,451 bytes；恢复没有再次传输或全量复制图片。

generation 1 在 E121 中途由 PyTorch pin-memory 线程报 `CUDA error: resource already mapped`。当时没有完整 E121，GPU 已释放。整个失败输出以同卷改名方式保存到：

`D:\ssh\AI\artifacts\sctsr_v4_formal_discovery_s007_r1_u_20260825_20a9558.failed_fence1_pin_memory_20260825`

归档包含 33 个文件、48,704,248 bytes，包含 quarantine 和未完成 ledger；没有删除中间产物。失败 runner receipt SHA-256 为 `3C52991DC657E9D19AE608819619DC50841E4D3F770050CEF0183A21D283FCFF`，generation-1 terminal digest 为 `7C91447A09579293A7DE07CE3F606678EBDA631579E4CBAB022F2D40A52BF653`，归档 receipt SHA-256 为 `1AE9A5F948D5F51899FD87A3D50D77964F4DA773CF5BF128DA8E91193B8D6711`。

恢复使用合法 fence generation 2，从完全相同的 E120 父 checkpoint 重新开始 E121；没有消费未完成的 E121。唯一训练运行时改动是 classification DataLoader 的 `pin_memory=False`，用于绕过 Windows/CUDA pinned-memory 映射异常；样本、顺序、batch、replay、模型、优化器、AMP、损失和数值计算均未改变。runtime shim SHA-256 为 `AE9D4375BB210AB893BD776692FD6DFD2E6FB52C126A773AC4A810A4AD97D192`；generation-2 token / acknowledgement SHA-256 分别为 `C4898F119C6D249E857F1B0EFE9430C1322695AD6162E36F6381CC12A50588E5` 和 `85B7206C93CB1AC470A0A4FFB5BE32FB86F2FFF71F2D7E98C3B668F42C5B7BCE`。

启动前还发现一条旧的只读状态探针异常占用约 26 GiB 内存；在确认它无 Python 训练子进程、GPU 为 0 后，仅终止该旧探针，空闲内存恢复到约 27.9 GiB，再启动 generation 2。没有训练重叠。

## 首个完整 epoch 验证

`P35_E121_VALIDATION.json` 是从 P35 原样拉回的最终验证 receipt，状态为 `PASS`，SHA-256 为 `F46D82A926E317B76955914D2F65C4B24C263310F9240B311560C79A2099F818`。

- E121 generation digest：`638B2C8DC8506C72A2266D7B256EC5F03F08CEA45D193F912601AD60B1286205`
- E121 checkpoint SHA-256：`4D5F604DFDAF2C4B4A4B0B3B765AD28E0A54775DFA883183D176CF79AA34B3FF`
- generation manifest 内 7 个文件的 bytes 和 SHA-256 全部匹配。
- occurrence ledger 为 120,600 行：120,000 base + 600 replay，共 938 个 base step。
- 600 个 replay ID 全部唯一，精确匹配冻结的 E121 schedule、重建后的 step-slot 映射和 R1 identity pool。
- replay 绑定 `selection_policy=R1_GLOBAL_RANDOM`、`identity_pool_id=R1_GLOBAL_RANDOM_POOL`，角色与 training identity manifest 一致。
- 验证快照时 E122 已完整完成，E123 正在运行；逻辑锁为 `ACTIVE / fence generation 2`，计划任务状态为 `Running`。

资源快照：GPU 7,196/24,576 MiB、66°C、222.89/370 W；系统空闲内存 19.23 GiB，C 盘空闲 56.06 GiB，D 盘空闲 896.05 GiB。无需温控干预。

该备注仅确认部署、恢复和训练数据链正确，不声明 SCTSR 的科学效果。失败尝试、claim 证据、epoch ledgers、checkpoints、caches 和其他中间产物均继续保存在 P35。
