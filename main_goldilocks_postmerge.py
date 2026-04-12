"""
机器 A+B 都跑完后, 在任意一台机器上运行此脚本。
执行跨机器组合验证: F-RD, F-TD, F-RTD, F-RTDC
以及可能在单机 combine 阶段被跳过的其他组合。

前置条件:
  1. 两台机器的 peak_results.json 已合并(手动拷贝或 git sync)
  2. 确保 research/results/stage1_formal/gate_goldilocks_campaign/peak_results.json
     包含 R, T, D, C 四个信号的 peak 信息

Usage:
    uv run main_goldilocks_postmerge.py
"""
import sys
import time

print("=" * 60)
print("  Post-merge: 跨机器组合验证")
print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

from scripts.run_goldilocks_campaign import main as _main

sys.argv = [sys.argv[0], "--phase", "combine", "--device", "0"]
try:
    _main()
except SystemExit:
    pass

print(f"\n{'=' * 60}")
print(f"  Post-merge complete")
print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'=' * 60}")
