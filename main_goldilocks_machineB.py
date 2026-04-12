"""
机器 B: D(抱团性, k-sweep) + C(扰动稳定性) + 部分组合验证
  D: 33 滑窗 (k={8,15,25} × 11 窗口) — 定位最优团尺度 + Goldilocks zone
  C: 11 滑窗 — 定位扰动稳定性 Goldilocks zone
  组合: F-D, F-C, F-RD, F-TD, F-RTD — 需等机器 A 的 peak 结果
  共 49 run (44 定峰 + 5 组合), 预计 ~5 天

注意: 组合验证阶段需要 R/T 的 peak 信息。
如果机器 A 还没跑完, 组合阶段会跳过缺失的信号,
等机器 A 完成后重新运行 --phase combine 即可补全。

Usage:
    uv run main_goldilocks_machineB.py
"""
import sys
import time

print("=" * 60)
print("  Machine B: D(k-sweep) + C + 组合")
print("  49 run, ~5 天")
print(f"  Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

from scripts.run_goldilocks_campaign import main as _main

# Phase 1: D 定峰 (k=8,15,25 各 11 窗口)
sys.argv = [sys.argv[0], "--phase", "peak", "--signal", "D", "--device", "0"]
try:
    _main()
except SystemExit:
    pass

# Phase 1: C 定峰
sys.argv = [sys.argv[0], "--phase", "peak", "--signal", "C", "--device", "0"]
try:
    _main()
except SystemExit:
    pass

# Phase 3: 组合验证 (B 负责的部分)
# 注意: 如果 A 的 R/T peak 还没出来, 部分组合会被跳过
sys.argv = [sys.argv[0], "--phase", "combine", "--device", "0"]
try:
    _main()
except SystemExit:
    pass

print(f"\n{'=' * 60}")
print(f"  Machine B complete")
print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'=' * 60}")
