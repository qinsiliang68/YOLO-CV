"""
机器 A: R(边界性) + T(训练动力学) + 随机对照 + 部分组合验证
  R: 11 滑窗 — 定位边界层 Goldilocks zone
  T: 11 滑窗 — 定位训练动力学 Goldilocks zone
  Rand50: 3 随机对照 — 证明"不是随便选 50 个都行"
  组合: F-R, F-T, F-RT, F-RTDC — 单路 + 交叉验证
  共 29 run, 预计 ~3 天

Usage:
    uv run main_goldilocks_machineA.py
"""
import sys
import time

print("=" * 60)
print("  Machine A: R + T + 对照 + 组合")
print("  29 run, ~3 天")
print(f"  Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

from scripts.run_goldilocks_campaign import main as _main

# Phase 1: R 定峰
sys.argv = [sys.argv[0], "--phase", "peak", "--signal", "R", "--device", "0"]
try:
    _main()
except SystemExit:
    pass

# Phase 1: T 定峰
sys.argv = [sys.argv[0], "--phase", "peak", "--signal", "T", "--device", "0"]
try:
    _main()
except SystemExit:
    pass

# Phase 2: 随机对照
sys.argv = [sys.argv[0], "--phase", "control", "--device", "0"]
try:
    _main()
except SystemExit:
    pass

# Phase 3: 组合验证 (A 负责的部分: F-R, F-T, F-RT, F-RTDC)
sys.argv = [sys.argv[0], "--phase", "combine", "--device", "0"]
try:
    _main()
except SystemExit:
    pass

print(f"\n{'=' * 60}")
print(f"  Machine A complete")
print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'=' * 60}")
