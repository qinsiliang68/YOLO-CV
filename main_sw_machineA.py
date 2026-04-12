"""
机器 A: 空间维度镜像对 R + D
  R = 边界性 (Risk) — 样本离门控决策边界多远
  D = 抱团性 (Density) — 样本在特征空间有没有相似邻居
  共 42 run, 预计 ~4.5 天

滑窗参数: 窗口 50 样本, 步长 10, 21 个窗口/信号
目的: 精确定位 R 和 D 各自的 Goldilocks peak 中心 μ 和宽度 w

Usage:
    uv run main_sw_machineA.py
"""
import sys
import time

print("=" * 60)
print("  Machine A: 空间维度镜像对 R + D")
print("  R (边界性) 21 run + D (抱团性) 21 run = 42 run")
print(f"  Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

from scripts.run_sliding_window_pipeline import main as _main

for signal in ["R", "D"]:
    print(f"\n{'#' * 60}")
    print(f"  Signal: {signal}")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 60}\n")
    sys.argv = [sys.argv[0], "--signal", signal, "--device", "0"]
    try:
        _main()
    except SystemExit:
        pass
    except Exception as exc:
        print(f"  {signal} FAILED: {exc}")
        import traceback
        traceback.print_exc()

print(f"\n{'=' * 60}")
print(f"  Machine A complete: R + D")
print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'=' * 60}")
