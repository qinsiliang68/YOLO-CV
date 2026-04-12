"""
机器 B: D(抱团性, k-sweep) + C(扰动稳定性) + 组合验证
  自动流程:
    1. D 定峰: 33 滑窗 (k={8,15,25} × 11 窗口)
    2. C 定峰: 11 滑窗
    3. 组合验证: F-D, F-C (跨机器组合需等 A 完成后用 postmerge)

Usage:
    uv run main_goldilocks_machineB.py
"""
import sys
import time

print("=" * 60)
print("  Machine B: D(k-sweep) + C + 组合")
print(f"  Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

from scripts.run_goldilocks_campaign import main as _main

for phase, signal, label in [
    ("peak", "D", "D 定峰 k-sweep (33 滑窗)"),
    ("peak", "C", "C 定峰 (11 滑窗)"),
    ("combine", "", "组合验证 (本机可完成部分)"),
]:
    print(f"\n{'#' * 60}")
    print(f"  {label}")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 60}\n")

    argv = [sys.argv[0], "--phase", phase, "--device", "0"]
    if signal:
        argv.extend(["--signal", signal])
    sys.argv = argv

    try:
        _main()
    except SystemExit:
        pass
    except Exception as exc:
        print(f"  FAILED: {exc}")
        import traceback
        traceback.print_exc()

print(f"\n{'=' * 60}")
print(f"  Machine B complete")
print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'=' * 60}")
