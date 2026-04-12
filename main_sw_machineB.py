"""
机器 2: 动态维度镜像对 T + C
  T = 训练动力学 (Trajectory) — 样本跨 epoch 是否持续对训练有帮助
  C = 扰动稳定性 (Consistency) — 样本在 TTA 扰动下预测跳不跳
  共 42 run, 预计 ~4.5 天

滑窗参数: 窗口 50 样本, 步长 10, 21 个窗口/信号
目的: 精确定位 T 和 C 各自的 Goldilocks peak 中心 μ 和宽度 w

注意: T 信号需要从 hn00 教师的 per_epoch_gate 逐样本预测中提取跨 epoch 方差。
如果 per_epoch_gate 目录不存在, T 会退化为 R 的代理。

Usage:
    uv run main_sw_machine2.py
"""
import sys
import time

print("=" * 60)
print("  Machine 2: 动态维度镜像对 T + C")
print("  T (训练动力学) 21 run + C (扰动稳定性) 21 run = 42 run")
print(f"  Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

from scripts.run_sliding_window_pipeline import main as _main

for signal in ["T", "C"]:
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
print(f"  Machine 2 complete: T + C")
print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'=' * 60}")
