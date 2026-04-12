"""滑窗扫描 - R → D → T → C 四信号按顺序执行, 共 84 run"""
import sys
import time

print("=" * 60)
print("  Sliding Window Pipeline: R → D → T → C")
print("  Total: 84 run, ~218 hours on single GPU")
print(f"  Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

from scripts.run_sliding_window_pipeline import main as _main

for signal in ["R", "D", "T", "C"]:
    print(f"\n{'#' * 60}")
    print(f"  Starting signal: {signal}")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 60}\n")

    sys.argv = [sys.argv[0], "--signal", signal, "--device", "0"]
    try:
        _main()
    except SystemExit:
        pass
    except Exception as exc:
        print(f"\n  Signal {signal} FAILED: {exc}")
        import traceback
        traceback.print_exc()
        print(f"  Continuing to next signal...\n")

print(f"\n{'=' * 60}")
print(f"  All signals complete.")
print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'=' * 60}")
