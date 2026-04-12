"""
机器 A: R(边界性) + T(训练动力学) + 随机对照 + 组合验证
  自动流程:
    1. 提取 T 信号 (如果 cache 不存在)
    2. R 定峰: 11 滑窗
    3. T 定峰: 11 滑窗
    4. 随机对照: 3 run
    5. 组合验证: F-R, F-T, F-RT, F-RTDC

Usage:
    uv run main_goldilocks_machineA.py
"""
import sys
import time
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
T_CACHE = REPO_ROOT / "research" / "materials" / "stage1_formal" / "gate_goldilocks_campaign" / "T_signal_cache.json"

print("=" * 60)
print("  Machine A: R + T + 对照 + 组合")
print(f"  Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# Step 0: 自动提取 T 信号 (如果 cache 不存在)
if not T_CACHE.exists():
    print("\n  T signal cache not found, extracting...")
    print("  (对 250 个候选样本在 7 个 checkpoint 上推理, 约 30 分钟)\n")
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from stage1_extract_T_signal import main as extract_T
    sys.argv = [sys.argv[0], "--device", "0"]
    try:
        extract_T()
    except SystemExit as e:
        if e.code and e.code != 0:
            print(f"\n  T extraction failed (exit {e.code}). T will use R as proxy.")
    except Exception as exc:
        print(f"\n  T extraction failed: {exc}. T will use R as proxy.")
    print()
else:
    data = json.loads(T_CACHE.read_text(encoding="utf-8"))
    nonzero = sum(1 for v in data.values() if v > 0)
    print(f"\n  T signal cache exists: {len(data)} samples ({nonzero} nonzero)\n")

# Step 1-5: 跑 campaign
from scripts.run_goldilocks_campaign import main as _main

for phase, signal, label in [
    ("peak", "R", "R 定峰 (11 滑窗)"),
    ("peak", "T", "T 定峰 (11 滑窗)"),
    ("control", "", "随机对照 (3 run)"),
    ("combine", "", "组合验证"),
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
print(f"  Machine A complete")
print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'=' * 60}")
