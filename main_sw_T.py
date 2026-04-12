"""滑窗扫描 - T 信号 (训练动力学), 21 run"""
from scripts.run_sliding_window_pipeline import main as _main
import sys
sys.argv = [sys.argv[0], "--signal", "T", "--device", "0"]
_main()
