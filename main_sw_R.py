"""滑窗扫描 - R 信号 (边界性), 21 run"""
from scripts.run_sliding_window_pipeline import main as _main
import sys
sys.argv = [sys.argv[0], "--signal", "R", "--device", "0"]
_main()
