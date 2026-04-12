"""滑窗扫描 - C 信号 (扰动稳定性), 21 run"""
from scripts.run_sliding_window_pipeline import main as _main
import sys
sys.argv = [sys.argv[0], "--signal", "C", "--device", "0"]
_main()
