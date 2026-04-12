"""滑窗扫描 - D 信号 (抱团性), 21 run"""
from scripts.run_sliding_window_pipeline import main as _main
import sys
sys.argv = [sys.argv[0], "--signal", "D", "--device", "0"]
_main()
