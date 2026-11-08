import sys
from pathlib import Path

# 让测试可以 import 本阶段目录下的 mini_trader 包
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
