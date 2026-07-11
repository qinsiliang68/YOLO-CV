from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage1_gapvalue240.prediction_worker import main


if __name__ == "__main__":
    raise SystemExit(main())
