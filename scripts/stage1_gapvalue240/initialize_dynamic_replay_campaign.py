from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage1_gapvalue240.campaign_layout import initialize_campaign_layout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register and initialize the isolated Stage1 dynamic replay campaign."
    )
    parser.add_argument(
        "--family-root",
        type=Path,
        default=REPO_ROOT / "artifacts/stage1_sample_value_experiments",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = initialize_campaign_layout(args.family_root)
    for key, value in result.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
