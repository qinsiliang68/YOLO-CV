from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from stage1_sctsr_v4.asset_registry import load_asset_registry
from stage1_sctsr_v4.serialization import sha256_file


def identity(path: Path) -> dict[str, object]:
    return {
        "path": path.resolve().as_posix(),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-root", type=Path)
    arguments = parser.parse_args()
    clean_root = Path.cwd().resolve()
    registry = load_asset_registry(clean_root / "configs" / "stage1_sctsr_v4" / "asset_registry_v1.json")
    record = next(asset for asset in registry.assets if asset.asset_id == "oof_metadata")
    clean = identity(clean_root / record.relative_path)
    expected = {"bytes": record.bytes, "sha256": record.sha256}
    matches = clean["bytes"] == expected["bytes"] and clean["sha256"] == expected["sha256"]
    result = {
        "schema_version": "stage1.sctsr.review.reproduction.v1",
        "check": "clean_checkout_registered_asset_identity",
        "asset_id": record.asset_id,
        "relative_path": record.relative_path,
        "expected": expected,
        "clean_checkout_observed": clean,
        "comparison_worktree_observed": (
            identity(arguments.comparison_root.resolve() / record.relative_path)
            if arguments.comparison_root is not None
            else None
        ),
        "status": "PASS" if matches else "FAIL",
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
