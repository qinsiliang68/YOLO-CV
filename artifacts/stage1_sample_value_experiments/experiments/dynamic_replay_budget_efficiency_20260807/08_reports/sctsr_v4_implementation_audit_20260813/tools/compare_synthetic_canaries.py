from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FILES = (
    "01_assets/SYNTHETIC_ASSET_REGISTRY.json",
    "08_receipts/CHECKPOINT_RESUME_RECEIPT.json",
    "08_receipts/SYNTHETIC_MECHANISM_AUDIT.json",
)


def load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    left_receipt = load(arguments.left / "08_receipts/SYNTHETIC_CANARY_RECEIPT.json")
    right_receipt = load(arguments.right / "08_receipts/SYNTHETIC_CANARY_RECEIPT.json")
    comparisons = {
        "source_tree_digest": left_receipt["source_tree_digest"] == right_receipt["source_tree_digest"],
        "parent_checkpoint_sha256": left_receipt["parent_checkpoint_sha256"] == right_receipt["parent_checkpoint_sha256"],
        "arms_completed": left_receipt["arms_completed"] == right_receipt["arms_completed"],
        "failure_injection_count": left_receipt["failure_injection_count"] == right_receipt["failure_injection_count"],
    }
    normalized_files = []
    for relative in FILES:
        left = load(arguments.left / relative)
        right = load(arguments.right / relative)
        equal = left == right
        comparisons[f"json_equal:{relative}"] = equal
        payload = (json.dumps(left, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        normalized_files.append({"relative_path": relative, "equal": equal, "stable_digest": hashlib.sha256(payload).hexdigest().upper()})
    report = {
        "schema_version": "stage1.sctsr.synthetic_determinism_comparison.v1",
        "status": "PASS" if all(comparisons.values()) else "FAIL",
        "semantic": "SYNTHETIC_NOT_SCIENTIFIC_RESULT",
        "training_seed": 20260606,
        "comparisons": comparisons,
        "stable_files": normalized_files,
        "excluded_from_byte_identity": [
            "absolute output_root paths",
            "timestamps and quarantine names",
            "artifact indexes containing absolute paths",
        ],
        "formal_training_started": False,
        "method_effectiveness_claimed": False,
    }
    arguments.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "comparison_count": len(comparisons)}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
