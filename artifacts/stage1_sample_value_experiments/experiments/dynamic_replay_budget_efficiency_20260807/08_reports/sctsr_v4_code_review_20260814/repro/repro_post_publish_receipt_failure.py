from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from stage1_sctsr_v4.epoch_transaction import EpochTransaction


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="sctsr_review_tx_") as raw:
        sandbox = Path(raw)
        transaction_root = sandbox / "03_epoch_transactions"
        tx = EpochTransaction(transaction_root, "REVIEW_RUN", 121, 1)
        tx.begin()
        tx.write_json("evidence.json", {"epoch": 121})

        def injected_failure(*_args, **_kwargs):
            raise OSError("INJECTED_DISK_FULL_AFTER_COMPLETE_RENAME")

        tx._append_receipt = injected_failure  # type: ignore[method-assign]
        caught = None
        try:
            tx.commit()
        except Exception as exc:  # expected injected path
            caught = f"{type(exc).__name__}: {exc}"

        quarantined = []
        quarantine_root = sandbox / "09_quarantine"
        if quarantine_root.exists():
            quarantined = sorted(path.name for path in quarantine_root.iterdir())
        result = {
            "schema_version": "stage1.sctsr.review.reproduction.v1",
            "check": "post_rename_receipt_failure_is_quarantined",
            "expected": {
                "complete_exists": False,
                "quarantined_generation_count": 1,
                "recovery_can_resume_previous_complete_prefix": True,
            },
            "observed": {
                "caught": caught,
                "inprogress_exists": tx.inprogress.exists(),
                "complete_exists": tx.complete.exists(),
                "quarantined_generation_count": len(quarantined),
                "quarantined": quarantined,
                "receipt_exists": tx.receipt_path.exists(),
                "artifact_index_exists": tx.artifact_index_path.exists(),
                "recovery_pointer_exists": tx.recovery_pointer_path.exists(),
            },
        }
        passed = (
            caught is not None
            and not tx.complete.exists()
            and len(quarantined) == 1
        )
        result["status"] = "PASS" if passed else "FAIL"
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
