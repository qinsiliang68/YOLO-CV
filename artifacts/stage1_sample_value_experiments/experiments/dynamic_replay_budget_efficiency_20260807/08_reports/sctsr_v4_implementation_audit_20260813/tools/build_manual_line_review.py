from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

REVIEWER_IDENTITY = "SELF_REVIEW_NOT_INDEPENDENT_REVIEW"


def stable_digest(value: object) -> str:
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.inprogress")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(temporary, path)


REVIEWS = (
    (
        "SA-280",
        "The base and replay backward calls precede one and only one frozen upstream optimizer_step invocation; counters advance only after that call.",
        (("stage1_sctsr_v4/ultralytics_overlay.py", 338, 340, ("run_ultralytics_fixed_step_epoch",)), ("stage1_sctsr_v4/ultralytics_overlay.py", 396, 431, ("run_ultralytics_fixed_step_epoch",))),
    ),
    (
        "SA-281",
        "Replay CE is unreduced per sample, summed, and divided by the immutable canonical denominator 128 in both the generic and YOLO execution paths.",
        (("stage1_sctsr_v4/fixed_step_runtime.py", 344, 371, ("run_fixed_step_epoch",)), ("stage1_sctsr_v4/ultralytics_overlay.py", 377, 397, ("run_ultralytics_fixed_step_epoch",))),
    ),
    (
        "SA-282",
        "The generic runtime orders unscale, norm measurement, clipping, scaler step, scaler update, skip detection, and EMA; the YOLO overlay delegates exactly once to the frozen upstream method with the same order.",
        (("stage1_sctsr_v4/fixed_step_runtime.py", 395, 443, ("run_fixed_step_epoch",)), ("stage1_sctsr_v4/ultralytics_overlay.py", 411, 429, ("run_ultralytics_fixed_step_epoch",)), ("YOLOv11/ultralytics/engine/trainer.py", 753, 761, ("BaseTrainer.optimizer_step",))),
    ),
    (
        "SA-283",
        "All _BatchNorm modules capture running_mean, running_var, and num_batches_tracked; restoration validates topology, nullability, and byte-identical digest.",
        (("stage1_sctsr_v4/bn_isolation.py", 18, 78, ("BatchNormSnapshot", "capture_batchnorm_buffers", "restore_batchnorm_buffers")), ("stage1_sctsr_v4/fixed_step_runtime.py", 321, 392, ("preserve_batchnorm_buffers",))),
    ),
    (
        "SA-284",
        "The replay RNG snapshot covers Python, NumPy, Torch CPU and every CUDA device, restores them in finally, and verifies the post-restore digest; base order and augmentation use independent counter domains.",
        (("stage1_sctsr_v4/rng_isolation.py", 16, 89, ("RngSnapshot", "capture_global_rng", "restore_global_rng", "replay_rng_domain")), ("stage1_sctsr_v4/base_rng.py", 28, 82, ("prepare_counter_domain_base_loader",))),
    ),
    (
        "SA-285",
        "R2 calls TerminalFieldGuard.project_rows before IdentityRecord construction or matching; the projection performs explicit whitelist lookups and does not enumerate forbidden values.",
        (("stage1_sctsr_v4/random_controls.py", 97, 124, ("build_r2_matched_random",)), ("stage1_sctsr_v4/terminal_field_guard.py", 9, 67, ("TerminalFieldGuard.project_row", "TerminalFieldGuard.reject_if_config_mentions_forbidden"))),
    ),
    (
        "SA-286",
        "OOM, malformed replay identity, cap overflow, AMP skipped steps, half-written evidence, and invalid generation paths raise without reducing batch, splitting a step, or publishing a canonical epoch.",
        (("stage1_sctsr_v4/fixed_step_runtime.py", 291, 320, ("run_fixed_step_epoch",)), ("stage1_sctsr_v4/fixed_step_runtime.py", 373, 439, ("run_fixed_step_epoch",)), ("stage1_sctsr_v4/epoch_transaction.py", 136, 163, ("EpochTransaction._validate_files",)), ("stage1_sctsr_v4/epoch_transaction.py", 225, 300, ("EpochTransaction.commit", "EpochTransaction.quarantine"))),
    ),
    (
        "SA-287",
        "Prepared training binds validation only to val_model/study, runtime policy seals test and blind_holdout, and endpoint publication hard-codes E200 val_op with ENDPOINT_ONLY_NOT_FOR_SELECTION.",
        (("stage1_sctsr_v4/formal_cli.py", 681, 715, ("validate_prepared_dataset_roles",)), ("stage1_sctsr_v4/formal_cli.py", 756, 786, ("_validate_runtime_policy",)), ("stage1_sctsr_v4/prediction_runtime.py", 225, 318, ("publish_formal_endpoint",))),
    ),
    (
        "SA-288",
        "Completion rejects prohibited side effects and requires explicit blockers; closeout only writes IMPLEMENTATION_ACCEPTANCE_PASS_NOT_TRAINING_AUTHORIZATION and never creates a release, assignment, gate, or pilot.",
        (("stage1_sctsr_v4/completion.py", 117, 184, ("CompletionAudit.validate",)), ("scripts/stage1_sctsr_v4/closeout_run.py", 34, 124, ("main.action",))),
    ),
    (
        "SA-289",
        "Every public SCTSR schema identity is enumerated in one exact registry; validation rejects any missing, stale, or extra entry, while field-level modules and tests enforce their frozen row contracts.",
        (("stage1_sctsr_v4/schema_registry.py", 10, 105, ("REQUIRED_SCHEMAS", "SchemaRegistry.validate")), ("configs/stage1_sctsr_v4/schema_registry_v1.json", 1, 73, ("schemas",))),
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    repository = arguments.repository_root.resolve()
    rows = []
    for review_id, finding, anchors in REVIEWS:
        anchor_rows = []
        for relative, start, end, symbols in anchors:
            lines = (repository / relative).read_text(encoding="utf-8").splitlines()
            if not (1 <= start <= end <= len(lines)):
                raise ValueError((relative, start, end, len(lines)))
            payload = ("\n".join(lines[start - 1 : end]) + "\n").encode("utf-8")
            anchor_rows.append(
                {
                    "relative_path": relative,
                    "start_line": start,
                    "end_line": end,
                    "line_sha256": hashlib.sha256(payload).hexdigest().upper(),
                    "reviewed_symbols": list(symbols),
                }
            )
        rows.append(
            {
                "review_id": review_id,
                "status": "PASS",
                "finding": finding,
                "residual_risk": "This is a self-review of implementation semantics; formal training and independent review remain unperformed.",
                "anchors": anchor_rows,
            }
        )
    core = {
        "schema_version": "stage1.sctsr.manual_line_review.v1",
        "implementation_source_commit": arguments.source_commit,
        "reviewer_identity": REVIEWER_IDENTITY,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reviews": rows,
    }
    atomic_write_json(arguments.output, {**core, "review_digest": stable_digest(core)})
    print(f"PASS reviews={len(rows)} anchors={sum(len(row['anchors']) for row in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
