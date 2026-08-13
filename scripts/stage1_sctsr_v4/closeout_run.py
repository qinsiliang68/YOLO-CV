from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage1_sctsr_v4.cli_support import add_output_argument, require_receipt_outside_artifact_root, run_cli
from stage1_sctsr_v4.completion import completion_audit_from_mapping
from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.run_validation import build_artifact_index, validate_run_tree
from stage1_sctsr_v4.serialization import atomic_write_json, load_json, sha256_file


def _resolve_evidence(repository_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repository_root / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish implementation-only completion after strict run and repository self-audit")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--implementation-audit", type=Path)
    parser.add_argument("--completion-output", type=Path, required=True)
    parser.add_argument("--allow-synthetic-columnar-fallback", action="store_true")
    add_output_argument(parser)
    arguments = parser.parse_args()

    def action():
        require_receipt_outside_artifact_root(arguments.output, arguments.run_root)
        if arguments.implementation_audit is None:
            raise SctsrError(
                ErrorCode.CLOSEOUT_NOT_VALIDATED,
                "A synthetic canary or run tree alone cannot prove full implementation completion",
            )
        repository_root = arguments.repository_root.resolve()
        validation = validate_run_tree(
            arguments.run_root,
            allow_synthetic_portable_fallback=arguments.allow_synthetic_columnar_fallback,
        )
        raw_audit = load_json(arguments.implementation_audit)
        audit = completion_audit_from_mapping(raw_audit)
        audit.validate(require_evidence=True)
        run_manifest = load_json(arguments.run_root / "RUN_MANIFEST.json")
        if str(run_manifest.get("source_tree_digest", "")).upper() != audit.source_tree_digest:
            raise SctsrError(ErrorCode.SOURCE_TREE_MISMATCH, "Run evidence and implementation audit bind different source trees")
        for check, evidence in audit.check_evidence.items():
            log = _resolve_evidence(repository_root, evidence.stdout_stderr_path)
            try:
                log.relative_to(repository_root)
            except ValueError as exc:
                raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Completion evidence log escapes repository root", failing_field=check) from exc
            if not log.is_file() or log.stat().st_size != evidence.stdout_stderr_bytes or sha256_file(log) != evidence.stdout_stderr_sha256:
                raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Completion evidence log bytes/SHA do not match", failing_field=check)
            for relative in (*evidence.reviewed_source_files, *evidence.reviewed_test_files):
                reviewed = _resolve_evidence(repository_root, relative)
                try:
                    reviewed.relative_to(repository_root)
                except ValueError as exc:
                    raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Reviewed file escapes repository root", failing_field=check) from exc
                if not reviewed.is_file():
                    raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Completion evidence names a missing reviewed file", failing_field=check, artifact_path=str(reviewed))
        ledger = _resolve_evidence(repository_root, audit.changed_file_ledger_path)
        if not ledger.is_file():
            raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Changed-file ledger is missing", artifact_path=str(ledger))
        completion = {
            "schema_version": "stage1.sctsr.closeout.v2",
            "status": "IMPLEMENTATION_ACCEPTANCE_PASS_NOT_TRAINING_AUTHORIZATION",
            "validation": validation,
            "implementation_audit_path": arguments.implementation_audit.resolve().as_posix(),
            "implementation_audit_sha256": sha256_file(arguments.implementation_audit),
            "source_tree_digest": audit.source_tree_digest,
            "commit_list": list(audit.commit_list),
            "changed_file_ledger_path": audit.changed_file_ledger_path,
            "formal_training_started": False,
            "engineering_gate_generated": False,
            "assignments_generated": False,
            "pilot_release_generated": False,
            "blind_holdout_opened": False,
            "selector_trained": False,
            "method_effectiveness_claimed": False,
            "formal_training_authorized": False,
        }
        atomic_write_json(arguments.completion_output, completion)
        if arguments.completion_output.resolve().is_relative_to(arguments.run_root.resolve()):
            atomic_write_json(arguments.run_root / "ARTIFACT_INDEX.json", build_artifact_index(arguments.run_root))
            completion["post_closeout_validation"] = validate_run_tree(
                arguments.run_root,
                allow_synthetic_portable_fallback=arguments.allow_synthetic_columnar_fallback,
            )
        return completion

    return run_cli("closeout_run", arguments.output, action)


if __name__ == "__main__":
    raise SystemExit(main())
