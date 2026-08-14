from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage1_sctsr_v4.cli_support import add_output_argument, run_cli
from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.run_intent import RunIntentContext, validate_run_intent_acknowledgement
from stage1_sctsr_v4.serialization import load_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate one SCTSR acknowledgement and its frozen runbook bytes")
    parser.add_argument("--acknowledgement", type=Path, required=True)
    parser.add_argument("--runbook-manifest", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    add_output_argument(parser)
    arguments = parser.parse_args()

    def action():
        raw = load_json(arguments.acknowledgement)
        if not isinstance(raw, dict):
            raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Acknowledgement is not a JSON object")
        names = {field.name for field in fields(RunIntentContext)}
        context = RunIntentContext.from_mapping({name: raw.get(name) for name in names})
        return validate_run_intent_acknowledgement(
            arguments.acknowledgement,
            expected_context=context,
            runbook_manifest_path=arguments.runbook_manifest,
            repository_root=arguments.repository_root,
        )

    return run_cli("validate_run_intent_acknowledgement", arguments.output, action)


if __name__ == "__main__":
    raise SystemExit(main())
