from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from stage1_sctsr_v4.asset_registry import load_asset_registry
from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.formal_pool_inputs import build_registered_r2, load_formal_pool_inputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    arguments = parser.parse_args()
    root = arguments.repository_root.resolve()
    registry_path = root / "configs" / "stage1_sctsr_v4" / "asset_registry_v1.json"
    registry = load_asset_registry(registry_path)
    inputs = load_formal_pool_inputs(registry, root)
    result = {
        "schema_version": "stage1.sctsr.review.reproduction.v1",
        "check": "real_r2_exact_quota_fail_closed",
        "repository_root": root.as_posix(),
        "base_count": len(inputs.base_records),
        "base_manifest_sha256": inputs.base_manifest_sha256,
        "preterminal_source_sha256": inputs.preterminal_source_sha256,
        "t_count": len(inputs.t_pool.records),
        "t_identity_digest": inputs.t_pool.spec.identity_digest,
        "expected_error": ErrorCode.R2_QUOTA_INFEASIBLE.value,
    }
    try:
        build_registered_r2(inputs, base_denominator=registry.base_denominator, selection_seed=20260812)
    except SctsrError as exc:
        shortfalls = exc.observed if isinstance(exc.observed, dict) else {}
        result["shortfall_strata"] = len(shortfalls)
        result["shortfall_occurrences"] = sum(int(value) for value in shortfalls.values())
        result["observed_error"] = exc.to_dict()
        result["status"] = "PASS" if exc.code is ErrorCode.R2_QUOTA_INFEASIBLE else "FAIL"
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["status"] == "PASS" else 1
    result["observed_error"] = None
    result["status"] = "FAIL"
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
