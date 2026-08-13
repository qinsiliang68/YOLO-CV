from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repository_root() -> Path:
    value = os.environ.get("SCTSR_AUDIT_REPOSITORY_ROOT")
    if not value:
        raise RuntimeError("SCTSR_AUDIT_REPOSITORY_ROOT is required")
    root = Path(value).resolve()
    if not (root / "stage1_sctsr_v4").is_dir():
        raise RuntimeError("SCTSR audit repository root is invalid")
    return root


def pytest_collection_modifyitems(items):
    """Adapt only fixtures superseded after the original red assertions.

    The historical test function names and assertion bodies remain byte-exact.
    Commit 03 later made runtime/assets identities required constructor inputs;
    commit 05 later added three mandatory RNG-evidence columns.  These two
    fixture adapters supply those newer preconditions so the original behavior
    assertions can be rerun under their exact pytest node IDs.
    """

    patched_modules = set()
    for item in items:
        module = item.module
        if module in patched_modules:
            continue
        patched_modules.add(module)
        if module.__name__.endswith("test_parent_lineage_hardening"):
            from stage1_sctsr_v4.common_parent import CommonParentSpec

            def current_spec():
                return CommonParentSpec(
                    "P",
                    7,
                    "A" * 64,
                    "B" * 64,
                    "C" * 64,
                    "D" * 64,
                    "E" * 64,
                    "F" * 64,
                )

            module._spec = current_spec
        if module.__name__.endswith("test_evidence_transaction_hardening"):
            original_step_row = module.step_row

            def current_step_row():
                row = original_step_row()
                row.update(
                    {
                        "rng_digest_before_base": "A" * 64,
                        "replay_rng_fork_digest": "A" * 64,
                        "replay_rng_fork_reason": "PRESENT",
                    }
                )
                return row

            module.step_row = current_step_row
