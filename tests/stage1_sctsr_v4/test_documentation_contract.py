from __future__ import annotations


def test_sctsr_documentation_records_current_fail_closed_state(repository_root):
    docs = repository_root / "docs" / "stage1_sctsr_v4"
    guide = (docs / "IMPLEMENTATION_GUIDE.md").read_text(encoding="utf-8")
    blockers = (docs / "KNOWN_BLOCKERS.md").read_text(encoding="utf-8")
    review = (docs / "INDEPENDENT_REVIEW_CHECKLIST.md").read_text(encoding="utf-8")
    addendum = (
        repository_root
        / "artifacts"
        / "stage1_sample_value_experiments"
        / "experiments"
        / "dynamic_replay_budget_efficiency_20260807"
        / "03_preregistration_v4_sctsr"
        / "SCTSR_R2_MATCHING_ADDENDUM_20260815.md"
    ).read_text(encoding="utf-8")

    assert "206" in guide and "IMPLEMENTATION_ACCEPTANCE_PASS_NOT_TRAINING_AUTHORIZATION" in guide
    assert "172" in blockers and "378" in blockers
    assert "R2 旧规格矛盾已解决，不再是当前阻断" in blockers
    assert "MINIMUM_OOF_GROUP_DISPLACEMENT_ZERO_OVERLAP_V1" in addendum
    assert "不为 U/F 各自另抽一套 3,000 IDs" in addendum
    assert "formal_training_started=false" in addendum
    assert "181 passed, 3 skipped" in blockers
    assert "SELF_AUDIT_FAIL" in blockers
    assert "不证明 SCTSR" in guide
    assert "不能伪称独立审稿人" in review


def test_sctsr_documentation_has_no_superseded_delivery_environment_claims(repository_root):
    docs = repository_root / "docs" / "stage1_sctsr_v4"
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            docs / "IMPLEMENTATION_GUIDE.md",
            docs / "KNOWN_BLOCKERS.md",
            docs / "TRAINING_SYSTEM_ARCHITECTURE.md",
            docs / "UPSTREAM_INTEGRATION.md",
        )
    )
    assert "This execution container did not contain" not in combined
    assert "had no PyArrow" not in combined
    assert "eleven taskbook CLIs" not in combined


def test_repository_documentation_indexes_the_held_sctsr_implementation(repository_root):
    root_readme = (repository_root / "README.md").read_text(encoding="utf-8")
    docs_readme = (repository_root / "docs" / "README.md").read_text(encoding="utf-8")
    assert "docs/stage1_sctsr_v4/IMPLEMENTATION_GUIDE.md" in root_readme
    assert "stage1_sctsr_v4/IMPLEMENTATION_GUIDE.md" in docs_readme
    assert "IMPLEMENTATION_ONLY_HELD" in docs_readme
