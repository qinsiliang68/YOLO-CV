from __future__ import annotations

from scripts.stage1_sctsr_v4.run_branch import revalidate_endpoint_dataset_bindings


def _binding() -> dict[str, object]:
    return {
        "dataset_binding": {
            "train_materialized_content_binding": {"role": "train"},
            "val_model_materialized_content_binding": {"role": "val_model"},
        }
    }


def test_terminal_only_finalization_reuses_current_invocation_validation(monkeypatch):
    monkeypatch.setattr(
        "stage1_sctsr_v4.dataset_adapter.revalidate_materialized_dataset_binding",
        lambda _binding: (_ for _ in ()).throw(AssertionError("terminal finalization repeated the byte pass")),
    )

    revalidate_endpoint_dataset_bindings(_binding(), terminal_epoch_complete=True)


def test_normal_training_keeps_endpoint_revalidation(monkeypatch):
    observed = []
    monkeypatch.setattr(
        "stage1_sctsr_v4.dataset_adapter.revalidate_materialized_dataset_binding",
        lambda binding: observed.append(binding["role"]),
    )

    revalidate_endpoint_dataset_bindings(_binding(), terminal_epoch_complete=False)

    assert observed == ["train", "val_model"]
