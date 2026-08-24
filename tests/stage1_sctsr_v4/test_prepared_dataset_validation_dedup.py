from __future__ import annotations

from types import SimpleNamespace

from stage1_sctsr_v4.formal_training import _revalidate_prepared_dataset_bindings
from stage1_sctsr_v4.serialization import stable_digest


def _prepared_binding() -> dict[str, object]:
    return {
        "dataset_binding": {
            "dataset_binding_digest": "A" * 64,
            "train_materialized_content_binding": {"role": "train"},
            "val_model_materialized_content_binding": {"role": "val_model"},
        }
    }


def test_fresh_same_invocation_binding_consumes_marker_without_rehash(monkeypatch):
    prepared = _prepared_binding()
    dataset = prepared["dataset_binding"]
    trainer = SimpleNamespace(
        _sctsr_fresh_materialized_dataset_binding_digest=stable_digest(dataset),
    )
    monkeypatch.setattr(
        "stage1_sctsr_v4.dataset_adapter.revalidate_materialized_dataset_binding",
        lambda _binding: (_ for _ in ()).throw(AssertionError("fresh binding was re-hashed")),
    )

    _revalidate_prepared_dataset_bindings(prepared, trainer=trainer)

    assert not hasattr(trainer, "_sctsr_fresh_materialized_dataset_binding_digest")


def test_nonfresh_binding_keeps_full_revalidation(monkeypatch):
    prepared = _prepared_binding()
    observed = []
    monkeypatch.setattr(
        "stage1_sctsr_v4.dataset_adapter.revalidate_materialized_dataset_binding",
        lambda binding: observed.append(binding["role"]),
    )

    _revalidate_prepared_dataset_bindings(prepared, trainer=SimpleNamespace())

    assert observed == ["train", "val_model"]
