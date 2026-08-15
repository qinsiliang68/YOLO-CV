from pathlib import Path
import pytest
from stage1_sctsr_v4.errors import ErrorCode,SctsrError
from stage1_sctsr_v4.training_system import assert_formal_authorized,bind_upstream,import_classification_trainer,prepare_classification_overrides,validate_sctsr_adapter_import,validate_upstream_manifest

def test_missing_upstream_tree_fails(tmp_path):
    with pytest.raises(SctsrError) as e:bind_upstream(tmp_path,verify_hashes=False)
    assert e.value.code is ErrorCode.UPSTREAM_BINDING_FAILED

def test_formal_authorization_guard(tmp_path):
    with pytest.raises(SctsrError):assert_formal_authorized('formal',tmp_path/'missing')

def test_upstream_binding_records_git_and_file_hashes(repository_root):
    binding=bind_upstream(repository_root)
    assert len(binding.source_git_blob_sha1)==6
    assert all(len(value)==40 for value in binding.source_git_blob_sha1.values())
    assert all(len(value)==64 for value in binding.source_file_sha256.values())
    assert len(validate_upstream_manifest(binding,repository_root/'integrations/ultralytics/UPSTREAM_FILES_MANIFEST.json'))==64
    trainer=import_classification_trainer(binding)
    assert trainer.__module__=='ultralytics.models.yolo.classify.train'


def test_sctsr_adapter_import_origin_is_exactly_bound(repository_root, tmp_path):
    import types

    binding = bind_upstream(repository_root)
    expected = repository_root / binding.adapter_relative_path
    valid = types.SimpleNamespace(__file__=str(expected))
    report = validate_sctsr_adapter_import(binding, valid)
    assert report["adapter_sha256"] == binding.adapter_file_sha256

    fake = tmp_path / "shadow" / "sctsr_classification_trainer.py"
    fake.parent.mkdir()
    fake.write_text("class SctsrClassificationTrainer: pass\n", encoding="utf-8")
    with pytest.raises(SctsrError) as caught:
        validate_sctsr_adapter_import(binding, types.SimpleNamespace(__file__=str(fake)))
    assert caught.value.code is ErrorCode.UPSTREAM_BINDING_FAILED

def test_trainer_overrides_preserve_full_canonical_lock(repository_root):
    binding=bind_upstream(repository_root)
    runtime={"model":"yolo11l-cls.pt","data":"DATA","device":0,"project":"PROJECT","name":"trainer","seed":7,"exist_ok":False,"resume":False}
    prepared=prepare_classification_overrides(binding,runtime)
    assert prepared["batch"]==128 and prepared["epochs"]==200 and prepared["imgsz"]==224
    assert prepared["warmup_epochs"]==3.0 and prepared["auto_augment"]=="randaugment"
    assert "world_size" not in prepared

def test_trainer_override_cannot_change_frozen_learner(repository_root):
    binding=bind_upstream(repository_root)
    runtime={"model":"yolo11l-cls.pt","data":"DATA","device":0,"project":"PROJECT","name":"trainer","seed":7,"exist_ok":False,"resume":False,"batch":64}
    with pytest.raises(SctsrError) as exc:prepare_classification_overrides(binding,runtime)
    assert exc.value.code is ErrorCode.CONFIGURATION_MISMATCH
