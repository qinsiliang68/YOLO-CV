import torch

from stage1_sctsr_v4.serialization import load_json, sha256_file

def test_canary_runs_all_eight_arms(canary_root):
    r=load_json(canary_root/'08_receipts/SYNTHETIC_CANARY_RECEIPT.json');assert r['status']=='PASS';assert len(r['arms_completed'])==8;assert r['failure_injection_count']==6

def test_canary_has_no_formal_side_effects(canary_root):
    m=load_json(canary_root/'RUN_MANIFEST.json')
    for k in ('formal_training_started','engineering_gate_generated','assignments_generated','pilot_release_generated','blind_holdout_opened','selector_trained','method_effectiveness_claimed'):assert m[k] is False

def test_canary_contains_real_checkpoint_and_ledgers(canary_root):
    assert list((canary_root/'05_checkpoints').glob('*.pt'));assert list((canary_root/'04_ledgers').rglob('*.parquet'))


def test_synthetic_fixture_covers_taskbook_minimums(synthetic_fixture):
    fixture = synthetic_fixture
    assert fixture.base_denominator % 40 == 0  # 25/1000 identity pool
    assert fixture.base_denominator % 200 == 0  # 5/1000 U replay
    assert fixture.base_denominator % 100 == 0  # 10/1000 F replay
    assert {record.y_true for record in fixture.base_records} == {0, 1}
    assert len({record.oof_fold for record in fixture.base_records}) > 1
    assert len({record.oof_group_id for record in fixture.base_records}) > 1
    assert all(len(groups) == 5 for groups in fixture.groups_by_pool.values())
    assert not ({row.sample_id for row in fixture.t_pool.records} & {row.sample_id for row in fixture.r2_result.pool.records})
    assert torch.equal(fixture.features['SYN_000000'], fixture.features['SYN_000001'])


def test_canary_exercises_real_checkpoint_resume(canary_root):
    receipt_path = canary_root / '08_receipts' / 'CHECKPOINT_RESUME_RECEIPT.json'
    receipt = load_json(receipt_path)
    assert receipt['status'] == 'PASS'
    assert receipt['checkpoint_epoch'] == 121
    assert receipt['resumed_epoch'] == 122
    assert receipt['uninterrupted_checkpoint_payload_digest'] == receipt['resumed_checkpoint_payload_digest']
    assert receipt['uninterrupted_optimizer_steps'] == receipt['resumed_optimizer_steps']
    assert receipt['uninterrupted_ema_updates'] == receipt['resumed_ema_updates']
    checkpoint = canary_root / receipt['resume_checkpoint_path']
    assert sha256_file(checkpoint) == receipt['resume_checkpoint_sha256_before']
    assert receipt['resume_checkpoint_sha256_before'] == receipt['resume_checkpoint_sha256_after']


def test_canary_predictions_contain_real_probability_ties(canary_root):
    manifest = load_json(canary_root / 'RUN_MANIFEST.json')
    tie_sizes = [summary['maximum_probability_tie_size'] for summary in manifest['branch_summaries'].values()]
    assert len(tie_sizes) == 8
    assert min(tie_sizes) >= 2


def test_canary_source_manifest_covers_dependency_lock_and_imported_upstream(canary_root):
    manifest = load_json(canary_root / '00_contract' / 'SOURCE_TREE_MANIFEST.json')
    included = set(manifest['include_paths'])
    assert 'uv.lock' in included
    assert '.gitattributes' in included
    assert 'YOLOv11/ultralytics' in included
    assert 'integrations/ultralytics' in included
    assert any(path.endswith('SCTSR_EXPERT_IMPLEMENTATION_TASKBOOK.md') for path in included)
