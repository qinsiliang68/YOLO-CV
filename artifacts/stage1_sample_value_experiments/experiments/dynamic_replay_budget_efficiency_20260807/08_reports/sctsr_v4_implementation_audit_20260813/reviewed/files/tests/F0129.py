from pathlib import Path

def test_overlay_contains_no_formal_queue_or_release(repository_root):
    bad=[]
    implementation_roots = (
        repository_root / 'stage1_sctsr_v4',
        repository_root / 'scripts' / 'stage1_sctsr_v4',
        repository_root / 'configs' / 'stage1_sctsr_v4',
        repository_root / 'integrations' / 'ultralytics',
    )
    for root in implementation_roots:
        for path in root.rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts:continue
            lower=path.as_posix().lower()
            if any(token in lower for token in ('pilot_release','formal_assignment','engineering_gate')):
                bad.append(path)
    assert bad==[]

def test_full_checkout_preserves_protected_historical_and_upstream_modules(repository_root):
    assert (repository_root/'stage1_dynamic_replay_v3').is_dir()
    assert (repository_root/'stage1_gapvalue240').is_dir()
    assert (repository_root/'YOLOv11/ultralytics').is_dir()
