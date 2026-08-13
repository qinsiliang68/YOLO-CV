from dataclasses import replace
import pytest
from stage1_sctsr_v4.errors import ErrorCode,SctsrError
from stage1_sctsr_v4.telemetry import sample_telemetry,validate_telemetry_for_closeout

def test_real_process_and_disk_telemetry(tmp_path):
    r=sample_telemetry(run_id='r',arm_id='NR',training_seed=1,epoch=1,run_path=tmp_path,artifact_path=tmp_path)
    validate_telemetry_for_closeout([r]);assert r.process_pid>0;assert r.run_volume_free

def test_fake_zero_critical_telemetry_rejected(tmp_path):
    r=sample_telemetry(run_id='r',arm_id='NR',training_seed=1,epoch=1,run_path=tmp_path,artifact_path=tmp_path)
    bad=replace(r,process_rss=0)
    with pytest.raises(SctsrError) as e:validate_telemetry_for_closeout([bad])
    assert e.value.code is ErrorCode.TELEMETRY_UNAVAILABLE

def test_bad_cadence_rejected(tmp_path):
    r=sample_telemetry(run_id='r',arm_id='NR',training_seed=1,epoch=1,run_path=tmp_path,artifact_path=tmp_path)
    rows=[r,replace(r,monotonic_seconds=r.monotonic_seconds+1),replace(r,monotonic_seconds=r.monotonic_seconds+3)]
    with pytest.raises(SctsrError):validate_telemetry_for_closeout(rows)
