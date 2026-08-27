from dataclasses import replace
import subprocess
import pytest
import stage1_sctsr_v4.telemetry as telemetry_module
from stage1_sctsr_v4.errors import ErrorCode,SctsrError
from stage1_sctsr_v4.telemetry import TelemetrySampler,sample_telemetry,validate_telemetry_for_closeout

def test_real_process_and_disk_telemetry(tmp_path):
    r=sample_telemetry(run_id='r',arm_id='NR',training_seed=1,epoch=1,run_path=tmp_path,artifact_path=tmp_path)
    validate_telemetry_for_closeout([r]);assert r.process_pid>0;assert r.run_volume_free

def test_fake_zero_critical_telemetry_rejected(tmp_path):
    r=sample_telemetry(run_id='r',arm_id='NR',training_seed=1,epoch=1,run_path=tmp_path,artifact_path=tmp_path)
    bad=replace(r,process_rss=0)
    with pytest.raises(SctsrError) as e:validate_telemetry_for_closeout([bad])
    assert e.value.code is ErrorCode.TELEMETRY_UNAVAILABLE

def test_cadence_jitter_does_not_abort_training_closeout(tmp_path):
    r=sample_telemetry(run_id='r',arm_id='NR',training_seed=1,epoch=1,run_path=tmp_path,artifact_path=tmp_path)
    rows=[r,replace(r,monotonic_seconds=r.monotonic_seconds+1),replace(r,monotonic_seconds=r.monotonic_seconds+3)]
    validate_telemetry_for_closeout(rows)


def test_nvidia_smi_timeout_cannot_overrun_formal_cadence(monkeypatch):
    observed = {}

    def timed_out_probe(*_args, **kwargs):
        observed["timeout"] = kwargs["timeout"]
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=kwargs["timeout"])

    monkeypatch.setattr(telemetry_module.shutil, "which", lambda _name: "nvidia-smi.exe")
    monkeypatch.setattr(telemetry_module.subprocess, "run", timed_out_probe)

    gpu, reason = telemetry_module._nvidia_smi()

    assert gpu is None
    assert reason == "NVIDIA_SMI_TIMEOUTEXPIRED"
    assert observed["timeout"] == telemetry_module.NVIDIA_SMI_TIMEOUT_SECONDS
    assert observed["timeout"] < telemetry_module.TELEMETRY_CADENCE_SECONDS


def test_sample_timestamp_is_provider_acquisition_start(tmp_path, monkeypatch):
    clock=[100.0]

    def slow_gpu_probe():
        clock[0]+=0.7
        return None,"NVIDIA_SMI_TIMEOUT"

    monkeypatch.setattr(telemetry_module.time,"monotonic",lambda:clock[0])
    monkeypatch.setattr(telemetry_module,"_nvidia_smi",slow_gpu_probe)
    row=sample_telemetry(
        run_id='r',arm_id='NR',training_seed=1,epoch=1,
        run_path=tmp_path,artifact_path=tmp_path,
    )

    assert row.monotonic_seconds==pytest.approx(100.0)
    assert clock[0]==pytest.approx(100.7)


def test_sampler_reanchors_after_slow_first_provider_sample(tmp_path, monkeypatch):
    base=sample_telemetry(run_id='r',arm_id='NR',training_seed=1,epoch=1,run_path=tmp_path,artifact_path=tmp_path)
    clock=[100.0]
    durations=iter((1.2,0.328,0.328))
    calls=[0]

    def fake_sample(**_kwargs):
        clock[0]+=next(durations)
        calls[0]+=1
        return replace(base,monotonic_seconds=clock[0])

    class FakeStop:
        def is_set(self):
            return calls[0]>=3

        def set(self):
            calls[0]=3

        def wait(self,delay):
            if calls[0]<3:
                clock[0]+=delay
            return calls[0]>=3

    monkeypatch.setattr(telemetry_module.time,'monotonic',lambda:clock[0])
    sampler=TelemetrySampler(
        run_id='r',arm_id='NR',training_seed=1,epoch=1,
        run_path=tmp_path,artifact_path=tmp_path,row_generation=1,
        sample_function=fake_sample,
    )
    sampler._stop=FakeStop()
    sampler._run()

    validate_telemetry_for_closeout(sampler.rows)
    assert sampler.rows[1].monotonic_seconds-sampler.rows[0].monotonic_seconds==pytest.approx(1.328)
