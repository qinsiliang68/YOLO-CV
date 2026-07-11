from pathlib import Path
import shutil
import pandas as pd
import pytest
import yaml
from stage1_gapvalue240.run_engine import prepare_run, run_all

ROOT=Path(__file__).resolve().parents[2]

def test_one_run_end_to_end_dry(tmp_path):
    repo=tmp_path/'repo'; repo.mkdir()
    (repo/'configs/stage1_gapvalue240').mkdir(parents=True)
    shutil.copy2(ROOT/'configs/stage1_gapvalue240/EXPERIMENT_CONTRACT.yaml',repo/'configs/stage1_gapvalue240/EXPERIMENT_CONTRACT.yaml')
    artifact=tmp_path/'artifact'; (artifact/'generated/selections/RUN_001').mkdir(parents=True)
    shutil.copy2(ROOT/'artifacts/stage1_sample_value_experiments/contracts/gapvalue240_v1_1/planned/planned_run_slot_matrix_v1_1.csv',artifact/'generated/frozen_experiment_matrix.csv')
    defect=pd.DataFrame({'canonical_image_relpath':[f'd{i:05d}' for i in range(60000)],'Filename':[f'd{i:05d}.png' for i in range(60000)],'label':1})
    normal=pd.DataFrame({'canonical_image_relpath':[f'n{i:05d}' for i in range(60000)],'Filename':[f'n{i:05d}.png' for i in range(60000)],'label':0})
    defect_path=tmp_path/'train.csv'; normal_path=tmp_path/'normal.csv'; defect.to_csv(defect_path,index=False); normal.to_csv(normal_path,index=False)
    sel=pd.DataFrame({'run_slot':'RUN_001','triad_id':'TRIAD_001','condition_id':'A01','arm':'T','training_seed':1,'selection_seed':1,
                      'rank':range(1,601),'sample_id':[f'n{i:05d}' for i in range(600)],'y_true':0,'oof_fold':'00',
                      'dynamic_bucket':'learnable_hard','mean_p_defect':.3,'correct_rate':.8,'std_p_defect':.2,
                      'replay_role':'normal_replay','source_method':'GapCritical-Strict'})
    sel.to_csv(artifact/'generated/selections/RUN_001/selection_manifest.csv',index=False)
    dataset=tmp_path/'dataset'; dataset.mkdir(); checkpoint=tmp_path/'yolo11l-cls.pt'; checkpoint.write_bytes(b'dry')
    val_defect=tmp_path/'val_model.csv'; val_normal=tmp_path/'normal_val_model.csv'
    defect.iloc[:2].to_csv(val_defect,index=False); normal.iloc[:2].to_csv(val_normal,index=False)
    machine={
      'machine_id':'machine_01','repo_root':str(repo),'dataset_root':str(dataset),'oof_raw_root':str(tmp_path/'raw'),
      'artifact_root':str(artifact),'output_root':str(tmp_path/'outputs'),'cache_root':str(tmp_path/'cache'),
      'local_scratch_root':str(tmp_path/'scratch'),'gpu_id':0,'num_workers':0,'python_executable':'python',
      'base_checkpoint':str(checkpoint),'train_manifest':str(defect_path),'normal_train_manifest':str(normal_path),
      'val_model_defect_manifest':str(val_defect),'val_model_normal_manifest':str(val_normal),
      'prediction_batch_size':256,'prediction_workers':0,'nvidia_smi_path':'definitely-not-installed','dry_run':True,'command_timeout_seconds':0
    }
    mp=tmp_path/'machine.yaml'; mp.write_text(yaml.safe_dump(machine),encoding='utf-8')
    result=run_all('RUN_001',mp)
    assert result.attempt_dir.exists()
    assert (result.attempt_dir/'08_status/VALIDATED').exists()
    assert (result.attempt_dir/'05_metrics/operational_metrics.json').exists()

    with pytest.raises(FileNotFoundError):
        prepare_run('RUN_002', mp)
    failed_attempts = list((tmp_path/'outputs/runs/RUN_002').glob('attempt_*.inprogress'))
    assert len(failed_attempts) == 1
    assert (failed_attempts[0]/'08_status/FAILED_INPUT').exists()
