import pandas as pd
from stage1_gapvalue240.manifests import build_replay_manifests

def test_additive_replay_and_phase_b_shape(tmp_path):
    d=pd.DataFrame({'canonical_image_relpath':[f'd{i}' for i in range(4)],'Filename':[f'd{i}.png' for i in range(4)],'x':range(4)})
    n=pd.DataFrame({'canonical_image_relpath':[f'n{i}' for i in range(4)],'Filename':[f'n{i}.png' for i in range(4)],'x':range(4)})
    d.to_csv(tmp_path/'d.csv',index=False); n.to_csv(tmp_path/'n.csv',index=False)
    s=pd.DataFrame({'run_slot':['RUN_001']*3,'rank':[1,2,3],
                    'sample_id':['n0','n1','d0'],'y_true':[0,0,1],
                    'replay_role':['normal_replay','normal_replay','defect_guard']})
    s.to_csv(tmp_path/'s.csv',index=False)
    out=build_replay_manifests(tmp_path/'d.csv',tmp_path/'n.csv',tmp_path/'s.csv',tmp_path/'out',expected_base_total=8)
    train=pd.read_csv(out.train_manifest); normal=pd.read_csv(out.normal_train_manifest)
    assert len(train)==5
    assert len(normal)==6
    assert len(pd.read_csv(out.audit_manifest))==11
    assert train.Filename.is_unique
    assert normal.Filename.is_unique
    assert train.iloc[-1].Filename.startswith('replay__RUN_001__00003__')
    assert normal.iloc[-2].Filename.startswith('replay__RUN_001__00001__')
    assert normal.iloc[-1].Filename.startswith('replay__RUN_001__00002__')
