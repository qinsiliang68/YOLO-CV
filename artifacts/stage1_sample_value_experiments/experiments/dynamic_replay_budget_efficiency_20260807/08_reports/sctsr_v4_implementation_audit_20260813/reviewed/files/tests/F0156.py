from stage1_sctsr_v4.arm_spec import ArmId
from stage1_sctsr_v4.synthetic_execution import run_synthetic_branch,run_synthetic_common_parent

def test_parent_then_branch(tmp_path,repository_root):
    parent=run_synthetic_common_parent(tmp_path/'parent',repository_root=repository_root,training_seed=9)
    branch=run_synthetic_branch(tmp_path/'branch',repository_root=repository_root,parent_checkpoint=parent['checkpoint_path'],arm_id=ArmId.T_U,training_seed=9)
    assert parent['optimizer_steps']>0;assert branch['optimizer_steps']==parent['optimizer_steps'];assert branch['replay_occurrences']>0
