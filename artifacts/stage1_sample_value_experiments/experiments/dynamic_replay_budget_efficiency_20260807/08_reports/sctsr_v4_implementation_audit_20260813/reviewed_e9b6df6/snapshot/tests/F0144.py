from stage1_sctsr_v4.arm_spec import ArmId
from stage1_sctsr_v4.schedule import validate_common_prefix,validate_u_f_parity

def test_all_eight_schedules_construct(phase1_schedules):
    plans=phase1_schedules;assert set(plans)==set(ArmId);assert all(len(p.epochs)==200 for p in plans.values())

def test_uniform_and_frontloaded_have_same_dose(phase1_schedules):
    p=phase1_schedules;validate_u_f_parity(p[ArmId.T_U],p[ArmId.T_F]);assert p[ArmId.T_U].total_occurrences==p[ArmId.T_F].total_occurrences==800

def test_every_t_identity_has_multiplicity_16(phase1_schedules):
    p=phase1_schedules;assert set(p[ArmId.T_U].multiplicity().values())=={16};assert set(p[ArmId.T_F].multiplicity().values())=={16}

def test_frontloaded_stops_after_160(phase1_schedules):
    p=phase1_schedules[ArmId.T_F];assert all(not p.epoch(e).sample_ids for e in range(161,201))

def test_fallback_and_stop_share_t_prefix(phase1_schedules):
    p=phase1_schedules;validate_common_prefix(p[ArmId.T_U],p[ArmId.T_TO_R2_AT_160]);validate_common_prefix(p[ArmId.T_U],p[ArmId.T_TO_NR_AT_160])

def test_fallback_is_r2_not_no_replay(phase1_schedules):
    p=phase1_schedules[ArmId.T_TO_R2_AT_160];assert p.epoch(161).identity_policy=='R2_MATCHED_RANDOM';assert p.epoch(161).sample_ids

def test_nr_has_no_replay(phase1_schedules):
    assert phase1_schedules[ArmId.NR].total_occurrences==0
