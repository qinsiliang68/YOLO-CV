from pathlib import Path
from stage1_gapvalue240.ranking import load_value_assets,direct_ranking

ROOT=Path(__file__).resolve().parents[2]
REF=ROOT/'artifacts/stage1_sample_value_experiments/contracts/gapvalue240_v1_1/frozen_inputs/reference_tables'

def test_real_candidate_counts_and_semantics():
    d=load_value_assets(REF/'sample_value_table.csv',REF/'train_oof_assignments.csv')
    strict=direct_ranking(d,'GapCritical-Strict').table
    clean=direct_ranking(d,'Confidence-Clean').table
    bottom=direct_ranking(d,'BottomGap-3000-stress-control').table.head(3000)
    assert len(d)==120000
    assert len(clean)==57541
    assert len(strict)==9574
    assert (bottom.raw_score<0).sum()==2844
