from pathlib import Path
from stage1_gapvalue240.contract import load_contract,validate_contract_semantics
from stage1_gapvalue240.matrix import build_run_specs

ROOT=Path(__file__).resolve().parents[2]

def test_contract_and_matrix_counts():
    c=load_contract(ROOT/'configs/stage1_gapvalue240/EXPERIMENT_CONTRACT.yaml')
    assert validate_contract_semantics(c)==[]
    specs=build_run_specs(c)
    assert len(specs)==240
    assert len({x.triad_id for x in specs})==80
    assert {x.arm for x in specs}=={'T','R1','R2'}
    assert sum(x.phase=='A' for x in specs)==171
    assert sum(x.phase=='B' for x in specs)==54
    assert sum(x.phase=='C' for x in specs)==15
