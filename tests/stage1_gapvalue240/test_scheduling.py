import pandas as pd

from stage1_gapvalue240.scheduling import order_triad_rows
from stage1_gapvalue240.shards import write_machine_shards


def _matrix():
    rows=[]
    for triad in range(1,81):
        for arm in ("T","R1","R2"):
            rows.append({"run_slot":f"RUN_{(triad-1)*3+(1 if arm=='T' else 2 if arm=='R1' else 3):03d}",
                         "triad_id":f"TRIAD_{triad:03d}","arm":arm})
    return pd.DataFrame(rows)


def test_machine_shards_round_robin_triads_across_hardware(tmp_path):
    write_machine_shards(_matrix(),tmp_path)
    m1=pd.read_csv(tmp_path/'machine_01_jobs.csv')
    m10=pd.read_csv(tmp_path/'machine_10_jobs.csv')
    assert m1.triad_id.drop_duplicates().tolist()==[f'TRIAD_{i:03d}' for i in range(1,72,10)]
    assert m10.triad_id.drop_duplicates().tolist()==[f'TRIAD_{i:03d}' for i in range(10,81,10)]
    assert len(m1)==len(m10)==24
    assert pd.read_csv(tmp_path/'machine_11_jobs.csv').empty
    assert pd.read_csv(tmp_path/'machine_12_jobs.csv').empty


def test_arm_order_is_counterbalanced_by_triad():
    matrix=_matrix()
    assert order_triad_rows(matrix[matrix.triad_id=='TRIAD_001']).arm.tolist()==['T','R1','R2']
    assert order_triad_rows(matrix[matrix.triad_id=='TRIAD_002']).arm.tolist()==['R1','R2','T']
    assert order_triad_rows(matrix[matrix.triad_id=='TRIAD_003']).arm.tolist()==['R2','T','R1']
