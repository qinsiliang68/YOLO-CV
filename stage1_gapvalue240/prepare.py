from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json
import pandas as pd
import numpy as np

from .assets import reference_root,validate_frozen_inputs
from .contract import Contract
from .identity import materialize_master_index
from .matrix import build_run_specs,write_matrix,RunSpec
from .matching import build_match_context
from .oof import build_oof_memmap
from .overlap import apply_overlap_gate
from .ranking import load_value_assets,direct_ranking,dynamic_scores,dynamic_ranking
from .seeds import derive_seed
from .selection import phase_a_selection,phase_b_selection,write_selection
from .queue_validation import validate_frozen_queues,write_frozen_queue_manifest
from .util import atomic_write_bytes,atomic_write_json,sha256_file

DIRECT_METHODS=["GapCritical-Strict","GapCritical-Global","Confidence-Clean","Boundary-0.5-Clean","Persistent-0.5-Clean",
                "EndpointTrend","BottomGap-3000-stress-control","FoldBalanced-Gap","GroupDiverse-Gap","LowConfidence-Defect",
                "Persistent-FN","GapGuard-Raw","GapGuard-ClassStrat"]
DYNAMIC_METHODS=["TailGap-Strict","WindowEarlyLate40","LOFO-GapStrict","Exclude178-GapStrict","GapResidual-Strict","GapCritical-Strict-TimeMatched"]


def _write_ranking(result,out_dir:Path)->Path:
    out_dir.mkdir(parents=True,exist_ok=True); p=out_dir/f"{result.method}.csv"
    atomic_write_bytes(p,result.table.to_csv(index=False).encode(),overwrite=True)
    atomic_write_json(out_dir/f"{result.method}.metadata.json",result.metadata|{"sha256":sha256_file(p)},overwrite=True)
    return p


def prepare_all(repo_root:str|Path,raw_oof_root:str|Path,artifact_root:str|Path,contract:Contract)->dict:
    repo_root=Path(repo_root).resolve(); artifact_root=Path(artifact_root).resolve(); artifact_root.mkdir(parents=True,exist_ok=True)
    validate_frozen_inputs(repo_root,contract,artifact_root/"review/frozen_input_validation.json")
    ref=reference_root(repo_root,contract); generated=artifact_root/"generated"; generated.mkdir(exist_ok=True)
    complete=generated/"PREPARATION_COMPLETE.json"
    if complete.exists():
        existing=json.loads(complete.read_text(encoding="utf-8"))
        validate_frozen_queues(generated/"frozen_experiment_matrix.csv",generated/"selection_index.csv",artifact_root,
                               generated/"machine_shards",generated/"QUEUE_VALIDATION.json")
        write_frozen_queue_manifest(generated,generated/"FROZEN_QUEUE_FILE_MANIFEST.csv")
        return existing|{"status":"READY_REUSED"}
    master=generated/"master_sample_index.csv"
    materialize_master_index(ref/"train_oof_assignments.csv",master,overwrite=True)
    data=load_value_assets(ref/"sample_value_table.csv",ref/"train_oof_assignments.csv")
    rankings:dict[str,pd.DataFrame]={}; rdir=generated/"rankings"
    precomputed=artifact_root/"precomputed_direct_assets/rankings"
    for method in DIRECT_METHODS:
        pre=precomputed/f"{method}.csv"
        if pre.exists():
            table=pd.read_csv(pre,dtype={"sample_id":"string","oof_fold":"string"}); table["oof_fold"]=table["oof_fold"].str.zfill(2)
            rankings[method]=table
            atomic_write_bytes(rdir/f"{method}.csv",table.to_csv(index=False).encode(),overwrite=True)
            atomic_write_json(rdir/f"{method}.metadata.json",{"source":"precomputed_frozen","source_sha256":sha256_file(pre),"rows":len(table)},overwrite=True)
        else:
            res=direct_ranking(data,method); _write_ranking(res,rdir); rankings[method]=res.table
    cache=build_oof_memmap(raw_oof_root,ref/"summary_input_manifest.csv",master,generated/"oof_cache")
    matrix=cache.open('r'); ids=pd.read_csv(cache.sample_ids_path,dtype={"sample_id":"string","oof_fold":"string"})
    # Ensure value table order matches the memmap order before assigning scores.
    if not data.sample_id.equals(ids.sample_id):
        order=pd.Series(np.arange(len(data)),index=data.sample_id)
        idx=ids.sample_id.map(order)
        if idx.isna().any(): raise RuntimeError("OOF cache/value identity mismatch")
        data=data.iloc[idx.to_numpy()].reset_index(drop=True)
    epoch=pd.read_csv(ref/"epoch_gap_metrics.csv")
    ds=dynamic_scores(matrix,ids.y_true.to_numpy(np.int8),ids.oof_fold.str.zfill(2).to_numpy(),epoch)
    for method,score in ds.items():
        res=dynamic_ranking(data,score,method); _write_ranking(res,rdir); rankings[method]=res.table
    main="GapCritical-Strict"; candidates=["TailGap-Strict","WindowEarlyLate40","LOFO-GapStrict","Exclude178-GapStrict"]
    replacements=["GapResidual-Strict","GapCritical-Global-B6000","GapCritical-Strict-TimeMatched"]
    # Alias the contract replacement to the existing global ranking.
    rankings["GapCritical-Global-B6000"]=rankings["GapCritical-Global"]
    retained,decisions=apply_overlap_gate(main,rankings,candidates,replacements,.95,3000)
    atomic_write_json(generated/"overlap_decisions.json",[asdict(x) for x in decisions],overwrite=True)
    replacement_map={d.candidate:d.retained_as for d in decisions if d.replaced}
    specs=build_run_specs(contract)
    updated=[]
    for s in specs:
        method=replacement_map.get(s.method,s.method); budget=s.budget
        if method=="GapCritical-Global-B6000": budget=6000
        cid=s.condition_id
        if method!=s.method or budget!=s.budget:
            cid=f"{s.condition_slot}_{method}_B{budget}"
            s=RunSpec(s.run_slot,s.triad_id,s.phase,s.condition_slot,cid,method,budget,s.guard_ratio,s.arm,s.training_seed,
                      derive_seed(contract.contract_id,cid,s.training_seed,s.arm),s.discovery_or_confirmation)
        updated.append(s)
    frozen=generated/"frozen_experiment_matrix.csv"; write_matrix(updated,frozen,overwrite=True)
    normal_top=rankings["GapCritical-Strict"].head(3000)
    qbins=contract.data["controls"]["r2"]["quantile_bins"]
    normal_match_context=build_match_context(data[(data.y_true==0)&data.is_clean],qbins)
    defect_match_context=build_match_context(data[(data.y_true==1)&data.is_clean],qbins)
    selection_index=[]
    for spec in updated:
        if spec.phase in {'A','C'}:
            ranking=rankings[spec.method]
            selection,audit=phase_a_selection(spec,ranking,data,contract,normal_match_context)
        else:
            selection,audit=phase_b_selection(spec,normal_top,rankings[spec.method],data,contract,defect_match_context)
        art=write_selection(selection,audit,generated/"selections"/spec.run_slot,overwrite=True)
        selection_index.append({"run_slot":spec.run_slot,"selection_manifest":str(art.csv_path.relative_to(artifact_root)),"sha256":art.sha256})
    atomic_write_bytes(generated/"selection_index.csv",pd.DataFrame(selection_index).to_csv(index=False).encode(),overwrite=True)
    from .shards import write_machine_shards
    write_machine_shards(pd.DataFrame([asdict(s) for s in updated]),generated/"machine_shards")
    queue_report=validate_frozen_queues(frozen,generated/"selection_index.csv",artifact_root,generated/"machine_shards",
                                        generated/"QUEUE_VALIDATION.json")
    report={"status":"READY","contract_sha256":contract.sha256,"matrix_sha256":sha256_file(frozen),"selection_count":len(selection_index),
            "run_count":len(updated),"triad_count":len({s.triad_id for s in updated}),"overlap_decisions":[asdict(x) for x in decisions],
            "queue_validation_status":queue_report["status"],"selection_index_sha256":queue_report["selection_index_sha256"]}
    atomic_write_json(complete,report,overwrite=False)
    write_frozen_queue_manifest(generated,generated/"FROZEN_QUEUE_FILE_MANIFEST.csv")
    return report
