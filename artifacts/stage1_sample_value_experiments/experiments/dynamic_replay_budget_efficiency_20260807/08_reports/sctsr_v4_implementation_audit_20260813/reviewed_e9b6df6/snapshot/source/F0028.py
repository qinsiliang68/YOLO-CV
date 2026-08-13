from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stage1_sctsr_v4.cli_support import add_output_argument, run_cli
from stage1_sctsr_v4.contracts import validate_contract_files
from stage1_sctsr_v4.schema_registry import SchemaRegistry
from stage1_sctsr_v4.serialization import load_json

def main() -> int:
    p=argparse.ArgumentParser(description="Validate the frozen SCTSR v4 contract")
    p.add_argument('--contract',type=Path,required=True);p.add_argument('--arms',type=Path,required=True);p.add_argument('--schemas',type=Path,required=True);add_output_argument(p);a=p.parse_args()
    def action():
        registry=SchemaRegistry.from_mapping(load_json(a.schemas));registry.validate()
        result=validate_contract_files(a.contract,a.arms)
        return {'contract_status':result.status,'contract_digest':result.contract_digest,'checks':dict(result.checks),'schema_registry_digest':registry.digest}
    return run_cli('validate_contract',a.output,action)
if __name__=='__main__': raise SystemExit(main())
