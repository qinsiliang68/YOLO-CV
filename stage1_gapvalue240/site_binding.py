from __future__ import annotations
from pathlib import Path
import json
from .contract import Contract
from .errors import ValidationError
from .util import atomic_write_json,sha256_file


def bind_checkpoint(contract:Contract,checkpoint:Path,binding_path:Path,machine_id:str)->dict:
    h=sha256_file(checkpoint); spec=contract.data["frozen_inputs"]["base_checkpoint"]
    if not h.startswith(str(spec["known_sha256_prefix"]).upper()) or not h.endswith(str(spec["known_sha256_suffix"]).upper()):
        raise ValidationError(f"Checkpoint SHA-256 does not match known prefix/suffix: {h}")
    if binding_path.exists():
        data=json.loads(binding_path.read_text())
        if data["full_sha256"]!=h: raise ValidationError(f"Checkpoint differs from site binding: {h} != {data['full_sha256']}")
        return data
    if machine_id!="machine_01": raise ValidationError("Only machine_01 may create the initial site asset binding")
    data={"contract_id":contract.contract_id,"checkpoint_filename":checkpoint.name,"full_sha256":h,"created_by":machine_id}
    atomic_write_json(binding_path,data)
    return data
