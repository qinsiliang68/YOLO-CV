from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


def main() -> None:
    repository = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(repository))
    sys.path.insert(0, str(repository / "tests" / "stage1_sctsr_v4"))

    from test_prediction_runtime import _image_registry
    from stage1_sctsr_v4.errors import SctsrError
    from stage1_sctsr_v4.prediction_runtime import (
        build_endpoint_input_binding,
        load_registered_image_records,
        validate_endpoint_input_binding,
    )
    from stage1_sctsr_v4.serialization import sha256_file

    with tempfile.TemporaryDirectory(prefix="sctsr_endpoint_binding_") as raw_temp:
        temp = Path(raw_temp)
        dataset = temp / "dataset"
        rows = [{"canonical_image_relpath": "Det/images/val_op/synthetic_a.png", "split": "val_op"}]
        image = dataset / rows[0]["canonical_image_relpath"]
        image.parent.mkdir(parents=True)
        image.write_bytes(b"synthetic-endpoint-input")
        content = {
            rows[0]["canonical_image_relpath"]: {
                "image_bytes": image.stat().st_size,
                "image_sha256": sha256_file(image),
            }
        }
        registry = _image_registry(temp, rows)
        records = load_registered_image_records(
            registry,
            repository_root=temp,
            dataset_root=dataset,
            split_role="val_op",
            expected_content=content,
        )
        ledger = temp / "content_ledger.parquet"
        ledger.write_bytes(b"synthetic-content-ledger")
        binding = build_endpoint_input_binding(
            records,
            registry=registry,
            repository_root=temp,
            dataset_root=dataset,
            split_role="val_op",
            content_ledger_path=ledger,
            content_ledger_sha256=sha256_file(ledger),
            content_ledger_identity_digest="B" * 64,
        )
        validate_endpoint_input_binding(binding)
        image.write_bytes(b"mutated-endpoint-input")
        mutation_rejected = None
        try:
            validate_endpoint_input_binding(binding)
        except SctsrError as error:
            mutation_rejected = error.code.value
        if mutation_rejected != "DATASET_CONTENT_MISMATCH":
            raise AssertionError("Endpoint input mutation was not rejected")
        print(
            json.dumps(
                {
                    "schema_version": "stage1.sctsr.training_output_fix_endpoint_binding_example.v1",
                    "scientific_role": "SYNTHETIC_SCHEMA_EVIDENCE_NOT_ENDPOINT_ACCESS",
                    "binding": binding,
                    "post_binding_mutation_rejected_with": mutation_rejected,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
