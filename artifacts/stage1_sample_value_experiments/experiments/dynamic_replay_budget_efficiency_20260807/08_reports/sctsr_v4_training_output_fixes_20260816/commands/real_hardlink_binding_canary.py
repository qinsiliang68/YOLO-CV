from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import torch


def main() -> None:
    repository = Path(sys.argv[1]).resolve()
    canonical_root = Path(sys.argv[2]).resolve()
    sys.path.insert(0, str(repository))

    from stage1_sctsr_v4.columnar import read_columnar
    from stage1_sctsr_v4.dataset_adapter import (
        DatasetIdentity,
        IdentityAugmentingDataset,
        revalidate_materialized_dataset_binding,
        validate_materialized_dataset_bytes,
    )
    from stage1_sctsr_v4.engineering_canary import select_engineering_canary_samples
    from stage1_sctsr_v4.errors import SctsrError

    selected = select_engineering_canary_samples(canonical_root, per_class=1)
    with tempfile.TemporaryDirectory(prefix="sctsr_real_hardlink_view_") as raw_temp:
        temp = Path(raw_temp)
        view = temp / "classification_view"
        physical_samples: list[tuple[str, int]] = []
        identities: list[DatasetIdentity] = []
        content: dict[str, dict[str, object]] = {}
        for sample in selected:
            class_name = "target_defect" if sample.label == 1 else "no_target"
            destination = view / "train" / class_name / sample.sample_id
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(sample.image_path, destination)
            relative = Path(sample.image_path).resolve().relative_to(canonical_root).as_posix()
            physical_samples.append((destination.as_posix(), sample.label))
            identities.append(DatasetIdentity(relative, sample.label, relative))
            content[relative] = {
                "image_bytes": sample.image_bytes,
                "image_sha256": sample.image_sha256,
            }

        class RealPhysicalDataset(torch.utils.data.Dataset):
            samples = tuple(physical_samples)

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, index):
                return {"img": torch.zeros((1, 2, 2)), "cls": torch.tensor(self.samples[index][1])}

        wrapped = IdentityAugmentingDataset(RealPhysicalDataset(), tuple(identities))
        binding = validate_materialized_dataset_bytes(
            wrapped,
            content,
            role="train",
            dataset_root=canonical_root,
            materialized_data_root=view,
            evidence_path=temp / "binding" / "real_materialized_rows.parquet",
        )
        revalidate_materialized_dataset_binding(binding)
        rows = read_columnar(binding["evidence"]["path"])
        if not all(row["samefile_as_canonical"] for row in rows):
            raise AssertionError("Real classification view is not hardlink-identical to canonical input")

        added = Path(physical_samples[0][0]).parent / "added_after_setup.png"
        added.write_bytes(b"not-registered")
        extra_file_rejected = None
        try:
            revalidate_materialized_dataset_binding(binding)
        except SctsrError as error:
            extra_file_rejected = error.code.value
        if extra_file_rejected != "DATASET_CONTENT_MISMATCH":
            raise AssertionError("Post-setup extra file was not rejected")

        print(
            json.dumps(
                {
                    "schema_version": "stage1.sctsr.real_hardlink_binding_canary.v1",
                    "status": "PASS",
                    "scientific_role": "ENGINEERING_CANARY_NOT_SCIENTIFIC_RESULT",
                    "canonical_dataset_root": binding["canonical_dataset_root"],
                    "materialized_data_root": binding["materialized_data_root"],
                    "row_count": binding["row_count"],
                    "binding_digest": binding["binding_digest"],
                    "materialized_content_digest": binding["materialized_content_digest"],
                    "all_rows_samefile_as_canonical": True,
                    "real_image_sha256": sorted(row["image_sha256"] for row in rows),
                    "post_setup_extra_file_rejected_with": extra_file_rejected,
                    "formal_training_started": False,
                    "test_accessed": False,
                    "blind_holdout_opened": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
