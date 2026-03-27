# SewerML Hierarchical Label Alignment

This note records the source-to-target label alignment used for cross-domain transfer experiments in this repository.

## Three-Class Alignment

| SewerML item | Aligned class | Reason |
| --- | --- | --- |
| ND | Normal | Explicit no-defect sample. |
| RB / OB / PF / DE / FS / IS / IN | StructuralDefect | These labels describe wall, joint, leakage, or geometric integrity problems. |
| RO / AF / BE / FO | FunctionalDefect | These labels mainly reduce flow capacity or create blockage risk. |
| GR / PH / PB / OS / OP / OK | Excluded during source-set extraction | They describe connection or construction attributes rather than core defect classes. |
| VA | Metadata only | Water level is operating state, not a defect label. |

## Six-Class Alignment

| SewerML item | Aligned class | Reason |
| --- | --- | --- |
| ND | Normal | Explicit no-defect sample. |
| RB / OB / PF | WallDamage | These are direct wall-body damage or manufacturing defects on the pipe body. |
| FS / IS | JointAnomaly | These are interface abnormalities centered on joint alignment or intrusion. |
| DE | Deformation | Deformation is visually and mechanically distinct enough to keep separate. |
| AF / BE | DepositAttachment | Both represent sediment or attached material buildup along the pipe wall. |
| RO | Roots | Tree roots are operationally important and visually distinctive. |
| IN | Excluded during source-set extraction | Kept out of the main 6-class task until target-domain semantics are finalized. |
| FO | Excluded during source-set extraction | Obstacle is visually broad and unstable across domains, so it is better used as an ablation label. |
| GR / PH / PB / OS / OP / OK | Excluded during source-set extraction | They are connection or construction descriptors, not primary defect targets. |
| VA | Metadata only | Water level is operating state, not a defect label. |

## Output Layout

- Raw SewerML data stays under `data/sewerml`.
- The active source training dataset is extracted under `YOLOv11/datasets/sewerml_cls6_train3000`.
- Holdout labels are excluded during extraction instead of being materialized as extra folders.
