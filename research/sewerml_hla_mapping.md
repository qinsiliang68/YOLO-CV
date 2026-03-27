# SewerML Hierarchical Label Alignment

This note records the source-to-target label alignment used for cross-domain transfer experiments in this repository.

## Three-Class Alignment

| SewerML item | Aligned class | Reason |
| --- | --- | --- |
| ND | Normal | Explicit no-defect sample. |
| RB / OB / PF / DE / FS / IS / IN | StructuralDefect | These labels describe wall, joint, leakage, or geometric integrity problems. |
| RO / AF / BE / FO | FunctionalDefect | These labels mainly reduce flow capacity or create blockage risk. |
| GR / PH / PB / OS / OP / OK | Holdout in `data/sewerml/research_holdout/cls3` | They describe connection or construction attributes rather than core defect classes. |
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
| IN | Holdout in `data/sewerml/research_holdout/cls6/IN` | Kept out of the main 6-class task until target-domain semantics are finalized. |
| FO | Holdout in `data/sewerml/research_holdout/cls6/FO` | Obstacle is visually broad and unstable across domains, so it is better used as an ablation label. |
| GR / PH / PB / OS / OP / OK | Holdout in `data/sewerml/research_holdout/cls6/ConstructionInfo` | They are connection or construction descriptors, not primary defect targets. |
| VA | Metadata only | Water level is operating state, not a defect label. |

## Output Layout

- Main aligned datasets are generated under `YOLOv11/datasets/sewerml_hla_cls3` and `YOLOv11/datasets/sewerml_hla_cls6`.
- Raw SewerML data stays under `data/sewerml`.
- Labels kept out of the main task are collected under `data/sewerml/research_holdout`.
