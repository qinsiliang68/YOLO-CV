# Readiness fix 08: real-image engineering canary

## Finding

The prior `test_real_yolo_integration.py` loaded the frozen YOLO11l checkpoint
but trained on all-zero/all-one synthetic tensors.  It did not prove that local
Sewer-ML image bytes could pass through the fixed-step runtime and canonical
Parquet/checkpoint/recovery/frontier publication path.

## Failure-first evidence

`RED_IMPORT.txt` records the first targeted collection failure: the test
contract existed before the engineering-canary module and failed with
`ModuleNotFoundError`。文件为 783 bytes，SHA-256
`2DB01A716A2696A08AEA715AC9ABCC95A681A0D24B5887C5A14F92910351DA13`。

## Fix

- Deterministically select only `train` and `normal_train` rows and bind every
  manifest/image byte count and SHA-256.
- Load the frozen local `yolo11l-cls.pt` through the repository-owned
  Ultralytics source and run one real base/replay forward/backward update.
- Require one optimizer update and one EMA update, with replay RNG and BatchNorm
  restoration checked by the production overlay.
- Publish real Zstd Parquet, a full state checkpoint, a 96-point FN frontier,
  one committed epoch transaction, and two injected recovery checks.
- Mark every output `ENGINEERING_CANARY_NOT_SCIENTIFIC_RESULT` and assert all
  formal side-effect flags remain false.

## Green evidence

`GREEN_BEHAVIOR.txt` records the targeted contract result: 3 tests passed；文件为
254 bytes，SHA-256
`788F402B9FBE68A111D51E17224084E8CD442F1867DB4702CE3B4EED14BC0DA3`。
The actual local GPU/image execution receipt is generated only after this
rollback unit is committed, so its source-tree identity refers to a clean
implementation commit rather than uncommitted canary code.

The first attempt to overwrite `RUNBOOK_MANIFEST_v1.json` was also deliberately
rejected by the immutable-publication contract.  The exact failure receipt is
preserved as `FAILED_IMMUTABLE_V1_REBUILD.json`; refreshed documents are bound
by a new v2 manifest instead of rewriting v1 history.
