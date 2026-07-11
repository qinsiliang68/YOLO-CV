# Runtime integrity v1.2

`RUNTIME_CONTRACT_v1_2.yaml` binds the immutable v1.1 science contract to the
frozen 240-row experiment matrix, 240-row selection index, all indexed selection
SHA-256 values, and the exact base checkpoint. It does not regenerate rankings
or selections.

Before distributing jobs, verify every frozen selection:

```powershell
uv run python scripts/stage1_gapvalue240/runtime_integrity.py all-selections --repo-root .
```

Each machine must create one durable asset report before formal training. The
default `existence` mode reads all eight manifests, proves their 384,000 sample
identities are disjoint, checks labels/splits, and checks every image exists.
Use `sha256` when a full content digest is required; it is intentionally slower.

```powershell
uv run python scripts/stage1_gapvalue240/runtime_integrity.py build-machine-assets `
  --machine-config configs/stage1_gapvalue240/machines/machine_01.yaml `
  --output D:/gapvalue240_outputs/machine_01/machine_asset_report.json `
  --image-verification existence
```

Formal preflight should call `validate_machine_asset_report(...)`. That cached
validation verifies the report snapshot and runtime identities without rescanning
384,000 images. Generate a new report whenever a manifest, dataset root,
checkpoint, runtime contract, matrix, or selection index changes.

Release validation requires Git tag `stage1-gapvalue240-runtime-v1.2.0` to point
to the current `HEAD`. The Python API has an explicit test-only override; the CLI
does not expose it. Dry runs have status `DRY_RUN_VALIDATED`, which is not in the
contract's aggregatable status list.
