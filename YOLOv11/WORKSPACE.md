# YOLOv11 Research Workspace

This folder has been prepared as a self-contained YOLOv11 workspace for your later experiments.

## Quick Start

```powershell
.\scripts\setup.ps1
.\scripts\train.ps1
.\scripts\val.ps1
.\scripts\test.ps1
.\scripts\predict.ps1
```

## What To Edit First

1. Put your dataset under `datasets/`
2. Create your own dataset YAML under `configs/datasets/`
3. Pass `-Data` or adjust defaults in `configs/runtime/*.json`
