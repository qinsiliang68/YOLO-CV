$ErrorActionPreference = "Stop"

$Repo = "C:\Users\ASUS\Desktop\ssh\AI\repos\YOLO-CV"
$Python = "C:\Users\ASUS\Desktop\ssh\AI\venvs\yolo-cv\Scripts\python.exe"
$Dataset = "C:\Users\ASUS\Desktop\ssh\AI\datasets\final_sewerml_dataset"
$PhaseRoot = "D:\ssh\AI\artifacts\stage1_phase1_hn_rn_20260623"
$WorkRoot = "C:\Users\ASUS\Desktop\ssh\AI\phase1_workdirs_c\stage1_phase1_hn_rn_20260623"
$Oof = Join-Path $Repo "artifacts\stage1_oof_predictions_calop_20260621\merged_10fold_20260622\oof_predictions_merged.csv"

function Assert-Path {
    param([string]$Path, [string]$Label)
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing ${Label}: $Path"
    }
}

Assert-Path $Repo "repo"
Assert-Path $Python "python"
Assert-Path $Dataset "dataset"
Assert-Path $Oof "oof predictions"
Assert-Path (Join-Path $Repo "yolo11l-cls.pt") "yolo11l weight"

$datasetItem = Get-Item -LiteralPath $Dataset
if ([bool]($datasetItem.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
    throw "Dataset path is a reparse point: $Dataset"
}
if ([IO.Path]::GetPathRoot($datasetItem.FullName).ToUpperInvariant() -ne "C:\") {
    throw "Dataset is not on C: $($datasetItem.FullName)"
}

New-Item -ItemType Directory -Force -Path (Split-Path $WorkRoot -Parent), $PhaseRoot | Out-Null
$workParent = Get-Item -LiteralPath (Split-Path $WorkRoot -Parent)
if ([bool]($workParent.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
    throw "Work root parent is a reparse point: $($workParent.FullName)"
}
if ([IO.Path]::GetPathRoot($workParent.FullName).ToUpperInvariant() -ne "C:\") {
    throw "Work root parent is not on C: $($workParent.FullName)"
}

Push-Location $Repo
try {
    & $Python -m py_compile `
        scripts\build_stage1_phase1_hn_rn_manifests_20260623.py `
        scripts\validate_stage1_phase1_hn_rn_manifests_20260623.py `
        scripts\run_stage1_phase1_hn_rn_pipeline_20260623.py `
        scripts\train_stage1_cls_sweep.py `
        scripts\evaluate_stage1_cls_gate.py
    if ($LASTEXITCODE -ne 0) { throw "py_compile failed" }

    & $Python scripts\build_stage1_phase1_hn_rn_manifests_20260623.py `
        --dataset-root $Dataset `
        --oof-predictions $Oof `
        --output-root $PhaseRoot `
        --replay-mode append `
        --max-q 20
    if ($LASTEXITCODE -ne 0) { throw "manifest build failed" }

    & $Python scripts\validate_stage1_phase1_hn_rn_manifests_20260623.py `
        --phase-root $PhaseRoot `
        --dataset-root $Dataset `
        --oof-predictions $Oof `
        --replay-mode append `
        --output-csv (Join-Path $PhaseRoot "validation_summary_pretrain.csv") `
        --output-json (Join-Path $PhaseRoot "validation_summary_pretrain.json")
    if ($LASTEXITCODE -ne 0) { throw "manifest validation failed" }
}
finally {
    Pop-Location
}

Write-Output "PHASE1_PREPARE_OK phase_root=$PhaseRoot work_root=$WorkRoot"
