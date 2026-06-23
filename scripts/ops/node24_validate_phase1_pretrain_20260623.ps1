$ErrorActionPreference = "Stop"

$Repo = "C:\Users\ASUS\Desktop\ssh\AI\repos\YOLO-CV"
$Python = "C:\Users\ASUS\Desktop\ssh\AI\venvs\yolo-cv\Scripts\python.exe"
$Dataset = "C:\Users\ASUS\Desktop\ssh\AI\datasets\final_sewerml_dataset"
$PhaseRoot = "D:\ssh\AI\artifacts\stage1_phase1_hn_rn_20260623"
$Oof = Join-Path $Repo "artifacts\stage1_oof_predictions_calop_20260621\merged_10fold_20260622\oof_predictions_merged.csv"

Push-Location $Repo
try {
    & $Python scripts\validate_stage1_phase1_hn_rn_manifests_20260623.py `
        --phase-root $PhaseRoot `
        --dataset-root $Dataset `
        --oof-predictions $Oof `
        --replay-mode append `
        --output-csv (Join-Path $PhaseRoot "validation_summary_pretrain_after_sync.csv") `
        --output-json (Join-Path $PhaseRoot "validation_summary_pretrain_after_sync.json")
    if ($LASTEXITCODE -ne 0) {
        throw "pretrain validation failed"
    }
}
finally {
    Pop-Location
}
Write-Output "PRETRAIN_VALIDATION_OK"
