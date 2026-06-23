$ErrorActionPreference = "Stop"

$Repo = "C:\Users\ASUS\Desktop\ssh\AI\repos\YOLO-CV"
$Python = "C:\Users\ASUS\Desktop\ssh\AI\venvs\yolo-cv\Scripts\python.exe"
$Dataset = "C:\Users\ASUS\Desktop\ssh\AI\datasets\final_sewerml_dataset"
$SmokeRoot = "D:\ssh\AI\artifacts\stage1_phase1_hn_rn_smoke_node24_20260623_1ep_v2"
$WorkRoot = "C:\Users\ASUS\Desktop\ssh\AI\phase1_workdirs_c\stage1_phase1_hn_rn_smoke_node24_20260623_1ep_v2"
$RunsRoot = "D:\ssh\AI\runs\stage1_phase1_hn_rn_smoke_node24_20260623_1ep_v2"
$EvalRoot = "D:\ssh\AI\artifacts\stage1_phase1_hn_rn_smoke_node24_20260623_1ep_v2\eval"
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
New-Item -ItemType Directory -Force -Path (Split-Path $WorkRoot -Parent), $RunsRoot | Out-Null

Push-Location $Repo
try {
    if (-not (Test-Path -LiteralPath (Join-Path $SmokeRoot "run_matrix.csv"))) {
        & $Python scripts\build_stage1_phase1_hn_rn_manifests_20260623.py `
            --dataset-root $Dataset `
            --oof-predictions $Oof `
            --output-root $SmokeRoot `
            --replay-mode append `
            --max-q 20
        if ($LASTEXITCODE -ne 0) { throw "smoke manifest build failed" }
    }

    & $Python scripts\validate_stage1_phase1_hn_rn_manifests_20260623.py `
        --phase-root $SmokeRoot `
        --dataset-root $Dataset `
        --oof-predictions $Oof `
        --run-id HN-01 `
        --replay-mode append `
        --output-csv (Join-Path $SmokeRoot "validation_HN-01_pretrain.csv") `
        --output-json (Join-Path $SmokeRoot "validation_HN-01_pretrain.json")
    if ($LASTEXITCODE -ne 0) { throw "smoke pretrain validation failed" }

    & $Python scripts\run_stage1_phase1_hn_rn_pipeline_20260623.py `
        --phase-root $SmokeRoot `
        --dataset-root $Dataset `
        --oof-predictions $Oof `
        --work-root $WorkRoot `
        --runs-root $RunsRoot `
        --eval-root $EvalRoot `
        --run-id HN-01 `
        --epochs 1 `
        --batch 64 `
        --eval-batch 64 `
        --workers 4 `
        --train-device 0 `
        --eval-device 0 `
        --eval-limit-per-class 64 `
        --exist-ok
    if ($LASTEXITCODE -ne 0) { throw "smoke pipeline failed" }
}
finally {
    Pop-Location
}

Write-Output "SMOKE_OK root=$SmokeRoot runs=$RunsRoot eval=$EvalRoot work=$WorkRoot"
