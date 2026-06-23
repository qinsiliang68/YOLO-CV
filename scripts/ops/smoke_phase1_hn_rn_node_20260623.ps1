param(
    [Parameter(Mandatory = $true)][string]$NodeLabel,
    [string]$RunId = "HN-01"
)

$ErrorActionPreference = "Stop"

$Repo = "C:\Users\ASUS\Desktop\ssh\AI\repos\YOLO-CV"
$Python = "C:\Users\ASUS\Desktop\ssh\AI\venvs\yolo-cv\Scripts\python.exe"
$Dataset = "C:\Users\ASUS\Desktop\ssh\AI\datasets\final_sewerml_dataset"
$SmokeRoot = "D:\ssh\AI\artifacts\stage1_phase1_hn_rn_smoke_${NodeLabel}_20260623_1ep"
$WorkRoot = "C:\Users\ASUS\Desktop\ssh\AI\phase1_workdirs_c\stage1_phase1_hn_rn_smoke_${NodeLabel}_20260623_1ep"
$RunsRoot = "D:\ssh\AI\runs\stage1_phase1_hn_rn_smoke_${NodeLabel}_20260623_1ep"
$EvalRoot = Join-Path $SmokeRoot "eval"
$Oof = Join-Path $Repo "artifacts\stage1_oof_predictions_calop_20260621\merged_10fold_20260622\oof_predictions_merged.csv"

function Assert-Path {
    param([string]$Path, [string]$Label)
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing ${Label}: $Path"
    }
}

function Assert-RealCPath {
    param([string]$Path, [string]$Label)
    $item = Get-Item -LiteralPath $Path
    if ([bool]($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "${Label} is a reparse point: $Path"
    }
    if ([IO.Path]::GetPathRoot($item.FullName).ToUpperInvariant() -ne "C:\") {
        throw "${Label} is not on C: $($item.FullName)"
    }
}

Assert-Path $Repo "repo"
Assert-Path $Python "python"
Assert-Path $Dataset "dataset"
Assert-Path $Oof "oof predictions"
Assert-Path (Join-Path $Repo "yolo11l-cls.pt") "yolo11l weight"
Assert-RealCPath $Dataset "dataset"
New-Item -ItemType Directory -Force -Path (Split-Path $WorkRoot -Parent), $RunsRoot | Out-Null
Assert-RealCPath (Split-Path $WorkRoot -Parent) "work root parent"

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
        --run-id $RunId `
        --replay-mode append `
        --output-csv (Join-Path $SmokeRoot "validation_${RunId}_pretrain.csv") `
        --output-json (Join-Path $SmokeRoot "validation_${RunId}_pretrain.json")
    if ($LASTEXITCODE -ne 0) { throw "smoke pretrain validation failed" }

    & $Python scripts\run_stage1_phase1_hn_rn_pipeline_20260623.py `
        --phase-root $SmokeRoot `
        --dataset-root $Dataset `
        --oof-predictions $Oof `
        --work-root $WorkRoot `
        --runs-root $RunsRoot `
        --eval-root $EvalRoot `
        --run-id $RunId `
        --epochs 1 `
        --batch 64 `
        --eval-batch 64 `
        --workers 4 `
        --train-device 0 `
        --eval-device 0 `
        --eval-limit-per-class 64 `
        --exist-ok
    if ($LASTEXITCODE -ne 0) { throw "smoke pipeline failed" }

    & $Python scripts\verify_stage1_phase1_hn_rn_outputs_20260623.py `
        --phase-root $SmokeRoot `
        --dataset-root $Dataset `
        --run-id $RunId `
        --allow-nonformal `
        --output-json (Join-Path $SmokeRoot "verification_${RunId}.json")
    if ($LASTEXITCODE -ne 0) { throw "smoke verifier failed" }
}
finally {
    Pop-Location
}

Write-Output "SMOKE_NODE_OK node_label=$NodeLabel run_id=$RunId root=$SmokeRoot runs=$RunsRoot eval=$EvalRoot work=$WorkRoot"
