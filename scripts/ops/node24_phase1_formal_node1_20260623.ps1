$ErrorActionPreference = "Stop"

$Repo = "C:\Users\ASUS\Desktop\ssh\AI\repos\YOLO-CV"
$Python = "C:\Users\ASUS\Desktop\ssh\AI\venvs\yolo-cv\Scripts\python.exe"
$Dataset = "C:\Users\ASUS\Desktop\ssh\AI\datasets\final_sewerml_dataset"
$PhaseRoot = "D:\ssh\AI\artifacts\stage1_phase1_hn_rn_20260623"
$WorkRoot = "C:\Users\ASUS\Desktop\ssh\AI\phase1_workdirs_c\stage1_phase1_hn_rn_20260623"
$RunsRoot = "D:\ssh\AI\runs\stage1_phase1_hn_rn_20260623"
$EvalRoot = "D:\ssh\AI\artifacts\stage1_phase1_hn_rn_20260623\eval"
$Oof = Join-Path $Repo "artifacts\stage1_oof_predictions_calop_20260621\merged_10fold_20260622\oof_predictions_merged.csv"
$ValidationCsv = Join-Path $PhaseRoot "validation_summary_pretrain_node24_formal_launch.csv"
$ValidationJson = Join-Path $PhaseRoot "validation_summary_pretrain_node24_formal_launch.json"

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

function Assert-NonCPathText {
    param([string]$Path, [string]$Label)
    $root = [IO.Path]::GetPathRoot([IO.Path]::GetFullPath($Path)).ToUpperInvariant()
    if ($root -eq "C:\") {
        throw "${Label} must not be on C: $Path"
    }
}

function Assert-FreeSpaceGb {
    param([string]$Drive, [double]$MinimumGb)
    $disk = Get-CimInstance Win32_LogicalDisk -Filter ("DeviceID='{0}:'" -f $Drive.TrimEnd(":"))
    if (-not $disk) { throw "Missing drive ${Drive}" }
    $freeGb = [math]::Round($disk.FreeSpace / 1GB, 2)
    Write-Output ("disk_{0}_freeGB={1}" -f $Drive.TrimEnd(":"), $freeGb)
    if ($freeGb -lt $MinimumGb) {
        throw "Drive ${Drive} free space too low: ${freeGb}GB < ${MinimumGb}GB"
    }
}

function Assert-NoGpuComputeProcess {
    $lines = @(& nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>$null)
    $busy = @($lines | Where-Object { $_ -and $_.Trim().Length -gt 0 })
    if ($busy.Count -gt 0) {
        throw "GPU compute process exists: $($busy -join '; ')"
    }
}

Assert-Path $Repo "repo"
Assert-Path $Python "python"
Assert-Path $Dataset "dataset"
Assert-Path $Oof "oof predictions"
Assert-Path (Join-Path $Repo "yolo11l-cls.pt") "yolo11l weight"
Assert-Path (Join-Path $PhaseRoot "run_matrix.csv") "formal run matrix"

Assert-RealCPath $Dataset "dataset"
New-Item -ItemType Directory -Force -Path (Split-Path $WorkRoot -Parent), $RunsRoot, $EvalRoot | Out-Null
Assert-RealCPath (Split-Path $WorkRoot -Parent) "work root parent"
Assert-NonCPathText $PhaseRoot "phase root"
Assert-NonCPathText $RunsRoot "runs root"
Assert-NonCPathText $EvalRoot "eval root"
Assert-FreeSpaceGb C 40
Assert-FreeSpaceGb D 80
Assert-NoGpuComputeProcess

Push-Location $Repo
try {
    & $Python scripts\validate_stage1_phase1_hn_rn_manifests_20260623.py `
        --phase-root $PhaseRoot `
        --dataset-root $Dataset `
        --oof-predictions $Oof `
        --replay-mode append `
        --output-csv $ValidationCsv `
        --output-json $ValidationJson
    if ($LASTEXITCODE -ne 0) { throw "formal pretrain validation failed" }

    & $Python scripts\run_stage1_phase1_hn_rn_pipeline_20260623.py `
        --phase-root $PhaseRoot `
        --dataset-root $Dataset `
        --oof-predictions $Oof `
        --work-root $WorkRoot `
        --runs-root $RunsRoot `
        --eval-root $EvalRoot `
        --node-index 1 `
        --epochs 200 `
        --batch 128 `
        --eval-batch 64 `
        --workers 4 `
        --train-device 0 `
        --eval-device 0
    if ($LASTEXITCODE -ne 0) { throw "formal node24 node-index 1 pipeline failed" }
}
finally {
    Pop-Location
}

Write-Output "FORMAL_NODE24_NODE1_OK phase=$PhaseRoot runs=$RunsRoot eval=$EvalRoot work=$WorkRoot"
