param(
    [switch]$DryRun,
    [switch]$Rerun
)

$ErrorActionPreference = "Stop"

function Require-Path {
    param(
        [string]$Path,
        [string]$Label
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "[missing] $Label -> $Path"
    }
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
Set-Location -LiteralPath $repoRoot

$entryConfig = Join-Path $repoRoot "YOLOv11\configs\runtime\main_entry.json"
$runtimeConfig = Join-Path $repoRoot "YOLOv11\configs\runtime\stage1_formal_gate_hn_x_sweep.json"
$manifest = Join-Path $repoRoot "research\materials\stage1_formal\gate_capacity\yolo11x_gate2_formal\best_epoch_manifest.json"
$datasetRoot = Join-Path $repoRoot "YOLOv11\datasets\sewerml_gate2_train7200"
$splitCsv = Join-Path $repoRoot "research\materials\stage1_formal\manifests\val_cal_op_split.csv"

Require-Path -Path $entryConfig -Label "main entry config"
Require-Path -Path $runtimeConfig -Label "x-sweep runtime config"
Require-Path -Path $manifest -Label "yolo11x base manifest"
Require-Path -Path $datasetRoot -Label "gate2 dataset"
Require-Path -Path $splitCsv -Label "formal val-cal/op split"

$meta = Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json
Require-Path -Path $meta.checkpoint_path -Label "yolo11x base checkpoint"

Write-Host "[ok] repo root: $repoRoot"
Write-Host "[ok] yolo11x base checkpoint: $($meta.checkpoint_path)"
Write-Host "[ok] dataset: $datasetRoot"
Write-Host "[ok] split csv: $splitCsv"

$command = @("uv", "run", "main.py", "--task", "stage1_formal_gate_hn_x_sweep")
if ($DryRun) {
    $command += "--dry-run"
}
if ($Rerun) {
    $command += "--rerun"
}

Write-Host "[run] $($command -join ' ')"
& $command[0] $command[1..($command.Length - 1)]
