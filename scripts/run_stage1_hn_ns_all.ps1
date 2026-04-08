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
$nConfig = Join-Path $repoRoot "YOLOv11\configs\runtime\stage1_formal_gate_hn_n_sweep.json"
$sConfig = Join-Path $repoRoot "YOLOv11\configs\runtime\stage1_formal_gate_hn_s_sweep.json"
$nManifest = Join-Path $repoRoot "research\materials\stage1_formal\gate_capacity\yolo11n_gate2_formal\best_epoch_manifest.json"
$sManifest = Join-Path $repoRoot "research\materials\stage1_formal\gate_capacity\yolo11s_gate2_formal\best_epoch_manifest.json"
$datasetRoot = Join-Path $repoRoot "YOLOv11\datasets\sewerml_gate2_train7200"
$splitCsv = Join-Path $repoRoot "research\materials\stage1_formal\manifests\val_cal_op_split.csv"

Require-Path -Path $entryConfig -Label "main entry config"
Require-Path -Path $nConfig -Label "n-sweep runtime config"
Require-Path -Path $sConfig -Label "s-sweep runtime config"
Require-Path -Path $nManifest -Label "yolo11n base manifest"
Require-Path -Path $sManifest -Label "yolo11s base manifest"
Require-Path -Path $datasetRoot -Label "gate2 dataset"
Require-Path -Path $splitCsv -Label "formal val-cal/op split"

$nMeta = Get-Content -LiteralPath $nManifest -Raw | ConvertFrom-Json
$sMeta = Get-Content -LiteralPath $sManifest -Raw | ConvertFrom-Json

Require-Path -Path $nMeta.checkpoint_path -Label "yolo11n base checkpoint"
Require-Path -Path $sMeta.checkpoint_path -Label "yolo11s base checkpoint"

Write-Host "[ok] repo root: $repoRoot"
Write-Host "[ok] yolo11n base checkpoint: $($nMeta.checkpoint_path)"
Write-Host "[ok] yolo11s base checkpoint: $($sMeta.checkpoint_path)"
Write-Host "[ok] dataset: $datasetRoot"
Write-Host "[ok] split csv: $splitCsv"

$command = @("uv", "run", "main.py", "--task", "stage1_formal_gate_hn_ns_all")
if ($DryRun) {
    $command += "--dry-run"
}
if ($Rerun) {
    $command += "--rerun"
}

Write-Host "[run] $($command -join ' ')"
& $command[0] $command[1..($command.Length - 1)]
