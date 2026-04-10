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

$configPath = Join-Path $repoRoot "YOLOv11\configs\runtime\stage1_formal_gate_info_sampling_lite_hn14_teacher.json"
$teacherManifest = Join-Path $repoRoot "research\materials\stage1_formal\gate_hn_m_sweep\hn14\best_epoch_manifest.json"
$datasetRoot = Join-Path $repoRoot "YOLOv11\datasets\sewerml_gate2_train7200"
$poolOutputDir = Join-Path $repoRoot "research\materials\stage1_formal\gate_hn_assets\yolo11m_hn14_train_normal_scores"

Require-Path -Path $configPath -Label "hn14-teacher lite config"
Require-Path -Path $teacherManifest -Label "hn14 teacher best_epoch_manifest"
Require-Path -Path $datasetRoot -Label "gate2 dataset"

$teacherMeta = Get-Content -LiteralPath $teacherManifest -Raw | ConvertFrom-Json
# Resolve checkpoint: prefer repo-relative checkpoint_path_relative, fall back to absolute checkpoint_path
if ($teacherMeta.PSObject.Properties["checkpoint_path_relative"]) {
    $teacherCheckpoint = Join-Path $repoRoot ([string]$teacherMeta.checkpoint_path_relative)
} else {
    $teacherCheckpoint = [string]$teacherMeta.checkpoint_path
}
Require-Path -Path $teacherCheckpoint -Label "hn14 teacher checkpoint"

Write-Host "[ok] repo root: $repoRoot"
Write-Host "[ok] hn14 teacher checkpoint: $teacherCheckpoint"
Write-Host "[ok] dataset: $datasetRoot"
Write-Host "[ok] pool output dir: $poolOutputDir"

if ($Rerun -and (Test-Path -LiteralPath $poolOutputDir)) {
    Write-Host "[clean] removing previous hn14 pool output: $poolOutputDir"
    Remove-Item -LiteralPath $poolOutputDir -Recurse -Force
}

$scoreCmd = @(
    "uv", "run", "scripts/stage1_score_train_normals.py",
    "--weights", $teacherCheckpoint,
    "--data-root", $datasetRoot,
    "--output-dir", $poolOutputDir,
    "--device", "0",
    "--imgsz", "640",
    "--batch", "1",
    "--chunk-size", "16",
    "--top-k", "250"
)

$liteCmd = @(
    "uv", "run", "scripts/stage1_formal_gate_info_sampling_lite.py",
    "--config", $configPath
)

if ($Rerun) {
    $liteCmd += "--rerun"
}

Write-Host "[run] $($scoreCmd -join ' ')"
if (-not $DryRun) {
    & $scoreCmd[0] $scoreCmd[1..($scoreCmd.Length - 1)]
}

Write-Host "[run] $($liteCmd -join ' ')"
if (-not $DryRun) {
    & $liteCmd[0] $liteCmd[1..($liteCmd.Length - 1)]
}
