param(
    [string]$CamManifest = "",
    [string]$Output = "",
    [string]$Thresholds = "",
    [double]$DefaultThreshold = 0.45,
    [double]$MinAreaRatio = 0.001,
    [double]$MaxAreaRatio = 0.85,
    [int]$MaxBoxes = 1,
    [ValidateSet("hardlink", "copy")]
    [string]$Mode = "hardlink",
    [switch]$KeepNormal
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$backendFile = Join-Path $repoRoot ".uv-torch-backend"
if (-not (Test-Path $backendFile)) {
    throw "Torch backend marker not found. Run .\\scripts\\setup.ps1 first."
}

$backend = (Get-Content $backendFile -Raw).Trim()
$args = @(
    "run", "--project", $repoRoot, "--frozen", "--extra", $backend,
    "python", "scripts/cam_to_pseudobox.py"
)

if ($CamManifest) { $args += @("--cam-manifest", $CamManifest) }
if ($Output) { $args += @("--output", $Output) }
if ($Thresholds) { $args += @("--thresholds", $Thresholds) }
$args += @("--default-threshold", $DefaultThreshold)
$args += @("--min-area-ratio", $MinAreaRatio)
$args += @("--max-area-ratio", $MaxAreaRatio)
$args += @("--max-boxes", $MaxBoxes)
$args += @("--mode", $Mode)
if ($KeepNormal.IsPresent) { $args += "--keep-normal" }

& uv @args
