param(
    [string]$Weights = "",
    [string]$Source = "",
    [string]$Output = "",
    [string]$LabelManifest = "",
    [string]$Device = "",
    [int]$Imgsz = 224,
    [double]$CropFraction = 1.0,
    [double]$Alpha = 0.45,
    [int]$Limit = 0,
    [switch]$SaveNpy,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$backendFile = Join-Path $repoRoot ".uv-torch-backend"
if (-not (Test-Path $backendFile)) {
    throw "Torch backend marker not found. Run .\\scripts\\setup.ps1 first."
}

$backend = (Get-Content $backendFile -Raw).Trim()
$args = @("run", "--project", $repoRoot, "--frozen", "--extra", $backend, "python", "scripts/export_cam.py")

if ($Weights) { $args += @("--weights", $Weights) }
if ($Source) { $args += @("--source", $Source) }
if ($Output) { $args += @("--output", $Output) }
if ($LabelManifest) { $args += @("--label-manifest", $LabelManifest) }
if ($Device) { $args += @("--device", $Device) }
if ($Imgsz -gt 0) { $args += @("--imgsz", $Imgsz) }
if ($CropFraction -gt 0) { $args += @("--crop-fraction", $CropFraction) }
if ($Alpha -gt 0) { $args += @("--alpha", $Alpha) }
if ($Limit -gt 0) { $args += @("--limit", $Limit) }
if ($SaveNpy.IsPresent) { $args += "--save-npy" }
if ($DryRun.IsPresent) { $args += "--dry-run" }

& uv @args
