param(
    [string]$Config = "configs/runtime/predict_detect_struct6_reviewed.json",
    [string]$Model = "",
    [string]$Source = "",
    [string]$Device = "",
    [double]$Conf = -1,
    [double]$Iou = -1,
    [int]$Imgsz = -1,
    [string]$Name = "",
    [string]$Project = "",
    [switch]$SaveTxt,
    [switch]$SaveConf,
    [string[]]$Extra = @()
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$repoRoot = Split-Path -Parent $root
$backendFile = Join-Path $repoRoot ".uv-torch-backend"
if (-not (Test-Path $backendFile)) {
    throw "Torch backend marker not found. Run scripts/setup.ps1 first."
}

$backend = (Get-Content $backendFile -Raw).Trim()
$configPath = Join-Path "YOLOv11" $Config

$args = @(
    "run", "--project", $repoRoot, "--frozen", "--extra", $backend,
    "python", "scripts/run_yolo_task.py",
    "--action", "predict",
    "--config", $configPath
)

if ($Model) { $args += @("--model", $Model) }
if ($Source) { $args += @("--source", $Source) }
if ($Device) { $args += @("--device", $Device) }
if ($Conf -ge 0) { $args += @("--conf", $Conf) }
if ($Iou -ge 0) { $args += @("--iou", $Iou) }
if ($Imgsz -gt 0) { $args += @("--imgsz", $Imgsz) }
if ($Name) { $args += @("--name", $Name) }
if ($Project) { $args += @("--project", $Project) }
if ($SaveTxt.IsPresent) { $args += "--save-txt" }
if ($SaveConf.IsPresent) { $args += "--save-conf" }
foreach ($item in $Extra) {
    if (-not [string]::IsNullOrWhiteSpace($item)) {
        $args += @("--set", $item)
    }
}
& uv @args
