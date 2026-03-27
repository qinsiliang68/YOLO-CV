param(
    [string]$Config = "configs/runtime/val_detect_struct6_reviewed.json",
    [string]$Data = "",
    [string]$Model = "",
    [string]$Device = "",
    [string]$Split = "",
    [int]$Batch = -1,
    [int]$Imgsz = -1,
    [string]$Name = "",
    [string]$Project = "",
    [switch]$SaveJson,
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
    "--action", "val",
    "--config", $configPath
)

if ($Data) { $args += @("--data", $Data) }
if ($Model) { $args += @("--model", $Model) }
if ($Device) { $args += @("--device", $Device) }
if ($Split) { $args += @("--split", $Split) }
if ($Batch -gt 0) { $args += @("--batch", $Batch) }
if ($Imgsz -gt 0) { $args += @("--imgsz", $Imgsz) }
if ($Name) { $args += @("--name", $Name) }
if ($Project) { $args += @("--project", $Project) }
if ($SaveJson.IsPresent) { $args += "--save-json" }
foreach ($item in $Extra) {
    if (-not [string]::IsNullOrWhiteSpace($item)) {
        $args += @("--set", $item)
    }
}
& uv @args
