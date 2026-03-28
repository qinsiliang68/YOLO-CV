param(
    [string]$Config = "YOLOv11/configs/runtime/cls_source_cls6.json",
    [string]$Data = "",
    [string]$Model = "",
    [string]$Device = "",
    [int]$Epochs = -1,
    [int]$Batch = -1,
    [int]$Imgsz = -1,
    [string]$Project = "",
    [string]$Name = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$backendFile = Join-Path $repoRoot ".uv-torch-backend"
if (-not (Test-Path $backendFile)) {
    throw "Torch backend marker not found. Run .\\scripts\\setup.ps1 first."
}

$backend = (Get-Content $backendFile -Raw).Trim()
$args = @("run", "--project", $repoRoot, "--frozen", "--extra", $backend, "python", "scripts/cls_pretrain.py", "--config", $Config)

if ($Data) { $args += @("--data", $Data) }
if ($Model) { $args += @("--model", $Model) }
if ($Device) { $args += @("--device", $Device) }
if ($Epochs -gt 0) { $args += @("--epochs", $Epochs) }
if ($Batch -gt 0) { $args += @("--batch", $Batch) }
if ($Imgsz -gt 0) { $args += @("--imgsz", $Imgsz) }
if ($Project) { $args += @("--project", $Project) }
if ($Name) { $args += @("--name", $Name) }
if ($DryRun.IsPresent) { $args += "--dry-run" }

& uv @args
