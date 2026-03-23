param(
    [string]$Config = "configs/runtime/train_detect.json",
    [string]$Data = "",
    [string]$Model = "",
    [string]$Device = "",
    [int]$Epochs = -1,
    [int]$Batch = -1,
    [int]$Imgsz = -1,
    [string]$Name = "",
    [string]$Project = "",
    [string[]]$Extra = @()
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$yolo = Join-Path $root ".venv\Scripts\yolo.exe"
if (-not (Test-Path $yolo)) {
    throw "Virtual environment not found. Run scripts/setup.ps1 first."
}

$configPath = Join-Path $root $Config
if (-not (Test-Path $configPath)) {
    throw "Config file not found: $configPath"
}

$cfg = Get-Content $configPath -Raw | ConvertFrom-Json

function Resolve-YoloValue([string]$Value, [switch]$Absolute) {
    if ([string]::IsNullOrWhiteSpace($Value)) {
        return $null
    }

    $candidate = Join-Path $root $Value
    if (Test-Path $candidate) {
        return (Resolve-Path $candidate).Path
    }

    if ($Absolute) {
        return (Join-Path $root $Value)
    }

    return $Value
}

function Add-YoloArg([ref]$ArgsRef, [string]$Key, $Value) {
    if ($null -eq $Value) {
        return
    }

    if ($Value -is [string] -and [string]::IsNullOrWhiteSpace($Value)) {
        return
    }

    if ($Value -is [bool]) {
        $ArgsRef.Value += "$Key=$($Value.ToString().ToLower())"
        return
    }

    $ArgsRef.Value += "$Key=$Value"
}

$args = @($cfg.task, "train")

$modelValue = if ($Model) { $Model } else { $cfg.model }
$dataValue = if ($Data) { $Data } else { $cfg.data }
$projectValue = if ($Project) { $Project } else { $cfg.project }
$nameValue = if ($Name) { $Name } else { $cfg.name }
$deviceValue = if ($Device) { $Device } else { $cfg.device }
$epochsValue = if ($Epochs -gt 0) { $Epochs } else { $cfg.epochs }
$batchValue = if ($Batch -gt 0) { $Batch } else { $cfg.batch }
$imgszValue = if ($Imgsz -gt 0) { $Imgsz } else { $cfg.imgsz }

Add-YoloArg ([ref]$args) "model" (Resolve-YoloValue $modelValue)
Add-YoloArg ([ref]$args) "data" (Resolve-YoloValue $dataValue)
Add-YoloArg ([ref]$args) "epochs" $epochsValue
Add-YoloArg ([ref]$args) "imgsz" $imgszValue
Add-YoloArg ([ref]$args) "batch" $batchValue
Add-YoloArg ([ref]$args) "device" $deviceValue
Add-YoloArg ([ref]$args) "workers" $cfg.workers
Add-YoloArg ([ref]$args) "project" (Resolve-YoloValue $projectValue -Absolute)
Add-YoloArg ([ref]$args) "name" $nameValue
Add-YoloArg ([ref]$args) "pretrained" $cfg.pretrained
Add-YoloArg ([ref]$args) "patience" $cfg.patience
Add-YoloArg ([ref]$args) "optimizer" $cfg.optimizer
Add-YoloArg ([ref]$args) "cache" $cfg.cache
Add-YoloArg ([ref]$args) "resume" $cfg.resume

foreach ($item in $Extra) {
    if (-not [string]::IsNullOrWhiteSpace($item)) {
        $args += $item
    }
}

& $yolo @args
