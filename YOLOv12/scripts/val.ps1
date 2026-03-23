param(
    [string]$Config = "configs/runtime/val_detect.json",
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

$args = @($cfg.task, "val")

$modelValue = if ($Model) { $Model } else { $cfg.model }
$dataValue = if ($Data) { $Data } else { $cfg.data }
$projectValue = if ($Project) { $Project } else { $cfg.project }
$nameValue = if ($Name) { $Name } else { $cfg.name }
$deviceValue = if ($Device) { $Device } else { $cfg.device }
$splitValue = if ($Split) { $Split } else { $cfg.split }
$batchValue = if ($Batch -gt 0) { $Batch } else { $cfg.batch }
$imgszValue = if ($Imgsz -gt 0) { $Imgsz } else { $cfg.imgsz }
$saveJsonValue = if ($SaveJson.IsPresent) { $true } else { $cfg.save_json }

if ([string]::IsNullOrWhiteSpace($dataValue)) {
    throw "No dataset YAML configured. Create your own file under configs\datasets\ and pass -Data or update configs/runtime/val_detect.json."
}

Add-YoloArg ([ref]$args) "model" (Resolve-YoloValue $modelValue)
Add-YoloArg ([ref]$args) "data" (Resolve-YoloValue $dataValue)
Add-YoloArg ([ref]$args) "split" $splitValue
Add-YoloArg ([ref]$args) "batch" $batchValue
Add-YoloArg ([ref]$args) "imgsz" $imgszValue
Add-YoloArg ([ref]$args) "device" $deviceValue
Add-YoloArg ([ref]$args) "project" (Resolve-YoloValue $projectValue -Absolute)
Add-YoloArg ([ref]$args) "name" $nameValue
Add-YoloArg ([ref]$args) "save_json" $saveJsonValue

foreach ($item in $Extra) {
    if (-not [string]::IsNullOrWhiteSpace($item)) {
        $args += $item
    }
}

& $yolo @args
