param(
    [string]$Config = "configs/runtime/predict_detect.json",
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

$args = @($cfg.task, "predict")

$modelValue = if ($Model) { $Model } else { $cfg.model }
$sourceValue = if ($Source) { $Source } else { $cfg.source }
$projectValue = if ($Project) { $Project } else { $cfg.project }
$nameValue = if ($Name) { $Name } else { $cfg.name }
$deviceValue = if ($Device) { $Device } else { $cfg.device }
$confValue = if ($Conf -ge 0) { $Conf } else { $cfg.conf }
$iouValue = if ($Iou -ge 0) { $Iou } else { $cfg.iou }
$imgszValue = if ($Imgsz -gt 0) { $Imgsz } else { $cfg.imgsz }
$saveTxtValue = if ($SaveTxt.IsPresent) { $true } else { $cfg.save_txt }
$saveConfValue = if ($SaveConf.IsPresent) { $true } else { $cfg.save_conf }

Add-YoloArg ([ref]$args) "model" (Resolve-YoloValue $modelValue)
Add-YoloArg ([ref]$args) "source" (Resolve-YoloValue $sourceValue)
Add-YoloArg ([ref]$args) "imgsz" $imgszValue
Add-YoloArg ([ref]$args) "conf" $confValue
Add-YoloArg ([ref]$args) "iou" $iouValue
Add-YoloArg ([ref]$args) "device" $deviceValue
Add-YoloArg ([ref]$args) "project" (Resolve-YoloValue $projectValue -Absolute)
Add-YoloArg ([ref]$args) "name" $nameValue
Add-YoloArg ([ref]$args) "save" $cfg.save
Add-YoloArg ([ref]$args) "save_txt" $saveTxtValue
Add-YoloArg ([ref]$args) "save_conf" $saveConfValue

foreach ($item in $Extra) {
    if (-not [string]::IsNullOrWhiteSpace($item)) {
        $args += $item
    }
}

& $yolo @args
