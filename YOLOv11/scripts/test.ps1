param(
    [string]$Config = "configs/runtime/val_detect_struct6_reviewed.json",
    [string]$Data = "",
    [string]$Model = "",
    [string]$Device = "",
    [int]$Batch = -1,
    [int]$Imgsz = -1,
    [string]$Project = "runs/test",
    [string]$Name = "exp",
    [switch]$SaveJson,
    [string[]]$Extra = @()
)

$params = @{
    Config = $Config
    Data = $Data
    Model = $Model
    Device = $Device
    Split = "test"
    Batch = $Batch
    Imgsz = $Imgsz
    Project = $Project
    Name = $Name
    Extra = $Extra
}

if ($SaveJson.IsPresent) {
    $params.SaveJson = $true
}

& (Join-Path $PSScriptRoot "val.ps1") @params
